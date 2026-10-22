"""
tracker.py
==========

Sistemin nesne takibi (multi-object tracking) katmani.

Takip algoritmasi olarak **BoT-SORT** kullanilir. Bu modul, `detector.py`
tarafindan uretilen tespitleri alir ve nesneleri frame'den frame'e
iliskilendirerek her nesneye gecici bir **Track ID** atar.

--------------------------------------------------------------------
ONEMLI NOT - TRACK ID KALICI KIMLIK DEGILDIR
--------------------------------------------------------------------
BoT-SORT yalnizca KISA SURELI takip icin kullanilir.

* Track ID'ler yalnizca aktif takip sureci boyunca gecerlidir.
* Bir nesne uzun sure goruntuden ciktiktan sonra tekrar gorunurse
  BoT-SORT ona YENI bir Track ID atayabilir.
* Track ID hicbir zaman kalici kimlik veya sahiplik bilgisi olarak
  kullanilmamalidir.

Kalici kimlik (Person UUID / Item UUID) Identity Layer'in
(`reid.py` + `association_memory.py`) sorumlulugundadir ve Track ID ile
karistirilmamalidir.
--------------------------------------------------------------------

ULTRALYTICS SURUM UYUMLULUGU
--------------------------------------------------------------------
ultralytics 8.4 ile takipci API'si degisti:
* `BOTSORT(args, frame_rate=...)` -> `BOTSORT(args)`; TRACK_BUFFER artik
  dogrudan kare sayisidir (eskiden frame_rate/30 ile olceklenirdi).
* Takipci girdiyi boolean maske ile dilimler (`results[mask]`).
* `yaml_load` kaldirildi, yerine `YAML.load` geldi.
Bu modul her iki API ile de calisacak sekilde yazilmistir.
--------------------------------------------------------------------

Tum yapilandirma degerleri `.env` dosyasindan okunur; kod icerisinde
sabit deger (magic number) kullanilmaz.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from ultralytics.trackers import BOTSORT
from ultralytics.utils import IterableSimpleNamespace

from src.config import Config, load_config
from src.detector import Detection

logger = logging.getLogger(__name__)

SUPPORTED_TRACKER = "botsort"

# Bounding box koordinat sayisi (x1, y1, x2, y2) - yapisal sabit
_BBOX_DIMS = 4


# =====================================================================
#  TAKIP VERI YAPISI
# =====================================================================


@dataclass
class Track:
    """Takip edilen tek bir nesnenin standart cikti formati.

    Diger moduller (association_memory, visualization) bu formati kullanir.

    UYARI: `track_id` GECICI bir kimliktir. Kalici kimlik icin Identity
    Layer tarafindan uretilen UUID kullanilmalidir.
    """

    track_id: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    detection_index: int  # Girdi tespit listesindeki karsiligin indeksi

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        return self.bbox

    def to_int_bbox(self) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.bbox
        return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

    def to_dict(self) -> Dict:
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "detection_index": self.detection_index,
        }


class _DetectionAdapter:
    """Detection listesini BoT-SORT'un bekledigi girdi formatina cevirir.

    ultralytics takipcileri `.conf`, `.cls`, `.xywh` alanlarina sahip ve
    boolean maske / indeks dizisi ile dilimlenebilen (`results[mask]`)
    bir nesne bekler. Dilimleme, ultralytics 8.4 ile ZORUNLU hale geldi.

    NOT: `xywhr` alani TANIMLANMAMALIDIR; tanimlanirsa ultralytics girdiyi
    donuk (obb) kutu olarak yorumlar.
    """

    __slots__ = ("conf", "cls", "xyxy", "xywh")

    def __init__(self, detections: Sequence[Detection] = ()) -> None:
        if not detections:
            self._set(
                np.empty((0, _BBOX_DIMS), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
            return
        self._set(
            np.array([d.bbox for d in detections], dtype=np.float32).reshape(-1, _BBOX_DIMS),
            np.array([d.confidence for d in detections], dtype=np.float32),
            np.array([d.class_id for d in detections], dtype=np.float32),
        )

    def _set(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls
        self.xywh = self._xyxy_to_xywh(xyxy)

    @staticmethod
    def _xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
        """(x1, y1, x2, y2) -> (merkez_x, merkez_y, genislik, yukseklik)"""
        xywh = np.empty_like(xyxy)
        xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) * 0.5
        xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) * 0.5
        xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
        xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
        return xywh

    def __getitem__(self, index) -> "_DetectionAdapter":
        """Boolean maske veya indeks dizisi ile alt kume dondurur."""
        subset = _DetectionAdapter.__new__(_DetectionAdapter)
        subset._set(
            self.xyxy[index].reshape(-1, _BBOX_DIMS),
            np.atleast_1d(self.conf[index]),
            np.atleast_1d(self.cls[index]),
        )
        return subset

    def __len__(self) -> int:
        return len(self.conf)


# =====================================================================
#  TRACKER
# =====================================================================


class Tracker:
    """BoT-SORT tabanli cok nesneli takip katmani.

    Kullanim:
        tracker = Tracker(class_names=detector.class_names)
        tracks = tracker.update(detections, frame)
    """

    # BoT-SORT sonuc satirindaki sutun indeksleri:
    # [x1, y1, x2, y2, track_id, score, cls, det_index]
    _COL_TRACK_ID = 4
    _COL_SCORE = 5
    _COL_CLS = 6
    _COL_DET_INDEX = 7
    _MIN_COLUMNS = 7

    def __init__(
        self,
        config: Optional[Config] = None,
        class_names: Optional[Mapping[int, str]] = None,
        frame_rate: Optional[int] = None,
    ) -> None:
        self.config = config or load_config()

        self.class_names: Dict[int, str] = dict(class_names or {})

        tracker_type = self.config.TRACKER_TYPE.lower()
        if tracker_type != SUPPORTED_TRACKER:
            raise ValueError(
                f"Desteklenmeyen TRACKER_TYPE: '{tracker_type}'. "
                f"Bu proje yalnizca '{SUPPORTED_TRACKER}' ile calisir."
            )
        self.tracker_type = tracker_type

        # frame_rate parametresi verilmisse .env degerini gecersiz kilar.
        self.frame_rate: int = int(frame_rate or self.config.TRACKER_FPS)

        self.args = self._build_args()
        self.tracker = self._create_tracker()
        self.frame_count: int = 0

        logger.info(
            "Tracker hazir | %s | fps=%d | track_buffer=%d kare (~%.1f sn) | "
            "match_thresh=%.2f | gmc=%s | dahili_reid=%s",
            self.tracker_type,
            self.frame_rate,
            self.max_lost_frames,
            self.max_lost_frames / max(self.frame_rate, 1),
            self.args.match_thresh,
            self.args.gmc_method,
            self.args.with_reid,
        )

    # ----------------------------------------------------------------
    #  YAPILANDIRMA
    # ----------------------------------------------------------------

    def _create_tracker(self) -> BOTSORT:
        """Kurulu ultralytics surumune uygun sekilde BOTSORT olusturur.

        8.3.x: BOTSORT(args, frame_rate=30)
        8.4.x: BOTSORT(args)   (frame_rate parametresi kaldirildi)
        """
        try:
            accepts_frame_rate = "frame_rate" in inspect.signature(BOTSORT.__init__).parameters
        except (TypeError, ValueError):  # pragma: no cover
            accepts_frame_rate = False
        if accepts_frame_rate:
            return BOTSORT(self.args, frame_rate=self.frame_rate)
        return BOTSORT(self.args)

    @staticmethod
    def _load_tracker_defaults() -> Dict:
        """Kurulu ultralytics surumunun botsort varsayilanlarini okur.

        Varsayilanlardan baslamak, surume ozgu yeni parametrelerin de
        eksiksiz tanimli olmasini garanti eder.
        """
        try:
            from ultralytics.utils.checks import check_yaml

            path = check_yaml("botsort.yaml")
            try:  # ultralytics >= 8.3.1xx
                from ultralytics.utils import YAML

                defaults = YAML.load(path)
            except ImportError:  # eski surumler
                from ultralytics.utils import yaml_load

                defaults = yaml_load(path)
            if defaults:
                return dict(defaults)
        except Exception as exc:  # pragma: no cover - surum farkliliklari
            logger.warning(
                "ultralytics botsort.yaml okunamadi (%s). Yalnizca Config'teki "
                "parametreler kullanilacak.",
                exc,
            )
        return {}

    def _build_args(self) -> IterableSimpleNamespace:
        """BoT-SORT parametrelerini Config uzerinden olusturur."""
        cfg = self._load_tracker_defaults()

        cfg["tracker_type"] = self.tracker_type
        cfg["track_high_thresh"] = self.config.TRACK_HIGH_THRESH
        cfg["track_low_thresh"] = self.config.TRACK_LOW_THRESH
        cfg["new_track_thresh"] = self.config.NEW_TRACK_THRESH
        cfg["match_thresh"] = self.config.MATCH_THRESH
        cfg["track_buffer"] = self.config.TRACK_BUFFER
        cfg["fuse_score"] = self.config.FUSE_SCORE
        cfg["gmc_method"] = self.config.GMC_METHOD
        cfg["proximity_thresh"] = self.config.PROXIMITY_THRESH
        cfg["appearance_thresh"] = self.config.APPEARANCE_THRESH
        cfg["with_reid"] = self.config.TRACKER_WITH_REID
        cfg["model"] = self.config.TRACKER_REID_MODEL

        self._validate_thresholds(cfg)
        cfg["with_reid"] = self._resolve_with_reid(cfg["with_reid"], cfg["model"])

        return IterableSimpleNamespace(**cfg)

    @staticmethod
    def _validate_thresholds(cfg: Dict) -> None:
        if cfg["track_low_thresh"] > cfg["track_high_thresh"]:
            raise ValueError(
                "TRACK_LOW_THRESH, TRACK_HIGH_THRESH degerinden buyuk olamaz "
                f"({cfg['track_low_thresh']} > {cfg['track_high_thresh']})."
            )
        if cfg["new_track_thresh"] < cfg["track_high_thresh"]:
            logger.warning(
                "NEW_TRACK_THRESH (%.2f) < TRACK_HIGH_THRESH (%.2f). Bu ayar "
                "gereksiz yeni Track ID uretimine yol acabilir.",
                cfg["new_track_thresh"],
                cfg["track_high_thresh"],
            )
        if cfg["track_buffer"] <= 0:
            raise ValueError("TRACK_BUFFER pozitif bir tam sayi olmalidir.")

    @staticmethod
    def _resolve_with_reid(with_reid: bool, model: str) -> bool:
        """BoT-SORT'un dahili ReID'ini guvenli sekilde etkinlestirir.

        'auto' modeli yalnizca ultralytics'in kendi `model.track()`
        akisinda calisir; tespitleri disaridan besledigimiz bu mimaride
        kullanilamaz.
        """
        if with_reid and str(model).lower() == "auto":
            logger.warning(
                "TRACKER_WITH_REID=true fakat TRACKER_REID_MODEL='auto'. Bu "
                "kombinasyon harici tespit akisinda desteklenmez; dahili ReID "
                "kapatildi. Etkinlestirmek icin gercek bir model yolu verin. "
                "(Kalici kimlik zaten reid.py tarafindan saglanacaktir.)"
            )
            return False
        return with_reid

    # ----------------------------------------------------------------
    #  GENEL API
    # ----------------------------------------------------------------

    def update(
        self, detections: Sequence[Detection], frame: Optional[np.ndarray] = None
    ) -> List[Track]:
        """Tek bir frame'in tespitlerini isleyerek guncel takipleri dondurur.

        Args:
            detections: `detector.detect()` ciktisi.
            frame: BGR goruntu. Global Motion Compensation (GMC) icin
                gereklidir. Kamera hareketi tahmininin tutarli olmasi
                icin her frame'de AYNI turden goruntu (ham veya on
                islenmis) verilmelidir.

        Returns:
            Aktif Track nesnelerinin listesi.
        """
        self.frame_count += 1

        adapter = _DetectionAdapter(detections)
        raw_tracks = self.tracker.update(adapter, frame)

        return self._parse_tracks(raw_tracks, detections)

    def reset(self) -> None:
        """Takip durumunu sifirlar (yeni bir video baslatilirken kullanilir)."""
        reset_fn = getattr(self.tracker, "reset", None)
        if callable(reset_fn):
            reset_fn()
        else:  # pragma: no cover - eski ultralytics surumleri
            self.tracker = self._create_tracker()
        self.frame_count = 0
        logger.info("Tracker sifirlandi.")

    @property
    def active_track_count(self) -> int:
        """Su an aktif olarak takip edilen nesne sayisi (HUD icin)."""
        return len(getattr(self.tracker, "tracked_stracks", []))

    @property
    def lost_track_count(self) -> int:
        """Gecici olarak kaybedilmis, henuz silinmemis takip sayisi."""
        return len(getattr(self.tracker, "lost_stracks", []))

    @property
    def max_lost_frames(self) -> int:
        """Bir Track ID'nin kayipken bellekte tutuldugu maksimum frame sayisi.

        8.4.x: `max_frames_lost`, 8.3.x: `max_time_lost`.
        """
        for attr in ("max_frames_lost", "max_time_lost"):
            value = getattr(self.tracker, attr, None)
            if value is not None:
                return int(value)
        return int(self.args.track_buffer)

    # ----------------------------------------------------------------
    #  SONUC DONUSTURME
    # ----------------------------------------------------------------

    def _parse_tracks(
        self, raw_tracks: np.ndarray, detections: Sequence[Detection]
    ) -> List[Track]:
        """BoT-SORT ciktisini Track listesine cevirir."""
        if raw_tracks is None or len(raw_tracks) == 0:
            return []

        raw = np.asarray(raw_tracks)
        if raw.ndim != 2 or raw.shape[1] < self._MIN_COLUMNS:
            logger.warning("Beklenmeyen tracker cikti sekli: %s", raw.shape)
            return []

        has_det_index = raw.shape[1] > self._COL_DET_INDEX

        tracks: List[Track] = []
        for row in raw:
            class_id = int(row[self._COL_CLS])
            det_index = int(row[self._COL_DET_INDEX]) if has_det_index else -1

            tracks.append(
                Track(
                    track_id=int(row[self._COL_TRACK_ID]),
                    bbox=(float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                    confidence=float(row[self._COL_SCORE]),
                    class_id=class_id,
                    class_name=self._class_name(class_id, det_index, detections),
                    detection_index=det_index,
                )
            )
        return tracks

    def _class_name(
        self, class_id: int, det_index: int, detections: Sequence[Detection]
    ) -> str:
        """Sinif adini once eslesen tespitten, sonra sinif haritasindan cozer."""
        if 0 <= det_index < len(detections):
            return detections[det_index].class_name
        return self.class_names.get(class_id, str(class_id))

    def __repr__(self) -> str:
        return (
            f"Tracker(type={self.tracker_type}, fps={self.frame_rate}, "
            f"track_buffer={self.args.track_buffer}, "
            f"match_thresh={self.args.match_thresh}, "
            f"active={self.active_track_count})"
        )
