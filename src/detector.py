"""
detector.py
===========

Sistemin nesne tespiti (object detection) katmani.

Tek sorumlulugu: bir frame'i alip on isleme adimlarindan gecirmek, YOLO
modeli ile cikarim (inference) yapmak ve standart bir tespit listesi
dondurmektir.

Bu modul icerisinde takip (tracking), kimlik (ReID), sahiplik (ownership)
veya gorsellestirme (visualization) mantigi bulunmaz.

Tum yapilandirma degerleri `.env` dosyasindan okunur; kod icerisinde
sabit deger (magic number) kullanilmaz.

ULTRALYTICS SURUM UYUMLULUGU
--------------------------------------------------------------------
ultralytics 8.4 ile FP16 parametresi `half=True` yerine `quantize=16`
oldu. Eski ad hala calisiyor fakat HER KAREDE kullanimdan kaldirma
uyarisi basiyor. Bu modul kurulu surume uygun parametreyi gonderir.
--------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from src.config import PROJECT_ROOT, Config, load_config
from src.distance import box_iou
from src.utils import get_device

logger = logging.getLogger(__name__)

# Kisi sinifinin adi. `association_memory.py` de ayni sabiti tanimlar; bu
# katman ondan daha altta oldugu icin oradan import EDILMEZ (bagimlilik yonu
# korunur). Deger COCO sinif adidir ve degismez.
PERSON_CLASS_NAME = "person"

# ultralytics'in `quantize` parametresinde FP16'yi ifade eden deger.
_FP16_QUANTIZE = 16


def _normalize_class_name(name: str) -> str:
    """'Cell Phone', 'cell_phone', 'cell phone' -> 'cell phone'."""
    return name.strip().lower().replace("_", " ").replace("-", " ")


def _supports_quantize_arg() -> bool:
    """Kurulu ultralytics `quantize` parametresini tanıyor mu? (>= 8.4)"""
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT

        return "quantize" in DEFAULT_CFG_DICT
    except Exception:  # pragma: no cover - cok eski surumler
        return False


# =====================================================================
#  TESPIT VERI YAPISI
# =====================================================================


@dataclass
class Detection:
    """Tek bir nesne tespitini temsil eden standart cikti formati.

    Diger moduller (tracker, association_memory, visualization) bu formati
    kullanir. Track ID veya UUID bilgisi bu katmanda uretilmez.
    """

    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        return self.bbox

    def to_int_bbox(self) -> Tuple[int, int, int, int]:
        # Bounding box koordinatlarını en yakın tam sayıya yuvarlayıp döndürür
        x1, y1, x2, y2 = self.bbox
        return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

    def to_dict(self) -> Dict:
        # Detection nesnesindeki tüm bilgileri bir Python sözlüğü (dict) olarak döndürür.
        return {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


# =====================================================================
#  GORUNTU ON ISLEME
# =====================================================================


class FramePreprocessor:
    """Frame'i modele vermeden once iyilestiren on isleme zinciri.

    Uygulama sirasi:
        1) CLAHE  (LAB renk uzayindeki L kanali uzerinde)
        2) Gamma duzeltmesi (onceden hesaplanmis LUT ile)

    CLAHE, renk bozulmasini onlemek icin BGR kanallarina ayri ayri degil,
    yalnizca parlaklik (L) kanalina uygulanir.
    """

    _LUT_SIZE = 256

    def __init__( # Sınıfın başlangıç ayarlarını yapar.
        self,
        clahe_enabled: bool,
        clahe_clip_limit: float,
        clahe_tile_grid_size: Tuple[int, int],
        gamma_enabled: bool,
        gamma: float,
    ) -> None:
        self.clahe_enabled = clahe_enabled
        self.gamma_enabled = gamma_enabled
        self.gamma = gamma

        self._clahe = (
            cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
            if clahe_enabled
            else None
        )
        # LUT bir kez hesaplanir, her frame'de yeniden uretilmez.
        self._gamma_lut = self._build_gamma_lut(gamma) if gamma_enabled else None

    @classmethod
    def _build_gamma_lut(cls, gamma: float) -> np.ndarray:
        # Gamma düzeltmesinde kullanılacak LUT (Lookup Table) tablosunu oluşturur.
        # Bu tablo sayesinde her karede yeniden hesaplama yapılmaz ve işlem daha hızlı gerçekleşir.
        if gamma <= 0:
            raise ValueError(f"GAMMA degeri pozitif olmalidir. Alinan: {gamma}")
        inv_gamma = 1.0 / gamma
        scale = cls._LUT_SIZE - 1
        table = np.array(
            [((i / scale) ** inv_gamma) * scale for i in range(cls._LUT_SIZE)],
            dtype=np.uint8,
        )
        return table

    def apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        # Görüntüye CLAHE uygular.
        if self._clahe is None:
            return frame
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)
        lightness = self._clahe.apply(lightness)
        merged = cv2.merge((lightness, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def apply_gamma(self, frame: np.ndarray) -> np.ndarray:
        # Gamma düzeltmesini uygular.
        if self._gamma_lut is None:
            return frame
        return cv2.LUT(frame, self._gamma_lut)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        processed = frame
        if self.clahe_enabled:
            processed = self.apply_clahe(processed)
        if self.gamma_enabled:
            processed = self.apply_gamma(processed)
        return processed


# =====================================================================
#  DETECTOR
# =====================================================================

# YOLO modelini yükleyip tek bir görüntü üzerinde nesne tespiti yapmak için kullanılır.
class Detector:
    """YOLO tabanli nesne tespit katmani.

    Kullanim:
        detector = Detector()
        detections = detector.detect(frame)
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or load_config()

        # --- Model ayarlari ---
        self.model_path: str = self.config.MODEL_PATH
        self.imgsz: int = self.config.MODEL_IMGSZ
        self.conf_threshold: float = self.config.CONF_THRESHOLD
        self.item_conf_threshold: float = self.config.ITEM_CONF_THRESHOLD
        self.iou_threshold: float = self.config.IOU_THRESHOLD
        self.max_det: int = self.config.MAX_DET
        self.verbose: bool = self.config.MODEL_VERBOSE
        self.cross_class_nms: bool = self.config.ITEM_CROSS_CLASS_NMS

        # --- Cihaz ve FP16 ---
        self.device: str = get_device(
            preferred=self.config.MODEL_DEVICE, use_gpu=self.config.USE_GPU
        )
        self.half: bool = self._resolve_half(self.config.USE_FP16, self.device)
        self._use_quantize_arg: bool = _supports_quantize_arg()

        # --- On isleme ---
        self.preprocessor = FramePreprocessor(
            clahe_enabled=self.config.CLAHE_ENABLED,
            clahe_clip_limit=self.config.CLAHE_CLIP_LIMIT,
            clahe_tile_grid_size=self.config.CLAHE_TILE_GRID_SIZE,
            gamma_enabled=self.config.GAMMA_ENABLED,
            gamma=self.config.GAMMA,
        )

        # --- Model yukleme ---
        self._validate_model_path()
        self.model = YOLO(self.model_path)
        self.model.to(self.device)

        # --- Hedef siniflarin indekslerini cozumle ---
        self.target_class_names: List[str] = list(self.config.TARGET_CLASSES)
        self.class_ids: List[int] = self._resolve_class_ids(self.target_class_names)
        self.class_names: Dict[int, str] = {
            cid: self.model.names[cid] for cid in self.class_ids
        }

        logger.info(
            "Detector hazir | model=%s | device=%s | fp16=%s | imgsz=%d | "
            "conf: kisi>=%.2f, esya>=%.2f (cikarim>=%.2f) | siniflar=%s",
            self.model_path,
            self.device,
            self.half,
            self.imgsz,
            self.conf_threshold,
            self.item_conf_threshold,
            self.inference_conf,
            self.class_names,
        )

    # ----------------------------------------------------------------
    #  KURULUM YARDIMCILARI
    # ----------------------------------------------------------------

    @staticmethod
    def _resolve_half(use_fp16: bool, device: str) -> bool:
        # FP16 (yarım hassasiyet) kullanımının uygun olup olmadığını kontrol eder. CPU kullanılıyorsa FP16'yı kapatır.
        """FP16 yalnizca CUDA uzerinde desteklenir."""
        if use_fp16 and device == "cpu":
            logger.warning("FP16 yalnizca CUDA uzerinde desteklenir. FP32'ye gecildi.")
            return False
        return use_fp16

    def _validate_model_path(self) -> None:
        # Model dosyasının belirtilen konumda olup olmadığını kontrol eder. Bulunamazsa Ultralytics'in otomatik indirme yapacağını bildirir.
        """Model dosyasi yerelde yoksa uyarir (Ultralytics otomatik indirir)."""
        candidate = Path(self.model_path)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / self.model_path
        if candidate.exists():
            self.model_path = str(candidate)
        else:
            logger.warning(
                "Model dosyasi bulunamadi: %s. Ultralytics indirmeyi deneyecek.",
                self.model_path,
            )

    def _resolve_class_ids(self, target_names: Sequence[str]) -> List[int]:
        # Hedef sınıf isimlerini modeldeki karşılık gelen sınıf ID'lerine dönüştürür.
        """Sinif isimlerini model.names uzerinden indekse cevirir.

        Sinif indeksleri koda gomulmez; boylece farkli bir model
        kullanildiginda .env disinda degisiklik gerekmez.
        """
        name_to_id = {
            _normalize_class_name(name): idx for idx, name in self.model.names.items()
        }

        resolved: List[int] = []
        for target in target_names:
            key = _normalize_class_name(target)
            if key not in name_to_id:
                raise ValueError(
                    f"'{target}' sinifi model tarafindan desteklenmiyor. "
                    f"TARGET_CLASSES degerini kontrol edin."
                )
            resolved.append(name_to_id[key])
        return sorted(set(resolved))

    # ----------------------------------------------------------------
    #  GENEL API
    # ----------------------------------------------------------------

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        # Görüntüye CLAHE ve Gamma ön işlemlerini uygular.
        """CLAHE + gamma on islemesi uygulanmis frame'i dondurur."""
        return self.preprocessor(frame)

    @property
    def inference_conf(self) -> float:
        # Modele gönderilecek güven eşiğini belirler. İnsan ve eşya için farklı eşikler kullanıldığı için en düşük eşik değeri döndürülür.
        """Modele verilen guven esigi.

        Sinif bazli esikler NMS sonrasinda `_parse_result` icinde
        uygulanir; bu nedenle cikarim, esiklerin EN DUSUGU ile yapilir.
        Aksi halde model dusuk guvenli esyayi hic dondurmez ve sinif
        bazli esigin bir anlami kalmaz.
        """
        return min(self.conf_threshold, self.item_conf_threshold)

    def conf_floor_for(self, class_name: str) -> float:
        # Nesnenin sınıfına göre kullanılacak güven eşiğini belirler. İnsan ve diğer nesneler için farklı eşikler uygulanmasını sağlar.
        """Sinifa gore uygulanacak guven esigini dondurur.

        Kisi tespiti kolaydir ve yuksek skor uretir; yere birakilmis,
        kendi uzerine cokmus, golgede kalan bir esya degildir. Tek bir
        esik kullanmak, esya duyarliligini kisi tespitinin gerektirdigi
        siki esige mahkum eder.
        """
        if _normalize_class_name(class_name) == PERSON_CLASS_NAME:
            return self.conf_threshold
        return self.item_conf_threshold

    @property
    def precision_kwargs(self) -> Dict:
        """FP16/FP32 secimini kurulu ultralytics surumune uygun bicimde verir.

        >= 8.4 : quantize=16 (FP16) | quantize=None (FP32)
        <  8.4 : half=True / half=False
        """
        if self._use_quantize_arg:
            return {"quantize": _FP16_QUANTIZE if self.half else None}
        return {"half": self.half}

    @property
    def infer_kwargs(self) -> Dict:
        # YOLO modeline her çıkarımda gönderilecek ortak parametreleri tek bir sözlük olarak döndürür.
        """Ultralytics cagrilarinda ortak kullanilan parametreler.

        `tracker.py` ayni ayarlarla calisabilmek icin bu sozlugu
        kullanabilir.
        """
        return {
            "imgsz": self.imgsz,
            "conf": self.inference_conf,
            "iou": self.iou_threshold,
            "device": self.device,
            "classes": self.class_ids,
            "max_det": self.max_det,
            "verbose": self.verbose,
            **self.precision_kwargs,
        }

    def warmup(self) -> None:
        # Model üzerinde boş bir görüntü ile ilk çıkarımı yaparak ilk karede oluşabilecek gecikmeyi azaltır.
        """FP16 cekirdeklerini onceden yukleyerek ilk frame gecikmesini azaltir."""
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.model.predict(source=dummy, **self.infer_kwargs)
        logger.info("Model warmup tamamlandi.")

    def detect(self, frame: np.ndarray, preprocess: bool = True) -> List[Detection]:
        # Bir görüntü üzerinde nesne tespiti yapar. İsteğe bağlı olarak ön işlem uygular ve sonuçları Detection listesi olarak döndürür.
        """Tek bir frame uzerinde nesne tespiti yapar.

        Args:
            frame: BGR formatinda goruntu.
            preprocess: CLAHE + gamma uygulanip uygulanmayacagi.

        Returns:
            Detection nesnelerinden olusan liste.
        """
        if frame is None or frame.size == 0:
            return []

        image = self.preprocess(frame) if preprocess else frame
        results = self.model.predict(source=image, **self.infer_kwargs)

        if not results:
            return []
        return self._parse_result(results[0])

    # ----------------------------------------------------------------
    #  SONUC DONUSTURME
    # ----------------------------------------------------------------

    def _parse_result(self, result) -> List[Detection]:
        """Ultralytics sonucunu Detection listesine cevirir."""
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        confidences = boxes.conf.cpu().numpy().astype(np.float32)
        class_ids = boxes.cls.cpu().numpy().astype(int)

        detections: List[Detection] = []
        dropped: List[Tuple[str, float]] = []

        for box, confidence, class_id in zip(xyxy, confidences, class_ids):
            resolved_id = int(class_id)
            name = self.model.names[resolved_id]
            score = float(confidence)

            # Sinif bazli esik. Cikarim en dusuk esikle yapildigi icin
            # eleme burada yapilmalidir.
            if score < self.conf_floor_for(name):
                dropped.append((name, score))
                continue

            x1, y1, x2, y2 = (float(value) for value in box)
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=score,
                    class_id=resolved_id,
                    class_name=name,
                )
            )

        # Esik ayari icin tani bilgisi: hangi nesne hangi skorla elendi.
        # ITEM_CONF_THRESHOLD kalibrasyonunda dogrudan kullanilir.
        if dropped and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Esik altinda elenen tespitler: %s",
                [(name, round(score, 3)) for name, score in dropped],
            )
        if self.cross_class_nms:
            detections = self._suppress_cross_class_duplicates(detections)
        return detections

    def _suppress_cross_class_duplicates(
        self, detections: List[Detection]
    ) -> List[Detection]:
        """Esya siniflari ARASINDA ust uste binen kutulardan en guclusunu tutar.

        Ultralytics NMS'i sinif bazlidir: ayni canta hem "backpack" hem
        "handbag" olarak (IoU ~0.9) donebilir. Bu iki kutu iki ayri Track,
        iki ayri Item UUID, iki ayri sahiplik ve sonunda iki ayri alarm
        demektir. Kisiler bu islemden etkilenmez.

        IoU kullanilir (kucuk kutuya gore ortusme degil): laptopun uzerinde
        duran telefon gibi farkli boyuttaki nesneler BIRBIRINI ELEMEZ.
        """
        persons = [d for d in detections if _normalize_class_name(d.class_name) == PERSON_CLASS_NAME]
        items = sorted(
            (d for d in detections if _normalize_class_name(d.class_name) != PERSON_CLASS_NAME),
            key=lambda d: d.confidence,
            reverse=True,
        )
        kept: List[Detection] = []
        for candidate in items:
            duplicate = next(
                (
                    k for k in kept
                    if k.class_name != candidate.class_name
                    and box_iou(k.bbox, candidate.bbox) >= self.iou_threshold
                ),
                None,
            )
            if duplicate is None:
                kept.append(candidate)
            elif logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Ayni esyanin ikinci sinifi elendi: %s(%.2f) ~ %s(%.2f)",
                    candidate.class_name, candidate.confidence,
                    duplicate.class_name, duplicate.confidence,
                )
        return persons + kept

    def __repr__(self) -> str:
        return (
            f"Detector(model={self.model_path}, device={self.device}, "
            f"fp16={self.half}, imgsz={self.imgsz}, "
            f"conf={self.conf_threshold}/{self.item_conf_threshold}, "
            f"classes={list(self.class_names.values())})"
        )
