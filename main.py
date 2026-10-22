"""
main.py
=======

Orkestrasyon katmani: tum modulleri sirasiyla calistirir ve sistemi yonetir.

--------------------------------------------------------------------
KARE BASINA AKIS
--------------------------------------------------------------------
    frame
      |
      v
    detector.py      -> on isleme (CLAHE + gamma) + YOLO tespiti
      |
      v
    tracker.py       -> BoT-SORT, gecici Track ID
      |
      +--- kisiler ---> utils.crop_bbox -> reid.py (OSNet)
      |                     |
      |                     v
      |                 Kalici Person UUID
      |                     |
      +--- esyalar ---------+--> association_memory.py
      |                          (sahiplik + Item UUID)
      v
    distance.py      -> sahip ile esya arasindaki olcekli mesafe (yaklasik metre)
      |
      v
    [BU DOSYA]       -> unutulma / terk edilme KARARI
      |
      v
    visualization.py -> cizim
--------------------------------------------------------------------

Unutulan esya (abandoned object) IS KURALI yalnizca bu dosyada bulunur.
Diger modullerin hicbiri bu karari vermez; `distance.py` sadece mesafeyi
hesaplar, `association_memory.py` sadece sonucu saklar.

Kullanim:
    python main.py
    python main.py --video assets/deneme.mp4 --output results/cikti.mp4
    python main.py --video 0 --no-save          # web kamerasi
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.association_memory import (
    PERSON_CLASS_NAME,
    ItemMemory,
    ItemStatus,
    MemoryManager,
    PersonCandidate,
    PersonMemory,
)
from src.config import Config, load_config
from src.detector import Detector
from src.distance import (
    box_diagonal,
    box_overlap_ratio,
    estimate_gap_m,
    euclidean_distance,
    get_box_center,
    touches_border,
)
from src.reid import ReID
from src.tracker import Tracker, Track
from src.utils import crop_bbox, describe_device, setup_logger
from src.visualization import VisualObject, Visualizer, WarningMessage

logger = logging.getLogger(__name__)

WINDOW_NAME = "Unutulan Esya Tespit Sistemi"
QUIT_KEYS = (ord("q"), ord("Q"), 27)  # q veya ESC
FPS_WINDOW = 30          # FPS ortalamasinin alindigi kare sayisi
PROGRESS_INTERVAL = 100  # Kac karede bir ilerleme loglanacagi

BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]

# HUD'da gosterilecek sinif basliklari
_HUD_LABELS = {
    "person": "Persons",
    "backpack": "Backpacks",
    "handbag": "Handbags",
    "suitcase": "Suitcases",
    "laptop": "Laptops",
    "cell phone": "Cell Phones",
}


# =====================================================================
#  FPS OLCUMU
# =====================================================================


class FPSMeter:
    """Isleme hizini olcer.

    Hesaplama burada yapilir; `visualization.py` yalnizca hazir degeri
    ekrana yazar.
    """

    def __init__(self, window: int = FPS_WINDOW) -> None:
        self._durations: deque = deque(maxlen=window)
        self._last: Optional[float] = None

    def tick(self) -> float:
        # Her yeni kare işlendiğinde çağrılır. İki kare arasındaki süreyi hesaplar, kaydeder ve güncel FPS değerini döndürür.
        now = time.perf_counter()
        if self._last is not None:
            self._durations.append(now - self._last)
        self._last = now
        return self.value

    @property
    def value(self) -> float:
        # Kaydedilen sürelerin ortalamasını kullanarak anlık ortalama FPS değerini hesaplayıp döndürür.
        if not self._durations:
            return 0.0
        average = sum(self._durations) / len(self._durations)
        return 1.0 / average if average > 0 else 0.0


# =====================================================================
#  UNUTULAN ESYA IS KURALI
# =====================================================================


@dataclass
class OwnerObservation:
    """Sahibin bu karedeki durumu (ham gozlem; karar `evaluate` icindedir).

    bbox        : sahip bu karede gorunuyorsa kutusu, degilse None
    last_bbox   : sahibin en son goruldugu kutu
    unseen_for  : kac saniyedir gorunmuyor (gorunuyorsa 0)
    exited      : kadraj kenarinda mi kayboldu (kenardan cikan kisi CIKMISTIR)
    """

    bbox: Optional[BBox]
    last_bbox: Optional[BBox] = None
    unseen_for: float = math.inf
    exited: bool = False


@dataclass
class _ItemTrace:
    """Bir esyanin karar icin gereken kisa sureli gecmisi."""

    away: float = 0.0                    # birikmis "sahipsiz ve gozetimsiz" sure (sn)
    last_eval: Optional[float] = None    # son degerlendirme zamani
    positions: Deque[Tuple[float, Point]] = field(default_factory=deque)
    # Sahip gorunurken olculen uzakliklar (zaman, metre): sahip kayboldugunda
    # esyadan UZAKLASIYOR muydu, yoksa yaninda mi duruyordu?
    owner_distances: Deque[Tuple[float, float]] = field(default_factory=deque)


class AbandonmentMonitor:
    """Bir esyanin unutulup unutulmadigina karar veren is kurali.

    Her karede, sahibi atanmis her esya icin SIRAYLA:

        1. SAHIP YANINDA  sahip gorunuyor ve uzakligi <= FORGET_DISTANCE_M
                          -> NORMAL, sayac SIFIRLANIR.
        2. TASINIYOR      esya son ITEM_MOTION_WINDOW sn'de kendi kosegeninin
                          ITEM_MOTION_TOLERANCE katindan fazla yer degistirdi
                          -> NORMAL, sayac SIFIRLANIR. Terk edilen esya durur.
        3. DURAKLAT       asagidakilerden biri dogruysa sayac ILERLEMEZ,
                          durum korunur:
                            - esya bu karede tespit edilemedi (kisa kayip)
                            - sahip kisa sure once esyanin yaninda, kadrajin
                              ortasinda ve uzaklasmiyorken kayboldu
                              (engelin arkasinda kalmis olabilir)
                            - esyanin >= ATTENDED_OVERLAP kadari HERHANGI bir
                              kisinin kutusunda (elde, sirtta, ayak dibinde)
        4. SAYAC ILERLER  sayac >= FORGET_TIME                     -> LOST
                          sayac >= ABANDONED_TIME ve
                          sahip uzakligi > ABANDONED_DISTANCE_M     -> ABANDONED

    Bir uzaklasma donemi icinde durum GERI GITMEZ (NORMAL -> LOST ->
    ABANDONED). Yalnizca 1. ve 2. maddeler durumu NORMAL'e dondurur;
    boylece uyari kare kare LOST <-> ABANDONED arasinda gidip gelmez.

    Mesafeler YAKLASIK METREDIR (distance.estimate_gap_m): kenar boslugu,
    sahibin kutu yuksekligi PERSON_HEIGHT_M kabul edilerek metreye
    cevrilir. Esikler bu sayede cozunurlukten ve kameraya uzakliktan
    bagimsizdir. Sahip goruntude yoksa uzaklik sonsuz kabul edilir.

    Neden bu kurallar (gozlenen hatalar ve genel karsiliklari):
      * Sabit piksel ve merkez mesafesi, masasinda oturan sahibin
        telefonunu "unutulmus" sayiyordu                        -> olcekli kenar mesafesi
      * Sahibinin ReID kimligi degisen yayalarin tasidigi cantalar
        "sahipsiz" sayiliyordu                                   -> 2 ve 3 (gozetim)
      * Sahibe bir an ulasilamayinca (direk arkasi, tespit kaybi)
        sayac ilerliyordu                                        -> 3
      * Durum LOST <-> ABANDONED arasinda gidip geliyordu        -> geri gitmeyen durum
    """

    def __init__(self, config: Config) -> None:
        self.forget_distance_m = config.FORGET_DISTANCE_M
        self.forget_time = config.FORGET_TIME
        self.abandoned_distance_m = config.ABANDONED_DISTANCE_M
        self.abandoned_time = config.ABANDONED_TIME
        self.attended_overlap = config.ATTENDED_OVERLAP
        self.motion_window = config.ITEM_MOTION_WINDOW
        self.motion_tolerance = config.ITEM_MOTION_TOLERANCE
        self.hold_time = config.OCCLUSION_HOLD_TIME
        self.exit_margin = config.EXIT_BORDER_MARGIN
        self.leaving_speed = config.OWNER_LEAVING_SPEED
        self.person_height_m = config.PERSON_HEIGHT_M
        self._traces: Dict[str, _ItemTrace] = {}

    # ----------------------------------------------------------------
    #  KANITLAR
    # ----------------------------------------------------------------

    def owner_distance_m(self, item: ItemMemory, owner_bbox: Optional[BBox]) -> float:
        """Sahibin esyaya yaklasik uzakligi (metre). Sahip yoksa sonsuz."""
        if owner_bbox is None or item.last_bbox is None:
            return math.inf
        return estimate_gap_m(item.last_bbox, owner_bbox, self.person_height_m)

    def observe_owner(
        self,
        owner: Optional[PersonMemory],
        current_bbox: Optional[BBox],
        now: float,
        frame_shape: Sequence[int],
    ) -> OwnerObservation:
        """Sahibin bu karedeki ham gozlemini hazirlar."""
        if current_bbox is not None:
            return OwnerObservation(bbox=current_bbox, last_bbox=current_bbox, unseen_for=0.0)
        if owner is None or owner.last_bbox is None:
            return OwnerObservation(bbox=None)
        return OwnerObservation(
            bbox=None,
            last_bbox=owner.last_bbox,
            unseen_for=max(0.0, now - owner.last_seen),
            exited=touches_border(owner.last_bbox, frame_shape, self.exit_margin),
        )

    def _record_owner_distance(self, trace: _ItemTrace, distance: float, now: float) -> None:
        trace.owner_distances.append((now, distance))
        while (
            len(trace.owner_distances) > 1
            and trace.owner_distances[1][0] <= now - self.motion_window
        ):
            trace.owner_distances.popleft()

    def _owner_was_leaving(self, trace: _ItemTrace) -> bool:
        """Sahip gorunur oldugu son pencerede esyadan uzaklasiyor muydu?

        Hiz METRE/SANIYE cinsindendir; esik (OWNER_LEAVING_SPEED) sahneden
        bagimsizdir: yuruyen insan ~1.3 m/s, oturan birinin kutu titremesi
        ~0.1-0.3 m/s uretir.
        """
        if len(trace.owner_distances) < 2:
            return False
        (t0, d0), (t1, d1) = trace.owner_distances[0], trace.owner_distances[-1]
        if t1 <= t0 or math.isinf(d0) or math.isinf(d1):
            return False
        return (d1 - d0) / (t1 - t0) > self.leaving_speed

    def owner_hidden_briefly(
        self, item: ItemMemory, owner: OwnerObservation, trace: _ItemTrace
    ) -> bool:
        """Gorunmeyen sahip, bir engelin arkasinda KISA SURE kalmis olabilir mi?

        Dort kosul birlikte aranir:
          * en fazla OCCLUSION_HOLD_TIME once goruldu,
          * kadraj kenarinda kaybolmadi (kenarda kaybolan kisi cikmistir),
          * kayboldugunda esyanin yanindaydi (<= FORGET_DISTANCE_M),
          * esyadan uzaklasmiyordu (kadraj ortasindaki bir kapidan cikan
            kisi bekletilmez; aksi halde gercek alarm gecikir).
        """
        if owner.bbox is not None or owner.last_bbox is None:
            return False
        if owner.unseen_for > self.hold_time or owner.exited:
            return False
        if self.owner_distance_m(item, owner.last_bbox) > self.forget_distance_m:
            return False
        return not self._owner_was_leaving(trace)

    def _record_position(self, trace: _ItemTrace, bbox: BBox, now: float) -> None:
        """Konum gecmisine bu kareyi ekler; pencerenin basinda en az
        `motion_window` once alinmis TEK bir konum birakir."""
        trace.positions.append((now, get_box_center(bbox)))
        while len(trace.positions) > 1 and trace.positions[1][0] <= now - self.motion_window:
            trace.positions.popleft()

    def is_moving(self, trace: _ItemTrace, bbox: BBox) -> bool:
        """Esya son pencerede kendi boyunun TOLERANS katindan fazla yer degistirdi mi?

        Esik esyanin kendi kosegeni cinsindendir; kucuk telefon ve buyuk
        valiz ayni kuralla degerlendirilir. Kutu titremesi ve kismi
        ortulme kosegenin cok altinda kalir; yuruyen birinin tasidigi
        canta ise saniyede kendi boyunun birkac kati yol alir.
        """
        if not trace.positions:
            return False
        diagonal = box_diagonal(bbox)
        if diagonal <= 0.0:
            return False
        _, oldest = trace.positions[0]
        return euclidean_distance(get_box_center(bbox), oldest) > self.motion_tolerance * diagonal

    def is_attended(self, item: ItemMemory, person_boxes: Sequence[BBox]) -> bool:
        """Esya HERHANGI bir kisinin kutusunun buyuk olcude icinde mi?

        Kimlige bakilmaz: elde tutulan, sirttaki veya birinin ayak dibinde
        duran esya gozetimsiz degildir. Sahibin ReID kimligi kopmus ya da
        sahiplik yanlis kurulmus olsa bile yanlis alarmi engeller. Yalnizca
        sayaci DURAKLATIR; durumu sifirlamaz (terk edilmis bir cantanin
        yanindan gecen yaya alarmi silmez).
        """
        if self.attended_overlap <= 0.0 or item.last_bbox is None:
            return False
        return any(
            box_overlap_ratio(item.last_bbox, box) >= self.attended_overlap
            for box in person_boxes
        )

    # ----------------------------------------------------------------
    #  KARAR
    # ----------------------------------------------------------------

    def evaluate(
        self,
        item: ItemMemory,
        owner: OwnerObservation,
        person_boxes: Sequence[BBox],
        now: float,
        visible: bool = True,
    ) -> ItemStatus:
        """Esyanin guncel durumunu belirler.

        Args:
            item: Sahibi atanmis esya.
            owner: Sahibin bu karedeki durumu (`observe_owner`).
            person_boxes: Bu karedeki TUM kisi kutulari (kimligi bilinmeyenler dahil).
            now: Video zamani (sn).
            visible: Esya bu karede tespit edildi mi? False ise son kutusu
                kullanilir ve sayac ilerlemez.
        """
        if item.last_bbox is None:
            return item.status

        trace = self._traces.setdefault(item.uuid, _ItemTrace())
        # Iki degerlendirme arasindaki sure. Esya OCCLUSION_HOLD_TIME'dan uzun
        # sure gorunmeyip geri geldiyse aradaki bosluk sayilmaz.
        step = 0.0
        if trace.last_eval is not None:
            gap = now - trace.last_eval
            step = gap if 0.0 < gap <= self.hold_time else 0.0
        trace.last_eval = now
        if visible:
            self._record_position(trace, item.last_bbox, now)

        distance = self.owner_distance_m(item, owner.bbox)
        if owner.bbox is not None:
            self._record_owner_distance(trace, distance, now)

        # 1) Sahip yaninda.
        if distance <= self.forget_distance_m:
            return self._reset(item, trace, "sahip yaninda")

        # 2) Esya tasiniyor.
        if visible and self.is_moving(trace, item.last_bbox):
            return self._reset(item, trace, "esya tasiniyor")

        # 3) Duraklat.
        if (
            not visible
            or self.owner_hidden_briefly(item, owner, trace)
            or self.is_attended(item, person_boxes)
        ):
            return item.status

        # 4) Sayac ilerler.
        trace.away += step
        target = ItemStatus.NORMAL
        if trace.away >= self.forget_time:
            target = ItemStatus.LOST
        if trace.away >= self.abandoned_time and distance > self.abandoned_distance_m:
            target = ItemStatus.ABANDONED

        # Bir uzaklasma donemi icinde durum geri gitmez.
        return target if target.severity > item.status.severity else item.status

    def _reset(self, item: ItemMemory, trace: _ItemTrace, reason: str) -> ItemStatus:
        if item.status is not ItemStatus.NORMAL:
            logger.debug("%s NORMAL'e dondu: %s", item.uuid, reason)
        trace.away = 0.0
        return ItemStatus.NORMAL

    def elapsed_away(self, item_uuid: str) -> float:
        """Esyanin birikmis sahipsiz/gozetimsiz suresi (sn)."""
        trace = self._traces.get(item_uuid)
        return 0.0 if trace is None else trace.away

    def prune(self, known_uuids: Sequence[str]) -> None:
        # Artık sistemde bulunmayan (bellekten silinmiş) eşyalara ait geçmişi temizler.
        """Bellekten dusmus esyalarin gecmisini temizler."""
        alive = set(known_uuids)
        for uuid in [u for u in self._traces if u not in alive]:
            del self._traces[uuid]


# =====================================================================
#  SISTEM
# =====================================================================


class ForgottenItemSystem:
    """Tum modulleri bir araya getiren orkestrasyon katmani."""

    def __init__(self, config: Config, frame_rate: Optional[int] = None) -> None:
        self.config = config

        self.detector = Detector(config)
        self.tracker = Tracker(
            class_names=self.detector.class_names, frame_rate=frame_rate, config=config
        )
        self.reid = ReID(config)
        self.memory = MemoryManager(config)
        self.visualizer = Visualizer(config)
        self.monitor = AbandonmentMonitor(config)

        # ReID galerisini hafizadan besle. `reid.py` galeriyi yalnizca
        # OKUR; kayitlari degistirmez.
        self.reid.set_gallery_provider(self.memory.get_gallery)

        # Track ID -> son ReID calistirilan kare. Kimligi bilinen bir
        # kisi icin OSNet'i her karede calistirmak gereksizdir.
        self._last_reid_frame: Dict[int, int] = {}
        self.fps_meter = FPSMeter()

        self._log_hardware()
        self._warmup()
        logger.info("Sistem hazir: %s", self.detector)

    # ----------------------------------------------------------------
    #  DONANIM / ISINDIRMA
    # ----------------------------------------------------------------

    def _log_hardware(self) -> None:
        """Hangi modelin hangi cihazda calistigini acikca loglar.

        GPU beklenirken CPU'da calisma, ancak FPS dusuklugunden fark
        edilen sessiz bir sorundur; bu satir onu ilk saniyede gorunur kilar.
        """
        info = describe_device(self.detector.device)
        if "name" in info:
            logger.info(
                "GPU: %s | %.1f GB | compute %s | torch %s | CUDA %s | cuDNN %s",
                info["name"], info["memory_gb"], info["capability"],
                info["torch"], info["cuda"], info["cudnn"],
            )
        else:
            logger.warning(
                "Sistem CPU uzerinde calisiyor (torch %s). Islem hizi dusuk olacaktir.",
                info.get("torch", "?"),
            )
        logger.info(
            "Cihazlar | YOLO: %s (fp16=%s) | OSNet: %s (fp16=%s)",
            self.detector.device, self.detector.half, self.reid.device, self.reid.half,
        )

    def _warmup(self) -> None:
        """CUDA cekirdeklerini ve cuDNN secimlerini ilk kareden ONCE yukler.

        Ilk cikarim GPU'da yuzlerce milisaniye surer; isindirma yapilmazsa
        bu gecikme ilk karelere ve FPS olcumune yansir.
        """
        if self.detector.device != "cpu":
            self.detector.warmup()
        if self.reid.device != "cpu":
            self.reid.warmup()

    # ----------------------------------------------------------------
    #  KARE ISLEME
    # ----------------------------------------------------------------

    def process_frame(
            # Tek bir video karesini işler. Sırasıyla nesne tespiti, takip, kimlik belirleme, sahip eşleştirme, unutulan eşya kontrolü,
            # bellek temizliği ve görselleştirme işlemlerini gerçekleştirir.
        self, frame: np.ndarray, timestamp: float, frame_index: int
    ) -> np.ndarray:
        """Tek bir kareyi bastan sona isler ve cizilmis kareyi dondurur.

        Args:
            frame: Ham BGR kare.
            timestamp: VIDEO zamani (saniye) = frame_index / fps.
                Gercek zaman degil; boylece sonuclar isleme hizindan
                bagimsiz ve tekrarlanabilir olur.
            frame_index: Kare sirasi.
        """
        # --- 1) Tespit ---
        processed = self.detector.preprocess(frame)
        detections = self.detector.detect(processed, preprocess=False)

        # --- 2) Takip ---
        # GMC (kamera hareketi telafisi) HAM kare ile beslenir: CLAHE
        # uyarlanir oldugundan ayni noktanin parlakligi kareden kareye
        # degisir ve optik akisin parlaklik sabitligi varsayimini bozar.
        tracks = self.tracker.update(detections, frame)

        persons = [t for t in tracks if t.class_name == PERSON_CLASS_NAME]
        items = [t for t in tracks if t.class_name != PERSON_CLASS_NAME]

        # --- 3) Kalici kimlik ---
        person_boxes, uuid_by_track = self._identify_persons(
            frame, persons, timestamp, frame_index
        )

        # --- 4-7) Sahiplik, karar, bakim, cizim ---
        return self.process_tracks(
            frame, persons, items, person_boxes, uuid_by_track, timestamp
        )

    def process_tracks(
        self,
        frame: np.ndarray,
        persons: Sequence[Track],
        items: Sequence[Track],
        person_boxes: Dict[str, BBox],
        uuid_by_track: Dict[int, str],
        timestamp: float,
    ) -> np.ndarray:
        """Kimligi cozulmus takiplerden itibaren kalan adimlari calistirir.

        Tespit/takip/ReID'den bagimsizdir; senaryo testleri bu noktadan
        baslayabilir.
        """
        # --- 4) Sahiplik ---
        # Aday listesi goruntudeki TUM kisileri icerir. Kimligi cozulmemis
        # kisiler UUID=None ile yer alir: sahip atanamazlar ama gercek sahip
        # olabilecekleri icin kanit toplarlar ve atamayi geciktirirler.
        candidates = [
            PersonCandidate(t.track_id, uuid_by_track.get(t.track_id), t.bbox)
            for t in persons
        ]
        all_person_boxes = [t.bbox for t in persons]
        item_records = self._register_items(items, candidates, timestamp)

        # Tespiti kisa sure kacan esyalar son kutulariyla karar ve cizime
        # dahil edilir; uyari kare kare yanip sonmez.
        visible_uuids = {item.uuid for item in item_records.values()}
        held = self.memory.recently_seen_items(
            timestamp, self.config.OCCLUSION_HOLD_TIME, exclude=visible_uuids
        )

        # --- 5) Unutulan esya karari ---
        warnings = self._evaluate_items(
            list(item_records.values()),
            held,
            person_boxes,
            all_person_boxes,
            timestamp,
            frame.shape,
        )

        # --- 6) Bellek bakimi ---
        self.memory.remove_expired_tracks(timestamp)
        self.monitor.prune(list(self.memory.items))

        # --- 7) Cizim ---
        return self._render(
            frame, persons, items, item_records, held, person_boxes, uuid_by_track, warnings
        )

    # ----------------------------------------------------------------
    #  3) KIMLIK COZUMLEME
    # ----------------------------------------------------------------

    def _identify_persons(
            # Takip edilen kişileri kalıcı UUID kimlikleriyle eşleştirir. Gerekli durumlarda ReID işlemini başlatır.
        self,
        frame: np.ndarray,
        persons: Sequence[Track],
        timestamp: float,
        frame_index: int,
    ) -> Tuple[Dict[str, BBox], Dict[int, str]]:
        """Her kisi Track'ini kalici bir Person UUID'ye baglar.

        KURAL: Bir UUID'ye bagli Track, takip surdukce o UUID'yi KORUR.
        Periyodik ReID (REID_INTERVAL) yalnizca galeriyi tazeler.

        Eski davranista tazeleme sonucu kararsiz (UNCERTAIN) cikinca kisi o
        karede kimliksiz kaliyor, esyasinin sahibi bir an "kadrajdan cikmis"
        gorunuyordu; benzerlik dusuk cikinca (kisi arkasini dondu, egildi)
        ayni kisiye ikinci bir UUID uretiliyordu. Kisa sureli takipte
        BoT-SORT'un Track surekliligi, tek bir crop'un gorunum
        benzerliginden daha guvenilir bir kanittir.

        Returns:
            (Person UUID -> bbox, Track ID -> Person UUID)
        """
        resolved: Dict[str, BBox] = {}
        uuid_by_track: Dict[int, str] = {}
        unbound: List[Track] = []
        refresh: List[Tuple[Track, str]] = []

        for track in persons:
            person = self.memory.find_person_by_track(track.track_id)
            if person is None:
                unbound.append(track)
                continue

            self.memory.observe_person(
                person.uuid, track.track_id, track.bbox, timestamp=timestamp
            )
            resolved[person.uuid] = track.bbox
            uuid_by_track[track.track_id] = person.uuid

            last = self._last_reid_frame.get(track.track_id)
            if last is None or frame_index - last >= self.config.REID_INTERVAL:
                refresh.append((track, person.uuid))

        if unbound or refresh:
            self._run_reid(
                frame, unbound, refresh, resolved, uuid_by_track, timestamp, frame_index
            )
        return resolved, uuid_by_track

    def _run_reid(
            # OSNet kullanarak kişilerin embeddinglerini çıkarır, mevcut kişilerle eşleştirir veya yeni bir kişi oluşturur.
        self,
        frame: np.ndarray,
        unbound: Sequence[Track],
        refresh: Sequence[Tuple[Track, str]],
        resolved: Dict[str, BBox],
        uuid_by_track: Dict[int, str],
        timestamp: float,
        frame_index: int,
    ) -> None:
        """OSNet ile yeni Track'ler icin kimlik tahmini ve bilinenler icin
        galeri tazeleme.

        Embedding'ler tek bir toplu cagrida uretilir (GPU verimliligi),
        eslestirme ise tek tek yapilir; boylece ayni UUID'nin ayni karede
        iki kisiye atanmasi engellenebilir.
        """
        jobs: List[Tuple[Track, Optional[str]]] = [(t, None) for t in unbound]
        jobs += [(t, uuid) for t, uuid in refresh]

        # Tazeleme denemesi crop kullanilamasa bile kaydedilir; aksi halde
        # kucuk/kesik crop'lu kisi icin her karede yeniden denenir.
        for track, bound_uuid in jobs:
            if bound_uuid is not None:
                self._last_reid_frame[track.track_id] = frame_index

        crops = [crop_bbox(frame, track.bbox) for track, _ in jobs]
        usable = [
            index
            for index, crop in enumerate(crops)
            if ReID.is_valid_crop(crop, self.config.REID_MIN_CROP_HEIGHT)
        ]
        if not usable:
            return

        embeddings = self.reid.extract_embeddings([crops[i] for i in usable])
        gallery = self.memory.get_gallery()
        used = set(resolved)  # Bu karede baska kisiye verilmis UUID'ler

        for index, embedding in zip(usable, embeddings):
            track, bound_uuid = jobs[index]

            if bound_uuid is not None:
                self._refresh_embedding(bound_uuid, embedding, gallery)
                continue

            result = (
                self.reid.match(embedding, gallery, exclude=used)
                if gallery
                else None
            )

            if result is not None and result.is_matched and result.uuid:
                self.memory.observe_person(
                    result.uuid, track.track_id, track.bbox, embedding, timestamp
                )
                uuid = result.uuid
            elif result is not None and result.is_uncertain:
                # Emin degiliz: yanlis UUID atamaktansa bu kareyi atla.
                # Kisi aday listesinde UUID=None olarak yer alir.
                continue
            else:
                person = self.memory.add_person(
                    track_id=track.track_id,
                    bbox=track.bbox,
                    embedding=embedding,
                    timestamp=timestamp,
                )
                uuid = person.uuid
                logger.info("Yeni kisi: %s (track=%d)", uuid, track.track_id)

            used.add(uuid)
            resolved[uuid] = track.bbox
            uuid_by_track[track.track_id] = uuid
            self._last_reid_frame[track.track_id] = frame_index

    def _refresh_embedding(
        self, uuid: str, embedding: np.ndarray, gallery: Dict[str, np.ndarray]
    ) -> None:
        """Bagli bir Track'in yeni embedding'ini galeriye ekler.

        Crop bariz bicimde farkli gorunuyorsa (ortulme, baska biriyle ust
        uste binme) embedding EKLENMEZ; galeri kirlenmez. UUID her durumda
        korunur.
        """
        stored = gallery.get(uuid)
        if stored is None or stored.size == 0:
            self.memory.update_embedding(uuid, embedding)
            return
        score = self.reid.similarity(embedding, stored)
        if score >= self.config.NEW_PERSON_THRESHOLD:
            self.memory.update_embedding(uuid, embedding)
        else:
            logger.debug(
                "%s icin tazeleme embedding'i eklenmedi (benzerlik %.2f < %.2f)",
                uuid, score, self.config.NEW_PERSON_THRESHOLD,
            )

    # ----------------------------------------------------------------
    #  4) SAHIPLIK
    # ----------------------------------------------------------------

    def _register_items(
            # Tespit edilen eşyaları hafızaya kaydeder ve uygun kişilere sahiplik atamasını yapar.
        self,
        items: Sequence[Track],
        candidates: Sequence[PersonCandidate],
        timestamp: float,
    ) -> Dict[int, ItemMemory]:
        """Esyalari hafizaya kaydeder ve gerekiyorsa sahiplerini atar.

        Sahiplik KARARI tamamen association_memory'deki
        register_item -> try_assign_owner zincirinde verilir. Burada
        mesafe kontrolunu tekrarlayan veya ATLAYAN kod bulunmamalidir.

        Returns:
            Track ID -> ItemMemory
        """
        records: Dict[int, ItemMemory] = {}
        for track in items:
            records[track.track_id] = self.memory.register_item(
                class_id=track.class_id,
                class_name=track.class_name,
                track_id=track.track_id,
                bbox=track.bbox,
                person_candidates=candidates,
                timestamp=timestamp,
            )
        return records

    # ----------------------------------------------------------------
    #  5) UNUTULAN ESYA KARARI
    # ----------------------------------------------------------------

    def _evaluate_items(
            # Her eşyanın durumunu değerlendirir.
        self,
        visible: Sequence[ItemMemory],
        held: Sequence[ItemMemory],
        person_boxes: Dict[str, BBox],
        all_person_boxes: Sequence[BBox],
        timestamp: float,
        frame_shape: Sequence[int],
    ) -> List[WarningMessage]:
        """Esyalarin durumunu gunceller ve uyarilari uretir.

        Bu karede gorunen esyalar ile tespiti kisa sure kacan (held)
        esyalar degerlendirilir. Uyarilar siddete gore siralanir.
        """
        alerts: List[Tuple[int, WarningMessage]] = []
        seen: set = set()
        batch = [(item, True) for item in visible] + [(item, False) for item in held]

        for item, is_visible in batch:
            if item.owner_uuid is None or item.uuid in seen:
                continue  # Sahibi bilinmeyen esya icin karar verilemez
            seen.add(item.uuid)

            owner = self.monitor.observe_owner(
                self.memory.find_person(item.owner_uuid),
                person_boxes.get(item.owner_uuid),
                timestamp,
                frame_shape,
            )
            previous = item.status
            status = self.monitor.evaluate(
                item, owner, all_person_boxes, timestamp, visible=is_visible
            )
            self.memory.set_status(item.uuid, status)

            if status is not previous:
                distance = self.monitor.owner_distance_m(item, owner.bbox)
                logger.info(
                    "%s (%s) durumu: %s -> %s | sahip=%s | uzaklik=%s | sure=%.1f sn",
                    item.uuid,
                    item.class_name,
                    previous.value,
                    status.value,
                    item.owner_uuid,
                    "kadraj disi" if math.isinf(distance) else f"~{distance:.1f} m",
                    self.monitor.elapsed_away(item.uuid),
                )

            if status is ItemStatus.ABANDONED:
                alerts.append((status.severity, WarningMessage(
                    "ABANDONED ITEM",
                    [f"{item.class_name.title()} {item.uuid}", f"Owner {item.owner_uuid}"],
                )))
            elif status is ItemStatus.LOST:
                alerts.append((status.severity, WarningMessage(
                    "WARNING",
                    [
                        f"Person {item.owner_uuid}",
                        f"Forgot {item.class_name.title()} {item.uuid}",
                    ],
                )))

        alerts.sort(key=lambda pair: -pair[0])
        return [message for _, message in alerts]

    # ----------------------------------------------------------------
    #  7) CIZIM
    # ----------------------------------------------------------------

    def _render(
            # Çizim için gerekli tüm bilgileri hazırlar ve sonuçların ekrana çizilmesini Visualizer sınıfına bırakır.
        self,
        frame: np.ndarray,
        persons: Sequence[Track],
        items: Sequence[Track],
        records: Dict[int, ItemMemory],
        held: Sequence[ItemMemory],
        person_boxes: Dict[str, BBox],
        uuid_by_track: Dict[int, str],
        warnings: Sequence[WarningMessage],
    ) -> np.ndarray:
        """Cizilecek verileri hazirlar ve `visualization.py`'ye devreder."""
        objects: List[VisualObject] = []
        associations: List[Tuple[BBox, BBox]] = []

        for track in persons:
            objects.append(
                VisualObject(
                    bbox=track.bbox,
                    class_name=track.class_name,
                    track_id=track.track_id,
                    uuid=uuid_by_track.get(track.track_id),
                    confidence=track.confidence,
                )
            )

        for track in items:
            item = records.get(track.track_id)
            if item is None:
                continue
            objects.append(self._item_visual(item, track.bbox, track.track_id, track.confidence))
            owner_bbox = person_boxes.get(item.owner_uuid or "")
            if owner_bbox is not None:
                associations.append((owner_bbox, track.bbox))

        # Tespiti kisa sure kacan ve alarmdaki esyalar son kutusuyla cizilir
        # (Track yok -> Track ID gosterilmez).
        for item in held:
            if item.status is ItemStatus.NORMAL or item.last_bbox is None:
                continue
            objects.append(self._item_visual(item, item.last_bbox, None, None))

        return self.visualizer.draw_frame(
            frame,
            objects=objects,
            associations=associations,
            warnings=warnings,
            stats=self._build_stats(persons, items, records, held),
            fps=self.fps_meter.value,
        )

    def _item_visual(
        self,
        item: ItemMemory,
        bbox: BBox,
        track_id: Optional[int],
        confidence: Optional[float],
    ) -> VisualObject:
        """Esya icin cizim nesnesi. Sahibi henuz atanmamissa lider aday
        "P003?" bicimiminde (kesinlesmemis) gosterilir."""
        owner = item.owner_uuid
        provisional = False
        if owner is None:
            owner = self.memory.provisional_owner(item.uuid)
            provisional = owner is not None
        return VisualObject(
            bbox=bbox,
            class_name=item.class_name,
            track_id=track_id,
            uuid=item.uuid,
            owner_uuid=owner,
            owner_provisional=provisional,
            confidence=confidence,
            status=None if item.status is ItemStatus.NORMAL else item.status.value,
        )

    def _build_stats(
            # Ekranda gösterilecek istatistikleri (kişi sayısı, eşya sayısı, unutulan eşya sayısı, aktif takip sayısı vb.) hesaplar.
        self,
        persons: Sequence[Track],
        items: Sequence[Track],
        records: Dict[int, ItemMemory],
        held: Sequence[ItemMemory] = (),
    ) -> Dict[str, int]:
        """HUD sayimlarini hazirlar (sayim burada yapilir, cizimde degil)."""
        counts = {label: 0 for label in _HUD_LABELS.values()}
        counts[_HUD_LABELS[PERSON_CLASS_NAME]] = len(persons)
        for track in items:
            label = _HUD_LABELS.get(track.class_name)
            if label:
                counts[label] += 1

        alarmed = {
            item.uuid
            for item in list(records.values()) + list(held)
            if item.status in (ItemStatus.LOST, ItemStatus.ABANDONED)
        }
        counts["Forgotten Items"] = len(alarmed)
        counts["Active Tracks"] = self.tracker.active_track_count
        return counts

    # ----------------------------------------------------------------
    #  ANA DONGU
    # ----------------------------------------------------------------

    def run(self, source, output_path: Optional[Path], show_window: bool) -> None:
        # Sistemin ana çalışma döngüsüdür. Videoyu kare kare okuyarak tüm işleme sürecini yürütür,
        # görüntüyü gösterir ve istenirse çıktı videosunu kaydeder.
        """Video kaynagini bastan sona isler."""
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"Video kaynagi acilamadi: {source}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or math.isnan(fps):
            fps = float(self.config.TRACKER_FPS)
            logger.warning("Video FPS okunamadi. %.0f varsayildi.", fps)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = self._create_writer(output_path, fps, width, height)
        frame_index = 0

        logger.info(
            "Isleme basliyor | %dx%d @ %.1f FPS | %s kare",
            width, height, fps, total if total > 0 else "?",
        )

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                # VIDEO zamani: sonuclar isleme hizindan bagimsiz olur.
                timestamp = frame_index / fps
                annotated = self.process_frame(frame, timestamp, frame_index)
                self.fps_meter.tick()

                if writer is not None:
                    writer.write(annotated)
                if show_window:
                    annotated=cv2.resize(annotated,(1280,720))
                    cv2.imshow(WINDOW_NAME, annotated)
                    if cv2.waitKey(1) & 0xFF in QUIT_KEYS:
                        logger.info("Kullanici tarafindan durduruldu.")
                        break

                frame_index += 1
                if frame_index % PROGRESS_INTERVAL == 0:
                    stats = self.memory.stats()
                    logger.info(
                        "Kare %d%s | %.1f FPS | kisi=%d esya=%d terk=%d",
                        frame_index,
                        f"/{total}" if total > 0 else "",
                        self.fps_meter.value,
                        stats["persons_total"],
                        stats["items_total"],
                        stats["items_abandoned"],
                    )
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if show_window:
                cv2.destroyAllWindows()

        self._log_summary(frame_index, output_path if writer is not None else None)

    def _create_writer(
            # Çıktı videosu kaydedilecekse VideoWriter nesnesini oluşturur ve hazırlar.
        self, output_path: Optional[Path], fps: float, width: int, height: int
    ) -> Optional[cv2.VideoWriter]:
        if output_path is None:
            return None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            logger.error("Cikti dosyasi olusturulamadi: %s", output_path)
            return None
        return writer

    def _log_summary(self, frame_count: int, output_path: Optional[Path]) -> None:
        # Video işleme tamamlandıktan sonra toplam kişi, eşya, sahiplik ilişkisi ve unutulan eşya gibi özet bilgileri log olarak yazdırır.
        stats = self.memory.stats()
        logger.info("=" * 58)
        logger.info("Islenen kare      : %d", frame_count)
        logger.info("Toplam kisi       : %d", stats["persons_total"])
        logger.info("Toplam esya       : %d", stats["items_total"])
        logger.info("Sahiplik iliskisi : %d", stats["associations"])
        logger.info("Terk edilmis esya : %d", stats["items_abandoned"])

        for item in self.memory.abandoned_items():
            logger.info(
                "  ! %s (%s) - sahip: %s",
                item.uuid, item.class_name, item.owner_uuid or "bilinmiyor",
            )
        if output_path is not None:
            logger.info("Cikti             : %s", output_path)
        logger.info("=" * 58)


# =====================================================================
#  GIRIS NOKTASI
# =====================================================================


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    # Komut satırından girilen parametreleri (video, output, no-save, no-show, env vb.) okuyup programa aktarır.
    parser = argparse.ArgumentParser(
        description="Kisi-esya sahipligi takibi ve unutulan esya tespiti."
    )
    parser.add_argument("--video", help="Video dosyasi veya kamera indeksi (ornek: 0)")
    parser.add_argument("--output", help="Cikti videosunun yolu")
    parser.add_argument("--no-save", action="store_true", help="Cikti videosu kaydetme")
    parser.add_argument("--no-show", action="store_true", help="Pencere acma")
    parser.add_argument("--env", help="Alternatif .env dosyasi")
    return parser.parse_args(argv)


def _resolve_source(config: Config):
    # Kullanılacak video kaynağını belirler. Girilen değer kamera indeksi ise kamerayı, dosya yolu ise ilgili video dosyasını kullanır.
    """Video kaynagini cozer; sayisal deger kamera indeksi kabul edilir."""
    value = config.VIDEO_PATH
    if str(value).isdigit():
        return int(value)
    path = config.resolve_path(value)
    if not path.exists():
        raise FileNotFoundError(f"Video bulunamadi: {path}")
    return str(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Programın ana başlangıç fonksiyonudur
    args = parse_args(argv)

    overrides = {}
    if args.video:
        overrides["VIDEO_PATH"] = args.video
    if args.output:
        overrides["OUTPUT_PATH"] = args.output
    if args.no_save:
        overrides["SAVE_OUTPUT"] = False
    if args.no_show:
        overrides["SHOW_WINDOW"] = False

    config = load_config(env_path=Path(args.env) if args.env else None, **overrides)
    setup_logger(
        level=config.LOG_LEVEL,
        log_file=config.resolve_path(config.LOG_FILE) if config.LOG_TO_FILE else None,
    )

    try:
        source = _resolve_source(config)
        output = config.resolve_path(config.OUTPUT_PATH) if config.SAVE_OUTPUT else None
        system = ForgottenItemSystem(config)
        system.run(source, output, config.SHOW_WINDOW)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.info("Kullanici tarafindan kesildi.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
