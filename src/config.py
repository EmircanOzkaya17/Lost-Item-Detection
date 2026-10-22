"""
config.py
=========

Sistemin **merkezi yapilandirma (configuration)** modulu.

Tek sorumlulugu: uygulamanin ihtiyac duydugu tum yapilandirma
parametrelerini okumak, dogrulamak ve diger modullere sunmaktir.

--------------------------------------------------------------------
BU DOSYADA BULUNMAYAN SEYLER
--------------------------------------------------------------------
* Algoritma
* Is kurali (business logic)
* Karar mekanizmasi
--------------------------------------------------------------------

Kurallar:
* Sistemdeki tum yapilandirmalar tek merkezden yonetilir.
* HICBIR modul `.env` dosyasini dogrudan okumaz; yalnizca bu modul okur.
* Tum moduller ayarlara yalnizca `Config` nesnesi uzerinden erisir.
* Kod icerisinde magic number kullanilmaz.
* Yeni bir parametre eklemek icin YALNIZCA `.env` ve bu dosya guncellenir
  (asagidaki `Config` sinifina bir alan eklemek yeterlidir; okuma ve tip
  donusumu alan tipinden otomatik yapilir).

Kullanim:
    from src.config import load_config
    config = load_config()
    detector = Detector(config)
"""

import logging
import os
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, get_args, get_origin, get_type_hints

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

Color = Tuple[int, int, int]  # BGR

_TRUE_VALUES = {"1", "true", "yes", "y", "on", "evet"}
_FALSE_VALUES = {"0", "false", "no", "n", "off", "hayir"}

_RGB_CHANNELS = 3
_GRID_DIMENSIONS = 2
_COLOR_MIN, _COLOR_MAX = 0, 255
_YOLO_STRIDE = 32

# Gecerli ReID arka uclari. 'boxmot' geriye donuk uyumluluk icin kabul
# edilir ve reid.py icinde 'builtin' arka ucuna yonlendirilir.
_REID_BACKENDS = ("auto", "builtin", "torchreid", "torchscript", "boxmot")

# Kaldirilan parametreler -> yerine gelen. `.env` dosyasinda hala
# duruyorlarsa sessizce yok sayilmalari yerine kullanici uyarilir.
_OBSOLETE_KEYS: Dict[str, str] = {
    "OWNERSHIP_DISTANCE": "OWNERSHIP_DISTANCE_M (yaklasik metre)",
    "OWNERSHIP_RETRY_DELAY": "OWNERSHIP_CONFIRM_TIME (biriken yakinlik suresi)",
    "FORGET_DISTANCE": "FORGET_DISTANCE_M (yaklasik metre)",
    "ABANDONED_DISTANCE": "ABANDONED_DISTANCE_M (yaklasik metre)",
    "OWNER_OVERLAP_TOLERANCE": "kaldirildi: kenar mesafesi ortusmeyi zaten 0 sayar "
    "(baskasinin tuttugu esya icin ATTENDED_OVERLAP)",
}


class ConfigError(ValueError):
    """Yapilandirma hatasi."""


# =====================================================================
#  CONFIG
# =====================================================================


@dataclass
class Config:
    """Uygulamanin tum yapilandirma parametreleri.

    Buradaki varsayilan degerler yalnizca `.env` dosyasinda ilgili anahtar
    bulunmadiginda kullanilir. Normal kullanimda tum degerler `.env`
    uzerinden yonetilmelidir.
    """

    # ---------------- Model ayarlari ----------------
    MODEL_PATH: str = "yolo11m.pt"
    MODEL_DEVICE: str = "cuda:0"
    USE_GPU: bool = True
    USE_FP16: bool = True
    MODEL_IMGSZ: int = 1280
    CONF_THRESHOLD: float = 0.25
    # Esya siniflari icin ayri (daha dusuk) guven esigi. Kisi tespiti kolaydir
    # ve yuksek skor uretir; yere birakilmis, deforme olmus bir esya degildir.
    # Tek esik kullanmak esya duyarliligini kisinin esigine mahkum eder.
    ITEM_CONF_THRESHOLD: float = 0.10
    IOU_THRESHOLD: float = 0.7
    MAX_DET: int = 300
    MODEL_VERBOSE: bool = False
    TARGET_CLASSES: List[str] = field(
        default_factory=lambda: [
            "person", "backpack", "handbag", "suitcase", "cell phone", "laptop",
        ]
    )
    # Ayni fiziksel esyanin farkli siniflarla (backpack/handbag/suitcase)
    # ust uste tespit edilmesini engeller: esya siniflari arasinda IoU >=
    # IOU_THRESHOLD olan kutulardan yalnizca en yuksek skorlu tutulur.
    # Aksi halde tek canta iki ayri UUID, iki ayri sahiplik ve iki alarm uretir.
    ITEM_CROSS_CLASS_NMS: bool = True

    # ---------------- Video ayarlari ----------------
    VIDEO_PATH: str = "assets/input.mp4"
    OUTPUT_PATH: str = "results/output.mp4"
    SAVE_OUTPUT: bool = True
    SHOW_WINDOW: bool = True

    # ---------------- Goruntu on isleme ----------------
    GAMMA: float = 1.4
    GAMMA_ENABLED: bool = True
    CLAHE_ENABLED: bool = True
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_GRID_SIZE: Tuple[int, int] = (8, 8)

    # ---------------- Tracker ayarlari ----------------
    TRACKER_TYPE: str = "botsort"
    TRACK_HIGH_THRESH: float = 0.5
    TRACK_LOW_THRESH: float = 0.1
    NEW_TRACK_THRESH: float = 0.6
    MATCH_THRESH: float = 0.85
    TRACK_BUFFER: int = 90
    TRACKER_FPS: int = 30
    FUSE_SCORE: bool = True
    TRACKER_WITH_REID: bool = False
    TRACKER_REID_MODEL: str = "auto"
    PROXIMITY_THRESH: float = 0.5
    APPEARANCE_THRESH: float = 0.25
    GMC_METHOD: str = "sparseOptFlow"

    # ---------------- ReID ayarlari ----------------
    OSNET_MODEL_PATH: str = "models/osnet_x1_0_msmt17.pt"
    OSNET_ARCH: str = "osnet_x1_0"
    REID_BACKEND: str = "auto"
    REID_DEVICE: str = "cuda:0"
    REID_USE_FP16: bool = True
    REID_INPUT_HEIGHT: int = 256
    REID_INPUT_WIDTH: int = 128
    EMBEDDING_SIZE: int = 512
    REID_BATCH_SIZE: int = 16
    REID_MIN_CROP_HEIGHT: int = 48
    REID_THRESHOLD: float = 0.65
    NEW_PERSON_THRESHOLD: float = 0.45
    REID_MATCH_STRATEGY: str = "max"
    # Kimligi bilinen bir Track icin embedding'in kac karede bir
    # tazelenecegi. Her karede OSNet calistirmak gereksiz pahalidir.
    # Tazeleme kisinin UUID'sini DEGISTIRMEZ; yalnizca galeriyi besler.
    REID_INTERVAL: int = 15

    # ---------------- Association Memory ayarlari ----------------
    MAX_EMBEDDING_HISTORY: int = 5
    PERSON_MEMORY_SIZE: int = 500
    MAX_TRACK_LOST_TIME: float = 30.0
    MEMORY_TIMEOUT: float = 300.0
    MEMORY_PURGE_ENABLED: bool = False
    # --- Olcek ---
    # Piksel mesafeleri yaklasik metreye cevirmek icin varsayilan kisi boyu.
    # Kisinin kutu yuksekligi bu kadar metre kabul edilir (distance.py).
    PERSON_HEIGHT_M: float = 1.7
    # --- Sahiplik atamasi (zamansal kanit) ---
    # Esyaya OWNERSHIP_DISTANCE_M'den yakin her kisi her karede kanit
    # biriktirir; kanit OWNERSHIP_MEMORY_TIME zaman sabitiyle sonumlenir.
    # Lider aday OWNERSHIP_CONFIRM_TIME saniyelik kanita ulasir VE ikinci
    # adayin en az OWNERSHIP_DOMINANCE kati kanita sahip olursa sahip atanir.
    OWNERSHIP_DISTANCE_M: float = 0.6
    OWNERSHIP_CONFIRM_TIME: float = 1.0
    OWNERSHIP_MEMORY_TIME: float = 5.0
    OWNERSHIP_DOMINANCE: float = 2.0
    ITEM_REMATCH_DISTANCE: float = 60.0
    ITEM_REMATCH_TIMEOUT: float = 120.0
    # Bir esya kaydinin "baska bir aktif takibe bagli" sayilmasi icin en fazla
    # kac saniye once gorulmus olmasi gerektigi. Rematch korumasinin olcutu
    # `active` bayragi DEGILDIR: o bayrak MAX_TRACK_LOST_TIME boyunca dusmez.
    ITEM_REMATCH_SEEN_GAP: float = 0.5
    # Yeniden eslesmede eski ve yeni kutu alanlari arasindaki en buyuk oran.
    # Ayni noktada beliren farkli boyutlu bir yanlis tespit, eski kaydin
    # sahibini ve alarm durumunu devralmamalidir.
    ITEM_REMATCH_MAX_AREA_RATIO: float = 2.0

    # ---------------- UUID formati ----------------
    UUID_PREFIX_PERSON: str = "P"
    UUID_PREFIX_BACKPACK: str = "B"
    UUID_PREFIX_HANDBAG: str = "H"
    UUID_PREFIX_SUITCASE: str = "S"
    UUID_PREFIX_LAPTOP: str = "L"
    UUID_PREFIX_CELL_PHONE: str = "C"
    UUID_PREFIX_DEFAULT: str = "X"
    UUID_PAD_WIDTH: int = 3

    # ---------------- Unutulan esya ayarlari ----------------
    # Mesafeler YAKLASIK METREDIR (PERSON_HEIGHT_M ile olceklenir).
    # Varsayilanlar PETS 2006/2007 terk edilmis bagaj tanimini izler:
    # sahip 2 m'den uzaklasirsa uyari, 3 m'den uzakta kalirsa alarm.
    FORGET_DISTANCE_M: float = 2.0
    FORGET_TIME: float = 10.0
    ABANDONED_DISTANCE_M: float = 3.0
    ABANDONED_TIME: float = 30.0
    # Esyanin bu orani HERHANGI bir kisinin kutusunun icindeyse (elde/sirtta
    # tasiniyor, birinin ayak dibinde) sayac DURAKLATILIR. 0.0 -> kapali.
    ATTENDED_OVERLAP: float = 0.5
    # Esya son ITEM_MOTION_WINDOW sn'de kendi kosegeninin
    # ITEM_MOTION_TOLERANCE katindan fazla yer degistirdiyse TASINIYOR
    # sayilir; tasinan esya unutulmus olamaz, sayac sifirlanir.
    ITEM_MOTION_WINDOW: float = 1.0
    ITEM_MOTION_TOLERANCE: float = 1.0
    # Kisa sureli gorunmezlik toleransi (sn). Esya veya sahip bu sureden
    # kisa bir sure tespit edilemezse sayac ILERLEMEZ, durum korunur.
    # Kadraj kenarindan cikan sahip icin uygulanmaz (EXIT_BORDER_MARGIN).
    OCCLUSION_HOLD_TIME: float = 1.0
    # Sahip kayboldugu sirada esyadan bu hizdan (metre/saniye) hizli
    # uzaklasiyorduysa "engel arkasinda" degil "gidiyor" kabul edilir ve
    # bekleme yapilmaz. Yurume ~1.3 m/s; oturan kisinin kutu titremesi < 0.3 m/s.
    OWNER_LEAVING_SPEED: float = 0.5
    # Goruntu kenarina bu oran (genislik/yukseklik) kadar yakin kaybolan
    # kisi "kadrajdan cikti" kabul edilir; bekleme yapilmaz.
    EXIT_BORDER_MARGIN: float = 0.02

    # ---------------- Gorsellestirme ayarlari ----------------
    SHOW_FPS: bool = True
    SHOW_TRACK_ID: bool = True
    SHOW_UUID: bool = True
    SHOW_ASSOCIATION_LINE: bool = True
    SHOW_WARNING_TEXT: bool = True
    SHOW_HUD: bool = True
    SHOW_OWNER: bool = True
    SHOW_CONFIDENCE: bool = False
    VIZ_LINE_THICKNESS: int = 2
    VIZ_FONT_SCALE: float = 0.5
    VIZ_AUTO_SCALE: bool = True
    VIZ_REFERENCE_HEIGHT: int = 1080
    VIZ_CORNER_STYLE: bool = True
    VIZ_CORNER_RATIO: float = 0.22
    VIZ_PANEL_ALPHA: float = 0.55
    VIZ_ASSOCIATION_THICKNESS: int = 1
    # Ayni anda gosterilecek en fazla uyari paneli (once ABANDONED).
    VIZ_MAX_WARNINGS: int = 3

    # ---------------- Loglama ayarlari ----------------
    LOG_LEVEL: str = "INFO"
    LOG_TO_FILE: bool = False
    LOG_FILE: str = "results/system.log"

    # ---------------- Bounding box renkleri (BGR) ----------------
    PERSON_COLOR: Color = (0, 255, 0)
    BACKPACK_COLOR: Color = (255, 0, 0)
    HANDBAG_COLOR: Color = (255, 0, 255)
    SUITCASE_COLOR: Color = (255, 255, 0)
    LAPTOP_COLOR: Color = (0, 165, 255)
    CELL_PHONE_COLOR: Color = (0, 255, 255)
    DEFAULT_COLOR: Color = (200, 200, 200)
    TEXT_COLOR: Color = (255, 255, 255)
    PANEL_COLOR: Color = (0, 0, 0)
    WARNING_COLOR: Color = (0, 0, 255)
    ASSOCIATION_LINE_COLOR: Color = (180, 180, 180)

    # ----------------------------------------------------------------
    #  GRUPLANMIS ERISIM
    #  (Yalnizca mevcut alanlari bir araya getirir; hesaplama yapmaz.)
    # ----------------------------------------------------------------

    @property
    def class_colors(self) -> Dict[str, Color]:
        """Sinif adi -> sabit BGR rengi."""
        return {
            "person": self.PERSON_COLOR,
            "backpack": self.BACKPACK_COLOR,
            "handbag": self.HANDBAG_COLOR,
            "suitcase": self.SUITCASE_COLOR,
            "laptop": self.LAPTOP_COLOR,
            "cell phone": self.CELL_PHONE_COLOR,
        }

    @property
    def uuid_prefixes(self) -> Dict[str, str]:
        """Sinif adi -> UUID oneki."""
        return {
            "person": self.UUID_PREFIX_PERSON,
            "backpack": self.UUID_PREFIX_BACKPACK,
            "handbag": self.UUID_PREFIX_HANDBAG,
            "suitcase": self.UUID_PREFIX_SUITCASE,
            "laptop": self.UUID_PREFIX_LAPTOP,
            "cell phone": self.UUID_PREFIX_CELL_PHONE,
            "_default": self.UUID_PREFIX_DEFAULT,
        }

    def resolve_path(self, value: str) -> Path:
        """Goreli bir yolu proje koku ile birlestirir."""
        candidate = Path(value)
        return candidate if candidate.is_absolute() else PROJECT_ROOT / value

    def as_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    # ----------------------------------------------------------------
    #  DOGRULAMA
    # ----------------------------------------------------------------

    def validate(self) -> None:
        """Eksik veya tutarsiz yapilandirmalari kontrol eder.

        Tum hatalar toplanip tek seferde bildirilir; kullanicinin hatalari
        teker teker kesfetmesi gerekmez.
        """
        errors: List[str] = []

        def in_range(name: str, low: float, high: float) -> None:
            value = getattr(self, name)
            if not low <= value <= high:
                errors.append(f"{name} degeri [{low}, {high}] araliginda olmalidir (alinan: {value}).")

        def positive(name: str) -> None:
            if getattr(self, name) <= 0:
                errors.append(f"{name} pozitif olmalidir (alinan: {getattr(self, name)}).")

        def non_negative(name: str) -> None:
            if getattr(self, name) < 0:
                errors.append(f"{name} negatif olamaz (alinan: {getattr(self, name)}).")

        # --- Model ---
        for name in ("CONF_THRESHOLD", "ITEM_CONF_THRESHOLD", "IOU_THRESHOLD"):
            in_range(name, 0.0, 1.0)
        for name in ("MODEL_IMGSZ", "MAX_DET", "GAMMA", "CLAHE_CLIP_LIMIT"):
            positive(name)
        if self.MODEL_IMGSZ % _YOLO_STRIDE:
            errors.append(
                f"MODEL_IMGSZ, {_YOLO_STRIDE} sayisinin kati olmalidir (alinan: {self.MODEL_IMGSZ})."
            )
        if not self.TARGET_CLASSES:
            errors.append("TARGET_CLASSES bos olamaz.")
        if any(size <= 0 for size in self.CLAHE_TILE_GRID_SIZE):
            errors.append(f"CLAHE_TILE_GRID_SIZE pozitif olmalidir (alinan: {self.CLAHE_TILE_GRID_SIZE}).")

        # --- Tracker ---
        if self.TRACKER_TYPE.lower() != "botsort":
            errors.append(
                f"Bu proje yalnizca 'botsort' ile calisir (alinan: '{self.TRACKER_TYPE}')."
            )
        for name in ("TRACK_HIGH_THRESH", "TRACK_LOW_THRESH", "NEW_TRACK_THRESH",
                     "MATCH_THRESH", "PROXIMITY_THRESH", "APPEARANCE_THRESH"):
            in_range(name, 0.0, 1.0)
        if self.TRACK_LOW_THRESH > self.TRACK_HIGH_THRESH:
            errors.append(
                f"TRACK_LOW_THRESH ({self.TRACK_LOW_THRESH}), TRACK_HIGH_THRESH "
                f"({self.TRACK_HIGH_THRESH}) degerinden buyuk olamaz."
            )
        # BoT-SORT, skoru TRACK_LOW_THRESH'ten kucuk/esit tespitleri tamamen
        # atar. ITEM_CONF_THRESHOLD bu degerin altinda kalirsa dedektorun
        # gecirdigi dusuk guvenli esyalar takipciye hic ulasmaz: esik
        # dusurulmus gibi gorunur ama davranis degismez. Sessiz kayip.
        if self.ITEM_CONF_THRESHOLD <= self.TRACK_LOW_THRESH:
            logger.warning(
                "ITEM_CONF_THRESHOLD (%.3f) <= TRACK_LOW_THRESH (%.3f). Bu araliktaki "
                "esya tespitleri takipci tarafindan atilir. TRACK_LOW_THRESH degerini "
                "ITEM_CONF_THRESHOLD altina cekin.",
                self.ITEM_CONF_THRESHOLD,
                self.TRACK_LOW_THRESH,
            )
        # Yeni Track yalnizca birinci eslestirme turundan dogar; o tur ise
        # skoru TRACK_HIGH_THRESH'in ustundeki tespitlerle beslenir. Dolayisiyla
        # yeni Track esigi pratikte max(TRACK_HIGH_THRESH, NEW_TRACK_THRESH)'tir.
        if self.ITEM_CONF_THRESHOLD < self.TRACK_HIGH_THRESH:
            logger.info(
                "ITEM_CONF_THRESHOLD (%.2f) < TRACK_HIGH_THRESH (%.2f): bu araliktaki "
                "esya tespitleri MEVCUT takibi surdurebilir fakat YENI Track "
                "olusturamaz. Yeni Track esigi = %.2f",
                self.ITEM_CONF_THRESHOLD,
                self.TRACK_HIGH_THRESH,
                max(self.TRACK_HIGH_THRESH, self.NEW_TRACK_THRESH),
            )
        for name in ("TRACK_BUFFER", "TRACKER_FPS"):
            positive(name)
        if self.GMC_METHOD.lower() not in ("sparseoptflow", "orb", "sift", "ecc", "none"):
            errors.append(
                f"Gecersiz GMC_METHOD: '{self.GMC_METHOD}'. Secenekler: "
                "sparseOptFlow, orb, sift, ecc, none"
            )

        # --- ReID ---
        for name in ("REID_THRESHOLD", "NEW_PERSON_THRESHOLD"):
            in_range(name, 0.0, 1.0)
        if self.NEW_PERSON_THRESHOLD > self.REID_THRESHOLD:
            errors.append(
                f"NEW_PERSON_THRESHOLD ({self.NEW_PERSON_THRESHOLD}), REID_THRESHOLD "
                f"({self.REID_THRESHOLD}) degerinden buyuk olamaz."
            )
        if self.REID_MATCH_STRATEGY.lower() not in ("max", "mean"):
            errors.append(
                f"REID_MATCH_STRATEGY 'max' veya 'mean' olmalidir (alinan: '{self.REID_MATCH_STRATEGY}')."
            )
        if self.REID_BACKEND.lower() not in _REID_BACKENDS:
            errors.append(
                f"Gecersiz REID_BACKEND: '{self.REID_BACKEND}'. Secenekler: "
                + ", ".join(_REID_BACKENDS[:-1])
            )
        for name in ("REID_INPUT_HEIGHT", "REID_INPUT_WIDTH", "EMBEDDING_SIZE",
                     "REID_BATCH_SIZE", "REID_INTERVAL"):
            positive(name)

        # --- Association Memory ---
        for name in ("MAX_EMBEDDING_HISTORY", "PERSON_MEMORY_SIZE",
                     "MAX_TRACK_LOST_TIME", "MEMORY_TIMEOUT", "UUID_PAD_WIDTH",
                     "ITEM_REMATCH_SEEN_GAP", "PERSON_HEIGHT_M", "OWNERSHIP_DISTANCE_M",
                     "OWNERSHIP_MEMORY_TIME"):
            positive(name)
        non_negative("OWNERSHIP_CONFIRM_TIME")
        if self.OWNERSHIP_DOMINANCE < 1.0:
            errors.append(
                f"OWNERSHIP_DOMINANCE en az 1.0 olmalidir (alinan: {self.OWNERSHIP_DOMINANCE})."
            )
        if self.OWNERSHIP_CONFIRM_TIME >= self.OWNERSHIP_MEMORY_TIME:
            # Surekli yakinlikta kanit en fazla ~OWNERSHIP_MEMORY_TIME'a ulasir.
            errors.append(
                f"OWNERSHIP_CONFIRM_TIME ({self.OWNERSHIP_CONFIRM_TIME}), "
                f"OWNERSHIP_MEMORY_TIME ({self.OWNERSHIP_MEMORY_TIME}) degerinden kucuk "
                "olmalidir; aksi halde biriken kanit esige hic ulasamaz."
            )
        if self.ITEM_REMATCH_MAX_AREA_RATIO < 1.0:
            errors.append(
                "ITEM_REMATCH_MAX_AREA_RATIO en az 1.0 olmalidir "
                f"(alinan: {self.ITEM_REMATCH_MAX_AREA_RATIO})."
            )
        if self.ITEM_REMATCH_SEEN_GAP > self.ITEM_REMATCH_TIMEOUT:
            errors.append(
                f"ITEM_REMATCH_SEEN_GAP ({self.ITEM_REMATCH_SEEN_GAP}), "
                f"ITEM_REMATCH_TIMEOUT ({self.ITEM_REMATCH_TIMEOUT}) degerinden "
                "buyuk olamaz; aksi halde hicbir esya yeniden eslesemez."
            )
        if self.MEMORY_TIMEOUT < self.MAX_TRACK_LOST_TIME:
            errors.append(
                f"MEMORY_TIMEOUT ({self.MEMORY_TIMEOUT}), MAX_TRACK_LOST_TIME "
                f"({self.MAX_TRACK_LOST_TIME}) degerinden kucuk olamaz."
            )

        # --- Unutulan esya ---
        in_range("ATTENDED_OVERLAP", 0.0, 1.0)
        in_range("EXIT_BORDER_MARGIN", 0.0, 0.5)
        for name in ("FORGET_DISTANCE_M", "FORGET_TIME", "ABANDONED_DISTANCE_M",
                     "ABANDONED_TIME", "ITEM_MOTION_WINDOW", "ITEM_MOTION_TOLERANCE",
                     "OCCLUSION_HOLD_TIME", "OWNER_LEAVING_SPEED"):
            positive(name)
        if self.ABANDONED_TIME < self.FORGET_TIME:
            errors.append(
                f"ABANDONED_TIME ({self.ABANDONED_TIME}), FORGET_TIME ({self.FORGET_TIME}) "
                "degerinden kucuk olamaz; terk edilme, unutulmadan sonra gelir."
            )
        if self.OWNERSHIP_DISTANCE_M > self.FORGET_DISTANCE_M:
            errors.append(
                f"OWNERSHIP_DISTANCE_M ({self.OWNERSHIP_DISTANCE_M}) "
                f"FORGET_DISTANCE_M ({self.FORGET_DISTANCE_M}) degerinden buyuk olamaz: "
                "sahip sayilmak icin gereken yakinlik, unutulmus sayilmak icin "
                "gereken uzakligi asamaz."
            )
        if self.ABANDONED_DISTANCE_M < self.FORGET_DISTANCE_M:
            errors.append(
                f"ABANDONED_DISTANCE_M ({self.ABANDONED_DISTANCE_M}), FORGET_DISTANCE_M "
                f"({self.FORGET_DISTANCE_M}) degerinden kucuk olamaz."
            )

        # --- Loglama ---
        if self.LOG_LEVEL.upper() not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            errors.append(
                f"Gecersiz LOG_LEVEL: '{self.LOG_LEVEL}'. Secenekler: "
                "DEBUG, INFO, WARNING, ERROR, CRITICAL"
            )

        # --- Gorsellestirme ---
        in_range("VIZ_PANEL_ALPHA", 0.0, 1.0)
        in_range("VIZ_CORNER_RATIO", 0.0, 0.5)
        for name in ("VIZ_LINE_THICKNESS", "VIZ_FONT_SCALE", "VIZ_REFERENCE_HEIGHT",
                     "VIZ_MAX_WARNINGS"):
            positive(name)

        # --- Renkler ---
        for f in fields(self):
            if not f.name.endswith("_COLOR"):
                continue
            value = getattr(self, f.name)
            if len(value) != _RGB_CHANNELS or not all(
                _COLOR_MIN <= channel <= _COLOR_MAX for channel in value
            ):
                errors.append(
                    f"{f.name} BGR formatinda 0-255 arasi uc deger olmalidir (alinan: {value})."
                )

        if errors:
            raise ConfigError(
                "Yapilandirma hatalari bulundu (.env dosyasini kontrol edin):\n  - "
                + "\n  - ".join(errors)
            )

    def __repr__(self) -> str:
        return (
            f"Config(model={self.MODEL_PATH}, imgsz={self.MODEL_IMGSZ}, "
            f"conf={self.CONF_THRESHOLD}/{self.ITEM_CONF_THRESHOLD} (kisi/esya), "
            f"tracker={self.TRACKER_TYPE}, reid_threshold={self.REID_THRESHOLD}, "
            f"classes={len(self.TARGET_CLASSES)})"
        )


# =====================================================================
#  .env OKUMA VE TIP DONUSUMU
# =====================================================================


def _strip_inline_comment(raw: str) -> str:
    return raw.split("#")[0].strip()


def _parse_bool(raw: str, name: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise ConfigError(f"{name} bir dogruluk degeri olmalidir (true/false). Alinan: '{raw}'")


def _parse_int_sequence(raw: str, name: str, expected: int) -> Tuple[int, ...]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != expected:
        raise ConfigError(f"{name} tam olarak {expected} deger icermelidir. Alinan: '{raw}'")
    try:
        return tuple(int(p) for p in parts)
    except ValueError as exc:
        raise ConfigError(f"{name} tam sayilardan olusmalidir. Alinan: '{raw}'") from exc


def _coerce(raw: str, target_type: Any, name: str) -> Any:
    """Ham `.env` metnini alanin tipine donusturur."""
    value = _strip_inline_comment(raw)

    if target_type is bool:
        return _parse_bool(value, name)
    if target_type is int:
        try:
            return int(value)
        except ValueError as exc:
            raise ConfigError(f"{name} tam sayi olmalidir. Alinan: '{raw}'") from exc
    if target_type is float:
        try:
            return float(value)
        except ValueError as exc:
            raise ConfigError(f"{name} sayi olmalidir. Alinan: '{raw}'") from exc
    if target_type is str:
        return value

    origin = get_origin(target_type)
    if origin is list:
        return [p.strip() for p in value.split(",") if p.strip()]
    if origin is tuple:
        return _parse_int_sequence(value, name, len(get_args(target_type)))

    raise ConfigError(f"{name} icin desteklenmeyen yapilandirma tipi: {target_type}")


def _field_default(f: Any) -> Any:
    return f.default_factory() if f.default is MISSING else f.default


def _warn_obsolete_keys() -> None:
    """Artik okunmayan anahtarlar `.env`'de duruyorsa kullaniciyi uyarir."""
    for key, replacement in _OBSOLETE_KEYS.items():
        if os.getenv(key) is not None:
            logger.warning(
                "%s artik kullanilmiyor ve YOK SAYILIYOR. Yerine: %s", key, replacement
            )


def load_config(
    env_path: Optional[Path] = None, override_env: bool = False, **overrides: Any
) -> Config:
    """`.env` dosyasini okuyarak tum yapilandirma degerlerini yukler.

    Args:
        env_path: `.env` dosyasinin yolu. Verilmezse proje kokundeki
            dosya kullanilir.
        override_env: True ise `.env` degerleri mevcut ortam
            degiskenlerini ezer.
        **overrides: Calisma aninda belirlenen degerler. Ornegin video
            FPS'i okunduktan sonra `load_config(TRACKER_FPS=fps)`.

    Returns:
        Dogrulanmis `Config` nesnesi.

    Raises:
        ConfigError: Eksik, hatali veya tutarsiz yapilandirma durumunda.
    """
    path = env_path or ENV_PATH
    if not Path(path).exists():
        logger.warning(".env dosyasi bulunamadi: %s. Varsayilan degerler kullanilacak.", path)
    load_dotenv(dotenv_path=path, override=override_env)
    _warn_obsolete_keys()

    hints = get_type_hints(Config)
    values: Dict[str, Any] = {}
    for f in fields(Config):
        raw = os.getenv(f.name)
        if raw is None or not raw.strip():
            values[f.name] = _field_default(f)
        else:
            values[f.name] = _coerce(raw, hints[f.name], f.name)

    unknown = set(overrides) - set(values)
    if unknown:
        raise ConfigError(f"Bilinmeyen yapilandirma parametresi: {sorted(unknown)}")
    values.update(overrides)

    config = Config(**values)
    config.validate()
    logger.info("Yapilandirma yuklendi: %s", config)
    return config


_cached_config: Optional[Config] = None


def get_config(reload: bool = False) -> Config:
    """Uygulama genelinde paylasilan yapilandirma nesnesini dondurur.

    Ilk cagride `.env` okunur, sonraki cagrilarda ayni nesne dondurulur.
    """
    global _cached_config
    if _cached_config is None or reload:
        _cached_config = load_config()
    return _cached_config
