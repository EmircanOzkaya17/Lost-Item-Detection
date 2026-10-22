"""
utils.py
========

Yalnizca **genel amacli yardimci fonksiyonlar**.

Bu modulun amaci, uygulamanin farkli bolumlerinde ortak olarak ihtiyac
duyulan ancak herhangi bir modulun dogal sorumluluguna girmeyen
fonksiyonlari tek bir yerde toplamaktir.

--------------------------------------------------------------------
BU DOSYAYA EKLENMEMESI GEREKENLER
--------------------------------------------------------------------
* YOLO cikarimi (inference)          -> detector.py
* BoT-SORT takip algoritmasi         -> tracker.py
* OSNet ReID islemleri               -> reid.py
* Association Memory yonetimi        -> association_memory.py
* Mesafe hesaplamalari               -> distance.py
* Gorsellestirme islemleri           -> visualization.py
* Unutulan esya tespiti / is kurali  -> ilgili katman

UUID uretimi de buraya AIT DEGILDIR. `generate_person_uuid()` ve
`generate_item_uuid()` fonksiyonlari `association_memory.py` icinde
bulunur; cunku UUID uretimi, benzersizligin garanti edilmesi ve
PersonMemory/ItemMemory ile iliskilendirilmesi tamamen kimlik ve
hafiza yonetiminin parcasidir.

Bu dosya zamanla "her seyin konuldugu dosya" haline gelmemelidir. Yeni
bir fonksiyon eklemeden once gercekten genel amacli olup olmadigi
degerlendirilmelidir.
--------------------------------------------------------------------

Bagimsizlik:
    `utils.py` projedeki hicbir modulu import etmez ve `.env` dosyasini
    okumaz. Ihtiyac duydugu degerleri parametre olarak alir; boylece
    bagimsiz, test edilebilir ve yeniden kullanilabilir kalir.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

__all__ = [
    "get_device",
    "describe_device",
    "crop_bbox",
    "clip_bbox",
    "resize_frame",
    "current_timestamp",
    "current_datetime",
    "setup_logger",
]

logger = logging.getLogger(__name__)

BBox = Sequence[float]
IntBBox = Tuple[int, int, int, int]

_CHANNELS = 3
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s"
_LOG_DATE_FORMAT = "%H:%M:%S"
_LOGGER_CONFIGURED_FLAG = "_unutulan_esya_logger_kuruldu"
_BYTES_PER_GIB = 1024 ** 3


# =====================================================================
#  1. CIHAZ (DEVICE) SECIMI
# =====================================================================


def _normalize_cuda_device(preferred: Optional[str], device_count: int) -> str:
    """'cuda', 'cuda:1', '0', 'gpu' gibi girdileri gecerli bir CUDA cihazina cevirir.

    Var olmayan bir GPU indeksi istenirse (ornegin tek GPU'lu makinede
    'cuda:1') 'cuda:0' kullanilir; aksi halde model yuklenirken CUDA
    'invalid device ordinal' hatasi verir.
    """
    text = (preferred or "cuda").strip().lower()
    if text in ("cuda", "gpu"):
        return "cuda:0"
    if text.isdigit():
        text = f"cuda:{text}"
    if text.startswith("cuda:"):
        index_text = text.split(":", 1)[1]
        if index_text.isdigit():
            index = int(index_text)
            if index < device_count:
                return f"cuda:{index}"
            logger.warning(
                "Istenen GPU (%s) bulunamadi; sistemde %d GPU var. cuda:0 kullanilacak.",
                text,
                device_count,
            )
            return "cuda:0"
    logger.warning("Taninmayan cihaz degeri '%s'; cuda:0 kullanilacak.", preferred)
    return "cuda:0"


def get_device(preferred: Optional[str] = None, use_gpu: bool = True) -> str:
    """Kullanilacak islem birimini belirler.

    GPU kullanilabilirligini kontrol ederek uygun cihaz bilgisini
    dondurur.

    Args:
        preferred: Istenen cihaz ('cuda', 'cuda:0', '0', 'cpu'). Verilmezse
            GPU varsa 'cuda:0', yoksa 'cpu' dondurulur.
        use_gpu: GPU kullanimi istenmiyorsa False.

    Returns:
        'cuda:N' veya 'cpu'

    Ornek:
        >>> get_device()
        'cuda:0'
        >>> get_device("cuda:0", use_gpu=False)
        'cpu'
    """
    if not use_gpu:
        return "cpu"
    if preferred is not None and preferred.strip().lower().startswith("cpu"):
        return "cpu"

    # torch yalnizca gerektiginde yuklenir; boylece bu modul torch
    # kurulu olmayan ortamlarda da import edilebilir.
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch bulunamadi. CPU kullanilacak.")
        return "cpu"

    if not torch.cuda.is_available():
        if torch.version.cuda is None:
            # En sik GPU sorunu: PyTorch'un CPU surumu kurulu.
            logger.warning(
                "CUDA bulunamadi: kurulu PyTorch (%s) CPU surumu. GPU icin "
                "'python install.py' ile CUDA surumunu kurun. CPU kullanilacak.",
                torch.__version__,
            )
        else:
            logger.warning(
                "CUDA bulunamadi: PyTorch CUDA %s icin derlenmis ama GPU/surucu "
                "erisilemiyor (nvidia-smi ile kontrol edin). CPU kullanilacak.",
                torch.version.cuda,
            )
        return "cpu"
    return _normalize_cuda_device(preferred, torch.cuda.device_count())


def describe_device(device: str) -> Dict[str, object]:
    """Cihaz hakkinda loglanabilir ozet bilgi dondurur.

    Returns:
        {'device', 'name', 'memory_gb', 'capability', 'torch', 'cuda', 'cudnn'}
        CPU icin yalnizca 'device' ve 'torch' doludur.
    """
    info: Dict[str, object] = {"device": device}
    try:
        import torch
    except ImportError:
        return info

    info["torch"] = torch.__version__
    if device == "cpu" or not torch.cuda.is_available():
        return info

    index = torch.device(device).index or 0
    props = torch.cuda.get_device_properties(index)
    info.update(
        name=props.name,
        memory_gb=round(props.total_memory / _BYTES_PER_GIB, 1),
        capability=f"{props.major}.{props.minor}",
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
    )
    return info


# =====================================================================
#  2-3. BOUNDING BOX DOGRULAMA VE CROP ALMA
# =====================================================================


def clip_bbox(bbox: BBox, frame_shape: Sequence[int]) -> IntBBox:
    """Bounding box koordinatlarini goruntu sinirlari icinde tutar.

    Gorevleri:
        * Negatif koordinatlari duzeltmek
        * Goruntu sinirlarini asan koordinatlari kirpmak
        * Ters girilmis koseleri duzeltmek (x1 > x2 gibi)
        * Gecerli bir bounding box dondurmek

    Args:
        bbox: (x1, y1, x2, y2)
        frame_shape: Goruntunun `shape` degeri; (h, w) veya (h, w, c).

    Returns:
        Piksel indeksleme icin kullanilabilir tam sayi kutusu.

    Not:
        Sonuc tam sayidir cunku bu fonksiyonun amaci piksel erisimidir.
        Geometrik hesaplamalar `distance.py` sorumlulugundadir.
    """
    if bbox is None or len(bbox) != 4:
        raise ValueError(f"Bounding box (x1, y1, x2, y2) olmalidir. Alinan: {bbox}")
    if frame_shape is None or len(frame_shape) < 2:
        raise ValueError(f"frame_shape en az (h, w) icermelidir. Alinan: {frame_shape}")

    height, width = int(frame_shape[0]), int(frame_shape[1])
    x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)

    # Ters girilmis koseleri duzelt
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))
    return x1, y1, x2, y2


def crop_bbox(frame: np.ndarray, bbox: BBox, copy: bool = False) -> np.ndarray:
    """Bounding box icerisindeki goruntuyu keserek dondurur.

    Ozellikle ReID (OSNet) icin kisi goruntusu (person crop) elde etmek
    amaciyla surekli kullanilir. Kod tekrarini onlemek icin merkezi bir
    yardimci fonksiyon olarak tasarlanmistir.

    Koordinatlar once `clip_bbox()` ile goruntu sinirlarina kirpilir;
    boylece kadraji tasan kutular hata uretmez.

    Args:
        frame: BGR goruntu.
        bbox: (x1, y1, x2, y2)
        copy: True ise bagimsiz bir kopya dondurur. Varsayilan olarak
            performans icin gorunum (view) dondurulur; donen dizi
            degistirilirse kaynak frame de degisir.

    Returns:
        Kesilmis goruntu. Kutu gecersiz veya bos ise (0, 0, 3) seklinde
        bos bir dizi.
    """
    if frame is None or frame.size == 0:
        return np.empty((0, 0, _CHANNELS), dtype=np.uint8)

    x1, y1, x2, y2 = clip_bbox(bbox, frame.shape)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, _CHANNELS), dtype=frame.dtype)

    crop = frame[y1:y2, x1:x2]
    return crop.copy() if copy else crop


# =====================================================================
#  4. GORUNTU BOYUTLANDIRMA
# =====================================================================


def resize_frame(
    frame: np.ndarray,
    size: Tuple[int, int],
    interpolation: Optional[int] = None,
) -> np.ndarray:
    """Frame'i istenilen cozunurluge donusturur.

    Farkli giris kaynaklari icin standart goruntu boyutlandirmasini
    saglar.

    Args:
        frame: BGR goruntu.
        size: (genislik, yukseklik) - OpenCV sirasi.
        interpolation: Verilmezse otomatik secilir: kucultmede
            INTER_AREA, buyutmede INTER_LINEAR. Bu secim kucultulen
            goruntulerde belirgin sekilde daha temiz sonuc verir.

    Returns:
        Boyutlandirilmis goruntu.
    """
    if frame is None or frame.size == 0:
        raise ValueError("resize_frame bos goruntu ile cagrilamaz.")
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Hedef boyut pozitif olmalidir. Alinan: {size}")

    if interpolation is None:
        shrinking = width * height < frame.shape[1] * frame.shape[0]
        interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(frame, (width, height), interpolation=interpolation)


# =====================================================================
#  5. ZAMAN FONKSIYONLARI
# =====================================================================


def current_timestamp() -> float:
    """Sure hesaplamalari icin zaman damgasi (saniye).

    `time.monotonic()` kullanilir: sistem saati degistirilse veya yaz
    saati uygulamasi devreye girse bile geri gitmez. Bu nedenle "kac
    saniyedir gorulmedi" gibi SURE hesaplari icin dogru secimdir.

    UYARI: Bu deger mutlak (takvim) zaman DEGILDIR; yalnizca farklari
    anlamlidir. Insan tarafindan okunabilir zaman icin
    `current_datetime()` kullanin.

    Not: Video isleme sirasinda gercek zaman yerine VIDEO zamani
    (frame_index / fps) kullanilmalidir; bu fonksiyon yalnizca acik bir
    zaman damgasi verilmediginde devreye girer. Bir calisma icinde tek
    bir zaman kaynagi kullanilmalidir.
    """
    return time.monotonic()


def current_datetime() -> datetime:
    """Log kayitlari ve olay zamanlari icin takvim zamani."""
    return datetime.now()


# =====================================================================
#  6. LOGGER OLUSTURMA
# =====================================================================


def setup_logger(
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    stream: Optional[object] = None,
) -> logging.Logger:
    """Uygulamanin merkezi log sistemini kurar.

    Yapilandirma tek bir noktadan yapilir; diger moduller yalnizca
    `logging.getLogger(__name__)` cagirarak bu ayarlari devralir.

    Tekrar cagrildiginda mevcut handler'lari temizler; boylece ayni log
    satiri birden fazla kez yazdirilmaz.

    Args:
        level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL' veya
            logging seviyesi tam sayisi.
        log_file: Verilirse loglar ayrica bu dosyaya yazilir. Ust
            dizinler otomatik olusturulur.
        stream: Konsol akisi. Varsayilan `sys.stderr`.

    Returns:
        Yapilandirilmis kok (root) logger.
    """
    if isinstance(level, str):
        resolved = logging.getLevelName(level.strip().upper())
        if not isinstance(resolved, int):
            raise ValueError(
                f"Gecersiz log seviyesi: '{level}'. "
                "DEBUG, INFO, WARNING, ERROR veya CRITICAL kullanin."
            )
        level = resolved

    root = logging.getLogger()
    if getattr(root, _LOGGER_CONFIGURED_FLAG, False):
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            handler.close()

    root.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)

    console = logging.StreamHandler(stream or sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    setattr(root, _LOGGER_CONFIGURED_FLAG, True)
    return root
