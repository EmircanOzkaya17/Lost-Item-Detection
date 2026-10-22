"""
reid.py
=======

Identity Layer - 1. modul: **Person Re-Identification (OSNet)**

Tek sorumlulugu: bir kisi crop'undan embedding uretmek ve verilen galeri
ile karsilastirarak kimlik TAHMINI yapmaktir.

--------------------------------------------------------------------
BU MODUL HAFIZA YONETIMI YAPMAZ
--------------------------------------------------------------------
`reid.py` hicbir kaydi olusturmaz, guncellemez veya silmez. UUID
uretmez. Galeri (gallery) disaridan parametre olarak verilir; bu
galerinin sahibi ve yoneticisi `association_memory.py`'dir.

Bu modul yalnizca sunlari yapar:
    crop -> on isleme -> OSNet -> embedding -> normalize -> benzerlik
--------------------------------------------------------------------

GPU NOTLARI
--------------------------------------------------------------------
* Cihaz secimi `USE_GPU` bayragina uyar (detector ile tutarli).
* Crop'lar CPU'da yalnizca yeniden boyutlandirilir ve uint8 olarak GPU'ya
  tasinir; float donusumu ve ImageNet normalizasyonu GPU'da yapilir.
  Boylece CPU->GPU veri transferi float32'ye gore 4 kat azalir.
* Giris boyutu sabit (256x128) oldugundan cuDNN otomatik ayari
  (benchmark) acilir.
--------------------------------------------------------------------

Kullanim:
    result = reid.identify(person_crop, gallery)
    if result.is_new:
        ...  # yeni kisi: UUID uretimi association_memory'nin isidir
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import cv2
import numpy as np
import torch

from src.config import PROJECT_ROOT, Config, load_config
from src.osnet import build_osnet, load_osnet_weights
from src.utils import get_device

logger = logging.getLogger(__name__)

# OSNet'in on egitimli agirliklari ImageNet istatistikleri ile
# normalize edilmis girdiler bekler. Bu degerler kullaniciya acik bir
# ayar degil, agirliklara ait sabitlerdir; bu nedenle .env'de degil
# burada tanimlanir.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_PIXEL_MAX = 255.0

# Galeri tipi: Person UUID -> (K, D) embedding matrisi
Gallery = Mapping[str, np.ndarray]


# =====================================================================
#  SONUC YAPILARI
# =====================================================================


class MatchStatus(Enum):
    """Kimlik tahmininin sonucu."""

    MATCHED = "matched"      # Mevcut bir kisi ile eslesti
    NEW = "new"              # Yeni bir kisi
    UNCERTAIN = "uncertain"  # Emin olunamadi; karar verilmemeli


@dataclass
class ReIDResult:
    """Kimlik tahmini sonucu.

    NOT: `uuid` yalnizca MATCHED durumunda doludur. Yeni UUID uretimi
    `association_memory.py`'nin sorumlulugundadir.
    """

    status: MatchStatus
    uuid: Optional[str]
    similarity: float
    embedding: Optional[np.ndarray] = None

    @property
    def is_matched(self) -> bool:
        return self.status is MatchStatus.MATCHED

    @property
    def is_new(self) -> bool:
        return self.status is MatchStatus.NEW

    @property
    def is_uncertain(self) -> bool:
        return self.status is MatchStatus.UNCERTAIN


# =====================================================================
#  MODEL ARKA UCLARI
# =====================================================================


def _load_builtin(arch: str, weights: Optional[Path], embedding_size: int):
    """Projeye gomulu OSNet (src/osnet.py) ile modeli yukler.

    Ek kutuphane gerektirmez; torchreid / boxmot agirlik dosyalarini
    dogrudan okur. Agirliksiz bir ReID modeli rastgele embedding uretir
    ve kimlik eslestirmesini anlamsiz kilar; bu nedenle dosya zorunludur.
    """
    if weights is None:
        raise FileNotFoundError(
            "builtin arka ucu icin egitilmis agirlik dosyasi gerekir "
            "(OSNET_MODEL_PATH). Indirmek icin: python install.py"
        )
    model = build_osnet(arch, feature_dim=embedding_size)
    load_osnet_weights(model, weights)
    return model


def _load_torchreid(arch: str, weights: Optional[Path], embedding_size: int):
    """deep-person-reid (torchreid) uzerinden OSNet yukler."""
    import torchreid  # type: ignore

    model = torchreid.models.build_model(
        name=arch, num_classes=embedding_size, pretrained=weights is None
    )
    if weights is not None:
        torchreid.utils.load_pretrained_weights(model, str(weights))
    return model


def _load_torchscript(arch: str, weights: Optional[Path], embedding_size: int):
    """TorchScript olarak disa aktarilmis bir OSNet modeli yukler."""
    if weights is None:
        raise FileNotFoundError("torchscript arka ucu icin OSNET_MODEL_PATH zorunludur.")
    return torch.jit.load(str(weights), map_location="cpu")


_BACKEND_LOADERS: Dict[str, Callable] = {
    "builtin": _load_builtin,
    "torchreid": _load_torchreid,
    "torchscript": _load_torchscript,
}

# 'boxmot' eski bir secenektir. boxmot 25.x ile OSNet'in kisi crop'u alan
# API'si kaldirildigi icin artik ayni agirlik dosyasini okuyan yerlesik
# arka uca yonlendirilir.
_BACKEND_ALIASES: Dict[str, str] = {"boxmot": "builtin"}

# 'auto' secildiginde deneme sirasi. Yerlesik arka uc ek bagimlilik
# gerektirmedigi icin ilk sirada.
_AUTO_ORDER = ("builtin", "torchreid", "torchscript")


# =====================================================================
#  ReID
# =====================================================================


class ReID:
    """OSNet tabanli kisi yeniden kimliklendirme (Person Re-Identification).

    Kullanim:
        reid = ReID()
        result = reid.identify(person_crop, gallery)
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        gallery_provider: Optional[Callable[[], Gallery]] = None,
    ) -> None:
        self.config = config or load_config()

        # `reid.identify(crop)` seklinde galerisiz cagri yapilabilmesi
        # icin opsiyonel bir saglayici. Bu bir hafiza YONETIMI degildir;
        # modul yalnizca okur, hicbir kaydi degistirmez.
        self._gallery_provider = gallery_provider

        self.arch: str = self.config.OSNET_ARCH
        self.embedding_size: int = self.config.EMBEDDING_SIZE
        self.batch_size: int = self.config.REID_BATCH_SIZE
        self.input_height: int = self.config.REID_INPUT_HEIGHT
        self.input_width: int = self.config.REID_INPUT_WIDTH
        self.min_crop_height: int = self.config.REID_MIN_CROP_HEIGHT

        self.reid_threshold: float = self.config.REID_THRESHOLD
        self.new_person_threshold: float = self.config.NEW_PERSON_THRESHOLD
        self.match_strategy: str = self.config.REID_MATCH_STRATEGY.lower()

        # USE_GPU=false ise ReID de CPU'da calisir (detector ile tutarli).
        self.device: str = get_device(
            preferred=self.config.REID_DEVICE, use_gpu=self.config.USE_GPU
        )
        self.half: bool = self.config.REID_USE_FP16 and self.device != "cpu"

        if self.device != "cpu":
            # Sabit giris boyutu -> cuDNN en hizli konvolusyon algoritmasini
            # ilk cagrida secip onbellege alir.
            torch.backends.cudnn.benchmark = True

        # Normalizasyon sabitleri hedef cihazda bir kez olusturulur.
        dtype = torch.float16 if self.half else torch.float32
        self._mean = torch.tensor(IMAGENET_MEAN * _PIXEL_MAX, device=self.device, dtype=dtype).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD * _PIXEL_MAX, device=self.device, dtype=dtype).view(1, 3, 1, 1)

        self.weights_path: Optional[Path] = self._resolve_weights(
            self.config.OSNET_MODEL_PATH
        )
        self.backend_name, self.model = self._build_model(
            self.config.REID_BACKEND.lower()
        )

        logger.info(
            "ReID hazir | %s (%s) | device=%s | fp16=%s | giris=%dx%d | "
            "esikler: eslesme>=%.2f, yeni<%.2f",
            self.arch,
            self.backend_name,
            self.device,
            self.half,
            self.input_height,
            self.input_width,
            self.reid_threshold,
            self.new_person_threshold,
        )

    # ----------------------------------------------------------------
    #  KURULUM
    # ----------------------------------------------------------------

    @staticmethod
    def _resolve_weights(raw_path: str) -> Optional[Path]:
        """OSNet agirlik dosyasinin yolunu cozer; yoksa None."""
        if not raw_path:
            return None
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / raw_path
        if not candidate.exists():
            logger.warning(
                "OSNet agirlik dosyasi bulunamadi: %s. Indirmek icin: python install.py",
                candidate,
            )
            return None
        return candidate

    def _build_model(self, backend: str):
        """Yapilandirilan arka uc ile OSNet modelini yukler."""
        if backend in _BACKEND_ALIASES:
            logger.warning(
                "REID_BACKEND='%s' artik desteklenmiyor; '%s' kullaniliyor.",
                backend,
                _BACKEND_ALIASES[backend],
            )
            backend = _BACKEND_ALIASES[backend]

        if backend != "auto" and backend not in _BACKEND_LOADERS:
            raise ValueError(
                f"Gecersiz REID_BACKEND: '{backend}'. "
                f"Secenekler: auto, {', '.join(_BACKEND_LOADERS)}"
            )
        candidates = _AUTO_ORDER if backend == "auto" else (backend,)

        errors: List[str] = []
        for name in candidates:
            loader = _BACKEND_LOADERS[name]
            try:
                model = loader(self.arch, self.weights_path, self.embedding_size)
                model = model.to(self.device).eval()
                if self.half:
                    model = model.half()
                return name, model
            except Exception as exc:
                errors.append(f"  - {name}: {exc}")
                logger.debug("ReID arka ucu '%s' yuklenemedi: %s", name, exc)

        raise RuntimeError(
            "OSNet modeli yuklenemedi. Denenen arka uclar:\n"
            + "\n".join(errors)
            + "\n\nCozum onerileri:\n"
            "  python install.py              (agirlik dosyasini models/ altina indirir)\n"
            "  OSNET_MODEL_PATH degerinin dogru oldugundan emin olun.\n"
            "  OSNET_ARCH degerinin agirlik dosyasiyla ayni mimari oldugundan emin olun."
        )

    @torch.inference_mode()
    def warmup(self) -> None:
        """cuDNN algoritma secimini ve CUDA cekirdeklerini onceden yukler.

        Aksi halde ilk kisinin goruldugu karede belirgin bir gecikme olur.
        """
        dummy = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
        self.extract_embeddings([dummy])
        if self.device != "cpu":
            torch.cuda.synchronize()
        logger.info("ReID warmup tamamlandi.")

    # ----------------------------------------------------------------
    #  ON ISLEME
    # ----------------------------------------------------------------

    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Tek bir kisi crop'unu OSNet girdi boyutuna getirir.

        BGR (H, W, 3) uint8  ->  RGB (H', W', 3) uint8

        Float donusumu ve normalizasyon `_to_batch` icinde, hedef
        cihazda (GPU) toplu olarak yapilir.
        """
        resized = cv2.resize(
            crop, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR
        )
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    def _to_batch(self, crops: Sequence[np.ndarray]) -> torch.Tensor:
        """Crop'lari normalize edilmis (N, 3, H, W) tensore cevirir."""
        stacked = np.ascontiguousarray(np.stack([self.preprocess(c) for c in crops], axis=0))
        tensor = torch.from_numpy(stacked)
        if self.device != "cpu":
            tensor = tensor.pin_memory().to(self.device, non_blocking=True)
        tensor = tensor.permute(0, 3, 1, 2)
        tensor = tensor.half() if self.half else tensor.float()
        return (tensor - self._mean) / self._std

    @staticmethod
    def is_valid_crop(crop: Optional[np.ndarray], min_height: int) -> bool:
        """Cok kucuk veya bos crop'lar guvenilir embedding uretmez."""
        if crop is None or crop.size == 0 or crop.ndim != 3:
            return False
        height, width = crop.shape[:2]
        return height >= min_height and width > 0

    # ----------------------------------------------------------------
    #  EMBEDDING
    # ----------------------------------------------------------------

    @staticmethod
    def normalize(embeddings: np.ndarray) -> np.ndarray:
        """L2 normalizasyonu.

        Normalize edildikten sonra cosine similarity, basit bir ic
        carpima (dot product) esittir.
        """
        array = np.atleast_2d(np.asarray(embeddings, dtype=np.float32))
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms = np.maximum(norms, np.finfo(np.float32).eps)
        return array / norms

    @torch.inference_mode()
    def extract_embeddings(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Birden fazla crop icin normalize edilmis embedding uretir.

        Returns:
            (N, D) float32 dizi.
        """
        if not crops:
            return np.empty((0, self.embedding_size), dtype=np.float32)

        outputs: List[np.ndarray] = []
        for start in range(0, len(crops), self.batch_size):
            batch = self._to_batch(crops[start : start + self.batch_size])
            features = self.model(batch)
            if isinstance(features, (tuple, list)):
                features = features[0]
            outputs.append(features.float().cpu().numpy())

        return self.normalize(np.concatenate(outputs, axis=0))

    def extract_embedding(self, crop: np.ndarray) -> np.ndarray:
        """Tek bir crop icin normalize edilmis embedding uretir. (D,)"""
        return self.extract_embeddings([crop])[0]

    # ----------------------------------------------------------------
    #  BENZERLIK
    # ----------------------------------------------------------------

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Iki vektor arasindaki kosinus benzerligi. [-1, 1]"""
        vec_a = np.asarray(a, dtype=np.float32).ravel()
        vec_b = np.asarray(b, dtype=np.float32).ravel()
        denominator = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        if denominator < np.finfo(np.float32).eps:
            return 0.0
        return float(np.dot(vec_a, vec_b) / denominator)

    def _gallery_score(self, embedding: np.ndarray, stored: np.ndarray) -> float:
        """Bir kisinin saklanan embedding'lerine karsi benzerlik skoru."""
        matrix = self.normalize(stored)
        if matrix.size == 0:
            return -1.0
        if self.match_strategy == "mean":
            mean_vector = self.normalize(matrix.mean(axis=0))[0]
            return float(np.dot(mean_vector, embedding))
        return float(np.max(matrix @ embedding))

    def similarity(self, embedding: np.ndarray, stored: np.ndarray) -> float:
        """Bir embedding'in TEK bir kisinin saklanan embedding'lerine benzerligi.

        `match()` ile ayni strateji (REID_MATCH_STRATEGY) kullanilir.
        Kimligi bilinen bir Track'in tazeleme embedding'inin galeriye
        eklenip eklenmeyecegine karar vermek icin kullanilir (karar
        `main.py`'dedir).

        Returns:
            Benzerlik skoru; saklanan embedding yoksa -1.0.
        """
        stored_matrix = np.asarray(stored, dtype=np.float32)
        if stored_matrix.size == 0:
            return -1.0
        query = self.normalize(embedding)[0]
        return self._gallery_score(query, stored_matrix)

    def match(
        self,
        embedding: np.ndarray,
        gallery: Gallery,
        exclude: Optional[Iterable[str]] = None,
    ) -> ReIDResult:
        """Bir embedding'i galeri ile karsilastirarak kimlik tahmini yapar.

        Args:
            embedding: Normalize edilmis (D,) embedding.
            gallery: Person UUID -> (K, D) embedding matrisi.
            exclude: Ayni frame'de baska bir kisiye zaten atanmis UUID'ler.
                Bir UUID'nin ayni karede iki kisiye verilmesini onler.
        """
        blocked = set(exclude or ())
        query = self.normalize(embedding)[0]

        best_uuid: Optional[str] = None
        best_score = -1.0
        for uuid, stored in gallery.items():
            if uuid in blocked:
                continue
            score = self._gallery_score(query, stored)
            if score > best_score:
                best_uuid, best_score = uuid, score

        if best_uuid is not None and best_score >= self.reid_threshold:
            status = MatchStatus.MATCHED
        elif best_score < self.new_person_threshold:
            status, best_uuid = MatchStatus.NEW, None
        else:
            # Iki esik arasindaki kararsiz bolge: yanlis UUID atamaktansa
            # karar vermemek daha guvenlidir.
            status, best_uuid = MatchStatus.UNCERTAIN, None

        return ReIDResult(
            status=status,
            uuid=best_uuid,
            similarity=max(best_score, 0.0),
            embedding=query,
        )

    # ----------------------------------------------------------------
    #  GENEL API
    # ----------------------------------------------------------------

    def identify(
        self,
        person_crop: np.ndarray,
        gallery: Optional[Gallery] = None,
        exclude: Optional[Iterable[str]] = None,
    ) -> ReIDResult:
        """Bir kisi crop'undan kimlik tahmini yapar."""
        if not self.is_valid_crop(person_crop, self.min_crop_height):
            return ReIDResult(MatchStatus.UNCERTAIN, None, 0.0, None)

        if gallery is None:
            gallery = self._gallery_provider() if self._gallery_provider else {}

        embedding = self.extract_embedding(person_crop)
        if not gallery:
            return ReIDResult(MatchStatus.NEW, None, 0.0, embedding)

        return self.match(embedding, gallery, exclude)

    def identify_batch(
        self, person_crops: Sequence[np.ndarray], gallery: Optional[Gallery] = None
    ) -> List[ReIDResult]:
        """Bir karedeki tum kisileri tek seferde isler.

        Eslesen UUID'ler otomatik olarak sonrakiler icin bloklanir; boylece
        ayni UUID ayni karede iki kisiye atanmaz.
        """
        if gallery is None:
            gallery = self._gallery_provider() if self._gallery_provider else {}

        valid_indices = [
            i
            for i, crop in enumerate(person_crops)
            if self.is_valid_crop(crop, self.min_crop_height)
        ]
        results: List[ReIDResult] = [
            ReIDResult(MatchStatus.UNCERTAIN, None, 0.0, None)
            for _ in range(len(person_crops))
        ]
        if not valid_indices:
            return results

        embeddings = self.extract_embeddings([person_crops[i] for i in valid_indices])

        used: set = set()
        for index, embedding in zip(valid_indices, embeddings):
            if not gallery:
                results[index] = ReIDResult(MatchStatus.NEW, None, 0.0, embedding)
                continue
            result = self.match(embedding, gallery, exclude=used)
            if result.is_matched and result.uuid is not None:
                used.add(result.uuid)
            results[index] = result
        return results

    def set_gallery_provider(self, provider: Callable[[], Gallery]) -> None:
        """Galeri saglayicisini baglar (ornegin `memory.get_gallery`)."""
        self._gallery_provider = provider

    def __repr__(self) -> str:
        return (
            f"ReID(arch={self.arch}, backend={self.backend_name}, "
            f"device={self.device}, fp16={self.half}, "
            f"threshold={self.reid_threshold})"
        )
