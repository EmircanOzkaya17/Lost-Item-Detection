"""
osnet.py
========

Projeye gomulu **OSNet (Omni-Scale Network)** uygulamasi.

Neden projenin icinde?
--------------------------------------------------------------------
OSNet'i disaridan saglayan kutuphanelerin API'leri surumden surume
kiriliyor:

* `boxmot` 25.x surumunde `boxmot.appearance` modulu tamamen kaldirildi;
  yeni API kisi crop'u degil, kare + tespit listesi bekliyor.
* `torchreid` PyPI'da guncel degil; GitHub'dan kurulumu Windows'ta C
  derleyicisi (MSVC) gerektiriyor.

Bu dosya ayni mimariyi bagimsiz olarak tanimlar ve torchreid / boxmot
model zoo'sundaki agirlik dosyalarini (ornegin `osnet_x1_0_msmt17.pt`)
DOGRUDAN yukleyebilir: katman adlari o dosyalardaki anahtarlarla
birebir aynidir.

Bu modul yalnizca ag tanimini ve agirlik yuklemeyi icerir. On isleme,
embedding normalizasyonu ve eslestirme `reid.py` icindedir.

Referans:
    Zhou et al., "Omni-Scale Feature Learning for Person
    Re-Identification", ICCV 2019.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

__all__ = [
    "OSNet",
    "OSNET_VARIANTS",
    "build_osnet",
    "load_osnet_weights",
    "WEIGHT_URLS",
]

# torchreid / boxmot model zoo'sundaki egitilmis agirliklarin indirme
# adresleri (Google Drive). install.py bu tabloyu kullanir.
WEIGHT_URLS: Dict[str, str] = {
    "osnet_x1_0_msmt17.pt": "https://drive.google.com/uc?id=112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M",
    "osnet_x1_0_market1501.pt": "https://drive.google.com/uc?id=1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA",
    "osnet_x1_0_dukemtmcreid.pt": "https://drive.google.com/uc?id=1QZO_4sNf4hdOKKKzKc-TZU9WW1v6zQbq",
    "osnet_x0_75_msmt17.pt": "https://drive.google.com/uc?id=1QEGO6WnJ-BmUzVPd3q9NoaO_GsPNlmWc",
    "osnet_x0_5_msmt17.pt": "https://drive.google.com/uc?id=1UT3AxIaDvS2PdxzZmbkLmjtiqq7AIKCv",
    "osnet_x0_25_msmt17.pt": "https://drive.google.com/uc?id=1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF",
    "osnet_ibn_x1_0_msmt17.pt": "https://drive.google.com/uc?id=1q3Sj2ii34NlfxA4LvmHdWO_75NDRmECJ",
}

# Agirlik dosyasindaki anahtarlarin en az bu orani modelle eslesmelidir;
# aksi halde dosya bu mimariye ait degildir (yanlis dosya / yanlis ARCH).
_MIN_MATCH_RATIO = 0.9
_CLASSIFIER_PREFIX = "classifier."


# =====================================================================
#  TEMEL KATMANLAR
#  (Adlar torchreid agirlik dosyalarindaki anahtarlarla ayni olmalidir.)
# =====================================================================


class ConvLayer(nn.Module):
    """Konvolusyon + BatchNorm (veya InstanceNorm) + ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        instance_norm: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False,
        )
        self.bn = (
            nn.InstanceNorm2d(out_channels, affine=True)
            if instance_norm
            else nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    """1x1 konvolusyon + BatchNorm + ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    """1x1 konvolusyon + BatchNorm (aktivasyon yok)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    """Hafif 3x3 konvolusyon: 1x1 dogrusal + derinlemesine (depthwise) 3x3."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3,
            padding=1, bias=False, groups=out_channels,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    """Farkli olceklerdeki akislari kanal bazinda agirliklandiran kapi.

    Ayni kapi (paylasilan agirliklar) dort akisin hepsine uygulanir.
    """

    _REDUCTION = 16

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = channels // self._REDUCTION
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=True)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.global_avgpool(x)
        weights = self.gate_activation(self.fc2(self.relu(self.fc1(weights))))
        return x * weights


class OSBlock(nn.Module):
    """Omni-scale blok: 1, 2, 3 ve 4 hafif konvolusyonluk dort akis."""

    _BOTTLENECK = 4

    def __init__(self, in_channels: int, out_channels: int, instance_norm: bool = False) -> None:
        super().__init__()
        mid = out_channels // self._BOTTLENECK
        self.conv1 = Conv1x1(in_channels, mid)
        self.conv2a = LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(*(LightConv3x3(mid, mid) for _ in range(2)))
        self.conv2c = nn.Sequential(*(LightConv3x3(mid, mid) for _ in range(3)))
        self.conv2d = nn.Sequential(*(LightConv3x3(mid, mid) for _ in range(4)))
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, out_channels)
        self.downsample = (
            Conv1x1Linear(in_channels, out_channels) if in_channels != out_channels else None
        )
        self.IN = nn.InstanceNorm2d(out_channels, affine=True) if instance_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        x1 = self.conv1(x)
        x2 = (
            self.gate(self.conv2a(x1))
            + self.gate(self.conv2b(x1))
            + self.gate(self.conv2c(x1))
            + self.gate(self.conv2d(x1))
        )
        out = self.conv3(x2) + identity
        if self.IN is not None:
            out = self.IN(out)
        return F.relu(out)


# =====================================================================
#  OSNet
# =====================================================================


class OSNet(nn.Module):
    """OSNet omurgasi.

    Egitim modunda siniflandirici logit'lerini, cikarim (eval) modunda
    `feature_dim` boyutlu embedding vektorunu dondurur. Bu projede model
    her zaman eval modunda kullanilir.
    """

    _LAYERS = (2, 2, 2)

    def __init__(
        self,
        channels: Sequence[int],
        num_classes: int = 1000,
        feature_dim: int = 512,
        ibn: bool = False,
    ) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError(f"OSNet icin 4 kanal degeri gerekir. Alinan: {channels}")
        self.feature_dim = feature_dim

        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, instance_norm=ibn)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(channels[0], channels[1], self._LAYERS[0], True, ibn)
        self.conv3 = self._make_layer(channels[1], channels[2], self._LAYERS[1], True, False)
        self.conv4 = self._make_layer(channels[2], channels[3], self._LAYERS[2], False, False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels[3], feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)
        self._init_params()

    @staticmethod
    def _make_layer(
        in_channels: int,
        out_channels: int,
        blocks: int,
        reduce_spatial_size: bool,
        instance_norm: bool,
    ) -> nn.Sequential:
        layers: List[nn.Module] = [OSBlock(in_channels, out_channels, instance_norm)]
        layers.extend(
            OSBlock(out_channels, out_channels, instance_norm) for _ in range(1, blocks)
        )
        if reduce_spatial_size:
            layers.append(
                nn.Sequential(Conv1x1(out_channels, out_channels), nn.AvgPool2d(2, stride=2))
            )
        return nn.Sequential(*layers)

    def _init_params(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.InstanceNorm2d)):
                if getattr(module, "weight", None) is not None:
                    nn.init.ones_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.conv1(x))
        x = self.conv5(self.conv4(self.conv3(self.conv2(x))))
        v = self.fc(torch.flatten(self.global_avgpool(x), 1))
        if not self.training:
            return v
        return self.classifier(v)


# Mimari adi -> (kanallar, IBN)
OSNET_VARIANTS: Dict[str, Tuple[Tuple[int, int, int, int], bool]] = {
    "osnet_x1_0": ((64, 256, 384, 512), False),
    "osnet_x0_75": ((48, 192, 288, 384), False),
    "osnet_x0_5": ((32, 128, 192, 256), False),
    "osnet_x0_25": ((16, 64, 96, 128), False),
    "osnet_ibn_x1_0": ((64, 256, 384, 512), True),
}


def build_osnet(arch: str, feature_dim: int = 512) -> OSNet:
    """Mimari adina gore bos (agirliksiz) bir OSNet olusturur."""
    key = arch.strip().lower()
    if key not in OSNET_VARIANTS:
        raise ValueError(
            f"Desteklenmeyen OSNET_ARCH: '{arch}'. Secenekler: {', '.join(OSNET_VARIANTS)}"
        )
    channels, ibn = OSNET_VARIANTS[key]
    return OSNet(channels=channels, feature_dim=feature_dim, ibn=ibn)


# =====================================================================
#  AGIRLIK YUKLEME
# =====================================================================


def _extract_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    """Farkli kayit bicimlerinden state_dict'i cikarir.

    torchreid: {'state_dict': {...}, 'epoch': ..} veya dogrudan state_dict.
    DataParallel ile kaydedilmisse anahtarlar 'module.' ile baslar.
    """
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            inner = checkpoint.get(key)
            if isinstance(inner, Mapping):
                return inner
            if isinstance(inner, nn.Module):
                return inner.state_dict()
        return checkpoint  # type: ignore[return-value]
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    raise ValueError(f"Taninmayan agirlik dosyasi bicimi: {type(checkpoint).__name__}")


def _torch_load(path: Path) -> object:
    """Agirlik dosyasini guvenli modda okur; gerekirse tam moda duser.

    PyTorch 2.6+ `weights_only=True` varsayilanini kullanir. Eski torchreid
    dosyalari icinde numpy skalerleri bulunabildigi icin guvenli mod
    basarisiz olabilir; bu durumda dosya YEREL ve kullanicinin kendi
    indirdigi dosya oldugundan tam mod ile okunur.
    """
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover - dosyaya bagli
        logger.debug("weights_only=True ile okunamadi (%s); tam mod deneniyor.", exc)
        return torch.load(str(path), map_location="cpu", weights_only=False)


def load_osnet_weights(model: OSNet, weights: Union[str, Path]) -> Tuple[int, int]:
    """Egitilmis agirliklari modele yukler.

    Siniflandirici (classifier) katmani yuklenmez: sinif sayisi egitim
    veri setine ozgudur ve embedding cikariminda kullanilmaz.

    Returns:
        (yuklenen anahtar sayisi, modeldeki toplam anahtar sayisi)

    Raises:
        RuntimeError: Dosya bu mimariye ait gorunmuyorsa.
    """
    path = Path(weights)
    if not path.exists():
        raise FileNotFoundError(f"OSNet agirlik dosyasi bulunamadi: {path}")

    state = _extract_state_dict(_torch_load(path))
    model_state = model.state_dict()

    matched: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for raw_key, value in state.items():
        key = raw_key[len("module."):] if raw_key.startswith("module.") else raw_key
        if key.startswith(_CLASSIFIER_PREFIX):
            continue
        target = model_state.get(key)
        if target is not None and hasattr(value, "shape") and target.shape == value.shape:
            matched[key] = value
        else:
            skipped.append(key)

    expected = [k for k in model_state if not k.startswith(_CLASSIFIER_PREFIX)]
    ratio = len(matched) / max(len(expected), 1)
    if ratio < _MIN_MATCH_RATIO:
        raise RuntimeError(
            f"'{path.name}' dosyasi bu mimariyle uyusmuyor: {len(matched)}/{len(expected)} "
            f"anahtar eslesti. OSNET_ARCH degerini ve dosyayi kontrol edin."
        )

    model_state.update(matched)
    model.load_state_dict(model_state)
    if skipped:
        logger.debug("Eslesmeyen %d anahtar atlandi: %s", len(skipped), skipped[:10])
    logger.info(
        "OSNet agirliklari yuklendi: %s (%d/%d anahtar)", path.name, len(matched), len(expected)
    )
    return len(matched), len(expected)
