#!/usr/bin/env python3
"""
install.py
==========

Projenin ihtiyac duydugu kutuphaneleri kontrol eder, **eksik veya yanlis
surumde olanlari** kurar ve GPU'nun gercekten kullanilabildigini dogrular.

Kullanim:
    python install.py                 # eksikleri kur, GPU'yu otomatik algila
    python install.py --dry-run       # hicbir sey kurma, sadece rapor ver
    python install.py --cuda cu130    # PyTorch CUDA etiketini elle sec
    python install.py --cuda cpu      # GPU yoksa / istemiyorsan
    python install.py --skip-weights  # OSNet agirligini indirme

Onemli ayrintilar:
    * PyTorch, ultralytics'ten ONCE ve her zaman PyTorch'un kendi paket
      deposundan (download.pytorch.org) kurulur. Aksi halde:
        - Windows'ta PyPI'daki torch CPU surumudur -> GPU kullanilamaz.
        - Linux'ta PyPI'daki torch (2.11+) CUDA 13 ister -> eski
          suruculerde calismaz.
    * Kurulu torch CPU surumuyse ve makinede NVIDIA GPU varsa "kurulu"
      sayilmaz; kaldirilip CUDA surumu kurulur.
    * Etiket secimi hem surucunun CUDA surumune hem de GPU mimarisine
      (compute capability) bakar: CUDA 13, Volta ve oncesi kartlari
      (compute < 7.5) desteklemez.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import importlib.util
import json
import platform
import re
import subprocess
import sys
import sysconfig
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
MIN_PYTHON = (3, 10)

OK, MISSING, OLD, FAIL, WARN = "[VAR]", "[YOK]", "[ESKI]", "[HATA]", "[UYARI]"

# --break-system-packages gibi tum pip cagrilarina eklenecek argumanlar
EXTRA_PIP_ARGS: List[str] = []

TORCH_INDEX = "https://download.pytorch.org/whl"
PYTORCH_HELP = "https://pytorch.org/get-started/locally/"

# Guncel PyTorch surumlerinin yayinlandigi CUDA etiketleri.
# (surucunun desteklemesi gereken en dusuk CUDA surumu, etiket)
CUDA_13_TAG = (13.0, "cu130")
CUDA_12_TAG = (12.6, "cu126")
# CUDA 11.8 destekleyen son PyTorch surumu; eski suruculer icin.
LEGACY_TAG = (11.8, "cu118")
LEGACY_TORCH_PINS = ["torch==2.7.1", "torchvision==0.22.1"]

# CUDA 13 derlemeleri bu compute capability'nin altini desteklemez
# (Maxwell, Pascal, Volta). Bu kartlar cu126 ile calisir.
CUDA13_MIN_CAPABILITY = 7.5
# Blackwell (RTX 50xx vb.) ve sonrasi; cu126 bu kartlari desteklemez.
BLACKWELL_CAPABILITY = 10.0

# OSNet agirligi (src/osnet.py ile ayni tablo; burada tekrar edilir ki
# install.py torch kurulu olmadan da calisabilsin).
OSNET_WEIGHTS = {
    "osnet_x1_0_msmt17.pt": "https://drive.google.com/uc?id=112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M",
}
DEFAULT_OSNET_FILE = "osnet_x1_0_msmt17.pt"


@dataclass
class Package:
    """Kurulumu kontrol edilecek tek bir paket."""

    import_name: str          # import ederken kullanilan ad (cv2)
    pip_name: str             # pip ile kurulurken kullanilan ad (opencv-python)
    purpose: str              # ne ise yaradigi
    min_version: Optional[str] = None
    # Surum sorgusu icin denenecek dagitim adlari. opencv gibi paketlerin
    # birden fazla dagitimi ayni modulu saglar.
    dist_names: Sequence[str] = field(default_factory=tuple)

    def candidates(self) -> Sequence[str]:
        return tuple(self.dist_names) or (self.pip_name,)


@dataclass
class GpuInfo:
    """nvidia-smi'den okunan GPU bilgisi."""

    name: str
    driver_cuda: float        # surucunun destekledigi en yuksek CUDA surumu
    capability: float         # compute capability (ornegin 8.6)


# ---------------------------------------------------------------------
#  PAKET LISTESI
#  requirements.txt duz `pip install -r` icindir; buradaki tablo ek
#  olarak import adlarini ve amaclari da tasir (kontrol icin gerekli).
# ---------------------------------------------------------------------

CORE_PACKAGES: List[Package] = [
    Package("numpy", "numpy", "Dizi ve embedding islemleri", "1.26.0"),
    Package(
        "cv2",
        "opencv-python",
        "Video okuma/yazma, CLAHE, cizim",
        "4.9.0",
        dist_names=("opencv-python", "opencv-contrib-python", "opencv-python-headless"),
    ),
    Package("dotenv", "python-dotenv", ".env dosyasinin okunmasi", "1.0.0"),
]

TORCH_PACKAGES: List[Package] = [
    Package("torch", "torch", "Derin ogrenme calisma zamani", "2.2.0"),
    Package("torchvision", "torchvision", "Goruntu modelleri ve donusumler", "0.17.0"),
]

YOLO_PACKAGES: List[Package] = [
    Package("ultralytics", "ultralytics", "YOLO tespiti + BoT-SORT takibi", "8.3.0"),
    # ultralytics takipcisi lap'i calisma aninda kurmaya calisir; bu
    # cevrimdisi ya da kisitli ortamlarda basarisiz olur. Onceden kurulur.
    Package("lap", "lap", "Takipcide macar algoritmasi (atama)", "0.5.12"),
    Package("gdown", "gdown", "OSNet agirligini Google Drive'dan indirme", "5.1.0"),
]

REQUIRED_DIRS = ("assets", "results", "models")


# =====================================================================
#  KONTROL YARDIMCILARI
# =====================================================================


def parse_version(text: str) -> Tuple[int, ...]:
    """'4.13.0.post1' -> (4, 13, 0)"""
    parts: List[int] = []
    for chunk in re.split(r"[.\-+_]", text):
        match = re.match(r"^(\d+)", chunk)
        if not match:
            break
        parts.append(int(match.group(1)))
    return tuple(parts) or (0,)


def version_satisfies(installed: Optional[str], minimum: Optional[str]) -> bool:
    """Surum yeterli mi? Surum okunamazsa yeterli kabul edilir."""
    if minimum is None or installed is None:
        return True
    return parse_version(installed) >= parse_version(minimum)


def installed_version(package: Package) -> Optional[str]:
    for name in package.candidates():
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def is_importable(import_name: str) -> bool:
    """Modulu CALISTIRMADAN varligini kontrol eder."""
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError):
        return False


def inspect(package: Package) -> Tuple[str, Optional[str]]:
    """Paketin durumunu dondurur: (OK | MISSING | OLD, surum)"""
    if not is_importable(package.import_name):
        return MISSING, None
    version = installed_version(package)
    if not version_satisfies(version, package.min_version):
        return OLD, version
    return OK, version


def run_python(code: str, timeout: int = 180) -> Tuple[int, str]:
    """Kodu AYRI bir Python surecinde calistirir.

    torch bu surece import edilirse, kaldirilip yeniden kurulduktan sonra
    eski surum bellekte kalir. Bu yuzden torch kontrolleri hep ayri
    surecte yapilir.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    return result.returncode, (result.stdout + result.stderr).strip()


# =====================================================================
#  ORTAM
# =====================================================================


def in_virtualenv() -> bool:
    """Sanal ortam (venv) icinde mi calisiyoruz?"""
    return sys.prefix != sys.base_prefix


def is_externally_managed() -> bool:
    """PEP 668: dagitim tarafindan yonetilen Python mu?"""
    stdlib = sysconfig.get_paths().get("stdlib")
    return bool(stdlib) and (Path(stdlib) / "EXTERNALLY-MANAGED").exists()


def print_venv_help() -> None:
    """Sanal ortam olusturma talimatlarini yazdirir."""
    windows = platform.system() == "Windows"
    activate = ".venv\\Scripts\\activate" if windows else "source .venv/bin/activate"
    print()
    print("  Bu Python dagitim tarafindan yonetiliyor (PEP 668); sisteme")
    print("  dogrudan kurulum yapilamaz. Iki secenek var:")
    print()
    print("  1) Sanal ortam kullanin (ONERILEN):")
    print(f"       {Path(sys.executable).name} -m venv .venv")
    print(f"       {activate}")
    print("       python install.py")
    print()
    print("  2) Sistem Python'una kurmayi zorlayin (isletim sisteminizin")
    print("     paketlerini bozabilir):")
    print("       python install.py --break-system-packages")
    print()


def pip_run(args: Sequence[str]) -> bool:
    """pip'i MEVCUT Python yorumlayicisi ile calistirir.

    `sys.executable -m pip` kullanmak kritiktir: sistemde birden fazla
    Python varsa paketlerin dogru ortama gitmesini garanti eder.
    """
    command = [sys.executable, "-m", "pip", *args]
    print(f"    $ pip {' '.join(args)}")
    try:
        result = subprocess.run(command, check=False)
    except OSError as exc:
        print(f"    {FAIL} pip calistirilamadi: {exc}")
        return False
    return result.returncode == 0


def pip_install(targets: Sequence[str], extra_args: Sequence[str] = ()) -> bool:
    return pip_run(["install", *EXTRA_PIP_ARGS, *extra_args, *targets])


# =====================================================================
#  GPU ALGILAMA VE PYTORCH SECIMI
# =====================================================================


def detect_gpu() -> Optional[GpuInfo]:
    """nvidia-smi ile GPU adini, surucu CUDA surumunu ve mimarisini okur."""
    try:
        summary = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if summary.returncode != 0:
        return None

    match = re.search(r"CUDA Version:\s*([\d.]+)", summary.stdout)
    driver_cuda = float(match.group(1)) if match else 0.0

    name, capability = "NVIDIA GPU", 0.0
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        first = query.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        name = parts[0] or name
        capability = float(parts[1]) if len(parts) > 1 else 0.0
    except (OSError, subprocess.TimeoutExpired, IndexError, ValueError):
        pass  # eski suruculer compute_cap sorgusunu desteklemez
    return GpuInfo(name=name, driver_cuda=driver_cuda, capability=capability)


def choose_torch_build(choice: str, gpu: Optional[GpuInfo]) -> Tuple[str, str, List[str]]:
    """PyTorch icin (etiket, aciklama, sabit surumler) dondurur."""
    if choice != "auto":
        pins = LEGACY_TORCH_PINS if choice == LEGACY_TAG[1] else []
        return choice, f"elle secildi: {choice}", pins
    if gpu is None:
        return "cpu", "NVIDIA GPU bulunamadi, CPU surumu", []

    cuda, cap = gpu.driver_cuda, gpu.capability
    old_arch = 0 < cap < CUDA13_MIN_CAPABILITY

    if cuda >= CUDA_13_TAG[0] and not old_arch:
        return CUDA_13_TAG[1], f"surucu CUDA {cuda}, compute {cap or '?'}", []
    if cuda >= CUDA_12_TAG[0]:
        note = f"surucu CUDA {cuda}, compute {cap or '?'}"
        if old_arch:
            note += " (eski mimari: CUDA 13 desteklemiyor)"
        if cap >= BLACKWELL_CAPABILITY:
            print(f"  {WARN} {gpu.name} (compute {cap}) icin surucunuz eski. cu126 bu karti")
            print("          desteklemez; NVIDIA surucusunu 580+ (CUDA 13) surumune guncelleyip")
            print("          install.py'yi yeniden calistirin.")
        return CUDA_12_TAG[1], note, []
    if cuda >= LEGACY_TAG[0]:
        print(f"  {WARN} Surucu CUDA {cuda}: guncel PyTorch icin eski. PyTorch 2.7.1 (cu118)")
        print("          kurulacak. Mumkunse NVIDIA surucusunu guncelleyin.")
        return LEGACY_TAG[1], f"eski surucu (CUDA {cuda})", LEGACY_TORCH_PINS
    return "cpu", f"surucu CUDA {cuda or 'bilinmiyor'} cok eski, CPU surumu", []


def installed_torch_cuda() -> Optional[str]:
    """Kurulu torch'un derlendigi CUDA surumu. 'cpu' veya None (kurulu degil)."""
    if not is_importable("torch"):
        return None
    code, out = run_python("import torch; print(torch.version.cuda or 'cpu')")
    return out.splitlines()[-1].strip() if code == 0 and out else "bozuk"


def install_torch(
    tag: str, description: str, pins: List[str], dry_run: bool, force: bool = False
) -> bool:
    """PyTorch'u dogru derleme ile kurar; yanlis derleme varsa degistirir.

    Args:
        force: Etiket elle secildiyse (--cuda cu126 gibi) kurulu torch
            hangi derleme olursa olsun degistirilir.
    """
    print("\n[2/7] PyTorch")
    print(f"  (secim: {tag} - {description})")
    report(TORCH_PACKAGES)

    current = installed_torch_cuda()
    wants_gpu = tag != "cpu"
    needs_replace = current is not None and (
        force or current == "bozuk" or (wants_gpu and current == "cpu")
    )
    needs_install = current is None or needs_replace or any(
        inspect(p)[0] is not OK for p in TORCH_PACKAGES
    )

    if current is not None:
        print(f"  Kurulu torch derlemesi: {'CUDA ' + current if current not in ('cpu', 'bozuk') else current}")
    if not needs_install:
        print("  -> Uygun surum kurulu, atlaniyor.")
        return True
    if dry_run:
        action = "degistirilecekti (CPU -> CUDA)" if needs_replace else "kurulacakti"
        print(f"  -> torch/torchvision {action}: {TORCH_INDEX}/{tag}")
        return True

    if needs_replace:
        reason = "elle secilen derleme" if force else "kurulu torch GPU'yu kullanamiyor"
        print(f"  {WARN} {reason}; kaldirilip {tag} kuruluyor.")
        pip_run(["uninstall", "-y", "torch", "torchvision", "torchaudio"])

    targets = pins or ["torch", "torchvision"]
    if not pip_install(targets, ["--index-url", f"{TORCH_INDEX}/{tag}"]):
        print(f"  {FAIL} PyTorch kurulamadi. Elle kurulum: {PYTORCH_HELP}")
        return False
    print("  -> Tamamlandi.")
    return True


# =====================================================================
#  ADIMLAR
# =====================================================================


def report(packages: Sequence[Package]) -> List[Package]:
    """Durumu yazdirir ve kurulmasi gerekenleri dondurur."""
    needed: List[Package] = []
    for package in packages:
        status, version = inspect(package)
        detail = f"{version}" if version else "kurulu degil"
        if status is OK:
            print(f"  {OK:7s} {package.pip_name:22s} {detail:14s} {package.purpose}")
        else:
            reason = (
                f"{detail} < {package.min_version} gerekli"
                if status is OLD
                else "kurulacak"
            )
            print(f"  {status:7s} {package.pip_name:22s} {reason:14s} {package.purpose}")
            needed.append(package)
    return needed


def install_group(
    title: str,
    packages: Sequence[Package],
    dry_run: bool,
    extra_args: Sequence[str] = (),
    note: str = "",
) -> bool:
    print(f"\n{title}")
    if note:
        print(f"  ({note})")
    needed = report(packages)

    if not needed:
        print("  -> Hepsi kurulu, atlaniyor.")
        return True
    if dry_run:
        print(f"  -> Kurulacakti: {', '.join(p.pip_name for p in needed)}")
        return True

    targets = [
        f"{p.pip_name}>={p.min_version}" if p.min_version else p.pip_name
        for p in needed
    ]
    if not pip_install(targets, extra_args):
        print(f"  {FAIL} Kurulum basarisiz: {', '.join(targets)}")
        return False
    print("  -> Tamamlandi.")
    return True


def ensure_directories(dry_run: bool) -> None:
    print("\n[5/7] Proje klasorleri")
    for name in REQUIRED_DIRS:
        path = PROJECT_ROOT / name
        if path.exists():
            print(f"  {OK:7s} {name}/")
        elif dry_run:
            print(f"  {MISSING:7s} {name}/  olusturulacakti")
        else:
            path.mkdir(parents=True, exist_ok=True)
            print(f"  {OK:7s} {name}/  olusturuldu")


def ensure_weights(dry_run: bool, skip_download: bool) -> bool:
    """YOLO ve OSNet agirliklarini kontrol eder; OSNet eksikse indirir."""
    print("\n[6/7] Model agirliklari")

    yolo = PROJECT_ROOT / "yolo11m.pt"
    if yolo.exists():
        print(f"  {OK:7s} yolo11m.pt ({yolo.stat().st_size / 1e6:.0f} MB)")
    else:
        print(f"  {MISSING:7s} yolo11m.pt  -> ultralytics ilk calistirmada otomatik indirir")

    target = PROJECT_ROOT / "models" / DEFAULT_OSNET_FILE
    if target.exists() and target.stat().st_size > 0:
        print(f"  {OK:7s} models/{DEFAULT_OSNET_FILE} ({target.stat().st_size / 1e6:.0f} MB)")
        return True

    url = OSNET_WEIGHTS[DEFAULT_OSNET_FILE]
    manual = (
        f"          Elle indirme: {url}\n"
        f"          Dosyayi models/{DEFAULT_OSNET_FILE} olarak kaydedin."
    )
    if dry_run or skip_download:
        print(f"  {MISSING:7s} models/{DEFAULT_OSNET_FILE}  indirilecekti")
        return True
    if not is_importable("gdown"):
        print(f"  {FAIL:7s} gdown kurulu degil, OSNet indirilemedi.\n{manual}")
        return False

    print(f"  {MISSING:7s} models/{DEFAULT_OSNET_FILE}  indiriliyor...")
    target.parent.mkdir(parents=True, exist_ok=True)
    code, out = run_python(
        "import gdown, sys; "
        f"sys.exit(0 if gdown.download({url!r}, {str(target)!r}, quiet=False) else 1)",
        timeout=900,
    )
    if code != 0 or not target.exists() or target.stat().st_size == 0:
        target.unlink(missing_ok=True)
        print(f"  {FAIL:7s} Indirme basarisiz (Google Drive kotasi olabilir).\n{manual}")
        if out:
            print("          " + out.splitlines()[-1][:200])
        return False
    print(f"  {OK:7s} models/{DEFAULT_OSNET_FILE} indirildi")
    return True


# Ayri surecte calisan GPU dogrulama betigi. Yalnizca "cuda mevcut mu"
# sorusu yetmez: kurulu torch, kartin mimarisi icin derlenmemisse
# ("no kernel image is available") ilk GPU isleminde coker.
_GPU_CHECK = r"""
import json, torch
info = {"torch": torch.__version__, "cuda": torch.version.cuda, "available": torch.cuda.is_available()}
if info["available"]:
    p = torch.cuda.get_device_properties(0)
    info.update(name=p.name, cap=f"{p.major}.{p.minor}", mem=round(p.total_memory / 1024**3, 1),
                arch_ok=f"sm_{p.major}{p.minor}" in torch.cuda.get_arch_list() or
                        any(a.startswith(f"compute_{p.major}") for a in torch.cuda.get_arch_list()))
    try:
        x = torch.randn(256, 256, device="cuda", dtype=torch.float16)
        conv = torch.nn.Conv2d(3, 8, 3).cuda().half()
        conv(torch.randn(1, 3, 64, 64, device="cuda", dtype=torch.float16))
        torch.cuda.synchronize()
        info["fp16_ok"] = bool(torch.isfinite((x @ x).float().sum()).item())
    except Exception as exc:
        info["fp16_ok"] = False
        info["error"] = str(exc)[:300]
print(json.dumps(info))
"""


def verify(gpu: Optional[GpuInfo]) -> bool:
    """Kurulum sonrasi gercek import denemesi ve GPU calisma testi."""
    print("\n[7/7] Dogrulama")
    everything_ok = True

    for import_name, label in (
        ("numpy", "numpy"),
        ("cv2", "opencv"),
        ("dotenv", "python-dotenv"),
        ("ultralytics", "ultralytics"),
        ("lap", "lap"),
    ):
        code, out = run_python(
            f"import {import_name} as m; print(getattr(m, '__version__', '?'))"
        )
        version = out.splitlines()[-1] if out else ""
        if code == 0:
            print(f"  {OK:7s} import {label:16s} {version}")
        else:
            print(f"  {FAIL:7s} import {label:16s} {version[:150]}")
            everything_ok = False

    code, out = run_python(_GPU_CHECK)
    try:
        info = json.loads(out.splitlines()[-1])
    except (IndexError, ValueError):
        print(f"  {FAIL:7s} import torch          {out[-300:]}")
        return False

    print(f"  {OK:7s} import torch            {info['torch']} (CUDA derlemesi: {info['cuda'] or 'yok'})")
    if info["available"]:
        print(f"  {OK:7s} GPU                     {info['name']} | compute {info['cap']} | {info['mem']} GB")
        if not info.get("arch_ok", True):
            print(f"  {FAIL:7s} Bu torch derlemesi kartinizin mimarisini (compute {info['cap']}) icermiyor.")
            everything_ok = False
        if info.get("fp16_ok"):
            print(f"  {OK:7s} GPU FP16 testi          basarili")
        else:
            print(f"  {FAIL:7s} GPU FP16 testi          {info.get('error', 'basarisiz')}")
            everything_ok = False
    elif gpu is not None:
        print(f"  {FAIL:7s} CUDA kullanilamiyor; ekran karti var ({gpu.name}) ama torch onu goremiyor.")
        print(f"          Kurulu torch derlemesi: {info['cuda'] or 'CPU'}. Cozum:")
        print("          python install.py --cuda cu126   (veya cu130)")
        everything_ok = False
    else:
        print(f"  {WARN:7s} CUDA yok               sistem CPU uzerinde calisacak")

    weights = PROJECT_ROOT / "models" / DEFAULT_OSNET_FILE
    if weights.exists():
        code, out = run_python(
            "import sys; sys.path.insert(0, %r)\n"
            "from src.osnet import build_osnet, load_osnet_weights\n"
            "m = build_osnet('osnet_x1_0'); print(load_osnet_weights(m, %r))"
            % (str(PROJECT_ROOT), str(weights))
        )
        if code == 0:
            print(f"  {OK:7s} OSNet agirligi          yuklenebiliyor")
        else:
            print(f"  {FAIL:7s} OSNet agirligi          {out.splitlines()[-1][:200] if out else ''}")
            everything_ok = False
    else:
        print(f"  {MISSING:7s} OSNet agirligi          models/{DEFAULT_OSNET_FILE} yok, reid.py calismaz")
        everything_ok = False

    return everything_ok


# =====================================================================
#  GIRIS NOKTASI
# =====================================================================


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Eksik kutuphaneleri tespit edip kurar ve GPU'yu dogrular."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Kurulum yapma, sadece durumu raporla"
    )
    parser.add_argument(
        "--cuda",
        default="auto",
        help="PyTorch derlemesi: auto | cpu | cu126 | cu130 | cu132 | cu118 (eski surucu)",
    )
    parser.add_argument(
        "--break-system-packages",
        action="store_true",
        help="PEP 668 korumasini asarak sistem Python'una kur (riskli)",
    )
    parser.add_argument(
        "--skip-weights", action="store_true", help="OSNet agirligini indirme"
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    print("=" * 70)
    print("  Unutulan Esya Tespit Sistemi - Kurulum")
    print("=" * 70)
    print(f"  Python : {platform.python_version()} ({sys.executable})")
    print(f"  Sistem : {platform.system()} {platform.machine()}")

    if sys.version_info < MIN_PYTHON:
        print(f"\n{FAIL} Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ gerekiyor.")
        return 1
    if not is_importable("pip"):
        print(f"\n{FAIL} pip bulunamadi. Once pip kurun.")
        return 1

    if in_virtualenv():
        print("  Ortam  : sanal ortam (venv)")
    elif is_externally_managed():
        print("  Ortam  : sistem Python'u (PEP 668 korumali)")
        if args.break_system_packages:
            EXTRA_PIP_ARGS.append("--break-system-packages")
            print(f"  {WARN} --break-system-packages verildi, koruma asiliyor.")
        elif not args.dry_run:
            print_venv_help()
            return 1
    else:
        print(f"  Ortam  : sistem Python'u  {WARN} sanal ortam onerilir")

    gpu = detect_gpu()
    if gpu is None:
        print("  GPU    : bulunamadi (nvidia-smi yok veya surucu kurulu degil)")
    else:
        print(
            f"  GPU    : {gpu.name} | surucu CUDA {gpu.driver_cuda or '?'} | "
            f"compute {gpu.capability or '?'}"
        )

    if args.dry_run:
        print("\n  ** DENEME MODU: hicbir sey kurulmayacak **")

    success = True

    # 1) Cekirdek paketler
    success &= install_group("[1/7] Cekirdek paketler", CORE_PACKAGES, args.dry_run)

    # 2) PyTorch - ultralytics'ten ONCE, her zaman PyTorch deposundan
    tag, description, pins = choose_torch_build(args.cuda.lower(), gpu)
    success &= install_torch(
        tag, description, pins, args.dry_run, force=args.cuda.lower() != "auto"
    )

    # 3) YOLO + BoT-SORT (+ lap, gdown)
    success &= install_group("[3/7] YOLO, takip ve indirme araclari", YOLO_PACKAGES, args.dry_run)

    # 4) ReID
    print("\n[4/7] ReID arka ucu (OSNet)")
    osnet_module = PROJECT_ROOT / "src" / "osnet.py"
    if osnet_module.exists():
        print(f"  {OK:7s} builtin               src/osnet.py (ek kutuphane gerekmez)")
    else:
        print(f"  {FAIL:7s} src/osnet.py bulunamadi; guncel proje dosyalarini kopyalayin.")
        success = False

    ensure_directories(args.dry_run)
    success &= ensure_weights(args.dry_run, args.skip_weights)

    if args.dry_run:
        print("\n" + "=" * 70)
        print("  Deneme modu tamamlandi. Gercek kurulum: python install.py")
        print("=" * 70)
        return 0

    verified = verify(gpu)

    print("\n" + "=" * 70)
    if success and verified:
        print("  Kurulum tamam. Calistirmak icin:  python main.py")
    else:
        print("  Kurulum eksik tamamlandi. Yukaridaki [HATA] satirlarina bakin.")
    print("=" * 70)
    return 0 if (success and verified) else 1


if __name__ == "__main__":
    sys.exit(main())
