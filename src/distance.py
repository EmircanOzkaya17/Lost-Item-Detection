"""
distance.py
===========

Yalnizca **geometrik hesaplamalar** yapan yardimci fonksiyonlar.

Bu modulun tek sorumlulugu, nesneler arasindaki konumsal iliskileri
hesaplamaktir.

--------------------------------------------------------------------
OLCEK: PIKSEL YERINE "KISI BOYU" VE YAKLASIK METRE
--------------------------------------------------------------------
Sabit piksel esikleri perspektife ve cozunurluge baglidir: 720p bir
kafede kameraya yakin oturan birinin kutusu 300 px, 4K bir istasyon
kamerasinda uzaktaki birininki 150 px olabilir. Ayni "80 px" birinde
30 cm, digerinde 1 m demektir; her sahne icin yeniden ayar gerekir.

Kisinin kutu yuksekligi, o kisinin bulundugu derinlikte dogal bir olcek
cubugudur (ortalama boy ~1.7 m). Kenar boslugu bu yukseklige bolunerek
birimsiz bir oran, PERSON_HEIGHT_M ile carpilarak YAKLASIK METRE elde
edilir (`estimate_gap_m`). Boylece esikler sahneden bagimsiz ve
anlasilir birimlerle (metre) verilebilir.

Bilinen sinirlar:
* Oturan kisinin kutusu kisa oldugundan mesafe biraz fazla cikar.
* Goruntude ust uste binen ama farkli derinlikteki kisi ve esya ayirt
  edilemez (kalibrasyonsuz tek kamerada derinlik bilinmez).
--------------------------------------------------------------------
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

__all__ = [
    "get_box_center",
    "euclidean_distance",
    "distance_between_boxes",
    "box_gap_distance",
    "box_overlap_ratio",
    "box_iou",
    "box_height",
    "box_area",
    "box_diagonal",
    "relative_gap",
    "estimate_gap_m",
    "touches_border",
    "is_farther_than",
    "find_nearest_person",
    "find_nearest_person_by_gap",
]

# (x1, y1, x2, y2)
BBox = Sequence[float]
# (cx, cy)
Point = Tuple[float, float]

_BOX_LENGTH = 4
_POINT_LENGTH = 2


def _validate_box(box: BBox) -> None:
    if box is None or len(box) != _BOX_LENGTH:
        raise ValueError(
            f"Bounding box (x1, y1, x2, y2) formatinda olmalidir. Alinan: {box}"
        )


def _validate_point(point: Sequence[float]) -> None:
    if point is None or len(point) != _POINT_LENGTH:
        raise ValueError(f"Nokta (x, y) formatinda olmalidir. Alinan: {point}")


# =====================================================================
#  1. BOUNDING BOX MERKEZINI HESAPLAMA
# =====================================================================


def get_box_center(box: BBox) -> Point:
    """Bounding box'in merkez koordinatlarini hesaplar.

    Diger tum mesafe hesaplamalarinin temelini olusturur.

    Args:
        box: (x1, y1, x2, y2)

    Returns:
        (cx, cy)
    """
    _validate_box(box)
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


# =====================================================================
#  2. IKI NOKTA ARASINDAKI OKLID MESAFESI
# =====================================================================


def euclidean_distance(point1: Sequence[float], point2: Sequence[float]) -> float:
    """Iki nokta arasindaki dogrusal (Euclidean) mesafeyi hesaplar.

    Args:
        point1: (cx1, cy1)
        point2: (cx2, cy2)

    Returns:
        distance
    """
    _validate_point(point1)
    _validate_point(point2)
    return math.hypot(point1[0] - point2[0], point1[1] - point2[1])


# =====================================================================
#  3. IKI BOUNDING BOX ARASINDAKI MESAFE
# =====================================================================


def distance_between_boxes(box1: BBox, box2: BBox) -> float:
    """Iki bounding box'in merkezleri arasindaki mesafeyi hesaplar.

        box1 -> get_box_center()
                                 -> euclidean_distance() -> distance
        box2 -> get_box_center()
    """
    return euclidean_distance(get_box_center(box1), get_box_center(box2))


# =====================================================================
#  4. MESAFE ESIGI KONTROLU
# =====================================================================


def is_farther_than(box1: BBox, box2: BBox, threshold: float) -> bool:
    """Iki nesne arasindaki mesafenin esik degerinden buyuk olup
    olmadigini kontrol eder.

    Ayni karsilastirma mantiginin farkli modullerde tekrar yazilmasini
    onler ve kodun okunabilirligini artirir.

    Args:
        box1: Birinci bounding box
        box2: Ikinci bounding box
        threshold: Mesafe esigi

    Returns:
        True  -> mesafe esikten buyuk
        False -> mesafe esige esit veya kucuk
    """
    return distance_between_boxes(box1, box2) > threshold


# =====================================================================
#  5. EN YAKIN KISIYI BULMA
# =====================================================================


def find_nearest_person(
    item_box: BBox, person_boxes: Sequence[BBox]
) -> Optional[int]:
    """Bir esyaya en yakin kisiyi bulur.

        Item Bounding Box -> Merkez Noktasi
        Tum Person Box'lari -> Merkez Noktalari
                            -> Mesafe Hesaplama
                            -> En Kucuk Mesafe
                            -> En Yakin Person

    Bu fonksiyon YALNIZCA geometrik yakinligi hesaplar. Kime sahiplik
    atanacagina karar vermek bu fonksiyonun sorumlulugunda DEGILDIR;
    bu karar Association Memory katmani tarafindan verilir. Bu nedenle
    kisi kimligi degil, listedeki INDEKSI dondurur.

    Args:
        item_box: Esyanin bounding box'i
        person_boxes: Ayni karedeki kisilerin bounding box'lari

    Returns:
        En yakin kisinin `person_boxes` icindeki indeksi.
        Liste bossa None.
    """
    if not person_boxes:
        return None

    item_center = get_box_center(item_box)
    distances = [
        euclidean_distance(item_center, get_box_center(box)) for box in person_boxes
    ]
    return min(range(len(distances)), key=distances.__getitem__)




# =====================================================================
#  6. IKI BOUNDING BOX ARASINDAKI KENAR MESAFESI
# =====================================================================


def box_gap_distance(box1: BBox, box2: BBox) -> float:
    """Iki bounding box'in KENARLARI arasindaki en kisa mesafe.

    Kutular kesisiyorsa 0.0 dondurur.

    Merkez mesafesinden farki: bir kisinin bbox merkezi govde
    hizasindadir. Yanindaki bir cantaya kenar mesafesi 0 iken merkez
    mesafesi yuzlerce piksel olabilir. Arka plandaki kucuk bir kisi ise
    merkez olcutunde daha yakin gorunur. "Bu esya bu kisinin yaninda mi"
    sorusunun dogru olcusu kenar mesafesidir.

    Args:
        box1: (x1, y1, x2, y2)
        box2: (x1, y1, x2, y2)

    Returns:
        Kenarlar arasi mesafe; kesisme varsa 0.0
    """
    _validate_box(box1)
    _validate_box(box2)
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2

    # Eksen bazinda ayrik mesafe; o eksende ortusme varsa 0.
    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return math.hypot(dx, dy)


def find_nearest_person_by_gap(
    item_box: BBox, person_boxes: Sequence[BBox]
) -> Optional[int]:
    """En yakin kisiyi KENAR mesafesine gore bulur.

    `find_nearest_person` ile ayni sozlesme: yalnizca geometrik yakinlik
    hesaplar, INDEKS dondurur, sahiplik karari vermez. Tek fark olcuttur.
    """
    if not person_boxes:
        return None
    distances = [box_gap_distance(item_box, box) for box in person_boxes]
    return min(range(len(distances)), key=distances.__getitem__)


# =====================================================================
#  7. IKI BOUNDING BOX ARASINDAKI ORTUSME ORANI
# =====================================================================


def box_overlap_ratio(box1: BBox, box2: BBox) -> float:
    """Iki bounding box'in ortusme orani. [0.0, 1.0]

    Kesisim alani, KUCUK olan kutunun alanina bolunur. Bu bir IoU
    (Intersection over Union) DEGILDIR.

    Neden IoU degil:
        Kisi kutusu bir cantadan cok daha buyuktur. Canta tamamen kisinin
        kutusunun icinde olsa bile IoU yalnizca ~0.15 cikar; dahasi bu oran
        perspektife gore degisir (yakindaki kisinin kutusu buyuk, uzaktakinin
        kucuk oldugu icin ayni fiziksel temas farkli IoU uretir). Kucuk
        kutunun alanina bolmek olcuyu boyut oranindan bagimsiz kilar:
        0.05 degeri her zaman "kucuk nesnenin alaninin %5'i digerinin
        kutusuna giriyor" anlamina gelir.

    Bu fonksiyon YALNIZCA geometri hesaplar. Ortusmenin ne ifade ettigine
    (sahip esyasinin yaninda mi, uyari bastirilmali mi) karar vermek bu
    modulun sorumlulugunda DEGILDIR.

    Args:
        box1: (x1, y1, x2, y2)
        box2: (x1, y1, x2, y2)

    Returns:
        Ortusme orani. Kesisme yoksa 0.0, kucuk kutu tamamen buyugun
        icindeyse 1.0.
    """
    _validate_box(box1)
    _validate_box(box2)

    # Ters girilmis koseler icin guvenli siralama.
    ax1, ax2 = sorted((float(box1[0]), float(box1[2])))
    ay1, ay2 = sorted((float(box1[1]), float(box1[3])))
    bx1, bx2 = sorted((float(box2[0]), float(box2[2])))
    by1, by2 = sorted((float(box2[1]), float(box2[3])))

    inter_width = min(ax2, bx2) - max(ax1, bx1)
    inter_height = min(ay2, by2) - max(ay1, by1)
    if inter_width <= 0.0 or inter_height <= 0.0:
        return 0.0

    smaller_area = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
    if smaller_area <= 0.0:
        return 0.0

    return min(1.0, (inter_width * inter_height) / smaller_area)


def box_iou(box1: BBox, box2: BBox) -> float:
    """Klasik IoU (Intersection over Union). [0.0, 1.0]

    `box_overlap_ratio`'dan farki: payda birlesim alanidir. Iki kutunun
    AYNI nesneyi gosterip gostermedigini olcmek icin uygundur (ornegin
    ayni cantanin hem "backpack" hem "handbag" olarak tespit edilmesi).
    """
    _validate_box(box1)
    _validate_box(box2)
    ax1, ax2 = sorted((float(box1[0]), float(box1[2])))
    ay1, ay2 = sorted((float(box1[1]), float(box1[3])))
    bx1, bx2 = sorted((float(box2[0]), float(box2[2])))
    by1, by2 = sorted((float(box2[1]), float(box2[3])))

    inter_width = min(ax2, bx2) - max(ax1, bx1)
    inter_height = min(ay2, by2) - max(ay1, by1)
    if inter_width <= 0.0 or inter_height <= 0.0:
        return 0.0
    intersection = inter_width * inter_height
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return 0.0 if union <= 0.0 else intersection / union


# =====================================================================
#  8. KUTU BOYUTLARI
# =====================================================================


def box_height(box: BBox) -> float:
    """Kutunun yuksekligi (piksel)."""
    _validate_box(box)
    return abs(float(box[3]) - float(box[1]))


def box_area(box: BBox) -> float:
    """Kutunun alani (piksel^2)."""
    _validate_box(box)
    return abs(float(box[2]) - float(box[0])) * abs(float(box[3]) - float(box[1]))


def box_diagonal(box: BBox) -> float:
    """Kutunun kosegen uzunlugu (piksel). Bir esyanin kendi olcek birimidir."""
    _validate_box(box)
    return math.hypot(float(box[2]) - float(box[0]), float(box[3]) - float(box[1]))


# =====================================================================
#  9. OLCEKTEN BAGIMSIZ YAKINLIK
# =====================================================================


def relative_gap(item_box: BBox, person_box: BBox) -> float:
    """Esya ile kisi kutusu arasindaki kenar boslugu, KISI BOYU cinsinden.

        relative_gap = box_gap_distance(esya, kisi) / kisi_kutu_yuksekligi

    0.0 -> kutular temas ediyor veya ortusuyor
    0.5 -> yarim boy kadar bosluk

    Returns:
        Birimsiz oran. Kisi kutusu dejenere ise (yukseklik 0) sonsuz.
    """
    height = box_height(person_box)
    if height <= 0.0:
        return math.inf
    return box_gap_distance(item_box, person_box) / height


def estimate_gap_m(item_box: BBox, person_box: BBox, person_height_m: float) -> float:
    """Esya ile kisi arasindaki boslugun YAKLASIK METRE karsiligi.

    Kisinin kutu yuksekligi `person_height_m` metre kabul edilerek piksel
    bosluk metreye cevrilir. Kalibrasyon gerektirmez; sonuc kisinin
    bulundugu derinlikte gecerli bir yaklasiktir.

    Neden merkez mesafesi degil: kisi kutusunun merkezi govde hizasindadir.
    Masada, kisinin hemen yanindaki bir telefonun merkezine olan uzaklik
    kisi boyunun %80'i kadar cikabilir; kenar boslugu ise ~%30'dur.
    """
    return relative_gap(item_box, person_box) * person_height_m


def touches_border(box: BBox, frame_shape: Sequence[int], margin_ratio: float) -> bool:
    """Kutu goruntu kenarina `margin_ratio` (genislik/yukseklik orani) kadar
    yakin mi? Kadrajdan cikmakta olan bir kisiyi, ortada bir engelin
    arkasina girip kisa sure kaybolan bir kisiden ayirmak icin kullanilir.
    """
    _validate_box(box)
    if frame_shape is None or len(frame_shape) < 2:
        return False
    height, width = float(frame_shape[0]), float(frame_shape[1])
    margin_x, margin_y = width * margin_ratio, height * margin_ratio
    x1, y1, x2, y2 = (float(v) for v in box)
    return (
        min(x1, x2) <= margin_x
        or min(y1, y2) <= margin_y
        or max(x1, x2) >= width - margin_x
        or max(y1, y2) >= height - margin_y
    )
