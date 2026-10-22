"""
visualization.py
================

Yalnizca **gorsellestirme** islemlerinden sorumlu katman.

Bu modulun tek amaci, sistem tarafindan uretilen bilgileri kullaniciya
anlasilir, okunabilir ve estetik bir sekilde ekrana cizmektir.

--------------------------------------------------------------------
BU DOSYADA BULUNMAYAN SEYLER
--------------------------------------------------------------------
* Person Re-Identification (ReID)
* Nesne takibi (Tracking)
* Association Memory yonetimi
* Sahiplik (ownership) karari verme
* Unutulan esya tespiti
* Alarm veya olay (event) uretme
* Mesafe hesaplama
* UUID olusturma veya guncelleme

Bu modul KARAR VERMEZ; yalnizca kendisine iletilen veriyi cizer.
--------------------------------------------------------------------

Bagimsizlik:
    `visualization.py` projedeki hicbir modulu import ETMEZ
    (detector, tracker, reid, association_memory, distance).
    Cizilecek veriler `VisualObject` ve `WarningMessage` gibi bu dosyaya
    ait sade yapilarla iletilir; donusumu `main.py` yapar.

Durumsuzluk (stateless):
    `Visualizer` yalnizca DEGISMEYEN yapilandirma (renkler, bayraklar,
    yazi boyutu) tutar. Frame'ler arasinda hicbir bilgi saklamaz;
    her cizim cagrisi bagimsizdir.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.config import Color, Config, load_config

logger = logging.getLogger(__name__)

BBox = Sequence[float]

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ITU-R BT.601 parlaklik katsayilari (BGR sirasinda). Etiket yazisinin koyu
# panel uzerinde okunabilir olup olmadigini olcmek icin kullanilir.
_LUMA_WEIGHTS_BGR = (0.114, 0.587, 0.299)
# Bu parlakligin altindaki renkler (saf mavi, saf kirmizi) koyu panel
# uzerinde okunmaz; YAZI rengi beyaza dogru acilir. Kutu rengi DEGISMEZ.
_MIN_TEXT_LUMA = 110.0
_TEXT_LIGHTEN_RATIO = 0.55
_WHITE = (255, 255, 255)
# Sahipligi kesinlesmemis (kanit biriken) esyada aday UUID'nin sonuna eklenir.
_PROVISIONAL_MARK = "?"


def _normalize_class_name(name: str) -> str:
    return name.strip().lower().replace("_", " ").replace("-", " ")


def _readable_text_color(color: Color) -> Color:
    """Koyu panel uzerinde okunabilir yazi rengi.

    Tonu (sinif kimligi) korunur; yalnizca cok koyu renkler beyaza dogru
    acilir. Ornek: mavi (255, 0, 0) -> acik mavi (255, 140, 140).
    """
    luma = sum(weight * channel for weight, channel in zip(_LUMA_WEIGHTS_BGR, color))
    if luma >= _MIN_TEXT_LUMA:
        return tuple(color)
    return tuple(
        int(round(channel + (white - channel) * _TEXT_LIGHTEN_RATIO))
        for channel, white in zip(color, _WHITE)
    )


# =====================================================================
#  CIZIM VERI YAPILARI
#  (Diger modullere bagimlilik olusmamasi icin burada tanimlanir.)
# =====================================================================


@dataclass
class VisualObject:
    """Ekrana cizilecek tek bir nesne.

    Tum alanlar diger moduller tarafindan SAGLANIR. Bu modul hicbirini
    hesaplamaz, uretmez veya dogrulamaz.
    """

    bbox: BBox
    class_name: str
    track_id: Optional[int] = None
    uuid: Optional[str] = None
    owner_uuid: Optional[str] = None
    # True ise `owner_uuid` kesinlesmis sahip degil, kanit toplayan lider
    # adaydir; etikette "P003?" olarak gosterilir.
    owner_provisional: bool = False
    confidence: Optional[float] = None
    status: Optional[str] = None  # Yalnizca etikette gosterilir


@dataclass
class WarningMessage:
    """Ekrana yazdirilacak uyari.

    Hangi uyarinin gosterilecegine BASKA moduller karar verir.
    """

    title: str
    lines: List[str] = field(default_factory=list)


# =====================================================================
#  VISUALIZER
# =====================================================================


class Visualizer:
    """Tum cizim islemlerini yapan gorsellestirme katmani.

    Kullanim:
        viz = Visualizer()
        viz.draw_objects(frame, objects)
        viz.draw_hud(frame, stats)
    """

    _LABEL_PADDING = 4
    _LINE_SPACING = 4
    _PANEL_MARGIN = 10
    _MIN_FONT_SCALE = 0.35

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or load_config()

        # --- Gosterim bayraklari ---
        self.show_fps = self.config.SHOW_FPS
        self.show_track_id = self.config.SHOW_TRACK_ID
        self.show_uuid = self.config.SHOW_UUID
        self.show_association_line = self.config.SHOW_ASSOCIATION_LINE
        self.show_warning_text = self.config.SHOW_WARNING_TEXT
        self.show_hud = self.config.SHOW_HUD
        self.show_owner = self.config.SHOW_OWNER
        self.show_confidence = self.config.SHOW_CONFIDENCE

        # --- Stil ---
        self.thickness = self.config.VIZ_LINE_THICKNESS
        self.font_scale = self.config.VIZ_FONT_SCALE
        self.auto_scale = self.config.VIZ_AUTO_SCALE
        self.reference_height = self.config.VIZ_REFERENCE_HEIGHT
        self.corner_style = self.config.VIZ_CORNER_STYLE
        self.corner_ratio = self.config.VIZ_CORNER_RATIO
        self.panel_alpha = self.config.VIZ_PANEL_ALPHA
        self.association_thickness = self.config.VIZ_ASSOCIATION_THICKNESS
        self.max_warnings = self.config.VIZ_MAX_WARNINGS

        # --- Renkler (BGR) ---
        self.default_color = self.config.DEFAULT_COLOR
        self.class_colors: Dict[str, Color] = dict(self.config.class_colors)
        self.text_color = self.config.TEXT_COLOR
        self.panel_color = self.config.PANEL_COLOR
        self.warning_color = self.config.WARNING_COLOR
        self.association_color = self.config.ASSOCIATION_LINE_COLOR

        logger.info(
            "Visualizer hazir | renkler=%s | otomatik_olcek=%s | en_fazla_uyari=%d",
            {k: v for k, v in self.class_colors.items()},
            self.auto_scale,
            self.max_warnings,
        )

    # ----------------------------------------------------------------
    #  2. SINIF BAZLI RENKLER
    # ----------------------------------------------------------------

    def color_for(self, class_name: str) -> Color:
        """Sinifa ait sabit rengi dondurur.

        Renk YALNIZCA sinifa gore belirlenir. Track ID, UUID, durum veya
        baska hicbir bilgi rengi degistirmez. Ekranda bes farkli kisi
        varsa besi de ayni yesil renkte cizilir.
        """
        return self.class_colors.get(_normalize_class_name(class_name), self.default_color)

    def label_text_color_for(self, class_name: str) -> Color:
        """Etiket yazisi: sinif rengi; koyu panelde okunmuyorsa acilmis hali."""
        return _readable_text_color(self.color_for(class_name))

    # ----------------------------------------------------------------
    #  OLCEKLENDIRME VE DUSUK SEVIYE CIZIM
    # ----------------------------------------------------------------

    def _scale(self, frame: np.ndarray) -> float:
        """Yazi boyutunu goruntu yuksekligine gore olceklendirir."""
        if not self.auto_scale or self.reference_height <= 0:
            return self.font_scale
        factor = frame.shape[0] / float(self.reference_height)
        return max(self._MIN_FONT_SCALE, self.font_scale * factor)

    def _line_thickness(self, frame: np.ndarray) -> int:
        if not self.auto_scale or self.reference_height <= 0:
            return self.thickness
        factor = frame.shape[0] / float(self.reference_height)
        return max(1, int(round(self.thickness * factor)))

    def _blend_rect(
        self, frame: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int], color: Color
    ) -> None:
        """Yari saydam dolgu dikdortgeni cizer."""
        height, width = frame.shape[:2]
        x1, y1 = max(0, p1[0]), max(0, p1[1])
        x2, y2 = min(width, p2[0]), min(height, p2[1])
        if x2 <= x1 or y2 <= y1:
            return
        roi = frame[y1:y2, x1:x2]
        overlay = np.empty_like(roi)
        overlay[:] = color
        cv2.addWeighted(overlay, self.panel_alpha, roi, 1.0 - self.panel_alpha, 0, dst=roi)

    def _text_block_size(
        self, lines: Sequence[str], scale: float, thickness: int
    ) -> Tuple[int, int, int]:
        """Cok satirli metnin (genislik, yukseklik, satir_yuksekligi) olcusu."""
        widths, heights = [], []
        for line in lines:
            (w, h), baseline = cv2.getTextSize(line, FONT, scale, thickness)
            widths.append(w)
            heights.append(h + baseline)
        line_height = max(heights) + self._LINE_SPACING
        return max(widths), line_height * len(lines), line_height

    def _draw_text_block(
        self,
        frame: np.ndarray,
        lines: Sequence[str],
        top_left: Tuple[int, int],
        text_color: Color,
        panel_color: Optional[Color] = None,
    ) -> Tuple[int, int]:
        """Arka plan panelli cok satirli metin cizer. (genislik, yukseklik)"""
        if not lines:
            return 0, 0

        scale = self._scale(frame)
        thickness = max(1, int(round(scale * 2)))
        block_w, block_h, line_h = self._text_block_size(lines, scale, thickness)
        pad = self._LABEL_PADDING

        x, y = top_left
        if panel_color is not None:
            self._blend_rect(
                frame,
                (x, y),
                (x + block_w + pad * 2, y + block_h + pad),
                panel_color,
            )

        for index, line in enumerate(lines):
            baseline_y = y + pad + line_h * (index + 1) - self._LINE_SPACING
            cv2.putText(
                frame,
                line,
                (x + pad, baseline_y),
                FONT,
                scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )
        return block_w + pad * 2, block_h + pad

    @staticmethod
    def _int_box(bbox: BBox) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

    @staticmethod
    def _box_center(bbox: BBox) -> Tuple[int, int]:
        """Cizim amacli merkez noktasi.

        NOT: Bu bir mesafe hesabi DEGILDIR; yalnizca cizgi ucunu
        belirlemek icin kullanilir. Mesafe hesaplari `distance.py`
        sorumlulugundadir.
        """
        x1, y1, x2, y2 = bbox
        return int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))

    # ----------------------------------------------------------------
    #  1. BOUNDING BOX CIZIMI
    # ----------------------------------------------------------------

    def draw_bbox(self, frame: np.ndarray, obj: VisualObject) -> np.ndarray:
        """Nesnenin bounding box'ini sinif rengiyle cizer."""
        x1, y1, x2, y2 = self._int_box(obj.bbox)
        color = self.color_for(obj.class_name)
        thickness = self._line_thickness(frame)

        # NOT: Eksen hizali kenarlarda cv2.LINE_AA kismi kapsama uretip
        # rengi soluklastirir. Keskin gorunum icin LINE_8 kullanilir;
        # LINE_AA yalnizca capraz cizgilerde ve yazilarda anlamlidir.
        if not self.corner_style:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_8)
            return frame

        # Kose vurgulu tasarim: ince govde + kalin koseler
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(1, thickness - 1), cv2.LINE_8)
        length = max(2, int(min(x2 - x1, y2 - y1) * self.corner_ratio))
        corner_thickness = thickness + 1
        for (cx, cy), (dx, dy) in (
            ((x1, y1), (1, 1)),
            ((x2, y1), (-1, 1)),
            ((x1, y2), (1, -1)),
            ((x2, y2), (-1, -1)),
        ):
            cv2.line(frame, (cx, cy), (cx + dx * length, cy), color, corner_thickness, cv2.LINE_8)
            cv2.line(frame, (cx, cy), (cx, cy + dy * length), color, corner_thickness, cv2.LINE_8)
        return frame

    # ----------------------------------------------------------------
    #  3-4-5. ETIKET / UUID / TRACK ID
    # ----------------------------------------------------------------

    def build_label_lines(self, obj: VisualObject) -> List[str]:
        """Etiket satirlarini olusturur.

            Person            Backpack
            Track : 12        Track : 35
            UUID : P001       UUID : B003
                              Owner : P001
        """
        lines = [obj.class_name.title()]
        if self.show_track_id and obj.track_id is not None:
            lines.append(f"Track : {obj.track_id}")
        if self.show_uuid and obj.uuid:
            lines.append(f"UUID : {obj.uuid}")
        if self.show_owner and obj.owner_uuid:
            mark = _PROVISIONAL_MARK if obj.owner_provisional else ""
            lines.append(f"Owner : {obj.owner_uuid}{mark}")
        if obj.status:
            lines.append(obj.status.upper())
        if self.show_confidence and obj.confidence is not None:
            lines.append(f"Conf : {obj.confidence:.2f}")
        return lines

    def draw_label(self, frame: np.ndarray, obj: VisualObject) -> np.ndarray:
        """Bounding box uzerine bilgi etiketini cizer."""
        lines = self.build_label_lines(obj)
        if not lines:
            return frame

        x1, y1, _, y2 = self._int_box(obj.bbox)
        scale = self._scale(frame)
        thickness = max(1, int(round(scale * 2)))
        _, block_h, _ = self._text_block_size(lines, scale, thickness)
        block_h += self._LABEL_PADDING

        # Etiket kutunun ustune sigmiyorsa altina cizilir.
        top = y1 - block_h
        if top < 0:
            top = min(y2, frame.shape[0] - block_h)
            top = max(0, top)

        self._draw_text_block(
            frame, lines, (x1, top), self.label_text_color_for(obj.class_name), self.panel_color
        )
        return frame

    def draw_object(self, frame: np.ndarray, obj: VisualObject) -> np.ndarray:
        """Bir nesnenin kutusunu ve etiketini birlikte cizer."""
        self.draw_bbox(frame, obj)
        self.draw_label(frame, obj)
        return frame

    def draw_objects(
        self, frame: np.ndarray, objects: Sequence[VisualObject]
    ) -> np.ndarray:
        for obj in objects:
            self.draw_object(frame, obj)
        return frame

    # ----------------------------------------------------------------
    #  6. KISI-ESYA ILISKI CIZGISI
    # ----------------------------------------------------------------

    def draw_association_line(
        self, frame: np.ndarray, person_bbox: BBox, item_bbox: BBox
    ) -> np.ndarray:
        """Kisi ile esyasi arasina baglanti cizgisi cizer.

        Bu cizgi YALNIZCA Association Memory tarafindan saglanan sahiplik
        bilgisine gore cizilir. Iliskinin dogru olup olmadigina bu modul
        karar VERMEZ.
        """
        if not self.show_association_line:
            return frame
        cv2.line(
            frame,
            self._box_center(person_bbox),
            self._box_center(item_bbox),
            self.association_color,
            max(1, self.association_thickness),
            cv2.LINE_AA,
        )
        return frame

    def draw_associations(
        self, frame: np.ndarray, pairs: Sequence[Tuple[BBox, BBox]]
    ) -> np.ndarray:
        """(kisi_bbox, esya_bbox) ciftlerinin tamamini cizer."""
        for person_bbox, item_bbox in pairs:
            self.draw_association_line(frame, person_bbox, item_bbox)
        return frame

    # ----------------------------------------------------------------
    #  7. UYARI GOSTERIMI
    # ----------------------------------------------------------------

    def draw_warnings(
        self, frame: np.ndarray, warnings: Sequence[WarningMessage]
    ) -> np.ndarray:
        """Uyarilari ekranin ust ortasina yazdirir.

            WARNING                 ABANDONED ITEM
            Person P001             Laptop L004
            Forgot Backpack B003    Owner P001

        En fazla VIZ_MAX_WARNINGS panel cizilir; fazlasi tek satirlik bir
        ozet panelde sayilir (paneller sahneyi kapatmasin). Siralama
        (onem) cagiran tarafindir.
        """
        if not self.show_warning_text or not warnings:
            return frame

        shown = list(warnings[: self.max_warnings])
        hidden = len(warnings) - len(shown)
        if hidden > 0:
            shown.append(WarningMessage(f"+{hidden} more warning(s)"))

        scale = self._scale(frame)
        thickness = max(1, int(round(scale * 2)))
        offset_y = self._PANEL_MARGIN
        text_color = _readable_text_color(self.warning_color)

        for warning in shown:
            lines = [warning.title] + list(warning.lines)
            block_w, block_h, _ = self._text_block_size(lines, scale, thickness)
            block_w += self._LABEL_PADDING * 2
            block_h += self._LABEL_PADDING

            left = max(0, (frame.shape[1] - block_w) // 2)
            self._blend_rect(
                frame,
                (left, offset_y),
                (left + block_w, offset_y + block_h),
                self.panel_color,
            )
            cv2.rectangle(
                frame,
                (left, offset_y),
                (left + block_w, offset_y + block_h),
                self.warning_color,
                max(1, self._line_thickness(frame)),
                cv2.LINE_8,
            )
            self._draw_text_block(
                frame, lines, (left, offset_y), text_color, panel_color=None
            )
            offset_y += block_h + self._PANEL_MARGIN

        return frame

    # ----------------------------------------------------------------
    #  8. FPS GOSTERIMI
    # ----------------------------------------------------------------

    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Anlik FPS degerini sag ust koseye yazdirir.

        FPS HESAPLANMAZ; hazir deger olarak alinir.
        """
        if not self.show_fps:
            return frame

        text = [f"FPS : {fps:.0f}"]
        scale = self._scale(frame)
        thickness = max(1, int(round(scale * 2)))
        block_w, _, _ = self._text_block_size(text, scale, thickness)
        left = frame.shape[1] - block_w - self._LABEL_PADDING * 2 - self._PANEL_MARGIN
        self._draw_text_block(
            frame, text, (max(0, left), self._PANEL_MARGIN), self.text_color, self.panel_color
        )
        return frame

    # ----------------------------------------------------------------
    #  9. FRAME BILGILERI (HUD)
    # ----------------------------------------------------------------

    def draw_hud(self, frame: np.ndarray, stats: Mapping[str, object]) -> np.ndarray:
        """Sistemin genel durumunu sol alt kosede ozetler.

            Persons          : 4
            Backpacks        : 3
            Forgotten Items  : 1
            Active Tracks    : 9

        Sayimlar BU MODULDE hesaplanmaz; hazir olarak alinir.
        """
        if not self.show_hud or not stats:
            return frame

        width = max(len(str(key)) for key in stats)
        lines = [f"{str(key):<{width}} : {value}" for key, value in stats.items()]

        scale = self._scale(frame)
        thickness = max(1, int(round(scale * 2)))
        _, block_h, _ = self._text_block_size(lines, scale, thickness)
        top = frame.shape[0] - block_h - self._LABEL_PADDING - self._PANEL_MARGIN
        self._draw_text_block(
            frame,
            lines,
            (self._PANEL_MARGIN, max(0, top)),
            self.text_color,
            self.panel_color,
        )
        return frame

    # ----------------------------------------------------------------
    #  BIRLESIK CIZIM
    # ----------------------------------------------------------------

    def draw_frame(
        self,
        frame: np.ndarray,
        objects: Sequence[VisualObject] = (),
        associations: Sequence[Tuple[BBox, BBox]] = (),
        warnings: Sequence[WarningMessage] = (),
        stats: Optional[Mapping[str, object]] = None,
        fps: Optional[float] = None,
    ) -> np.ndarray:
        """Tum gorsel bilesenleri dogru sirayla cizen kolaylik fonksiyonu.

        Yalnizca diger cizim fonksiyonlarini cagirir; kendi basina hicbir
        karar vermez. Iliski cizgileri kutularin altinda kalmasi icin
        once cizilir.
        """
        self.draw_associations(frame, associations)
        self.draw_objects(frame, objects)
        if stats:
            self.draw_hud(frame, stats)
        if fps is not None:
            self.draw_fps(frame, fps)
        self.draw_warnings(frame, warnings)
        return frame

    def __repr__(self) -> str:
        return f"Visualizer(classes={list(self.class_colors)}, auto_scale={self.auto_scale})"
