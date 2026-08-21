import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, to_hex

from PyQt6.QtCore import Qt, QUrl, QRectF, QPointF
from PyQt6.QtGui import (
    QAction,
    QColor,
    QDesktopServices,
    QFont,
    QGuiApplication,
    QIcon,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QDialog,
)

APP_NAME = "Researcher Analysis toolkit — Color"
MIN_COLORS = 8


def resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS  # type: ignore[attr-defined]
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


LOGO_PNG = resource_path("ResearcherLogo.png")
LOGO_ICO = resource_path("ResearcherLogo.ico")


@dataclass
class ColorEntry:
    index: int
    hex_code: str
    role: str


@dataclass
class PaletteSpec:
    name: str
    class_name: str
    source: str
    colors: List[str]
    family: str = "General scientific"
    intuitive_for_diverging: bool = False
    intuitive_for_sequential: bool = False
    good_on_light_bg: float = 0.9
    good_on_dark_bg: float = 0.85
    perceptual_score: float = 0.85
    colorblind_score: float = 0.85
    cyclic_score: float = 0.30
    small_variation_score: float = 0.80
    notes: str = ""


# --------------------------
# Palette utilities
# --------------------------
def sample_cmap(name: str, n: int = MIN_COLORS, start: float = 0.04, stop: float = 0.96) -> List[str]:
    cmap = cm.get_cmap(name)
    points = np.linspace(start, stop, n)
    return [to_hex(cmap(float(p)), keep_alpha=False).upper() for p in points]


def sample_custom_gradient(colors: List[str], n: int = MIN_COLORS) -> List[str]:
    cmap = LinearSegmentedColormap.from_list("custom_grad", colors)
    points = np.linspace(0.0, 1.0, n)
    return [to_hex(cmap(float(p)), keep_alpha=False).upper() for p in points]


def hex_to_rgb01(hex_code: str) -> Tuple[float, float, float]:
    hex_code = hex_code.strip().lstrip("#")
    return tuple(int(hex_code[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def rgb01_to_hex(rgb: Tuple[float, float, float]) -> str:
    vals = [max(0, min(255, int(round(c * 255)))) for c in rgb]
    return "#%02X%02X%02X" % tuple(vals)


def luminance(hex_code: str) -> float:
    def channel(v: float) -> float:
        if v <= 0.03928:
            return v / 12.92
        return ((v + 0.055) / 1.055) ** 2.4

    r, g, b = hex_to_rgb01(hex_code)
    r_l, g_l, b_l = channel(r), channel(g), channel(b)
    return 0.2126 * r_l + 0.7152 * g_l + 0.0722 * b_l


def contrast_ratio(hex1: str, hex2: str) -> float:
    l1 = luminance(hex1)
    l2 = luminance(hex2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def avg_step_distance(colors: List[str]) -> float:
    rgbs = [hex_to_rgb01(c) for c in colors]
    dists = []
    for i in range(len(rgbs) - 1):
        a = np.array(rgbs[i])
        b = np.array(rgbs[i + 1])
        dists.append(float(np.linalg.norm(a - b)))
    return float(np.mean(dists)) if dists else 0.0


def monotonic_luminance_score(colors: List[str]) -> float:
    vals = [luminance(c) for c in colors]
    diffs = np.diff(vals)
    pos = np.sum(diffs >= 0)
    neg = np.sum(diffs <= 0)
    score = max(pos, neg) / max(1, len(diffs))
    return float(score)


def centre_symmetry_score(colors: List[str]) -> float:
    vals = [luminance(c) for c in colors]
    n = len(vals)
    if n < 2:
        return 1.0
    half = n // 2
    if half == 0:
        return 1.0
    diffs = []
    for i in range(half):
        left = vals[i]
        right = vals[-(i + 1)]
        diffs.append(1.0 - min(1.0, abs(left - right)))
    return float(np.mean(diffs)) if diffs else 1.0


def is_rainbow_like(name: str) -> bool:
    name_l = name.lower()
    return any(k in name_l for k in ["spectral", "rainbow", "hsv", "jet"])


def luminance_span_score(colors: List[str]) -> float:
    vals = [luminance(c) for c in colors]
    if not vals:
        return 0.0
    span = max(vals) - min(vals)
    return float(min(1.0, span / 0.72))


# --------------------------
# Curated palette library
# --------------------------
PALETTE_LIBRARY: Dict[str, List[PaletteSpec]] = {
    "Sequential": [
        PaletteSpec(
            name="Viridis Pro",
            class_name="Sequential",
            source="matplotlib: viridis",
            family="Publication-safe",
            colors=sample_cmap("viridis"),
            intuitive_for_sequential=False,
            good_on_light_bg=0.97,
            good_on_dark_bg=0.86,
            perceptual_score=0.96,
            colorblind_score=0.95,
            small_variation_score=0.95,
            notes="Strong publication-safe default for ordered continuous data.",
        ),
        PaletteSpec(
            name="Cividis Pro",
            class_name="Sequential",
            source="matplotlib: cividis",
            family="Colorblind-safe",
            colors=sample_cmap("cividis"),
            intuitive_for_sequential=False,
            good_on_light_bg=0.95,
            good_on_dark_bg=0.84,
            perceptual_score=0.94,
            colorblind_score=0.98,
            small_variation_score=0.90,
            notes="Excellent accessibility and stable grayscale behaviour.",
        ),
        PaletteSpec(
            name="Magma Pro",
            class_name="Sequential",
            source="matplotlib: magma",
            family="Dark-background",
            colors=sample_cmap("magma"),
            good_on_light_bg=0.90,
            good_on_dark_bg=0.96,
            perceptual_score=0.91,
            colorblind_score=0.88,
            small_variation_score=0.89,
            notes="High visual drama and strong contrast; especially good on dark background.",
        ),
        PaletteSpec(
            name="Thermal Pro",
            class_name="Sequential",
            source="custom",
            family="Scientific heatmap",
            colors=sample_custom_gradient(["#0B3C5D", "#328CC1", "#D9B310", "#F05D23"]),
            intuitive_for_sequential=True,
            good_on_light_bg=0.89,
            good_on_dark_bg=0.80,
            perceptual_score=0.80,
            colorblind_score=0.72,
            small_variation_score=0.74,
            notes="Useful when a warm-high semantic is desired for temperature-like variables.",
        ),
        PaletteSpec(
            name="Inferno Heatmap",
            class_name="Sequential",
            source="matplotlib: inferno",
            family="Scientific heatmap",
            colors=sample_cmap("inferno"),
            intuitive_for_sequential=True,
            good_on_light_bg=0.91,
            good_on_dark_bg=0.97,
            perceptual_score=0.94,
            colorblind_score=0.89,
            small_variation_score=0.93,
            notes="High-contrast scientific heatmap palette for dense fields and continuous maps.",
        ),
        PaletteSpec(
            name="Plasma Heatmap",
            class_name="Sequential",
            source="matplotlib: plasma",
            family="Scientific heatmap",
            colors=sample_cmap("plasma"),
            intuitive_for_sequential=True,
            good_on_light_bg=0.92,
            good_on_dark_bg=0.91,
            perceptual_score=0.93,
            colorblind_score=0.87,
            small_variation_score=0.91,
            notes="Strong heatmap option with vivid mid-range separation.",
        ),
        PaletteSpec(
            name="Density Blue",
            class_name="Sequential",
            source="custom",
            family="Publication-safe",
            colors=sample_custom_gradient(["#F7FBFF", "#D6EAF8", "#7FB3D5", "#2E86C1", "#154360"]),
            intuitive_for_sequential=False,
            good_on_light_bg=0.98,
            good_on_dark_bg=0.74,
            perceptual_score=0.92,
            colorblind_score=0.94,
            small_variation_score=0.90,
            notes="Clean density/intensity palette for publication-safe blue heatmaps.",
        ),
        PaletteSpec(
            name="Publication Gray",
            class_name="Sequential",
            source="custom",
            family="Grayscale-safe",
            colors=sample_custom_gradient(["#F8F9FA", "#E9ECEF", "#CED4DA", "#6C757D", "#212529"]),
            intuitive_for_sequential=False,
            good_on_light_bg=0.99,
            good_on_dark_bg=0.70,
            perceptual_score=0.90,
            colorblind_score=1.00,
            small_variation_score=0.88,
            notes="Grayscale-safe sequential palette for print and thesis figures.",
        ),
    ],
    "Diverging": [
        PaletteSpec(
            name="Blue-Orange Pro",
            class_name="Diverging",
            source="custom",
            family="Publication-safe",
            colors=sample_custom_gradient(["#2B59C3", "#A7C6FF", "#F7F7F7", "#FFD2A6", "#C75B12"]),
            intuitive_for_diverging=True,
            good_on_light_bg=0.96,
            good_on_dark_bg=0.82,
            perceptual_score=0.91,
            colorblind_score=0.90,
            small_variation_score=0.89,
            notes="Balanced cool-warm diverging map with readable centre.",
        ),
        PaletteSpec(
            name="PuOr Pro",
            class_name="Diverging",
            source="matplotlib: PuOr",
            family="General diverging",
            colors=sample_cmap("PuOr"),
            intuitive_for_diverging=True,
            good_on_light_bg=0.92,
            good_on_dark_bg=0.80,
            perceptual_score=0.88,
            colorblind_score=0.84,
            small_variation_score=0.83,
            notes="Good when negative-positive semantics should feel distinct.",
        ),
        PaletteSpec(
            name="BrBG Pro",
            class_name="Diverging",
            source="matplotlib: BrBG",
            family="General diverging",
            colors=sample_cmap("BrBG"),
            intuitive_for_diverging=False,
            good_on_light_bg=0.90,
            good_on_dark_bg=0.79,
            perceptual_score=0.85,
            colorblind_score=0.80,
            small_variation_score=0.79,
            notes="Solid diverging option; use when blue/orange semantics are not needed.",
        ),
        PaletteSpec(
            name="Coolwarm Soft",
            class_name="Diverging",
            source="matplotlib: coolwarm",
            family="General diverging",
            colors=sample_cmap("coolwarm"),
            intuitive_for_diverging=True,
            good_on_light_bg=0.90,
            good_on_dark_bg=0.78,
            perceptual_score=0.80,
            colorblind_score=0.76,
            small_variation_score=0.78,
            notes="Familiar and readable, but less robust than the top diverging choices.",
        ),
        PaletteSpec(
            name="Red-Blue Publication",
            class_name="Diverging",
            source="custom",
            family="Publication-safe",
            colors=sample_custom_gradient(["#08306B", "#6BAED6", "#F7F7F7", "#FCAE91", "#99000D"]),
            intuitive_for_diverging=True,
            good_on_light_bg=0.95,
            good_on_dark_bg=0.80,
            perceptual_score=0.92,
            colorblind_score=0.88,
            small_variation_score=0.87,
            notes="Publication-style red-blue diverging palette for positive/negative deviation maps.",
        ),
        PaletteSpec(
            name="Colorblind Diverging",
            class_name="Diverging",
            source="custom",
            family="Colorblind-safe",
            colors=sample_custom_gradient(["#3B4CC0", "#8DB0FE", "#F7F7F7", "#F7B89C", "#B40426"]),
            intuitive_for_diverging=True,
            good_on_light_bg=0.95,
            good_on_dark_bg=0.81,
            perceptual_score=0.90,
            colorblind_score=0.91,
            small_variation_score=0.86,
            notes="Accessibility-focused diverging palette with balanced centre contrast.",
        ),
    ],
    "Multi-sequential": [
        PaletteSpec(
            name="Dual-Side Pro",
            class_name="Multi-sequential",
            source="custom",
            family="Publication-safe",
            colors=sample_custom_gradient(["#0B5394", "#6FA8DC", "#F7F7F7", "#F9CB9C", "#B45F06"]),
            intuitive_for_diverging=False,
            good_on_light_bg=0.95,
            good_on_dark_bg=0.80,
            perceptual_score=0.90,
            colorblind_score=0.89,
            small_variation_score=0.87,
            notes="Useful when two ordered branches meet near a transition.",
        ),
        PaletteSpec(
            name="PRGn Pro",
            class_name="Multi-sequential",
            source="matplotlib: PRGn",
            family="General multi-sequential",
            colors=sample_cmap("PRGn"),
            good_on_light_bg=0.87,
            good_on_dark_bg=0.77,
            perceptual_score=0.80,
            colorblind_score=0.70,
            small_variation_score=0.74,
            notes="Readable, but green-purple combinations can be less accessible.",
        ),
    ],
    "Cyclic": [
        PaletteSpec(
            name="Twilight Pro",
            class_name="Cyclic",
            source="matplotlib: twilight",
            family="Phase / cyclic",
            colors=sample_cmap("twilight", start=0.0, stop=1.0),
            good_on_light_bg=0.88,
            good_on_dark_bg=0.93,
            perceptual_score=0.89,
            colorblind_score=0.84,
            cyclic_score=0.98,
            small_variation_score=0.80,
            notes="Best default for phase, angle, and periodic variables.",
        ),
        PaletteSpec(
            name="Phase Loop",
            class_name="Cyclic",
            source="custom",
            family="Phase / cyclic",
            colors=sample_custom_gradient(["#313695", "#74ADD1", "#F7F7F7", "#FDAE61", "#A50026", "#313695"]),
            good_on_light_bg=0.86,
            good_on_dark_bg=0.83,
            perceptual_score=0.78,
            colorblind_score=0.75,
            cyclic_score=0.92,
            small_variation_score=0.72,
            notes="Distinct cyclic palette with stronger colour identity.",
        ),
        PaletteSpec(
            name="Phase Publication",
            class_name="Cyclic",
            source="custom",
            family="Publication-safe",
            colors=sample_custom_gradient(["#1B2A97", "#5FA8D3", "#EAF2F8", "#F6AE2D", "#C73E1D", "#1B2A97"]),
            good_on_light_bg=0.89,
            good_on_dark_bg=0.85,
            perceptual_score=0.82,
            colorblind_score=0.79,
            cyclic_score=0.94,
            small_variation_score=0.75,
            notes="Publication-friendly cyclic map for phase and angular fields.",
        ),
    ],
    "Categorical": [
        PaletteSpec(
            name="Category Pro",
            class_name="Categorical",
            source="custom",
            family="General categorical",
            colors=[
                "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
                "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"
            ],
            good_on_light_bg=0.93,
            good_on_dark_bg=0.82,
            perceptual_score=0.84,
            colorblind_score=0.83,
            small_variation_score=0.60,
            notes="General-purpose categorical set with clear separation.",
        ),
        PaletteSpec(
            name="Muted Category",
            class_name="Categorical",
            source="custom",
            family="Publication-safe",
            colors=[
                "#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1",
                "#9C755F", "#76B7B2", "#EDC948", "#AF7AA1", "#FF9DA7"
            ],
            good_on_light_bg=0.95,
            good_on_dark_bg=0.79,
            perceptual_score=0.82,
            colorblind_score=0.80,
            small_variation_score=0.56,
            notes="Softer but still distinct; better for multi-series manuscript plots.",
        ),
        PaletteSpec(
            name="Okabe-Ito Line Plot",
            class_name="Categorical",
            source="custom",
            family="Line-plot / colorblind-safe",
            colors=[
                "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
                "#56B4E9", "#F0E442", "#000000", "#7F7F7F", "#999933"
            ],
            good_on_light_bg=0.97,
            good_on_dark_bg=0.83,
            perceptual_score=0.88,
            colorblind_score=0.99,
            small_variation_score=0.62,
            notes="Excellent line-plot palette with strong colorblind safety.",
        ),
        PaletteSpec(
            name="Publication Muted Line Plot",
            class_name="Categorical",
            source="custom",
            family="Line-plot / publication-safe",
            colors=[
                "#355070", "#6D597A", "#B56576", "#E56B6F", "#EAAC8B",
                "#5B8E7D", "#457B9D", "#8D99AE", "#BC6C25", "#6C757D"
            ],
            good_on_light_bg=0.96,
            good_on_dark_bg=0.80,
            perceptual_score=0.85,
            colorblind_score=0.86,
            small_variation_score=0.60,
            notes="Muted publication-safe series colors for manuscript line plots.",
        ),
        PaletteSpec(
            name="Grayscale-safe Series",
            class_name="Categorical",
            source="custom",
            family="Grayscale-safe",
            colors=[
                "#111111", "#2E2E2E", "#4B4B4B", "#686868", "#858585",
                "#A2A2A2", "#BFBFBF", "#D0D0D0", "#E0E0E0", "#F0F0F0"
            ],
            good_on_light_bg=0.98,
            good_on_dark_bg=0.62,
            perceptual_score=0.78,
            colorblind_score=1.00,
            small_variation_score=0.58,
            notes="Grayscale-safe categorical option for print-focused figures.",
        ),
    ],
}


ROLE_LABELS = {
    "Sequential": ["Lowest", "Very low", "Low", "Low-mid", "Mid", "Mid-high", "High", "Very high", "Near maximum", "Maximum"],
    "Diverging": ["Strong negative", "Negative", "Moderate negative", "Weak negative", "Near centre", "Near centre", "Weak positive", "Moderate positive", "Positive", "Strong positive"],
    "Multi-sequential": ["Side A minimum", "Side A low", "Side A moderate", "Approaching transition", "Transition", "Transition", "Side B low", "Side B moderate", "Side B high", "Side B maximum"],
    "Cyclic": ["0°", "40°", "80°", "120°", "160°", "200°", "240°", "280°", "320°", "360°"],
    "Categorical": [f"Category {i}" for i in range(1, 11)],
}


# --------------------------
# Professional selection logic
# --------------------------
def build_entries(colors: List[str], labels: List[str], n: int) -> List[ColorEntry]:
    return [
        ColorEntry(i + 1, colors[i], labels[min(i, len(labels) - 1)])
        for i in range(min(len(colors), n))
    ]


def class_from_logic(ordered: bool, centric: bool, diverging: bool, periodic: bool) -> str:
    if not ordered:
        return "Categorical"
    if periodic:
        return "Cyclic"
    if centric:
        return "Diverging" if diverging else "Multi-sequential"
    return "Sequential"


def type_from_logic(class_name: str, small_variation: bool) -> str:
    if class_name == "Categorical":
        return "Categorical"
    return "Continuous" if small_variation else "Discrete"


def combination_from_logic(empty_values: bool, intuitive: bool, light_bg: bool, centric: bool) -> str:
    rules = []
    if empty_values:
        rules.append("Use the figure background for missing or masked values. Prefer clearly distinct end colours with a soft central transition. Best suited to light-background figures.")
    if intuitive:
        if centric:
            rules.append("Prefer semantically distinct poles with a gentle centre.")
        else:
            rules.append("Prefer intuitive low-to-high semantic direction.")
    else:
        rules.append("Prioritise perceptual robustness over semantic colour convention.")
    rules.append("Optimised for light background." if light_bg else "Optimised for dark background.")
    return " ".join(rules)


def score_palette(
    palette: PaletteSpec,
    class_name: str,
    type_name: str,
    intuitive: bool,
    light_bg: bool,
    small_variation: bool,
    empty_values: bool,
) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    score = 0.0

    colors = palette.colors[:MIN_COLORS]
    background = "#FFFFFF" if light_bg else "#101418"

    # Core fit by class
    score += 30.0
    reasons.append(f"Matches required class: {class_name}.")

    # Perceptual quality
    score += 18.0 * palette.perceptual_score
    reasons.append(f"Perceptual robustness score: {palette.perceptual_score:.2f}.")

    # Accessibility
    score += 14.0 * palette.colorblind_score
    reasons.append(f"Colour-vision accessibility score: {palette.colorblind_score:.2f}.")

    # Background compatibility
    bg_fit = palette.good_on_light_bg if light_bg else palette.good_on_dark_bg
    score += 12.0 * bg_fit
    reasons.append(f"Background compatibility score: {bg_fit:.2f}.")

    # Small-variation handling
    if small_variation:
        score += 12.0 * palette.small_variation_score
        reasons.append(f"Small-variation visibility score: {palette.small_variation_score:.2f}.")
    else:
        score += 7.0 * max(0.65, palette.small_variation_score)

    # Type preference
    if type_name == "Continuous":
        step_score = min(1.0, avg_step_distance(colors) / 0.22)
        mono_score = monotonic_luminance_score(colors)
        type_fit = 0.55 * step_score + 0.45 * mono_score
        score += 10.0 * type_fit
        reasons.append(f"Continuous-gradient quality score: {type_fit:.2f}.")
    elif type_name == "Discrete":
        step_score = min(1.0, avg_step_distance(colors) / 0.18)
        score += 8.0 * step_score
        reasons.append(f"Discrete separation score: {step_score:.2f}.")

    # Semantic intuition
    if intuitive:
        if class_name == "Sequential":
            sem = 1.0 if palette.intuitive_for_sequential else 0.55
        elif class_name in ["Diverging", "Multi-sequential"]:
            sem = 1.0 if palette.intuitive_for_diverging else 0.60
        else:
            sem = 0.75
        score += 7.0 * sem
        reasons.append(f"Semantic intuition score: {sem:.2f}.")
    else:
        score += 4.0

    # Cyclic behaviour
    if class_name == "Cyclic":
        score += 14.0 * palette.cyclic_score
        reasons.append(f"Cyclic continuity score: {palette.cyclic_score:.2f}.")
    else:
        if palette.cyclic_score > 0.8:
            score -= 3.0

    # Diverging / multi-sequential symmetry
    if class_name in ["Diverging", "Multi-sequential"]:
        sym = centre_symmetry_score(colors)
        score += 8.0 * sym
        reasons.append(f"Centre-balance score: {sym:.2f}.")

    # Contrast against background for endpoints
    endpoint_contrast = (contrast_ratio(colors[0], background) + contrast_ratio(colors[-1], background)) / 2.0
    endpoint_score = min(1.0, endpoint_contrast / 5.5)
    score += 8.0 * endpoint_score
    reasons.append(f"Endpoint contrast score: {endpoint_score:.2f}.")

    # Empty values handling
    if empty_values:
        bg_overlap = min(contrast_ratio(c, background) for c in colors)
        overlap_score = min(1.0, bg_overlap / 1.8)
        score += 5.0 * overlap_score
        reasons.append(f"Masked-value separability score: {overlap_score:.2f}.")

    # Publication / print stability bonus
    lum_span = luminance_span_score(colors)
    score += 5.0 * lum_span
    reasons.append(f"Luminance-span score: {lum_span:.2f}.")

    name_l = palette.name.lower()
    notes_l = palette.notes.lower()

    if "publication" in name_l or "publication" in notes_l:
        score += 2.5
        reasons.append("Bonus applied for publication-oriented palette design.")

    if "colorblind" in name_l or "accessibility" in notes_l:
        score += 2.0
        reasons.append("Bonus applied for accessibility-oriented palette design.")

    if type_name == "Categorical" and ("line plot" in name_l or "series" in name_l):
        score += 2.5
        reasons.append("Bonus applied for multi-series / line-plot suitability.")

    # Penalty for rainbow-like palettes in professional use
    if is_rainbow_like(palette.name):
        score -= 10.0
        reasons.append("Penalty applied for rainbow-like behaviour.")

    return score, reasons


# --------------------------
# Custom widgets
# --------------------------
class GradientPreview(QFrame):
    def __init__(self):
        super().__init__()
        self.colors: List[str] = ["#000000", "#FFFFFF"]
        self.setMinimumHeight(62)
        self.setMaximumHeight(62)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_colors(self, colors: List[str]):
        self.colors = colors[:]
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)
        path = QPainterPath()
        path.addRoundedRect(rect, 14.0, 14.0)

        grad = QLinearGradient(QPointF(rect.left(), rect.top()), QPointF(rect.right(), rect.top()))
        if not self.colors:
            self.colors = ["#000000", "#FFFFFF"]

        n = max(1, len(self.colors) - 1)
        for i, c in enumerate(self.colors):
            grad.setColorAt(i / n if n else 0.0, QColor(c))

        painter.fillPath(path, grad)
        painter.setPen(QPen(QColor("#C9D5E3"), 1.0))
        painter.drawPath(path)


class ColorCard(QFrame):
    def __init__(self, entry: ColorEntry):
        super().__init__()
        self.setObjectName("ColorCard")
        self.setMinimumHeight(116)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.swatch = QFrame()
        self.swatch.setFixedHeight(36)
        self.swatch.setStyleSheet(
            f"background:{entry.hex_code}; border-radius:10px; border:1px solid #8BA0B5;"
        )

        top = QHBoxLayout()

        self.idx_label = QLabel(f"Colour {entry.index}")
        self.idx_label.setStyleSheet("font-size:12px; font-weight:700; color:#112031;")

        self.role_label = QLabel(entry.role)
        self.role_label.setStyleSheet("font-size:11px; color:#516172;")
        self.role_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        top.addWidget(self.idx_label)
        top.addStretch(1)
        top.addWidget(self.role_label)

        self.hex_label = QLabel(entry.hex_code)
        self.hex_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.hex_label.setStyleSheet(
            "font-family: Consolas, 'Courier New', monospace; font-size:12px; color:#203040;"
        )

        layout.addLayout(top)
        layout.addWidget(self.swatch)
        layout.addWidget(self.hex_label)

    def update_entry(self, entry: ColorEntry):
        self.idx_label.setText(f"Colour {entry.index}")
        self.role_label.setText(entry.role)
        self.hex_label.setText(entry.hex_code)
        self.swatch.setStyleSheet(
            f"background:{entry.hex_code}; border-radius:10px; border:1px solid #8BA0B5;"
        )


# --------------------------
# Main window
# --------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1400, 900)
        self.setMinimumSize(1080, 720)

        if os.path.exists(LOGO_ICO):
            self.setWindowIcon(QIcon(LOGO_ICO))

        self.current_entries: List[ColorEntry] = []
        self.current_score: float = 0.0
        self.card_cache = []
        self._build_ui()
        self._update_result()
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        left = QFrame()
        left.setObjectName("Panel")
        left.setMinimumWidth(380)
        left.setMaximumWidth(460)

        left_outer_layout = QVBoxLayout(left)
        left_outer_layout.setContentsMargins(0, 0, 0, 0)
        left_outer_layout.setSpacing(0)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(12)

        left_scroll.setWidget(left_content)
        left_outer_layout.addWidget(left_scroll)

        header = QVBoxLayout()

        # ---------- Logo ----------
        logo_row = QHBoxLayout()

        self.logo_label = QLabel()
        if os.path.exists(LOGO_PNG):
            from PyQt6.QtGui import QPixmap
            pix = QPixmap(LOGO_PNG)
            self.logo_label.setPixmap(pix.scaledToHeight(50, Qt.TransformationMode.SmoothTransformation))

        logo_row.addWidget(self.logo_label)
        logo_row.addStretch()

        header.addLayout(logo_row)

        # ---------- Title ----------
        title = QLabel(APP_NAME)
        title.setStyleSheet("font-size:24px; font-weight:800; color:#0F1C2E;")
        title.setStyleSheet("font-size:24px; font-weight:800; color:#0F1C2E;")
        subtitle = QLabel("Professional colour recommendation logic for scientific figures")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("font-size:12px; color:#5B6B7E; font-weight:600;")
        header.addWidget(title)
        header.addWidget(subtitle)
        left_layout.addLayout(header)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)

        self.ordered_combo = self._make_combo(["Yes", "No"])
        self.centric_combo = self._make_combo(["Yes", "No"])
        self.diverging_combo = self._make_combo(["Yes", "No"])
        self.periodic_combo = self._make_combo(["Yes", "No"])
        self.small_var_combo = self._make_combo(["Yes", "No"])
        self.empty_combo = self._make_combo(["Yes", "No"])
        self.intuitive_combo = self._make_combo(["Yes", "No"])
        self.light_bg_combo = self._make_combo(["Yes", "No"])
        self.palette_combo = self._make_combo(["Auto"])
        self.num_colors_combo = self._make_combo([str(i) for i in range(3, 101)])
        self.num_colors_combo.setCurrentText(str(MIN_COLORS))

        form.addRow(self._make_label("Ordered"), self.ordered_combo)
        form.addRow(self._make_label("Centric value"), self.centric_combo)
        form.addRow(self._make_label("Diverging / converging"), self.diverging_combo)
        form.addRow(self._make_label("Circular / periodic"), self.periodic_combo)
        form.addRow(self._make_label("Important small variations"), self.small_var_combo)
        form.addRow(self._make_label("Empty values present"), self.empty_combo)
        form.addRow(self._make_label("Prefer intuitive colours"), self.intuitive_combo)
        form.addRow(self._make_label("Light figure background"), self.light_bg_combo)
        form.addRow(self._make_label("Palette option"), self.palette_combo)
        form.addRow(self._make_label("Number of colours"), self.num_colors_combo)
        left_layout.addLayout(form)

        self.article_notes = QCheckBox("Show publication reminders")
        self.article_notes.setChecked(True)
        left_layout.addWidget(self.article_notes)

        btn_row = QHBoxLayout()
        run_btn = QPushButton("Generate Recommendation")
        copy_btn = QPushButton("Copy Summary")
        run_btn.clicked.connect(self._update_result)
        copy_btn.clicked.connect(self._copy_summary)
        btn_row.addWidget(run_btn)
        btn_row.addWidget(copy_btn)
        left_layout.addLayout(btn_row)

        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setMinimumHeight(180)
        left_layout.addWidget(self.summary_box)

        link_row = QHBoxLayout()
        about_btn = QPushButton("About")
        about_btn.clicked.connect(self._show_about)
        link_row.addWidget(about_btn)
        left_layout.addLayout(link_row)
        left_layout.addStretch(1)

        right = QFrame()
        right.setObjectName("Panel")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(12)

        self.class_label = QLabel("Class: -")
        self.type_label = QLabel("Type: -")
        self.combo_label = QLabel("Colour combination: -")
        self.palette_label = QLabel("Palette: -")
        self.score_label = QLabel("Professional score: -")
        for lab in [self.class_label, self.type_label, self.combo_label, self.palette_label, self.score_label]:
            lab.setWordWrap(True)
            lab.setStyleSheet("font-size:14px; font-weight:700; color:#102338;")
            right_layout.addWidget(lab)

        preview_wrap = QFrame()
        preview_wrap.setObjectName("SubPanel")
        preview_layout = QVBoxLayout(preview_wrap)
        preview_layout.setContentsMargins(14, 14, 14, 14)
        preview_layout.setSpacing(10)
        preview_title = QLabel("Gradient Preview")
        preview_title.setStyleSheet("font-size:13px; font-weight:800; color:#14273A;")
        self.gradient_preview = GradientPreview()
        preview_layout.addWidget(preview_title)
        preview_layout.addWidget(self.gradient_preview)
        right_layout.addWidget(preview_wrap)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.cards_widget = QWidget()
        self.cards_layout = QGridLayout(self.cards_widget)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setHorizontalSpacing(12)
        self.cards_layout.setVerticalSpacing(12)
        scroll.setWidget(self.cards_widget)
        right_layout.addWidget(scroll, 1)

        root.addWidget(left, 0)
        root.addWidget(right, 1)

        self._build_menu()
        self._apply_style()

        for combo in [
            self.ordered_combo,
            self.centric_combo,
            self.diverging_combo,
            self.periodic_combo,
            self.small_var_combo,
            self.empty_combo,
            self.intuitive_combo,
            self.light_bg_combo,
            self.palette_combo,
            self.num_colors_combo,
        ]:
            combo.currentTextChanged.connect(self._update_result)

    def _build_menu(self):
        copy_action = QAction("Copy Summary", self)
        copy_action.triggered.connect(self._copy_summary)
        self.menuBar().addAction(copy_action)

    def _apply_style(self):
        self.setStyleSheet(
            """
            QWidget {
                background: #EDF3F8;
                color: #162433;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QFrame#Panel {
                background: #FFFFFF;
                border: 1px solid #D6E0EA;
                border-radius: 20px;
            }
            QFrame#SubPanel {
                background: #F7FAFD;
                border: 1px solid #D7E2EC;
                border-radius: 16px;
            }
            QFrame#ColorCard {
                background: #F9FBFD;
                border: 1px solid #DCE6F0;
                border-radius: 16px;
            }
            QLabel { background: transparent; }
            QComboBox, QTextEdit {
                background: #F8FBFE;
                border: 1px solid #C9D5E2;
                border-radius: 12px;
                padding: 8px;
                font-size: 12px;
                min-height: 22px;
            }
            QPushButton {
                background: #0E5A8A;
                color: white;
                border: none;
                border-radius: 12px;
                padding: 10px 14px;
                font-weight: 800;
                min-height: 22px;
            }
            QPushButton:hover {
                background: #0B476D;
            }
            QCheckBox {
                font-size: 12px;
                font-weight: 600;
                color: #425466;
            }
            QMenuBar {
                background: #F4F8FB;
                border-bottom: 1px solid #D7E2EC;
            }
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: #EAF0F6;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #B7C7D8;
                min-height: 24px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            """
        )

    def _make_label(self, text: str) -> QLabel:
        lab = QLabel(text)
        lab.setStyleSheet("font-size:12px; font-weight:700; color:#27384B;")
        lab.setMinimumWidth(180)
        return lab

    def _make_combo(self, items: List[str]) -> QComboBox:
        combo = QComboBox()
        combo.addItems(items)
        combo.setMinimumWidth(180)
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return combo

    def _refresh_palette_options(self, class_name: str):
        current = self.palette_combo.currentText()
        self.palette_combo.blockSignals(True)
        self.palette_combo.clear()
        self.palette_combo.addItem("Auto")
        for spec in PALETTE_LIBRARY[class_name]:
            self.palette_combo.addItem(spec.name)
        idx = self.palette_combo.findText(current)
        self.palette_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.palette_combo.blockSignals(False)

    def _clear_cards(self):
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)

    def _populate_cards(self, entries: List[ColorEntry]):
        self._clear_cards()

        width = self.centralWidget().width()
        if width >= 1700:
            cols = 5
        elif width >= 1350:
            cols = 4
        elif width >= 1050:
            cols = 3
        else:
            cols = 2

        while len(self.card_cache) < len(entries):
            dummy = ColorEntry(0, "#FFFFFF", "")
            self.card_cache.append(ColorCard(dummy))

        for i, entry in enumerate(entries):
            card = self.card_cache[i]
            card.update_entry(entry)
            row, col = divmod(i, cols)
            self.cards_layout.addWidget(card, row, col)
            card.show()

        for card in self.card_cache[len(entries):]:
            card.hide()
            card.setParent(None)

        self.cards_widget.update()

    def _update_card(self, card, entry: ColorEntry):
        card.entry = entry

        color = QColor(entry.hex_code)
        r, g, b, a = color.getRgbF()
        brightness = 0.299*r + 0.587*g + 0.114*b
        text_color = "#000000" if brightness > 0.65 else "#FFFFFF"

        swatch = card.findChild(QFrame)
        swatch.setStyleSheet(f"background:{entry.hex_code}; border-radius:10px; border:1px solid #8BA0B5;")

        idx_label = card.findChild(QLabel, "idx_label")
        idx_label.setText(f"Colour {entry.index}")
        idx_label.setStyleSheet(f"font-size:12px; font-weight:700; color:{text_color};")

        role_label = card.findChild(QLabel, "role_label")
        role_label.setText(entry.role)
        role_label.setStyleSheet(f"font-size:11px; color:rgba(0,0,0,0.6);")

        hex_label = card.findChild(QLabel)
        hex_label.setText(entry.hex_code)
        hex_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        hex_label.setStyleSheet(f"font-family: Consolas, 'Courier New', monospace; font-size:12px; color:#203040;")

    def _pick_palette(self, class_name: str, type_name: str, intuitive: bool, light_bg: bool, small_variation: bool, empty_values: bool) -> Tuple[PaletteSpec, float, List[str]]:
        manual_name = self.palette_combo.currentText()
        candidates = PALETTE_LIBRARY[class_name]

        if manual_name != "Auto":
            for spec in candidates:
                if spec.name == manual_name:
                    score, reasons = score_palette(spec, class_name, type_name, intuitive, light_bg, small_variation, empty_values)
                    reasons.insert(0, "Manual palette override applied.")
                    return spec, score, reasons

        best_spec = candidates[0]
        best_score = -1e9
        best_reasons: List[str] = []
        for spec in candidates:
            score, reasons = score_palette(spec, class_name, type_name, intuitive, light_bg, small_variation, empty_values)
            if score > best_score:
                best_spec = spec
                best_score = score
                best_reasons = reasons
        return best_spec, best_score, best_reasons

    def _summary_text(self, class_name: str, type_name: str, combo_rule: str, palette: PaletteSpec, score: float, reasons: List[str], entries: List[ColorEntry]) -> str:
        lines = [
            f"Software: {APP_NAME}",
            f"Class: {class_name}",
            f"Type: {type_name}",
            f"Palette: {palette.name}",
            f"Palette family: {palette.family}",
            f"Source: {palette.source}",
            f"Professional score: {score:.1f} / 100+",
            f"Colour combination rule: {combo_rule}",
            "",
            "Selection logic summary:",
        ]
        for reason in reasons[:8]:
            lines.append(f"- {reason}")

        lines.extend(["", f"Suggested colours ({len(entries)} HEX codes):"])
        for entry in entries:
            lines.append(f"{entry.index:02d}. {entry.hex_code}  |  {entry.role}")

        if self.article_notes.isChecked():
            lines.extend([
                "",
                "Publication reminders:",
                "- Avoid rainbow-like maps for quantitative gradients.",
                "- Keep colourbar labels explicit and physically meaningful.",
                "- Reserve neutral colour or background for masked / missing values.",
                "- Check grayscale readability before manuscript submission.",
            ])

        return "\n".join(lines)

    def _palette_colors_for_count(self, palette: PaletteSpec, class_name: str, n: int) -> List[str]:
        if class_name == "Categorical":
            base = palette.colors[:]
            if not base:
                return []
            return [base[i % len(base)] for i in range(n)]

        if palette.source.startswith("matplotlib:"):
            cmap_name = palette.source.split(":", 1)[1].strip()
            if palette.name == "Twilight Pro":
                return sample_cmap(cmap_name, n=n, start=0.0, stop=1.0)
            return sample_cmap(cmap_name, n=n)

        return sample_custom_gradient(palette.colors, n=n)

    def _update_result(self):
        num_colors = int(self.num_colors_combo.currentText())

        ordered = self.ordered_combo.currentText() == "Yes"
        centric = self.centric_combo.currentText() == "Yes"
        diverging = self.diverging_combo.currentText() == "Yes"
        periodic = self.periodic_combo.currentText() == "Yes"
        small_variation = self.small_var_combo.currentText() == "Yes"
        empty_values = self.empty_combo.currentText() == "Yes"
        intuitive = self.intuitive_combo.currentText() == "Yes"
        light_bg = self.light_bg_combo.currentText() == "Yes"

        class_name = class_from_logic(ordered, centric, diverging, periodic)
        self._refresh_palette_options(class_name)
        type_name = type_from_logic(class_name, small_variation)
        combo_rule = combination_from_logic(empty_values, intuitive, light_bg, centric)

        palette, score, reasons = self._pick_palette(class_name, type_name, intuitive, light_bg, small_variation, empty_values)
        display_colors = self._palette_colors_for_count(palette, class_name, num_colors)
        entries = build_entries(display_colors, ROLE_LABELS[class_name], num_colors)
        self.current_entries = entries
        self.current_score = score

        self.class_label.setText(f"Class: {class_name}")
        self.type_label.setText(f"Type: {type_name}")
        self.combo_label.setText(f"Colour combination: {combo_rule}")
        self.palette_label.setText(f"Palette: {palette.name}  |  Family: {palette.family}  |  Source: {palette.source}")
        self.score_label.setText(f"Professional score: {score:.1f}")
        self.gradient_preview.set_colors(display_colors)
        self.summary_box.setPlainText(self._summary_text(class_name, type_name, combo_rule, palette, score, reasons, entries))
        self._populate_cards(entries)
    def _copy_summary(self):
        QGuiApplication.clipboard().setText(self.summary_box.toPlainText())
        QMessageBox.information(self, APP_NAME, "Summary copied to clipboard.")

    def _show_about(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("About")
        dialog.setMinimumWidth(520)

        if os.path.exists(LOGO_ICO):
            dialog.setWindowIcon(QIcon(LOGO_ICO))

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        if os.path.exists(LOGO_PNG):
            logo_label = QLabel()
            logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pix = QPixmap(LOGO_PNG)
            logo_label.setPixmap(pix.scaledToHeight(80, Qt.TransformationMode.SmoothTransformation))
            layout.addWidget(logo_label)

        title = QLabel(APP_NAME)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setWordWrap(True)
        title.setStyleSheet("font-size:18px; font-weight:800; color:#0F1C2E;")
        layout.addWidget(title)

        description = QLabel(
            "A professional scientific tool for intelligent color selection "
            "in data visualization.\n\n"
            "The software applies perceptual accuracy, accessibility, and "
            "publication-oriented standards to recommend high-quality color "
            "palettes for scientific figures.\n\n"
            "Designed to support clear, reproducible, and publication-ready "
            "visual results for research and academic writing."
        )
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setStyleSheet("font-size:12px; color:#405166;")
        layout.addWidget(description)

        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(dialog.accept)
        layout.addWidget(ok_btn)

        dialog.setStyleSheet(
            """
            QDialog { background: #FFFFFF; }
            QLabel { background: transparent; }
            QPushButton {
                background: #0E5A8A;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px 18px;
                font-weight: 800;
            }
            QPushButton:hover { background: #0B476D; }
            """
        )

        dialog.exec()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_entries:
            self._populate_cards(self.current_entries)


if __name__ == "__main__":
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("researcher.standard.color.toolkit")

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setFont(QFont("Segoe UI", 10))

    if os.path.exists(LOGO_ICO):
        app.setWindowIcon(QIcon(LOGO_ICO))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
