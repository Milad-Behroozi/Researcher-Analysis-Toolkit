"""Physics-guided utilities for automatic EIS circuit selection.

The functions in this module deliberately avoid GUI state and the
``impedance.py`` fitting API wherever possible.  This keeps the physical
feature extraction and candidate-screening logic deterministic, testable, and
fast.  The fitting front end in :mod:`eis_auto_v2` adds model-specific
diagnostics after each candidate has been fitted.

The automatic workflow is a *candidate-ranking* system.  Equivalent circuits
are often non-unique over a finite frequency window, so this module reports
ambiguity rather than presenting a statistically close model as a unique
physical answer.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.signal import find_peaks, savgol_filter


PHYSICS_ENGINE_VERSION = "2.0.0"

_TOKEN_RE = re.compile(r"CPE\d+|Wo\d+|Ws\d+|[RCWLG]\d+")
_RELAXATION_RE = re.compile(
    r"p\(R\d+(?:-(?:W|Wo|Ws)\d+)?,(?:CPE|C)\d+\)"
    r"|p\((?:CPE|C)\d+,R\d+(?:-(?:W|Wo|Ws)\d+)?\)"
)


def circuit_tokens(circuit: str) -> list[str]:
    """Return element tokens without confusing CPE with C or W with Wo/Ws."""
    return _TOKEN_RE.findall(str(circuit or ""))


def circuit_element_families(circuit: str) -> set[str]:
    families: set[str] = set()
    for token in circuit_tokens(circuit):
        if token.startswith("CPE"):
            families.add("CPE")
        elif token.startswith(("Wo", "Ws", "W")):
            families.add("W")
        else:
            families.add(token[0])
    return families


def circuit_relaxation_count(circuit: str) -> int:
    """Count recognizable R||C/R||CPE relaxation blocks."""
    count = len(_RELAXATION_RE.findall(str(circuit or "")))
    if count:
        return count
    # Fallback for unusual but valid nested strings.
    return min(
        str(circuit or "").count("p("),
        sum(t.startswith(("CPE", "C")) for t in circuit_tokens(circuit)),
    )


def circuit_warburg_tokens(circuit: str) -> list[str]:
    return [t for t in circuit_tokens(circuit) if t.startswith(("W", "Wo", "Ws"))]


def _safe_savgol(values: np.ndarray, preferred: int = 11) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < 5:
        return values.copy()
    window = min(max(5, int(preferred)), int(values.size))
    if window % 2 == 0:
        window -= 1
    if window < 5:
        return values.copy()
    return savgol_filter(values, window_length=window, polyorder=2, mode="interp")


def _linear_tail_metrics(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or np.ptp(x) <= np.finfo(float).eps:
        return 90.0, 0.0
    design = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > np.finfo(float).eps else 0.0
    angle = abs(float(np.degrees(np.arctan(slope))))
    return float(np.clip(angle, 0.0, 90.0)), float(np.clip(r2, -1.0, 1.0))


def _peak_frequencies(
    logf: np.ndarray,
    signal: np.ndarray,
    frequencies: np.ndarray,
    *,
    min_height: float,
    min_separation_decades: float = 0.45,
    prominence_fraction: float = 0.08,
) -> tuple[list[float], list[float], np.ndarray]:
    smooth = _safe_savgol(np.asarray(signal, dtype=float))
    dynamic = max(float(np.ptp(smooth)), 0.0)
    prominence = max(float(min_height) * 0.15, prominence_fraction * dynamic)
    step = float(np.median(np.diff(logf))) if logf.size > 1 else 1.0
    distance = max(1, int(round(min_separation_decades / max(abs(step), 1e-9))))
    peaks, props = find_peaks(
        smooth,
        height=float(min_height),
        prominence=prominence,
        distance=distance,
    )
    prominences = np.asarray(props.get("prominences", np.zeros(peaks.size)), dtype=float)
    if peaks.size:
        strongest = max(float(np.max(prominences)), np.finfo(float).eps)
        keep = prominences >= max(prominence, 0.25 * strongest)
        peaks = peaks[keep]
        prominences = prominences[keep]
    return (
        [float(frequencies[i]) for i in peaks],
        [float(v) for v in prominences],
        smooth,
    )


def analyze_eis_physics(
    frequency_hz: Sequence[float],
    impedance: Sequence[complex],
    phase_deg: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Extract conservative physical features from an EIS spectrum.

    Low-frequency metrics are always calculated from the *lowest-frequency*
    end after sorting.  This corrects the legacy implementation, which sorted
    frequencies ascending but then inspected the final (high-frequency) rows.
    """
    f = np.asarray(frequency_hz, dtype=float)
    z = np.asarray(impedance, dtype=complex)
    phase = None if phase_deg is None else np.asarray(phase_deg, dtype=float)

    mask = np.isfinite(f) & np.isfinite(z.real) & np.isfinite(z.imag) & (f > 0)
    if phase is not None:
        mask &= np.isfinite(phase)
    f, z = f[mask], z[mask]
    if phase is not None:
        phase = phase[mask]
    if f.size < 5:
        raise ValueError("At least five finite positive-frequency EIS points are required.")

    order = np.argsort(f)
    f, z = f[order], z[order]
    if phase is None:
        phase = np.angle(z, deg=True)
    else:
        phase = phase[order]

    logf = np.log10(f)
    phase_magnitude = np.maximum(-phase, 0.0)
    if float(np.max(phase_magnitude)) < 3.0:
        phase_magnitude = np.abs(phase)

    phase_freqs, phase_strengths, phase_smooth = _peak_frequencies(
        logf,
        phase_magnitude,
        f,
        min_height=4.0,
    )

    nyquist_y = -z.imag
    nyquist_positive = np.maximum(nyquist_y, 0.0)
    nyquist_height = max(0.01 * float(np.ptp(nyquist_positive)), 1e-12)
    nyquist_freqs, nyquist_strengths, nyquist_smooth = _peak_frequencies(
        logf,
        nyquist_positive,
        f,
        min_height=nyquist_height,
        prominence_fraction=0.05,
    )

    n_tail = max(5, min(10, int(math.ceil(0.18 * f.size))))
    low = slice(0, n_tail)
    tail_angle, tail_r2 = _linear_tail_metrics(z.real[low], nyquist_y[low])
    low_phase = float(np.median(np.abs(phase[low])))
    log_mag = np.log10(np.clip(np.abs(z[low]), 1e-30, np.inf))
    if np.ptp(logf[low]) > 0:
        magnitude_slope = float(np.polyfit(logf[low], log_mag, 1)[0])
    else:
        magnitude_slope = 0.0

    tail_span = float(
        np.hypot(np.ptp(z.real[low]), np.ptp(nyquist_y[low]))
        / max(np.hypot(np.ptp(z.real), np.ptp(nyquist_y)), 1e-30)
    )
    tail_is_resolved = tail_span >= 0.08 and tail_r2 >= 0.75

    # A moderate Warburg contribution can leave a clear ~45° low-frequency
    # Nyquist tail while the total phase remains below 18° because R0 and the
    # relaxation branch still dominate |Z|.  Accept that weaker but resolved
    # morphology only when the tail spans a substantial fraction of the plot
    # and |Z| increases toward low frequency.  The span/linearity gates keep a
    # noisy, already-closed RC arc from being labelled as diffusion.
    semi_infinite = bool(
        (
            tail_is_resolved
            and 28.0 <= tail_angle <= 62.0
            and 18.0 <= low_phase <= 72.0
        )
        or (
            tail_span >= 0.15
            and tail_r2 >= 0.65
            and 28.0 <= tail_angle <= 62.0
            and magnitude_slope <= -0.05
        )
    )
    finite_open = bool(
        tail_is_resolved
        and tail_angle >= 65.0
        and low_phase >= 50.0
        and magnitude_slope <= -0.25
    )
    finite_short = bool(
        tail_is_resolved
        and 5.0 <= tail_angle <= 28.0
        and low_phase <= 35.0
        and abs(magnitude_slope) <= 0.35
    )

    if finite_open:
        diffusion_kind = "open"
    elif semi_infinite:
        diffusion_kind = "semi-infinite"
    elif finite_short:
        diffusion_kind = "short"
    else:
        diffusion_kind = "none"

    # An inductive loop must be a *resolved, contiguous* positive-Z'' feature.
    # Counting isolated positive points classified ordinary Gaussian noise as
    # an inductor in the legacy search.  Work on a short smooth trace, set the
    # threshold from both the spectrum scale and the local noise estimate, and
    # require at least three adjacent samples above it.
    imag_smooth = _safe_savgol(z.imag, preferred=7)
    smooth_residual = z.imag - imag_smooth
    residual_center = float(np.median(smooth_residual))
    noise_sigma = 1.4826 * float(
        np.median(np.abs(smooth_residual - residual_center))
    )
    imag_scale = max(float(np.max(np.abs(z.imag))), 1e-30)
    inductive_threshold = max(0.04 * imag_scale, 3.0 * noise_sigma, 1e-12)
    inductive_mask = imag_smooth > inductive_threshold

    max_inductive_run = 0
    current_run = 0
    for is_positive in inductive_mask:
        current_run = current_run + 1 if bool(is_positive) else 0
        max_inductive_run = max(max_inductive_run, current_run)

    inductive_fraction = float(np.mean(inductive_mask))
    has_inductive = bool(max_inductive_run >= 3)

    apparent_loops = max(1, len(phase_freqs))
    # Nyquist maxima provide supporting evidence only; CPE and diffusion tails
    # can hide phase maxima, while noise can create small Nyquist ripples.
    if len(phase_freqs) == 0 and len(nyquist_freqs) > 1:
        apparent_loops = min(2, len(nyquist_freqs))

    separations: list[float] = []
    if len(phase_freqs) > 1:
        ordered_logs = sorted(math.log10(v) for v in phase_freqs if v > 0)
        separations = [float(b - a) for a, b in zip(ordered_logs, ordered_logs[1:])]

    delta_real = float(np.ptp(z.real))
    delta_imag = float(np.ptp(nyquist_y))
    aspect_ratio = delta_imag / max(delta_real, 1e-30)
    shape_mode = "linear" if aspect_ratio > 3.0 else "arc"

    notes: list[str] = []
    if apparent_loops == 1:
        notes.append("one apparent relaxation is resolvable")
    else:
        notes.append(f"{apparent_loops} apparent relaxations are resolvable")
    if diffusion_kind != "none":
        notes.append(f"{diffusion_kind} diffusion signature detected")
    if has_inductive:
        notes.append("inductive response detected")
    if separations and min(separations) < 0.5:
        notes.append("relaxations are strongly overlapping")

    return {
        "n_points": int(f.size),
        "frequency_min_hz": float(np.min(f)),
        "frequency_max_hz": float(np.max(f)),
        "frequency_span_decades": float(np.ptp(logf)),
        "apparent_loops": int(apparent_loops),
        "phase_peak_frequencies_hz": phase_freqs,
        "phase_peak_prominences": phase_strengths,
        "nyquist_peak_frequencies_hz": nyquist_freqs,
        "nyquist_peak_prominences": nyquist_strengths,
        "minimum_peak_separation_decades": (
            float(min(separations)) if separations else None
        ),
        "tail_angle_deg": float(tail_angle),
        "tail_linearity_r2": float(tail_r2),
        "tail_span_fraction": float(tail_span),
        "low_frequency_phase_magnitude_deg": float(low_phase),
        "low_frequency_logmag_slope": float(magnitude_slope),
        "diffusion_kind": diffusion_kind,
        "has_diffusion": diffusion_kind != "none",
        "has_inductive": has_inductive,
        "inductive_fraction": inductive_fraction,
        "inductive_threshold_ohm": float(inductive_threshold),
        "maximum_inductive_run_points": int(max_inductive_run),
        "shape_mode": shape_mode,
        "max_phase_magnitude_deg": float(np.max(phase_smooth)),
        "notes": notes,
    }


def _allowed(circuit: str, allowed_elements: Iterable[str] | None) -> bool:
    if allowed_elements is None:
        return True
    allowed_set = {str(v).upper() for v in allowed_elements}
    return all(family.upper() in allowed_set for family in circuit_element_families(circuit))


def build_physics_guided_candidates(
    features: Mapping[str, Any],
    level: str = "L1",
    allowed_elements: Iterable[str] | None = None,
    learned_circuits: Iterable[str] | None = None,
) -> list[str]:
    """Build a compact deterministic library from observed EIS morphology."""
    level = str(level or "L1").upper()
    depth = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}.get(level, 1)
    loops = max(1, int(features.get("apparent_loops", 1)))
    diffusion = str(features.get("diffusion_kind", "none"))
    inductive = bool(features.get("has_inductive", False))

    candidates: list[str] = []

    def add(*items: str) -> None:
        for item in items:
            if item and item not in candidates and _allowed(item, allowed_elements):
                candidates.append(item)

    # Always begin with standard one-relaxation electrode-interface models.
    add("R0-p(R1,CPE1)", "R0-p(R1,C1)")

    # Two-time-constant models are used when resolved, and remain available as
    # diagnostic alternatives from Level 2 upward.
    if loops >= 2 or depth >= 2:
        add(
            "R0-p(R1,CPE1)-p(R2,CPE2)",
            "R0-p(R1,C1)-p(R2,C2)",
            "R0-p(R1,CPE1)-p(R2,C2)",
            "R0-p(R1,C1)-p(R2,CPE2)",
        )

    if loops >= 3 and depth >= 3:
        add(
            "R0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)",
            "R0-p(R1,C1)-p(R2,C2)-p(R3,C3)",
        )

    if diffusion == "semi-infinite":
        add(
            "R0-p(R1,CPE1)-W1",
            "R0-p(R1-W1,CPE1)",
            "R0-p(R1,C1)-W1",
            "R0-p(R1-W1,C1)",
        )
        if loops >= 2 or depth >= 2:
            add(
                "R0-p(R1,CPE1)-p(R2,CPE2)-W1",
                "R0-p(R1,CPE1)-p(R2-W1,CPE2)",
            )
    elif diffusion == "open":
        add(
            "R0-p(R1,CPE1)-Wo1",
            "R0-p(R1-Wo1,CPE1)",
            "R0-p(R1,C1)-Wo1",
        )
        if loops >= 2 or depth >= 2:
            add("R0-p(R1,CPE1)-p(R2-Wo1,CPE2)")
    elif diffusion == "short":
        add(
            "R0-p(R1,CPE1)-Ws1",
            "R0-p(R1-Ws1,CPE1)",
            "R0-p(R1,C1)-Ws1",
        )
        if loops >= 2 or depth >= 2:
            add("R0-p(R1,CPE1)-p(R2-Ws1,CPE2)")
    elif depth >= 3:
        # Weak tails can be ambiguous.  Advanced levels inspect one canonical
        # representative per diffusion boundary, never arbitrary W chains.
        add(
            "R0-p(R1,CPE1)-W1",
            "R0-p(R1-Wo1,CPE1)",
            "R0-p(R1-Ws1,CPE1)",
        )

    if inductive:
        add("R0-p(R1,CPE1)-p(R2,L1)", "R0-p(R1,C1)-p(R2,L1)")

    if depth >= 4:
        add(
            "R0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)",
            "R0-p(R1,CPE1)-p(R2-W1,CPE2)",
            "R0-p(R1,CPE1)-p(R2-Wo1,CPE2)",
            "R0-p(R1,CPE1)-p(R2-Ws1,CPE2)",
        )

    # Learned circuits are admitted only after the same physical screening.
    if learned_circuits:
        for circuit in learned_circuits:
            diagnostic = physics_compatibility(circuit, features, allowed_elements)
            if not diagnostic["hard_reject"]:
                add(str(circuit))

    return candidates


def physics_compatibility(
    circuit: str,
    features: Mapping[str, Any],
    allowed_elements: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return physical consistency flags and a soft information penalty."""
    circuit = str(circuit or "")
    reasons: list[str] = []
    hard_reject = False
    penalty = 0.0

    if not circuit.startswith("R0"):
        hard_reject = True
        reasons.append("circuit does not begin with series resistance R0")
    if not _allowed(circuit, allowed_elements):
        hard_reject = True
        reasons.append("contains a user-disabled element family")

    warburg = circuit_warburg_tokens(circuit)
    if len(warburg) > 1:
        hard_reject = True
        reasons.append("multiple diffusion elements are not allowed in the guided search")
    if re.search(r"-(?:W|Wo|Ws)\d+-(?:W|Wo|Ws)\d+", circuit):
        hard_reject = True
        reasons.append("serial diffusion-only chain is physically underconstrained")

    diffusion_kind = str(features.get("diffusion_kind", "none"))
    if warburg:
        token = warburg[0]
        token_kind = "open" if token.startswith("Wo") else "short" if token.startswith("Ws") else "semi-infinite"
        if diffusion_kind == "none":
            penalty += 18.0
            reasons.append("diffusion element lacks a resolved low-frequency signature")
        elif token_kind != diffusion_kind:
            penalty += 12.0
            reasons.append(f"{token_kind} diffusion conflicts with the observed {diffusion_kind} tail")
    elif diffusion_kind != "none":
        penalty += 4.0
        reasons.append("model omits a resolved diffusion signature")

    has_l = "L" in circuit_element_families(circuit)
    inductive = bool(features.get("has_inductive", False))
    if has_l and not inductive:
        hard_reject = True
        reasons.append("inductor used without an observed inductive response")
    elif inductive and not has_l:
        penalty += 5.0
        reasons.append("model omits an observed inductive response")

    expected = max(1, int(features.get("apparent_loops", 1)))
    model_loops = circuit_relaxation_count(circuit)
    loop_mismatch = abs(model_loops - expected)
    penalty += 6.0 * loop_mismatch
    if loop_mismatch:
        reasons.append(
            f"model has {model_loops} relaxation block(s), data resolve {expected}"
        )
    if expected >= 1 and model_loops == 0 and not warburg:
        hard_reject = True
        reasons.append("model has no relaxation or diffusion element")

    if hard_reject:
        status = "inconsistent"
    elif penalty >= 12.0:
        status = "weak"
    elif penalty > 0.0:
        status = "plausible"
    else:
        status = "consistent"

    return {
        "physical_status": status,
        "physical_penalty": float(penalty),
        "hard_reject": bool(hard_reject),
        "physical_reasons": reasons,
        "model_relaxations": int(model_loops),
        "loop_mismatch": int(loop_mismatch),
        "warburg_count": int(len(warburg)),
    }


def _extract_parameter_names(circuit: str) -> list[str]:
    names: list[str] = []
    for token in circuit_tokens(circuit):
        if token.startswith(("CPE", "Wo", "Ws")):
            names.extend([f"{token}_0", f"{token}_1"])
        else:
            names.append(token)
    return names


def parameter_identifiability(
    circuit: str,
    model: Any,
    frequency_hz: Sequence[float],
) -> dict[str, Any]:
    """Assess bounds, uncertainty, and resolvability of fitted time constants."""
    flags: list[str] = []
    if model is None or not hasattr(model, "parameters_"):
        return {
            "identifiability_status": "failed",
            "identifiability_rank": 3,
            "bound_hits": 999999,
            "identifiability_flags": ["fitted parameters unavailable"],
            "characteristic_frequencies_hz": [],
            "minimum_time_constant_separation_decades": None,
            "maximum_relative_standard_error": None,
        }

    names = _extract_parameter_names(circuit)
    values = np.asarray(model.parameters_, dtype=float)
    if values.size != len(names) or not np.all(np.isfinite(values)):
        return {
            "identifiability_status": "failed",
            "identifiability_rank": 3,
            "bound_hits": 999999,
            "identifiability_flags": ["parameter vector is invalid"],
            "characteristic_frequencies_hz": [],
            "minimum_time_constant_separation_decades": None,
            "maximum_relative_standard_error": None,
        }

    params = dict(zip(names, values))
    bound_hits = 0
    for name, value in params.items():
        if name.startswith("CPE") and name.endswith("_1"):
            if value <= 0.403 or value >= 0.997:
                bound_hits += 1
                flags.append(f"{name} is at a CPE-exponent bound")
        elif value <= 1e-28:
            bound_hits += 1
            flags.append(f"{name} collapsed onto the positive numerical bound")

    relative_se: list[float] = []
    conf = getattr(model, "conf_", None)
    if conf is not None:
        conf = np.asarray(conf, dtype=float)
        if conf.size == values.size:
            relative_se = [
                float(abs(c) / max(abs(v), 1e-30))
                for c, v in zip(conf, values)
                if np.isfinite(c) and np.isfinite(v)
            ]
    max_relative_se = max(relative_se) if relative_se else None
    if max_relative_se is not None and max_relative_se > 1.0:
        flags.append("at least one parameter has relative standard error above 100%")
    elif max_relative_se is not None and max_relative_se > 0.5:
        flags.append("at least one parameter has relative standard error above 50%")

    characteristic: list[float] = []
    indices = sorted({
        int(match.group(1))
        for name in names
        for match in [re.match(r"(?:CPE|C)(\d+)(?:_[01])?$", name)]
        if match
    })
    for idx in indices:
        resistance = params.get(f"R{idx}")
        if resistance is None or resistance <= 0:
            continue
        if f"CPE{idx}_0" in params and f"CPE{idx}_1" in params:
            q = params[f"CPE{idx}_0"]
            exponent = params[f"CPE{idx}_1"]
            if q > 0 and 0 < exponent <= 1:
                fc = (1.0 / (resistance * q)) ** (1.0 / exponent) / (2.0 * math.pi)
                if np.isfinite(fc) and fc > 0:
                    characteristic.append(float(fc))
        elif f"C{idx}" in params and params[f"C{idx}"] > 0:
            fc = 1.0 / (2.0 * math.pi * resistance * params[f"C{idx}"])
            if np.isfinite(fc) and fc > 0:
                characteristic.append(float(fc))

    f = np.asarray(frequency_hz, dtype=float)
    f = f[np.isfinite(f) & (f > 0)]
    if f.size:
        fmin, fmax = float(np.min(f)), float(np.max(f))
        for fc in characteristic:
            if fc < fmin / 10.0 or fc > fmax * 10.0:
                flags.append(f"characteristic frequency {fc:.3g} Hz lies outside the measured window")

    minimum_separation = None
    if len(characteristic) > 1:
        logs = sorted(math.log10(fc) for fc in characteristic)
        separations = [b - a for a, b in zip(logs, logs[1:])]
        minimum_separation = float(min(separations))
        if minimum_separation < 0.5:
            flags.append("fitted relaxation times are strongly overlapping")
        elif minimum_separation < 1.0:
            flags.append("fitted relaxation times are weakly resolved")

    rank = 0
    if bound_hits or (max_relative_se is not None and max_relative_se > 1.0):
        rank = 2
    elif flags:
        rank = 1
    status = {0: "good", 1: "weak", 2: "poor"}[rank]
    return {
        "identifiability_status": status,
        "identifiability_rank": int(rank),
        "bound_hits": int(bound_hits),
        "identifiability_flags": flags,
        "characteristic_frequencies_hz": characteristic,
        "minimum_time_constant_separation_decades": minimum_separation,
        "maximum_relative_standard_error": max_relative_se,
    }


def residual_diagnostics(
    frequency_hz: Sequence[float],
    impedance: Sequence[complex],
    fitted_impedance: Sequence[complex],
) -> dict[str, Any]:
    f = np.asarray(frequency_hz, dtype=float)
    z = np.asarray(impedance, dtype=complex)
    zfit = np.asarray(fitted_impedance, dtype=complex)
    mask = (
        np.isfinite(f)
        & np.isfinite(z.real)
        & np.isfinite(z.imag)
        & np.isfinite(zfit.real)
        & np.isfinite(zfit.imag)
        & (f > 0)
    )
    f, z, zfit = f[mask], z[mask], zfit[mask]
    if f.size < 4:
        return {"residual_structure": np.inf, "residual_structure_rank": 2}
    order = np.argsort(f)
    scale = np.clip(np.abs(z[order]), 1e-30, np.inf)
    residual = (z[order] - zfit[order]) / scale

    def lag_one(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        if values.size < 4 or np.std(values[:-1]) <= 1e-15 or np.std(values[1:]) <= 1e-15:
            return 0.0
        return float(np.corrcoef(values[:-1], values[1:])[0, 1])

    structure = max(abs(lag_one(residual.real)), abs(lag_one(residual.imag)))
    rank = 0 if structure < 0.35 else 1 if structure < 0.65 else 2
    return {
        "residual_structure": float(structure),
        "residual_structure_rank": int(rank),
    }


def annotate_result_physics(
    result: dict[str, Any],
    features: Mapping[str, Any],
    frequency_hz: Sequence[float],
    impedance: Sequence[complex],
    allowed_elements: Iterable[str] | None = None,
) -> dict[str, Any]:
    circuit = str(result.get("circuit", ""))
    result.update(physics_compatibility(circuit, features, allowed_elements))

    model = result.get("model")
    ident = parameter_identifiability(circuit, model, frequency_hz)
    result.update(ident)
    # Preserve the historical bound-hit field while using the stricter v2
    # diagnostics as the source of truth.
    result["bound_hits"] = int(ident.get("bound_hits", result.get("bound_hits", 999999)))

    if model is not None:
        try:
            zfit = model.predict(np.asarray(frequency_hz, dtype=float))
            result.update(residual_diagnostics(frequency_hz, impedance, zfit))
        except Exception:
            result.update({"residual_structure": np.inf, "residual_structure_rank": 2})
    else:
        result.update({"residual_structure": np.inf, "residual_structure_rank": 2})

    penalty = float(result.get("physical_penalty", 0.0))
    penalty += 15.0 * int(result.get("identifiability_rank", 3))
    penalty += 6.0 * int(result.get("residual_structure_rank", 2))
    penalty += 20.0 * min(int(result.get("bound_hits", 0)), 3)
    result["selection_penalty"] = float(penalty)

    bic = float(result.get("bic", np.inf))
    aicc = float(result.get("aicc", result.get("aic", np.inf)))
    result["physics_adjusted_bic"] = bic + penalty if np.isfinite(bic) else np.inf
    result["physics_adjusted_aicc"] = aicc + penalty if np.isfinite(aicc) else np.inf
    return result


def annotate_model_confidence(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Attach delta-AICc/Akaike weights and return an ambiguity summary."""
    eligible = [
        item
        for item in results
        if item.get("status") == "OK"
        and not item.get("hard_reject", False)
        and np.isfinite(float(item.get("physics_adjusted_aicc", np.inf)))
    ]
    if not eligible:
        return {
            "selection_status": "no-valid-model",
            "top_weight": 0.0,
            "delta_to_second": np.inf,
        }

    eligible.sort(key=lambda r: float(r.get("physics_adjusted_aicc", np.inf)))
    best = float(eligible[0]["physics_adjusted_aicc"])
    deltas = np.asarray([
        max(0.0, float(r["physics_adjusted_aicc"]) - best) for r in eligible
    ])
    raw = np.exp(-0.5 * np.clip(deltas, 0.0, 1400.0))
    weights = raw / max(float(np.sum(raw)), np.finfo(float).tiny)
    for result, delta, weight in zip(eligible, deltas, weights):
        result["delta_aicc_physics"] = float(delta)
        result["akaike_weight"] = float(weight)

    top_weight = float(weights[0])
    delta_second = float(deltas[1]) if deltas.size > 1 else np.inf
    if delta_second < 2.0 or top_weight < 0.70:
        status = "ambiguous"
    elif delta_second < 6.0:
        status = "moderate-confidence"
    else:
        status = "preferred"
    return {
        "selection_status": status,
        "top_weight": top_weight,
        "delta_to_second": delta_second,
        "eligible_models": int(len(eligible)),
    }


def run_lin_kk_diagnostic(
    frequency_hz: Sequence[float],
    impedance: Sequence[complex],
) -> dict[str, Any]:
    """Run impedance.py Lin-KK when available; otherwise return unavailable."""
    try:
        from impedance.validation import linKK
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        return {"available": False, "status": "unavailable", "message": str(exc)}

    try:  # pragma: no cover - exercised in the packaged application runtime
        f = np.asarray(frequency_hz, dtype=float)
        z = np.asarray(impedance, dtype=complex)
        max_m = max(3, min(50, int(f.size // 2)))
        m, mu, zfit, residual_real, residual_imag = linKK(
            f,
            z,
            c=0.85,
            max_M=max_m,
            fit_type="complex",
            add_cap=False,
        )
        residual_rms = float(
            np.sqrt(np.mean(np.asarray(residual_real) ** 2 + np.asarray(residual_imag) ** 2))
        )
        status = "pass" if residual_rms <= 0.05 else "review"
        return {
            "available": True,
            "status": status,
            "M": int(m),
            "mu": float(mu),
            "residual_rms": residual_rms,
        }
    except Exception as exc:
        return {"available": True, "status": "failed", "message": str(exc)}
