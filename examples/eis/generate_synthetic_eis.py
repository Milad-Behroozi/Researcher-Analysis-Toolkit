"""Generate the documented two-relaxation EIS validation datasets.

The generator uses only the Python standard library so that the numerical
benchmark can be recreated before the toolkit dependencies are installed.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent
RANDOM_SEED = 20260723
POINT_COUNT = 81

PARAMETERS = {
    "R0_ohm": 10.0,
    "R1_ohm": 10.0,
    "Q1_S_s_n": 1.0e-4,
    "n1": 0.85,
    "R2_ohm": 15.0,
    "Q2_S_s_n": 1.0e-3,
    "n2": 0.50,
}


def parallel_r_cpe(resistance: float, q: float, exponent: float, omega: float) -> complex:
    """Return the impedance of a resistance in parallel with a CPE."""
    admittance = (1.0 / resistance) + q * (1j * omega) ** exponent
    return 1.0 / admittance


def model_impedance(frequency_hz: float) -> complex:
    """R0-p(R1,CPE1)-p(R2,CPE2), with the two branches in series."""
    omega = 2.0 * math.pi * frequency_hz
    z1 = parallel_r_cpe(PARAMETERS["R1_ohm"], PARAMETERS["Q1_S_s_n"], PARAMETERS["n1"], omega)
    z2 = parallel_r_cpe(PARAMETERS["R2_ohm"], PARAMETERS["Q2_S_s_n"], PARAMETERS["n2"], omega)
    return PARAMETERS["R0_ohm"] + z1 + z2


def logarithmic_frequencies() -> list[float]:
    return [10.0 ** (6.0 - 8.0 * index / (POINT_COUNT - 1)) for index in range(POINT_COUNT)]


def generate(noise_fraction: float, output_name: str) -> None:
    rng = random.Random(RANDOM_SEED)
    rows = []
    for frequency in logarithmic_frequencies():
        impedance = model_impedance(frequency)
        if noise_fraction:
            sigma = noise_fraction * abs(impedance)
            impedance = complex(
                impedance.real + rng.gauss(0.0, sigma),
                impedance.imag + rng.gauss(0.0, sigma),
            )
        rows.append((frequency, impedance.real, impedance.imag))

    with (OUTPUT_DIR / output_name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frequency_Hz", "Z_real_ohm", "Z_imag_ohm"])
        writer.writerows((f"{f:.12g}", f"{zr:.12g}", f"{zi:.12g}") for f, zr, zi in rows)


def write_metadata() -> None:
    metadata = {
        "description": "Synthetic EIS benchmark for automatic circuit-selection testing.",
        "generating_circuit": "R0-p(R1,CPE1)-p(R2,CPE2)",
        "parameters": PARAMETERS,
        "frequency_range_Hz": [1.0e-2, 1.0e6],
        "frequency_distribution": "logarithmic, descending",
        "point_count": POINT_COUNT,
        "noise_definition": "Independent Gaussian noise added to the real and imaginary components with sigma = noise_fraction * |Z|.",
        "random_seed": RANDOM_SEED,
        "datasets": {
            "EIS_two_relaxation_noise0.csv": {"noise_fraction": 0.0},
            "EIS_two_relaxation_noise3.csv": {"noise_fraction": 0.03},
        },
    }
    (OUTPUT_DIR / "benchmark_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    generate(0.0, "EIS_two_relaxation_noise0.csv")
    generate(0.03, "EIS_two_relaxation_noise3.csv")
    write_metadata()


if __name__ == "__main__":
    main()
