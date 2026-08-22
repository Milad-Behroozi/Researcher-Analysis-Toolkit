# Reproducibility examples

These examples exercise different stages of the Researcher Analysis Toolkit. They are provided to demonstrate software operation and numerical consistency and are not intended to establish the electrochemical, optical, or photophysical performance of a material.

| Folder           | Input                              | Suggested module | Reference anchor*                                                             |
| ---------------- | ---------------------------------- | ---------------- | ----------------------------------------------------------------------------- |
| `tafel/`         | Polarization data                  | LSV / Tafel      | apparent Tafel slope ≈ 164.98 mV dec⁻¹                                        |
| `mott_schottky/` | Potential and impedance columns    | Mott–Schottky    | flat-band potential ≈ −0.169 V vs RHE for the illustrated n-type fit          |
| `trpl/`          | Time and PL intensity              | TRPL             | average lifetime ≈ 0.50635 ns for the illustrated biexponential fit           |
| `tauc/`          | Wavelength and diffuse reflectance | Tauc             | direct-transition estimate ≈ 3.273 eV                                         |
| `eis/`           | Frequency and complex impedance    | EIS              | three predefined generating circuits evaluated at 0, 1, 3, and 5% added noise |

* The reported anchors reproduce the displayed examples only when the same model, preprocessing procedure, reference scale, fitting interval, and analysis settings are selected. See `validation_manifest.json` for the corresponding assumptions and numerical tolerances.

## Recommended smoke-test sequence

1. Run `EIS_single_relaxation_noise0.csv` and confirm recovery of the documented generating circuit without parameter-bound violations.
2. Run `EIS_two_relaxation_noise3.csv` and `EIS_diffusion_noise5.csv` to reproduce the automatic topology-ranking examples reported in Table S4 of the Supporting Information.
3. Evaluate the complete 0, 1, 3, and 5% noise series using the user-defined fitting workflow and compare fit quality, bound hits, parameter uncertainties, correlations, and physical plausibility.
4. Load each experimental example in its corresponding module and reproduce the associated image in `reference_outputs/` using the documented analysis choices.
5. Record the software version, fitting interval, selected model, transformations, reference scale, and diagnostic results with every exported analysis.

The synthetic EIS datasets can be regenerated deterministically using:

```text
python examples/eis/generate_synthetic_eis.py
```
