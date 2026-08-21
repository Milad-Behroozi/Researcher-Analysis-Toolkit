# Reproducibility examples

These examples test different stages of the toolkit. They are not intended to
establish the electrochemical or optical performance of a material.

| Folder | Input | Suggested module | Reference anchor* |
|---|---|---|---|
| `tafel/` | Polarization data | LSV / Tafel | apparent slope ≈ 164.98 mV dec⁻¹ |
| `mott_schottky/` | Potential and impedance columns | Mott–Schottky | flat-band intercept ≈ 0.169 V for the illustrated n-type fit |
| `trpl/` | Time and PL intensity | TRPL | average lifetime ≈ 0.50645 ns for the illustrated biexponential fit |
| `tauc/` | Wavelength and diffuse reflectance | Tauc | direct-transition estimate ≈ 3.273 eV |
| `eis/` | Frequency and complex impedance | EIS | known two-relaxation generating circuit and parameters |

\* Anchors reproduce the displayed examples only when the same model,
preprocessing, reference scale, and fitting regions are selected. See
`validation_manifest.json` for the assumptions and tolerances.

## Recommended smoke-test sequence

1. Run the noise-free EIS file and confirm that the documented generating
   circuit is fitted without bound violations.
2. Repeat with the 3% noise file and compare the physically valid candidates,
   diagnostics, and recovered parameters—not only a single error metric.
3. Load each experimental example in its module and reproduce the associated
   image in `reference_outputs/` using the documented manual choices.
4. Record software version, fit range, model, transformations, and diagnostics
   with every exported result.

The EIS files can be regenerated deterministically with:

```text
python examples/eis/generate_synthetic_eis.py
```

