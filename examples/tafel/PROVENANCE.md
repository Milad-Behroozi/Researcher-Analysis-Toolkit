# Provenance — Tafel/LSV example

- Source type: experimental polarization data supplied by the project owner.
- Repository transformation: the instrument header was removed because it
  contained a private local filesystem path; the 1,800 numeric records were
  retained unchanged and written as UTF-8 CSV.
- Renaming: the original sample-specific filename was generalized to an
  oxygen-deficient tungsten oxide example.
- Columns: `Potential_V` and `Current_A`.
- Reference output: `../reference_outputs/Tafel_reference_output.png`.

The analysis requires the user to define electrode area, reference conversion,
pH, reaction mode, current normalization, and a defensible fitting window.

