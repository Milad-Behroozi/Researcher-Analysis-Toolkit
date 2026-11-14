# Researcher Analysis Toolkit

**Researcher Analysis Toolkit for Integrated and Traceable Evaluation of 
Electrochemical, Photoelectrochemical, and Spectroscopic Data**

Researcher Analysis Toolkit combines independent analysis modules in one
launcher so that experimental data can be processed, fitted, visualized, and
exported without writing code. The project is intended to improve consistency
and reproducibility while keeping the analyst responsible for selecting valid
models, fitting regions, reference scales, and physical assumptions.

## Included modules

- Electrochemical: CV, LSV, Tafel, Mott–Schottky, potential conversion, and EIS
- EIS: user-defined circuit fitting, tiered automatic circuit selection, and
  circuit/plot generation
- Photoelectrochemical: ABPE, IPCE, faradaic efficiency, and STH efficiency
- Optical/spectroscopic: Tauc plot, TRPL, and XPS
- Reporting: publication-oriented plots, tables, and export utilities

The Level 5 EIS option interfaces with AutoEIS for complex model exploration.
It is one optional part of the broader toolkit and is not the core of the
software.

## Quick start on Windows

1. Download or clone this repository.
2. Double-click `run.bat`.
3. In the launcher, click **Initialize Environment** once.
4. Select a module, load one of the files in `examples/`, and click **Run**.

Python 3.11 is the supported runtime. If it is missing, `run.bat` downloads the
official Python 3.11.9 x64 installer from python.org and verifies its SHA-256
checksum before installation. The first dependency installation can take time,
especially for AutoEIS/JAX/Julia-related packages.

### Start from a terminal

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python launcher.py
```

## Reproducibility examples

The `examples/` directory contains four representative experimental datasets,
a deterministic synthetic EIS benchmark at two noise levels, provenance notes,
reference output images, and a machine-readable validation manifest. Start with
[`examples/README.md`](examples/README.md).

The reported example values are smoke-test anchors under the documented fitting
choices. Results that depend on a user-selected window, baseline, reference
electrode, pH, or model must not be interpreted as universal expected values.

Run the repository checks with:

```powershell
py -3.11 -m unittest discover -s tests -v
```

## Build the Windows installer

Run:

```text
packaging\build_windows.bat
```

The script creates an isolated packaging environment, builds the launcher with
PyInstaller, and compiles `Setup.exe` when Inno Setup 6 or 7 is installed. See
`packaging/BUILD_SETUP_FA.md` and `packaging/RELEASE_CHECKLIST.md`.

## Scientific-use notice

- Inspect the raw data and diagnostic outputs before accepting a fit.
- Automatic region or circuit suggestions are candidates, not proof of a
  mechanism.
- Report preprocessing, normalization, reference conversion, fit window,
  constraints, and goodness-of-fit measures with published results.
- Independently verify important results with controls and domain knowledge.

This repository is a **v1.0.0 release candidate** until the clean-install test
matrix in `packaging/RELEASE_CHECKLIST.md` is completed on another Windows
computer. The precise completed/pending checks are listed in
[`VALIDATION_STATUS.md`](VALIDATION_STATUS.md).

```markdown
## Citation and license

If you use the Researcher Analysis Toolkit in your research, please cite the associated article.

M. Behroozi, S. Sadighi, R. Luque and E. Doustkhah, “Researcher Analysis Toolkit for integrated and traceable evaluation of electrochemical, photoelectrochemical, and spectroscopic data,” *RSC Advances*, 2026. [DOI 10.1039/D6RA08618G](https://doi.org/10.1039/D6RA08618G)

Citation metadata are available in [`CITATION.cff`](CITATION.cff).

Software license selection is pending.
```


