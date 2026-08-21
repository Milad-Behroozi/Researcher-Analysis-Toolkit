# Researcher Analysis Toolkit — Quick User Guide

## First launch

1. Open **Researcher Analysis Toolkit**.
2. Confirm that the Python status is **Installed**.
3. Select **Initialize Environment**.
4. Keep the program open while the required scientific libraries are installed.
5. Wait for the message **Environment initialized successfully**.

The initialization is required only on the first run or after the package list
has been updated.  Use **Repair** only if a dependency installation was
interrupted or the environment no longer starts correctly.

## Running an analysis

1. Choose a category tab.
2. Select the required module.  EIS contains manual, automatic, and plotting
   submodules.
3. Select **Run** or double-click the module name.
4. Load the experimental data within the selected module.
5. Review the selected fitting region, diagnostics, units, and reference-scale
   settings before exporting results.

## Automatic EIS

Levels 1–4 use the toolkit's organized circuit library.  Level 5 interfaces
with AutoEIS for more complex candidate-model generation and statistical
evaluation.  Its first use can take longer because the Julia/AutoEIS backend
may need one-time initialization.  Automatic suggestions should always be
reviewed using physical plausibility, residuals, parameter bounds, and
identifiability diagnostics.

## Input and output safety

- Work from a copy of irreplaceable raw data.
- Do not delete or overwrite the original measurement file.
- Verify column assignments and units before starting an analysis.
- Treat automatically selected fitting windows as suggestions that require
  scientific review.
- Save exported results in a dedicated project folder.

## Troubleshooting

- **Python not found:** allow the included Python installer to finish, close
  the launcher, and open it again.
- **Dependency installation failed:** check the internet connection, then use
  **Repair**.
- **A module is missing:** reinstall the software instead of copying a single
  module manually.
- **AutoEIS first run is slow:** allow the initial backend setup to complete.
- **Windows SmartScreen warning:** unsigned research software can trigger this
  warning.  Distribute the installer together with a published checksum and,
  when possible, use a trusted code-signing certificate.

