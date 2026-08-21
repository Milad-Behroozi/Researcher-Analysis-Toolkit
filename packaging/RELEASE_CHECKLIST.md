# Windows release checklist

Complete this checklist before uploading version 1.0.0 to a public repository.

## Build

- [ ] Run `packaging/build_windows.bat` on Windows x64.
- [ ] Confirm that the build ends without a PyInstaller or Inno Setup error.
- [ ] Confirm that the Setup EXE and SHA-256 text file are present in
      `packaging/output`.

## Clean-machine installation

- [ ] Test on Windows 10 or 11 x64 without an existing toolkit installation.
- [ ] Install for the current user.
- [ ] Confirm that the Start-menu shortcut opens the graphical launcher.
- [ ] Confirm that the included Python runtime is detected.
- [ ] Run **Initialize Environment** and confirm successful completion.
- [ ] Close and reopen the toolkit.

## Scientific smoke tests

- [ ] Load one representative file in CV, LSV, and Tafel.
- [ ] Run manual EIS and one Levels 1–4 automatic EIS example.
- [ ] Initialize and run one Level 5 AutoEIS example.
- [ ] Load one representative Mott–Schottky, Tauc, TRPL, and XPS dataset.
- [ ] Check one PEC-efficiency module such as IPCE or ABPE.
- [ ] Confirm that plots, spreadsheets, and reports export to a user-selected
      writable folder.
- [ ] Compare key numerical outputs with the validated reference values used in
      the article or Supporting Information.
- [ ] Run the noise-free and 3% synthetic EIS benchmarks in `examples/eis` and
      record the ranked candidate circuits and diagnostic outputs.

## Public release metadata

- [ ] Replace the inactive article link with the DOI after publication.
- [ ] Add the final citation.
- [ ] Select and add a software license approved by all authors/institutions.
- [x] Add representative input data and documented smoke-test anchors.
- [ ] Add authors, affiliations, acknowledgments, and contact information.
- [ ] Publish the source archive, Setup EXE, and SHA-256 checksum under the same
      `v1.0.0` release tag.
