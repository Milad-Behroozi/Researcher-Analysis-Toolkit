# Packaging-only changes for version 1.0.0

The analytical calculations and fitting routines were not modified while the
Windows release files were prepared.

Changes made for distribution:

- Renamed `lancher.py` to `launcher.py`.
- Added installed/frozen application-path detection.
- Changed the Auto EIS launcher target from `bootstrap.py` to `eis_auto.py` so
  it uses the same application environment as the other modules.
- Replaced the legacy AutoEIS bootstrap with a compatibility wrapper that no
  longer creates a second virtual environment.
- Added supported Python 3.11 discovery and per-user runtime installation.
- Added verified download of the official Python runtime when the binary is not
  present in a source checkout.
- Added automatic replacement of virtual environments created by an
  incompatible Python version, preventing python311.dll conflicts.
- Replaced temporary article/guide links with an inactive Article button and a
  local user guide.
- Replaced the placeholder Mott–Schottky report website with the developer's
  existing GitHub profile.
- Removed the development launcher log from the release copy.
- Added PyInstaller and Inno Setup build files, documentation, and checksum
  generation.
