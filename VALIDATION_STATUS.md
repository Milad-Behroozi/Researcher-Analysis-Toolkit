# Validation status — v1.0.0 release candidate

## Completed in this source package

- All root Python files parse successfully.
- Every module target declared by the launcher exists.
- Experimental example files pass structural and finite-value checks.
- The synthetic EIS generator is deterministic and its datasets match the documented row count, frequency order, topology, parameters, noise level, and seed.
- The validation manifest resolves to existing files.
- Python 3.11 is enforced for the application environment; incompatible environments are replaced automatically.
- The Python runtime downloader verifies the official installer by SHA-256.
- `Setup.exe` was built on Windows, and its SHA-256 checksum was published with the release.
- Clean installation and module-launch testing were completed successfully on a second Windows x64 computer.

The repository checks run through `tests/test_repository_structure.py` and GitHub Actions. The Windows installation checks were performed manually on two computers.

## Still required before calling v1.0.0 fully validated

- Build `Setup.exe` on Windows and verify its published checksum.
- Install on a second clean Windows 10/11 x64 computer.
- Run at least one example through every main module group.
- Compare exported numerical values and diagnostics with the documented manual
  anchors.
- Confirm Level 5 AutoEIS initialization on a clean computer.
- Confirm write/export behavior in a normal user-selected folder.

Until these items are complete, describe the package as a **release candidate**
rather than a universally validated installer.

