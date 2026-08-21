# Contributing

Contributions that improve numerical correctness, diagnostics, file-format
support, accessibility, documentation, or reproducible testing are welcome.

Before proposing a change:

1. Create a focused branch and avoid unrelated formatting changes.
2. Preserve original experimental data and document every transformation.
3. Add a minimal input and expected diagnostic for numerical changes.
4. Run `python -m unittest discover -s tests -v`.
5. Describe scientific assumptions, tested platforms, and any known limits.

Never commit patient data, credentials, private filesystem paths, unpublished
third-party datasets, or large generated build directories.

