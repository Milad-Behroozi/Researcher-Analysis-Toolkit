# Building the Windows Installer

1. Keep your internet connection active during the build. If Python 3.11 is unavailable, the build script downloads the official Python 3.11.9 installer from `python.org` and verifies its SHA-256 checksum.
2. Install **Inno Setup** from its official website. The build script supports Inno Setup 6 and 7.
3. Open the `packaging` folder.
4. Double-click `build_windows.bat`.
5. Keep the build window open until the build finishes.

The installer is saved to:

```text
packaging\output\Researcher_Analysis_Toolkit_v1.0.0_Setup.exe
```

Test the installer on a clean Windows system. When you launch the toolkit for the first time, click **Initialize Environment** and wait for it to finish. Then test at least one sample file in each main module.

Windows SmartScreen may display a warning the first time you run the unsigned installer. For the official release, publish the SHA-256 checksum alongside the installer so users can verify the integrity of their download.
