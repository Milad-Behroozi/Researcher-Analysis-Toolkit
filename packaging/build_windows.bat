@echo off
setlocal EnableExtensions
chcp 65001 >nul

REM Normalize both paths without a trailing backslash. A quoted Windows
REM argument ending in a backslash can be misparsed when the project path
REM contains spaces.
for %%I in ("%~dp0.") do set "PACKAGING_DIR=%%~fI"
for %%I in ("%PACKAGING_DIR%\..") do set "PROJECT_DIR=%%~fI"

set "LAUNCHER=%PROJECT_DIR%\launcher.py"
set "ICON=%PROJECT_DIR%\Researcherlogo.ico"
set "BUILD_ENV=%PROJECT_DIR%\.packaging-venv"
set "BUILD_PY=%BUILD_ENV%\Scripts\python.exe"
set "DIST_DIR=%PROJECT_DIR%\dist"
set "APP_DIST=%DIST_DIR%\ResearcherAnalysisToolkit"
set "WORK_DIR=%PROJECT_DIR%\build"
set "PYTHON_INSTALLER=%PROJECT_DIR%\python-3.11.9-amd64.exe"

echo ============================================================
echo Researcher Analysis Toolkit - Windows Release Builder
echo ============================================================
echo Project: %PROJECT_DIR%
echo.

if not exist "%LAUNCHER%" (
    echo ERROR: launcher.py was not found.
    goto :failed
)

if not exist "%ICON%" (
    echo ERROR: Researcherlogo.ico was not found.
    goto :failed
)

set "BASE_PY="
for /f "usebackq delims=" %%I in (`py -3.11 -c "import sys; print(sys.executable)" 2^>nul`) do set "BASE_PY=%%I"

if not defined BASE_PY (
    for /f "usebackq delims=" %%I in (`python -c "import sys; print(sys.executable if sys.version_info[:2] == (3, 11) else '')" 2^>nul`) do set "BASE_PY=%%I"
)

if not defined BASE_PY (
    if not exist "%PYTHON_INSTALLER%" (
        echo Downloading the verified Python 3.11 installer...
        powershell -NoProfile -ExecutionPolicy Bypass -File "%PACKAGING_DIR%\download_python_runtime.ps1"
        if errorlevel 1 goto :failed
    )

    echo Installing Python 3.11 for the current user...
    start /wait "" "%PYTHON_INSTALLER%" /passive InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_launcher=1 Include_test=0
    set "BASE_PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
)

if not exist "%BASE_PY%" (
    echo ERROR: Python 3.11 could not be found or installed.
    goto :failed
)

if not exist "%BUILD_PY%" (
    echo Creating isolated packaging environment...
    "%BASE_PY%" -m venv "%BUILD_ENV%"
    if errorlevel 1 goto :failed
)

echo Installing packaging tools...
"%BUILD_PY%" -m pip install --disable-pip-version-check --upgrade pip setuptools wheel
if errorlevel 1 goto :failed

"%BUILD_PY%" -m pip install --disable-pip-version-check PyInstaller==6.22.2 PyQt6==6.6.1 PyQt6-Qt6==6.6.1 PyQt6-sip==13.6.0
if errorlevel 1 goto :failed

echo Building the graphical launcher...
"%BUILD_PY%" -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --onedir ^
    --windowed ^
    --contents-directory . ^
    --name ResearcherAnalysisToolkit ^
    --icon "%ICON%" ^
    --version-file "%PACKAGING_DIR%\version_info.txt" ^
    --distpath "%DIST_DIR%" ^
    --workpath "%WORK_DIR%" ^
    --specpath "%PACKAGING_DIR%" ^
    "%LAUNCHER%"
if errorlevel 1 goto :failed

if not exist "%APP_DIST%\ResearcherAnalysisToolkit.exe" (
    echo ERROR: PyInstaller did not create the expected executable.
    goto :failed
)

echo Copying analysis modules and release resources...
copy /Y "%PROJECT_DIR%\*.py" "%APP_DIST%\" >nul
copy /Y "%PROJECT_DIR%\requirements.txt" "%APP_DIST%\" >nul
copy /Y "%PROJECT_DIR%\Researcherlogo.ico" "%APP_DIST%\" >nul
copy /Y "%PROJECT_DIR%\Researcherlogo.png" "%APP_DIST%\" >nul
copy /Y "%PROJECT_DIR%\README.md" "%APP_DIST%\" >nul
copy /Y "%PROJECT_DIR%\USER_GUIDE.md" "%APP_DIST%\" >nul
copy /Y "%PYTHON_INSTALLER%" "%APP_DIST%\" >nul

set "ISCC="
if exist "%ProgramFiles%\Inno Setup 7\ISCC.exe" set "ISCC=%ProgramFiles%\Inno Setup 7\ISCC.exe"
if not defined ISCC if exist "%ProgramFiles(x86)%\Inno Setup 7\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 7\ISCC.exe"
if not defined ISCC if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not defined ISCC if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"

if not defined ISCC (
    echo.
    echo Launcher build completed successfully.
    echo Inno Setup 7 or 6 was not found, so Setup.exe was not compiled.
    echo Install Inno Setup and run this build file again.
    echo Prepared application folder:
    echo %APP_DIST%
    goto :success
)

echo Compiling Setup.exe with Inno Setup...
pushd "%PACKAGING_DIR%"
"%ISCC%" "ResearcherAnalysisToolkit.iss"
set "ISCC_RESULT=%errorlevel%"
popd
if not "%ISCC_RESULT%"=="0" goto :failed

set "SETUP_FILE=%PACKAGING_DIR%\output\Researcher_Analysis_Toolkit_v1.0.0_Setup.exe"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p='%SETUP_FILE%'; $h=(Get-FileHash -Algorithm SHA256 -LiteralPath $p).Hash.ToLower(); Set-Content -LiteralPath ($p + '.sha256.txt') -Value ($h + '  ' + [IO.Path]::GetFileName($p)) -Encoding ascii"

echo.
echo Setup created successfully:
echo %SETUP_FILE%
echo SHA-256 checksum:
echo %SETUP_FILE%.sha256.txt

:success
echo.
echo Build completed.
pause
exit /b 0

:failed
echo.
echo BUILD FAILED. Review the messages above.
pause
exit /b 1
