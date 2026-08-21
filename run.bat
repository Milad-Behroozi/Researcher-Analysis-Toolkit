@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "APP_DIR=%~dp0"
cd /d "%APP_DIR%"

REM Installed release: launch the packaged GUI directly.
if exist "%APP_DIR%ResearcherAnalysisToolkit.exe" (
    start "" "%APP_DIR%ResearcherAnalysisToolkit.exe"
    exit /b 0
)

REM Source release fallback.
set "SCRIPT=%APP_DIR%launcher.py"
if not exist "%SCRIPT%" (
    echo ERROR: launcher.py was not found beside run.bat.
    pause
    exit /b 1
)

set "BASE_PY="
for /f "usebackq delims=" %%I in (`py -3.11 -c "import sys; print(sys.executable)" 2^>nul`) do set "BASE_PY=%%I"

if not defined BASE_PY (
    for /f "usebackq delims=" %%I in (`python -c "import sys; print(sys.executable if sys.version_info[:2] == (3, 11) else '')" 2^>nul`) do set "BASE_PY=%%I"
)

if not defined BASE_PY (
    if not exist "%APP_DIR%python-3.11.9-amd64.exe" if exist "%APP_DIR%packaging\download_python_runtime.ps1" (
        echo Downloading the verified Python 3.11 runtime...
        powershell -NoProfile -ExecutionPolicy Bypass -File "%APP_DIR%packaging\download_python_runtime.ps1"
    )
    if exist "%APP_DIR%python-3.11.9-amd64.exe" (
        echo Installing the included Python 3.11 runtime...
        start /wait "" "%APP_DIR%python-3.11.9-amd64.exe" /passive InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_launcher=1 Include_test=0
        set "BASE_PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    )
)

if not exist "%BASE_PY%" (
    echo ERROR: A compatible Python 3.11 runtime could not be found or installed.
    pause
    exit /b 1
)

set "APP_DATA=%LOCALAPPDATA%\ResearcherToolkit"
set "VENV_DIR=%APP_DATA%\venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "VENV_PYW=%VENV_DIR%\Scripts\pythonw.exe"

if exist "%VENV_PY%" (
    "%VENV_PY%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>nul
    if errorlevel 1 (
        echo Removing an incompatible Researcher Toolkit environment...
        rmdir /S /Q "%VENV_DIR%"
    )
)

if not exist "%VENV_PY%" (
    echo Creating the Researcher Toolkit environment...
    "%BASE_PY%" -m venv "%VENV_DIR%"
    if errorlevel 1 goto :error
)

"%VENV_PY%" -c "import PyQt6" >nul 2>nul
if errorlevel 1 (
    echo Installing launcher components...
    "%VENV_PY%" -m pip install --disable-pip-version-check PyQt6==6.6.1 PyQt6-Qt6==6.6.1 PyQt6-sip==13.6.0
    if errorlevel 1 goto :error
)

start "" "%VENV_PYW%" "%SCRIPT%"
exit /b 0

:error
echo.
echo ERROR: The launcher environment could not be prepared.
echo Check your internet connection and try again.
pause
exit /b 1
