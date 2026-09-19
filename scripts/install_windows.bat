@echo off
setlocal

echo.
echo ============================================
echo   GameBus Data Analyzer - Installation
echo ============================================
echo.

REM Move to repository root.
cd /d "%~dp0\.."

echo [1/6] Checking Python 3.14+...
py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 14) else 1)" >nul 2>nul

if errorlevel 1 (
    echo.
    echo ERROR: Python 3.14 or newer is not installed or not available.
    echo.
    echo Please install Python 3.14 or newer and make sure
    echo it is available through the Windows Python launcher.
    echo.
    pause
    exit /b 1
)

echo [2/6] Removing old virtual environment...

if exist ".venv" (
    rmdir /s /q ".venv"

    if errorlevel 1 (
        echo.
        echo ERROR: Failed to remove the existing .venv folder.
        echo.
        echo Close terminals, editors, or running applications
        echo that may be using the virtual environment and try again.
        echo.
        pause
        exit /b 1
    )
)

echo [3/6] Creating virtual environment...
py -3 -m venv .venv

if errorlevel 1 (
    echo.
    echo ERROR: Failed to create the virtual environment.
    echo.
    pause
    exit /b 1
)

echo [4/6] Preparing pip...

".venv\Scripts\python.exe" -m ensurepip --upgrade

if errorlevel 1 (
    echo.
    echo ERROR: Failed to bootstrap pip.
    echo.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel

if errorlevel 1 (
    echo.
    echo ERROR: Failed to upgrade pip.
    echo.
    pause
    exit /b 1
)

echo [5/6] Installing GameBus Data Analyzer dependencies...

".venv\Scripts\python.exe" -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install project dependencies.
    echo.
    echo Please check:
    echo - your internet connection
    echo - whether Python package downloads are allowed
    echo - whether requirements.txt is present in the project root
    echo.
    pause
    exit /b 1
)

echo [6/6] Checking installed dependencies...

".venv\Scripts\python.exe" -m pip check

if errorlevel 1 (
    echo.
    echo ERROR: The environment contains incompatible dependencies.
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================
echo Installation completed successfully.
echo.
echo To start GameBus Data Analyzer:
echo   Double-click scripts\run_app.bat
echo ============================================
echo.

pause
endlocal