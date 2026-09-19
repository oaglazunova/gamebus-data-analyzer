@echo off
setlocal

REM Move to repository root.
cd /d "%~dp0\.."

echo.
echo ============================================
echo   GameBus Data Analyzer
echo ============================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: The virtual environment was not found.
    echo.
    echo Run:
    echo   scripts\install_windows.bat
    echo.
    echo first.
    echo.
    pause
    exit /b 1
)

if not exist "streamlit_app.py" (
    echo ERROR: streamlit_app.py was not found.
    echo.
    echo Make sure this script is inside the scripts folder
    echo of the GameBus Data Analyzer repository.
    echo.
    pause
    exit /b 1
)

echo Starting GameBus Data Analyzer...
echo.
echo Close this window or press Ctrl+C to stop the app.
echo.

".venv\Scripts\python.exe" -m streamlit run streamlit_app.py

if errorlevel 1 (
    echo.
    echo ERROR: GameBus Data Analyzer stopped with an error.
    echo.
    pause
    exit /b 1
)

endlocal