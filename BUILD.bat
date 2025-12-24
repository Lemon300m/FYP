@echo off
REM One-click PyInstaller compiler for Deepfake Detector
REM Final output:  DeepfakeDetector.exe in current folder only

setlocal enabledelayedexpansion

echo. 
echo ==========================================
echo  Deepfake Detector - One-Click Compiler
echo ==========================================
echo. 

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Ensure PyInstaller
python -m pip show pyinstaller >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing PyInstaller...
    python -m pip install pyinstaller || exit /b 1
)

REM Install dependencies
python -m pip install -r requirements.txt || exit /b 1

REM Extract Haar Cascade
python -c "import cv2, shutil; shutil.copy(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml','haarcascade_frontalface_default.xml')" || exit /b 1

REM ===== AUTO CLEAN (NO PROMPTS) =====
echo Cleaning previous build artifacts...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist __pycache__ rmdir /s /q __pycache__
for %%f in (*.spec) do del /f /q "%%f"

REM ===== BUILD =====
echo Building executable...
python -m PyInstaller ^
    --name DeepfakeDetector ^
    --onefile ^
    --windowed ^
    --add-data "haarcascade_frontalface_default.xml;." ^
    --hidden-import cv2 ^
    --hidden-import numpy ^
    --hidden-import sklearn ^
    --hidden-import PIL ^
    --hidden-import pystray ^
    --hidden-import win10toast ^
    --hidden-import screeninfo ^
    --hidden-import mss ^
    --hidden-import tensorflow ^
    --hidden-import keras ^
    realtime_deepfake_detector.py || goto :fail

REM ===== MOVE EXE TO CURRENT FOLDER =====
if exist "dist\DeepfakeDetector.exe" (
    move /y "dist\DeepfakeDetector.exe" ".\" >nul
) else (
    echo ERROR: Executable not found! 
    goto :fail
)

REM ===== FINAL CLEANUP =====
rmdir /s /q dist
rmdir /s /q build
for %%f in (*.spec) do del /f /q "%%f"

REM Ask user if they want to delete source files
echo.
echo ==========================================
echo  CLEANUP OPTIONS
echo ==========================================
set /p DeleteSource="Delete source files (keeping only .exe and models)? (y/n, default=n): "
if /i "%DeleteSource%"=="y" (
    echo Removing source files...
    for %%f in (*.py) do del /f /q "%%f"
    for %%f in (*.json) do del /f /q "%%f"
    for %%f in (*.txt) do del /f /q "%%f"
    for %%f in (*.md) do del /f /q "%%f"
    for %%f in (*.bat) do del /f /q "%%f"
    if exist __pycache__ rmdir /s /q __pycache__
    if exist screenshot rmdir /s /q screenshot
    if exist self_learning_data rmdir /s /q self_learning_data
    echo Source files removed.
)

echo.
echo ==========================================
echo  ✅ BUILD SUCCESSFUL
echo ==========================================
echo. 
echo Output: DeepfakeDetector.exe
echo. 

pause
exit /b 0

:fail
echo.
echo ✖ BUILD FAILED
echo. 
pause
exit /b 1