@echo off
REM One-click PyInstaller compiler for Deepfake Detector
REM Just double-click this file to build the executable

setlocal enabledelayedexpansion

echo.
echo ==========================================
echo  Deepfake Detector - One-Click Compiler
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

echo ✓ Python found
echo.

REM Check if PyInstaller is installed
python -m pip show pyinstaller >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing PyInstaller...
    python -m pip install pyinstaller
    if %errorlevel% neq 0 (
        echo Error: Failed to install PyInstaller
        pause
        exit /b 1
    )
)

echo ✓ PyInstaller ready
echo.

REM Install/update dependencies from requirements.txt
echo Installing dependencies from requirements.txt...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Warning: Some dependencies may have failed to install
)
echo ✓ Dependencies ready
echo.

REM Extract Haar Cascade file from OpenCV
echo Extracting Haar Cascade classifier...
python -c "import cv2, shutil, os; src = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'; shutil.copy(src, 'haarcascade_frontalface_default.xml'); print(f'Copied to: {os.path.abspath(\"haarcascade_frontalface_default.xml\")}')"
if %errorlevel% neq 0 (
    echo Error: Failed to extract Haar Cascade file
    pause
    exit /b 1
)
echo ✓ Haar Cascade extracted
echo.

REM Ask if user wants to clean old builds
set /p CLEAN="Clean old builds? (y/n, default: n): "
if /i "%CLEAN%"=="y" (
    echo.
    echo Cleaning old builds...
    if exist build rmdir /s /q build
    if exist dist rmdir /s /q dist
    if exist __pycache__ rmdir /s /q __pycache__
    echo ✓ Cleaned
    echo.
)

REM Build the executable
echo Building executable...
echo.

python -m PyInstaller ^
    --name=DeepfakeDetector ^
    --onefile ^
    --windowed ^
    --add-data="haarcascade_frontalface_default.xml;." ^
    --hidden-import=cv2 ^
    --hidden-import=numpy ^
    --hidden-import=sklearn ^
    --hidden-import=PIL ^
    --hidden-import=pystray ^
    --hidden-import=win10toast ^
    --hidden-import=screeninfo ^
    --hidden-import=mss ^
    --distpath=dist ^
    --workpath=build ^
    --specpath=. ^
    realtime_deepfake_detector.py

if %errorlevel% neq 0 (
    echo.
    echo ✖ Build failed!
    pause
    exit /b 1
)

echo.
echo ==========================================
echo  Setting up distribution folder...
echo ==========================================
echo.

REM Create required directories in dist
cd dist
if not exist "self_learning_data\real" mkdir "self_learning_data\real"
if not exist "self_learning_data\fake" mkdir "self_learning_data\fake"
if not exist "model_archive" mkdir "model_archive"
echo ✓ Created required directories

REM Copy trained model if it exists
if exist "..\deepfake_model.pkl" (
    copy "..\deepfake_model.pkl" . >nul 2>&1
    echo ✓ Copied trained model
) else (
    echo ⚠  No trained model found - you'll need to train one first
)

REM Copy config files if they exist
if exist "..\config.json" (
    copy "..\config.json" . >nul 2>&1
    echo ✓ Copied config.json
)

if exist "..\default.json" (
    copy "..\default.json" . >nul 2>&1
    echo ✓ Copied default.json
)

cd ..

echo.
echo ==========================================
echo  ✅ Build Complete!
echo ==========================================
echo.
echo 📁 Executable location: dist\DeepfakeDetector.exe
echo.
echo 📦 Distribution folder contains:
echo    • DeepfakeDetector.exe
echo    • self_learning_data\ (folder)
echo    • model_archive\ (folder)
if exist "dist\deepfake_model.pkl" (
    echo    • deepfake_model.pkl (trained model) ✓
) else (
    echo    • deepfake_model.pkl (MISSING - train a model first!)
)
echo.
echo 💡 To distribute: Copy the entire 'dist' folder
echo    All files and folders must stay together!
echo.

set /p OPEN="Open dist folder? (y/n): "
if /i "%OPEN%"=="y" (
    start dist
)

echo.
pause