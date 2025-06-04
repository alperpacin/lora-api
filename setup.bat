@echo off
REM setup.bat - Script to set up LoRA training environment for Windows

REM Set base directory to script location
cd /d "%~dp0"

REM Create Python virtual environment
echo Creating Python virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing dependencies...
python -m pip install -r requirements.txt

REM Create necessary directories
echo Creating necessary directories...
mkdir scripts\stable 2>nul
mkdir huggingface 2>nul
mkdir "%USERPROFILE%\lora_data\train_images" 2>nul
mkdir "%USERPROFILE%\lora_data\lora_jobs" 2>nul

REM Copy training script to scripts directory
echo Setting up training scripts...
copy /Y train_network.py scripts\stable\

REM Configure accelerate
echo Configuring accelerate...
accelerate config default

echo ========================================================================
echo Setup completed successfully!
echo.
echo To run the API server:
echo   1. Activate the virtual environment: venv\Scripts\activate.bat
echo   2. Start the server: uvicorn api_server:app --host 0.0.0.0 --port 8000
echo ========================================================================

REM Keep the window open
pause 