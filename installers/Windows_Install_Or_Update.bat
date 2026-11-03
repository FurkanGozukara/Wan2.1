@echo off

echo WARNING. For this auto installer to work you need to have installed Python 3.12.10, Git, FFmpeg, cuDNN, CUDA, Visual C++ Tools
echo This tutorial shows how to install requirements (1-time only) with 1-clik installer scripts : https://youtu.be/ptuLL0C72Ms

set UV_SKIP_WHEEL_FILENAME_CHECK=1
set UV_LINK_MODE=copy
REM Compile the libraries to bytecode during install; otherwise the first app start does it and looks frozen
set UV_COMPILE_BYTECODE=1

set "INSTALLER_DIR=%~dp0"
set "VENV_PYTHON=%INSTALLER_DIR%Wan2.1\venv\Scripts\python.exe"
cd /d "%INSTALLER_DIR%" || goto install_failed

git clone https://github.com/FurkanGozukara/Wan2.1 Wan2.1

cd Wan2.1 || goto install_failed

git remote set-url origin https://github.com/FurkanGozukara/Wan2.1

git reset --hard

git pull || goto install_failed

REM A venv made with another Python version (3.10 in older releases) cannot be reused
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" -c "import sys; raise SystemExit(not sys.version_info[:2] == (3, 12))" >nul 2>&1
    if errorlevel 1 (
        echo Existing venv is not Python 3.12. Removing it so it can be recreated...
        rmdir /s /q venv
        if exist venv (
            echo Could not remove the old venv. Close the app and any window using it, then run this installer again.
            goto install_failed
        )
    )
)

if not exist "venv\Scripts\python.exe" (
    py --version >nul 2>&1
    if not errorlevel 1 (
        echo Python launcher is available. Generating Python 3.12 VENV
        py -3.12 -m venv venv
    ) else (
        echo Python launcher is not available, generating VENV with default Python. Make sure that it is 3.12
        python -m venv venv
    )
)

call .\venv\Scripts\activate.bat || goto install_failed

"%VENV_PYTHON%" -c "import sys; raise SystemExit(not sys.version_info[:2] == (3, 12))" || (
    echo The virtual environment is not Python 3.12. Install Python 3.12 and run this installer again.
    goto install_failed
)

echo installing requirements

"%VENV_PYTHON%" -m pip install --upgrade pip || goto install_failed

"%VENV_PYTHON%" -m pip install uv || goto install_failed

cd /d "%INSTALLER_DIR%" || goto install_failed

REM Pin the target even when inherited UV_PYTHON or UV_SYSTEM_PYTHON selects system Python
"%VENV_PYTHON%" -m uv pip install --python "%VENV_PYTHON%" -r requirements.txt --index-strategy unsafe-best-match || goto install_failed

cd Wan2.1 || goto install_failed

git clone https://github.com/SECourses/DiffSynth-Studio

cd DiffSynth-Studio || goto install_failed

git remote set-url origin https://github.com/SECourses/DiffSynth-Studio

git reset --hard

git pull || goto install_failed

REM Its dependencies are in the unified requirements file; its own pins (old transformers) must not be installed
"%VENV_PYTHON%" -m uv pip install --python "%VENV_PYTHON%" -e . --no-deps --build-constraints "%INSTALLER_DIR%uv_build_constraints.txt" || goto install_failed

cd ..

git clone https://github.com/FurkanGozukara/Practical-RIFE Practical-RIFE

cd Practical-RIFE || goto install_failed

git remote set-url origin https://github.com/FurkanGozukara/Practical-RIFE

git reset --hard

git pull || goto install_failed

cd /d "%INSTALLER_DIR%" || goto install_failed

"%VENV_PYTHON%" Download_RIFE.py || goto install_failed

echo.
echo Verifying the installed libraries...
"%VENV_PYTHON%" "Wan2.1/verify_install.py" || goto install_failed

REM Show completion message
echo.
echo Wan 2.1 installed. Now run Windows_Download_Models.bat to download the models you want, then Start_APP.bat
echo Virtual environment made and installed properly - save logs to send me if any errors occurs

REM Pause to keep the command prompt open
pause
exit /b 0

:install_failed
echo Installation failed. Please save the full console log and send it for debugging.
pause
exit /b 1
