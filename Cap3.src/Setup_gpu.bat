@echo off
echo ================================================================
echo  DP Fault AI — GPU Environment Setup (GTX 1080 Ti / CUDA 11.8)
echo ================================================================

REM ── Step 1: create venv if not exists ───────────────────────────
if not exist venv (
    echo Creating virtual environment...
    py -3 -m venv venv
) else (
    echo Virtual environment already exists — skipping creation
)

REM ── Step 2: activate ────────────────────────────────────────────
call venv\Scripts\activate.bat
echo Venv activated: %VIRTUAL_ENV%

REM ── Step 3: upgrade pip ─────────────────────────────────────────
echo Upgrading pip...
python -m pip install --upgrade pip

REM ── Step 4: install torch with CUDA 11.8 ────────────────────────
echo.
echo Installing PyTorch with CUDA 11.8 support...
echo (this is a large download ~2GB, please wait)
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu118

REM ── Step 5: install all other packages ──────────────────────────
echo.
echo Installing remaining packages...
pip install -r requirements_gpu.txt

REM ── Step 6: verify CUDA is visible ──────────────────────────────
echo.
echo Verifying CUDA detection...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND')"

echo.
echo ================================================================
echo  Setup complete.
echo  If CUDA available = False, install CUDA 11.8 from:
echo  https://developer.nvidia.com/cuda-11-8-0-download-archive
echo ================================================================
pause
