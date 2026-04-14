@echo off
echo ================================================================
echo  DP Fault AI — GPU Environment Setup (GTX 1080 Ti / CUDA 11.8)
echo ================================================================

REM ── Шаг 1: Явное указание пути к Python 3.13 ─────────────────────
set "PYTHON_EXE=C:\Python\Python313\python.exe"

echo Detecting Python version...
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python 3.13 not found at %PYTHON_EXE%
    pause
    exit /b
)

"%PYTHON_EXE%" --version

REM ── Шаг 2: Создание venv (используем конкретный путь) ─────────────
if not exist venv (
    echo Creating virtual environment using Python 3.13...
    "%PYTHON_EXE%" -m venv venv
) else (
    echo Virtual environment already exists — skipping creation
)

REM ── Шаг 3: Активация ────────────────────────────────────────────
call venv\Scripts\activate.bat
echo Venv activated: %VIRTUAL_ENV%

REM ── Шаг 4: Обновление pip ────────────────────────────────────────
echo Upgrading pip...
python -m pip install --upgrade pip

REM ── Шаг 5: Установка Torch (CUDA 11.8) ──────────────────────────
echo.
echo Installing PyTorch with CUDA 11.8 support...
pip install torch --index-url https://download.pytorch.org/whl/cu118

REM ── Шаг 6: Остальные пакеты ──────────────────────────────────────
echo.
echo Installing remaining packages...
pip install -r requirements_gpu.txt

REM ── Шаг 7: Проверка ──────────────────────────────────────────────
echo.
echo Verifying CUDA and torch installation...
python -c "import torch; print('Torch version :', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU           :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND')"

echo.
echo ================================================================
echo  Setup complete.
echo ================================================================
pause
