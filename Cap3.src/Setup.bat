@echo off
echo Setting up Cap3_PY environment...

REM Use py launcher to ensure Python 3
if not exist venv (
    echo Creating virtual environment...
    py -3 -m venv venv
)

echo Installing packages...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

echo Setup complete!
pause