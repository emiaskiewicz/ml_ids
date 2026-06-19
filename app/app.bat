@echo off
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
    echo Tworzenie srodowiska wirtualnego...
    python -m venv .venv
)

echo Aktywowanie srodowiska wirtualnego...
call ".venv\Scripts\activate.bat"

echo Instalowanie zaleznosci...
python -m pip install -q -r requirements.txt

echo Uruchamianie aplikacji...
python app\app.py

pause