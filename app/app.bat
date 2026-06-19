@echo off
cd /d "%~dp0.."

if not exist "requirements.txt" (
    echo Nie znaleziono pliku requirements.txt.
    pause
    exit /b 1
)

if not exist "app\app.py" (
    echo Nie znaleziono pliku app\app.py.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Tworzenie srodowiska wirtualnego...
    python -m venv .venv
    if errorlevel 1 (
        echo Nie udalo sie utworzyc srodowiska wirtualnego.
        pause
        exit /b 1
    )
)

echo Aktywowanie srodowiska wirtualnego...
call ".venv\Scripts\activate.bat"

echo Instalowanie zaleznosci...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Nie udalo sie zainstalowac zaleznosci.
    pause
    exit /b 1
)

echo Pobieranie plikow Git LFS wymaganych przez aplikacje...

git lfs version >nul 2>&1
if errorlevel 1 (
    echo Nie znaleziono Git LFS.
    echo Zainstaluj Git LFS albo pobierz pelne pliki danych i modeli recznie.
    pause
    exit /b 1
)

git lfs pull
if errorlevel 1 (
    echo Nie udalo sie pobrac plikow Git LFS.
    pause
    exit /b 1
)

echo Uruchamianie aplikacji...
python app\app.py

pause