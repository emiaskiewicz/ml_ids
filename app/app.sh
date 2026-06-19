#!/bin/bash

cd "$(dirname "$0")/.." || {
    echo "Nie udalo sie przejsc do folderu projektu."
    exit 1
}

if [ ! -f "requirements.txt" ]; then
    echo "Nie znaleziono pliku requirements.txt."
    exit 1
fi

if [ ! -f "app/app.py" ]; then
    echo "Nie znaleziono pliku app/app.py."
    exit 1
fi

if [ ! -f ".venv/bin/python" ]; then
    echo "Tworzenie srodowiska wirtualnego..."
    python3 -m venv .venv

    if [ $? -ne 0 ]; then
        echo "Nie udalo sie utworzyc srodowiska wirtualnego."
        exit 1
    fi
fi

echo "Aktywowanie srodowiska wirtualnego..."
source ".venv/bin/activate"

if [ $? -ne 0 ]; then
    echo "Nie udalo sie aktywowac srodowiska wirtualnego."
    exit 1
fi

echo "Instalowanie zaleznosci..."
python -m pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Nie udalo sie zainstalowac zaleznosci."
    exit 1
fi

echo "Pobieranie plikow Git LFS wymaganych przez aplikacje..."

if command -v git >/dev/null 2>&1 && git lfs version >/dev/null 2>&1; then
    git lfs pull --include="data/split/**/test.csv,final_models/**"

    if [ $? -ne 0 ]; then
        echo "Nie udalo sie pobrac plikow Git LFS."
        echo "Sprawdz, czy Git LFS jest zainstalowany i czy repozytorium ma dostep do plikow."
        exit 1
    fi
else
    echo "Nie znaleziono Git LFS."
    echo "Zainstaluj Git LFS albo pobierz pelne pliki danych i modeli recznie."
    exit 1
fi

echo "Uruchamianie aplikacji..."
python app/app.py

if [ $? -ne 0 ]; then
    echo "Aplikacja zakonczyla dzialanie z bledem."
    exit 1
fi