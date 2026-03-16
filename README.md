# ML Intrusion Detection System - comparison

### Repozytorium dotyczące cześci badawczej do pracy licencjackiej "Porównanie metod detekcji anomalii w ruchu sieciowym"

# Uruchamianie projektu

## 1. Base preprocessing

Skrypt ```base_data.py``` odpowiada za wstępne przygotowanie danych.

Wykonywane operacje:
- wczytanie wszystkich plików CSV ze zbioru CICIDS2017
- usunięcie kolumn identyfikacyjnych oraz znaczników czasu
- obsługa wartości nieskończonych (inf, -inf)
- usunięcie wierszy z brakującymi wartościami
- usunięcie zduplikowanych rekordów
- normalizacja nazw kolumn
- normalizacja wartości w kolumnie ```Label```
- usunięcie cech o stałej wartości
- zapis przetworzonego zbioru danych

### Uruchomienie:

    python src/preprocessing/base_data.py

Po wykonaniu skrypt zapisuje dane do:

    data/cleaned/cicids2017_preprocessed.csv

## 2. Target preprocessing

Skrypt ```target_data.py``` przygotowuje zmienną celu dla klasyfikacji.

Na tym etapie problem został sprowadzony do klasyfikacji binarnej:

    BENIGN → 0
    ATTACK → 1

gdzie ```ATTACK``` oznacza wszystkie inne klasy niż ```BENIGN```

W zbiorze pozostaje również oryginalna kolumna ```Label```.

### Uruchomienie:

    python src/preprocessing/target_data.py

Wynikowy zbiór danych zostaje zapisany do:

    data/processed/cicids2017_binary.csv

## Logowanie

Skrypty korzystają z centralnego logowania. Szczegółowe logi zapisują się w katalogu:

    logs/

Logi zawierają m.in.:
- informacje o przetwarzanych plikach
- liczbę usuniętych rekordów
- końcowy rozmiar datasetu
- rozkład klas