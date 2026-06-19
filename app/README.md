# CLI do testowania finalnych modeli IDS

Program pozwala wybrac model i poziom trudnosci danych z menu w konsoli.
Sciezki sa ustawione automatycznie:

- modele: `final_models/<nazwa_modelu>/model/`
- dane testowe: `data/split/<easy|medium|hard>/test.csv`

## Wymagania

Do uruchomienia aplikacji wymagany jest Python zainstalowany na komputerze.
Zalecana wersja Pythona: 3.11.

Projekt byl przygotowywany i testowany dla Pythona 3.11. W przypadku innych wersji Pythona czesc zaleznosci, szczegolnie `torch`, moze wymagac innej wersji pakietu albo dodatkowej instalacji.

Komenda `python` musi dzialac w konsoli.

Nie trzeba osobno instalowac `venv`, poniewaz jest to standardowy modul Pythona.
Plik `app.bat` moze utworzyc srodowisko wirtualne komenda:

```bat
python -m venv .venv
```

Przy pierwszym uruchomieniu potrzebny jest dostep do internetu, poniewaz zaleznosci sa instalowane z pliku `requirements.txt`.

W projekcie musza byc dostepne:

- `requirements.txt`
- `app/app.py`
- `final_models/`
- `data/split/`

## Uruchomienie

Z glownego folderu projektu:

```bash
python app/app.py
```

Na Windows mozna tez uruchomic:

```bash
app/app.bat
```

## Dostepne opcje

Modele:

1. Logistic Regression
2. Decision Tree
3. SVM
4. MLP
5. Autoencoder
6. CNN

Poziomy trudnosci:

1. Easy
2. Medium
3. Hard

Program pyta tez o liczbe rekordow do testu. Pusta wartosc oznacza `1000`, a `all` oznacza caly plik `test.csv`.

## Wyniki

Program wyswietla:

- accuracy
- precision
- recall
- F1-score
- ROC-AUC
- average precision
- macierz pomylek
- classification report
