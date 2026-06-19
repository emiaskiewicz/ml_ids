# CLI do testowania finalnych modeli IDS

Aplikacja konsolowa umożliwia uruchomienie zapisanych modeli końcowych wykorzystanych w pracy licencjackiej. Program pozwala wybrać model, wariant danych oraz liczbę rekordów testowych, a następnie wykonuje predykcję i wyświetla podstawowe metryki klasyfikacji.

Aplikacja nie trenuje modeli od nowa. Do działania wykorzystuje zapisane artefakty modeli oraz przygotowane pliki testowe.

## Struktura wykorzystywana przez aplikację

Ścieżki do modeli i danych są ustawione automatycznie:

```text
final_models/<nazwa_modelu>/model/
data/split/<easy|medium|hard>/test.csv
```

W projekcie muszą być dostępne następujące elementy:

```text
requirements.txt
app/app.py
app/app.bat
app/app.sh
final_models/
data/split/
src/
```

Folder `final_models/` zawiera zapisane modele końcowe, natomiast folder `data/split/` zawiera pliki testowe dla wariantów `easy`, `medium` oraz `hard`.

## Wymagania

Do uruchomienia aplikacji wymagany jest Python zainstalowany na komputerze.

Zalecana wersja:

```text
Python 3.11
```

Projekt był przygotowywany i testowany dla Pythona 3.11. W przypadku innych wersji Pythona część zależności może wymagać zmiany wersji pakietu albo osobnej instalacji zgodnej ze środowiskiem uruchomieniowym.

Wymagany jest również Git LFS, ponieważ dane testowe są przechowywane z użyciem Git Large File Storage.

Przykładowa instalacja Git LFS w systemie Linux Ubuntu/Debian:

```bash
sudo apt install git-lfs
git lfs install
```

W systemie Windows Git LFS można zainstalować za pomocą instalatora dostępnego na oficjalnej stronie Git LFS. Po instalacji należy otworzyć Git Bash i wykonać:

```bash
git lfs install
```

Przy pierwszym uruchomieniu wymagany jest dostęp do internetu, ponieważ skrypty uruchomieniowe tworzą środowisko wirtualne i instalują zależności z pliku `requirements.txt`; sam proces może potrwać dłużej szczególnie ze względu na instalację biblioteki torch oraz ewentualnych zależności CUDA/NVIDIA.

Nie trzeba osobno instalować modułu `venv`, ponieważ jest on standardowym modułem Pythona.

## Uruchomienie w systemie Windows

Najprostszy sposób uruchomienia aplikacji w systemie Windows to użycie pliku:

```bat
app\app.bat
```

Plik można uruchomić dwuklikiem albo z poziomu terminala:

```bat
app\app.bat
```

Skrypt wykonuje kolejno następujące operacje:

1. przechodzi do głównego folderu projektu,
2. sprawdza obecność wymaganych plików,
3. tworzy środowisko wirtualne `.venv`, jeśli jeszcze nie istnieje,
4. aktywuje środowisko wirtualne,
5. instaluje zależności z pliku `requirements.txt`,
6. uruchamia aplikację `app/app.py`.

Aplikację można również uruchomić ręcznie z głównego folderu projektu:

```bat
python app\app.py
```

## Uruchomienie w systemie Linux

W systemie Linux aplikację można uruchomić za pomocą skryptu:

```bash
app/app.sh
```

Przed pierwszym uruchomieniem należy nadać plikowi uprawnienia wykonywania:

```bash
chmod +x app/app.sh
```

Następnie można uruchomić aplikację:

```bash
./app/app.sh
```

Skrypt wykonuje kolejno następujące operacje:

1. przechodzi do głównego folderu projektu,
2. sprawdza obecność wymaganych plików,
3. tworzy środowisko wirtualne `.venv`, jeśli jeszcze nie istnieje,
4. aktywuje środowisko wirtualne,
5. instaluje zależności z pliku `requirements.txt`,
6. uruchamia aplikację `app/app.py`.

Aplikację można również uruchomić ręcznie z głównego folderu projektu:

```bash
python3 app/app.py
```

albo po aktywacji środowiska wirtualnego:

```bash
python app/app.py
```

## Dostępne modele

Program umożliwia wybór jednego z zapisanych modeli końcowych:

1. Logistic Regression
2. Decision Tree
3. SVM
4. MLP
5. Autoencoder
6. CNN

## Dostępne warianty danych

Program umożliwia wybór jednego z trzech wariantów danych:

1. Easy
2. Medium
3. Hard

Warianty odpowiadają przygotowanym plikom testowym znajdującym się w folderze:

```text
data/split/
```

## Liczba testowanych rekordów

Po wyborze modelu i wariantu danych program pyta o liczbę rekordów do przetestowania.

Dostępne opcje:

```text
pusta wartość    - użycie domyślnej liczby 1000 rekordów
liczba całkowita - użycie wskazanej liczby rekordów
all              - użycie całego pliku test.csv
```

Przykłady:

```text
1000
5000
all
```

Należy pamiętać, że opcja `all` może wydłużyć czas działania programu.

## Wyniki

Po wykonaniu predykcji program wyświetla:

* accuracy,
* precision,
* recall,
* F1-score,
* ROC-AUC,
* average precision,
* macierz pomyłek,
* raport klasyfikacji.

Wyniki są wyświetlane bezpośrednio w konsoli.

## Uwagi dotyczące zależności

Modele klasyczne, takie jak Logistic Regression, Decision Tree oraz SVM, korzystają głównie z bibliotek `scikit-learn`, `pandas`, `numpy` oraz `joblib`.

Modele MLP, CNN oraz Autoencoder wymagają biblioteki `torch`, ponieważ zostały zapisane jako modele PyTorch. W przypadku problemów z instalacją tej biblioteki należy dobrać wersję PyTorch zgodną z używaną wersją Pythona oraz systemem operacyjnym.
