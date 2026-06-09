## Jak uruchomić projekt

### 1. Wymagania wstępne
Python (Ja używałem wersji 3.11.9, 3.14 nie ma tensora)

### 2. Konfiguracja środowiska
Zalecamy użycie środowiska wirtualnego, aby uniknąć konfliktów bibliotek:

```bash
# Utworzenie środowiska wirtualnego
python -m venv venv

# Aktywacja (Windows)
venv\Scripts\activate

# Aktywacja (Linux/macOS)
source venv/bin/activate
```

### 3. Uruchomienie programu:
```bash
# Instalacja bibliotek
pip install numpy matplotlib seaborn scikit-learn tensorflow opencv-python-headless streamlit scikit-image

# Uruchomienie treningu
python src/training/train.py

# Uruchomienie aplikacji
streamlit run src/app.py
```