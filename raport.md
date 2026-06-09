# Raport Badawczy: Asystent Recyklingu AI z Wyjaśnialnym AI (XAI)

## 1. Wstęp i Cel Projektu
Celem projektu jest stworzenie inteligentnego systemu do klasyfikacji odpadów na podstawie obrazu. Projekt realizuje założenia ścieżki "Aplikacja/Serwis" (50% część badawcza, 50% wdrożenie w postaci interfejsu webowego Streamlit). Modele trenowano na zbiorze TrashNet, obejmującym 6 klas: karton, szkło, metal, papier, plastik oraz odpady zmieszane (trash). Dodatkowo zaimplementowano techniki Wyjaśnialnego AI (XAI), aby zrozumieć proces decyzyjny sieci neuronowych.

## 2. Preprocessing i Zaawansowana Inżynieria Cech (Feature Engineering)
Zdjęcia zostały ujednolicone do rozdzielczości 128x128 pikseli. Zastosowano techniki augmentacji danych (obroty, przesunięcia, odbicia lustrzane). W początkowej fazie klasyczne modele korzystały z surowych pikseli, co dawało bardzo słabe rezultaty. 

W docelowym rozwiązaniu zaimplementowano **zaawansowaną inżynierię cech**. Zamiast surowych pikseli, z obrazów wyekstrahowano:
* **HOG (Histogram of Oriented Gradients):** deskryptor wyłapujący krawędzie i kształty (ignorujący kolor),
* **Średni kolor RGB** oraz **Wariancję koloru** (reprezentującą teksturę materiału).

## 3. Eksperyment 1: Modele Klasyczne (Baseline z HOG)
Dzięki zastosowaniu deskryptora HOG oraz cech koloru, skuteczność klasycznych algorytmów uległa drastycznej poprawie:
* **KNN:** 75.00% (ogromny skok względem 25% na surowych pikselach)
* **Random Forest:** 31.25%
* **SVM (Linear):** 31.25%

**Wniosek:** Algorytm KNN w połączeniu z odpowiednio zaprojektowaną inżynierią cech potrafi osiągnąć skuteczność porównywalną z głębokimi sieciami neuronowymi. Udowadnia to fundamentalną zasadę Data Science: jakość danych wejściowych i cech jest równie ważna co skomplikowanie samego modelu.

## 4. Eksperyment 2: Głębokie Sieci Neuronowe (Transfer Learning)
W ramach badań przetestowano własne architektury CNN oraz potężny model ResNet50. Obie opcje odrzucono – własne CNN okazało się zbyt proste, a ResNet50 "przewymiarowany" (przeuczał się lub zatrzymywał na skuteczności ~30%).
Docelowym modelem został lekki i zoptymalizowany **MobileNetV2**, który po dotrenowaniu (Transfer Learning) osiągnął najwyższą ogólną dokładność: **75.35%**.

## 5. Analiza Zwycięskiego Modelu (MobileNetV2)

### Krzywa Uczenia
Wykres uczenia wykazuje klasyczny, zdrowy przebieg. Dokładność treningowa rośnie do niemal 90%, podczas gdy walidacyjna stabilizuje się w okolicach 73-75%. Mechanizm wczesnego zatrzymywania (Early Stopping) zapobiegł przeuczeniu (overfittingowi), przerywając trening w optymalnym momencie.

### Macierz Pomyłek
Szczegółowa analiza macierzy pomyłek wskazuje na logiczne problemy modelu z rozróżnianiem materiałów o podobnej strukturze wizualnej:
* **Największe pomyłki:** Model aż 18 razy sklasyfikował **plastik jako szkło** (co jest zrozumiałe w przypadku przezroczystych butelek PET) oraz 14 razy pomylił **karton z papierem**.
* **Mocne strony:** Model świetnie radzi sobie z wyizolowanym papierem (105 poprawnych predykcji) oraz szkłem (75 poprawnych).
* Metryka F1-Score dla klasy *trash* (odpady zmieszane) wynosi zaledwie 0.44, co wynika z faktu, że klasa ta jest "workiem" na przedmioty o całkowicie losowych kształtach i kolorach, przez co model nie potrafi znaleźć dla niej jednego, spójnego wzorca.

## 6. Aspekt Badawczy: Explainable AI (Grad-CAM) i Odkrycie "Shortcut Learning"
Kluczowym elementem projektu było zaimplementowanie w aplikacji algorytmu **Grad-CAM**, który generuje mapy ciepła, wskazując obszary obrazu determinujące decyzję sieci.

**Analiza wizualna ujawniła zjawisko "Background Bias" (Efekt Mądrego Hansa).**
W wielu przypadkach model MobileNetV2 nie klasyfikował odpadu na podstawie jego fizycznej faktury, lecz uczył się tzw. skrótów (shortcut learning) ze zbioru treningowego. Przykładowo, klasyfikując metalową puszkę, model ignorował jej środek (nadruk), a skupiał się na trawie w tle lub cieniach rzucanych przez obiekt. Algorytm zauważył statystyczną korelację między pewnymi typami teł w zbiorze TrashNet a konkretną klasą odpadu. Odkrycie to jest niezwykle cenne i dowodzi, że sama wysoka skuteczność w metrykach nie gwarantuje, że model "rozumie" problem w sposób ludzki.

## 7. Podsumowanie
Projekt zakończył się sukcesem, dostarczając funkcjonalną aplikację w bibliotece Streamlit. Wykorzystano w nim zarówno klasyczne metody widzenia komputerowego (HOG), nowoczesne podejścia głębokiego uczenia (MobileNetV2), jak i zaawansowane techniki wyjaśnialności (Grad-CAM). Aplikacja jest gotowa do demonstracji na żywo, umożliwiając skanowanie odpadów z kamery lub dysku.
