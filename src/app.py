import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import os

# --- KONFIGURACJA ŚCIEŻEK ---
MODEL_PATH = 'models/best_trash_model.h5'
LEARNING_CURVE_PATH = 'reports/figures/learning_curve.png'
CONFUSION_MATRIX_PATH = 'reports/figures/confusion_matrix.png'
REPORT_PATH = 'reports/global/raport_badawczy.txt'

st.set_page_config(page_title="Asystent Recyklingu AI", page_icon="♻️", layout="wide")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Nie znaleziono pliku modelu. Upewnij się, że uruchomiłeś skrypt trenujący i plik istnieje w: {MODEL_PATH}")
    st.stop()

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
translations = {
    'cardboard': ('Karton', 'Wyrzuć do NIEBIESKIEGO pojemnika (Papier).'),
    'glass': ('Szkło', 'Wyrzuć do ZIELONEGO pojemnika (Szkło).'),
    'metal': ('Metal', 'Wyrzuć do ŻÓŁTEGO pojemnika (Metale i Tworzywa Sztuczne).'),
    'paper': ('Papier', 'Wyrzuć do NIEBIESKIEGO pojemnika (Papier).'),
    'plastic': ('Plastik', 'Wyrzuć do ŻÓŁTEGO pojemnika (Metale i Tworzywa Sztuczne).'),
    'trash': ('Odpady Zmieszane', 'Wyrzuć do CZARNEGO pojemnika (Zmieszane).')
}

st.title("♻️ Asystent Recyklingu AI")
st.markdown("Inteligentny system klasyfikacji odpadów stworzony na przedmiot **Inteligencja Obliczeniowa**.")

tab1, tab2, tab3 = st.tabs(["📸 Skaner Odpadów", "📈 Raport Badawczy", "ℹ️ O Projekcie"])

with tab1:
    st.header("Sprawdź, gdzie wyrzucić swój śmieć")
    
    input_method = st.radio("Wybierz metodę wprowadzania obrazu:", ("Wgraj zdjęcie z dysku", "Użyj kamery"))
    
    img_display = None
    if input_method == "Wgraj zdjęcie z dysku":
        uploaded_file = st.file_uploader("Wgraj zdjęcie śmiecia (JPG, PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            img_display = Image.open(uploaded_file)
    else:
        camera_file = st.camera_input("Zrób zdjęcie śmiecia")
        if camera_file:
            img_display = Image.open(camera_file)

    if img_display is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_display, caption='Przeanalizowany obraz', use_container_width=True)
        with col2:
            with st.spinner('Analiza obrazu przez sztuczną inteligencję...'):
                img = img_display.resize((128, 128))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0
                
                predictions = model.predict(img_array)[0]
                predicted_idx = np.argmax(predictions)
                predicted_class = class_names[predicted_idx]
                confidence = predictions[predicted_idx]
                
                polish_name, advice = translations[predicted_class]
                
                st.success(f"### Wynik: {polish_name}")
                st.info(f"**Instrukcja:** {advice}")
                st.write(f"Pewność modelu: **{confidence*100:.2f}%**")
                
                st.markdown("#### Rozkład prawdopodobieństwa:")
                df_probs = pd.DataFrame({
                    'Klasa': [translations[name][0] for name in class_names],
                    'Prawdopodobieństwo': predictions
                })
                fig = px.bar(df_probs, x='Prawdopodobieństwo', y='Klasa', orientation='h', 
                             color='Prawdopodobieństwo', color_continuous_scale='Blues')
                fig.update_layout(showlegend=False, height=300, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Analiza Porównawcza Modeli")
    st.markdown("Zgodnie z wymogami projektu, zrealizowano część badawczą w której przetestowano algorytmy klasyczne oraz głębokie sieci neuronowe.")
    
    col3, col4 = st.columns(2)
    with col3:
        if os.path.exists(LEARNING_CURVE_PATH):
            st.image(LEARNING_CURVE_PATH, caption='Krzywa uczenia modelu sieci neuronowej')
        else:
            st.warning("Brak wykresu krzywej uczenia.")
            
    with col4:
        if os.path.exists(CONFUSION_MATRIX_PATH):
            st.image(CONFUSION_MATRIX_PATH, caption='Macierz pomyłek modelu docelowego')
        else:
            st.warning("Brak macierzy pomyłek.")
            
    if os.path.exists(REPORT_PATH):
        with st.expander("Pokaż szczegółowy raport tekstowy (Classification Report)"):
            with open(REPORT_PATH, 'r', encoding='utf-8') as f:
                st.code(f.read())

with tab3:
    st.header("Technologie Użyte w Projekcie")
    st.markdown("""
    * **Język:** Python 3.11
    * **Klasyfikacja Obrazów / Deep Learning:** TensorFlow, Keras, MobileNetV2
    * **Klasyfikatory bazowe (Baseline):** scikit-learn (Random Forest, KNN)
    * **Przetwarzanie obrazu:** OpenCV / Pillow, ImageDataGenerator
    * **Interfejs Użytkownika:** Streamlit
    * **Wizualizacja Danych:** Matplotlib, Seaborn, Plotly
    """)