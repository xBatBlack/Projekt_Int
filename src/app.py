import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import os
import cv2

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
    st.error(f"Nie znaleziono pliku modelu. Upewnij się, że uruchomiłeś skrypt trenujący!")
    st.stop()

#FUNKCJE DLA WYJAŚNIALNEGO AI (GRAD-CAM)
def make_gradcam_heatmap(img_array, full_model):
    """Tworzy mapę ciepła pokazującą, na które piksele patrzył model"""
    base_model = full_model.layers[0] # Pobieramy wbudowanego MobileNetV2
    last_conv_layer = base_model.get_layer('out_relu') # Ostatnia warstwa z cechami

    # Budujemy pod-model pobierający cechy i przewidywania
    last_conv_model = tf.keras.Model(base_model.inputs, last_conv_layer.output)
    
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer in full_model.layers[1:]:
        x = layer(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_model(img_array)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.5):
    """Nakłada mapę ciepła na oryginalny obrazek"""
    img_arr = np.array(img)
    heatmap_resized = cv2.resize(heatmap, (img_arr.shape[1], img_arr.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    superimposed_img = cv2.addWeighted(heatmap_color, alpha, img_arr, 1 - alpha, 0)
    return superimposed_img

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
st.markdown("Inteligentny system z Explainable AI (XAI) do klasyfikacji odpadów.")

tab1, tab2, tab3 = st.tabs(["📸 Skaner Odpadów", "📈 Raport Badawczy", "ℹ️ O Projekcie"])

with tab1:
    st.header("Sprawdź, gdzie wyrzucić swój śmieć")
    input_method = st.radio("Wybierz metodę wprowadzania obrazu:", ("Wgraj zdjęcie z dysku", "Użyj kamery"))
    
    img_display = None
    if input_method == "Wgraj zdjęcie z dysku":
        uploaded_file = st.file_uploader("Wgraj zdjęcie śmiecia (JPG, PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file: img_display = Image.open(uploaded_file).convert('RGB')
    else:
        camera_file = st.camera_input("Zrób zdjęcie śmiecia")
        if camera_file: img_display = Image.open(camera_file).convert('RGB')

    if img_display is not None:
        col1, col2 = st.columns(2)
        
        # Przygotowanie obrazu
        img = img_display.resize((128, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        with col2:
            st.subheader("Rozpoznanie Sztucznej Inteligencji")
            predictions = model.predict(img_array)[0]
            predicted_idx = np.argmax(predictions)
            predicted_class = class_names[predicted_idx]
            confidence = predictions[predicted_idx]
            
            polish_name, advice = translations[predicted_class]
            
            st.success(f"**Wynik:** {polish_name} ({confidence*100:.1f}%)")
            st.info(f"**Instrukcja:** {advice}")
            
            # Explainable AI (Grad-CAM)
            try:
                heatmap = make_gradcam_heatmap(img_array, model)
                heatmap_img = overlay_heatmap(img_display, heatmap)
            except Exception as e:
                heatmap_img = None
                st.error("Błąd podczas generowania mapy ciepła.")

        with col1:
            st.subheader("Analiza Obrazu")
            if heatmap_img is not None:
                img_col1, img_col2 = st.columns(2)
                with img_col1:
                    st.image(img_display, caption='Oryginał', use_container_width=True)
                with img_col2:
                    st.image(heatmap_img, caption='Uwaga Modelu (Grad-CAM)', use_container_width=True)
            else:
                st.image(img_display, caption='Przeanalizowany obraz', width=300)

with tab2:
    st.header("Analiza Porównawcza Modeli")
    st.markdown("Raport obejmujący modele z inżynierią cech HOG oraz architekturę splotową MobileNetV2.")
    
    col3, col4 = st.columns(2)
    with col3:
        if os.path.exists(LEARNING_CURVE_PATH): st.image(LEARNING_CURVE_PATH, caption='Krzywa uczenia modelu sieci neuronowej')
    with col4:
        if os.path.exists(CONFUSION_MATRIX_PATH): st.image(CONFUSION_MATRIX_PATH, caption='Macierz pomyłek modelu docelowego')
            
    if os.path.exists(REPORT_PATH):
        with st.expander("Pokaż szczegółowy raport tekstowy"):
            with open(REPORT_PATH, 'r', encoding='utf-8') as f: st.code(f.read())

with tab3:
    st.header("Metodologia Badawcza")
    st.markdown("""
    ### 1. Inżynieria Cech (HOG)
    Dla modeli klasycznych zastosowano analizę Histogramu Zorientowanych Gradientów (HOG), aby pomóc prostym algorytmom dostrzegać krawędzie i kształty odpadów.

    ### 2. Wyjaśnialne AI (Explainable AI - XAI)
    Aplikacja wykorzystuje technikę **Grad-CAM**, która potrafi "zajrzeć do głowy" sieci neuronowej. Oblicza ona gradienty z ostatniej warstwy splotowej, generując "mapę ciepła", która podświetla w aplikacji na czerwono te elementy obrazu (np. gwint butelki, narożnik kartonu), które zadecydowały o zaklasyfikowaniu śmiecia.
    """)