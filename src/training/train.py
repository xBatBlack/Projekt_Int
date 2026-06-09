import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
from skimage.feature import hog
from skimage.color import rgb2gray

DATASET_PATH = 'data/raw/dataset-resized'
MODEL_SAVE_PATH = 'models/best_trash_model.h5'
FIGURES_DIR = 'reports/figures'
REPORT_DIR = 'reports/global'

os.makedirs('models', exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

print("ETAP 1: Preprocessing i Augmentacja")
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Nie znaleziono folderu z danymi: {DATASET_PATH}. Sprawdź strukturę plików!")

train_gen = datagen.flow_from_directory(DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='training')
val_gen = datagen.flow_from_directory(DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='validation', shuffle=False)
class_names = list(train_gen.class_indices.keys())

# --- INŻYNIERIA CECH: HOG (Histogram of Oriented Gradients) ---
def extract_advanced_features(images):
    """Wyciąga krawędzie (HOG) oraz informacje o kolorze dla starych modeli ML"""
    features = []
    for img in images:
        gray_img = rgb2gray(img)
        hog_feat = hog(gray_img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False)
        mean_col = np.mean(img, axis=(0, 1)) # Średni kolor RGB
        std_col = np.std(img, axis=(0, 1))   # Tekstura (Wariancja koloru)
        features.append(np.hstack([hog_feat, mean_col, std_col]))
    return np.array(features)

print("\nETAP 2: Modele Klasyczne z HOG (sklearn)")
# Zbieramy większą próbkę dla klasycznych algorytmów (5 paczek = 160 obrazków)
X_train_list, y_train_list = [], []
for _ in range(5):
    X_b, y_b = next(train_gen)
    X_train_list.append(X_b)
    y_train_list.append(y_b)
X_train_cls = np.concatenate(X_train_list)
y_train_cls = np.argmax(np.concatenate(y_train_list), axis=1)

X_val_list, y_val_list = [], []
for _ in range(3):
    X_b, y_b = next(val_gen)
    X_val_list.append(X_b)
    y_val_list.append(y_b)
X_val_cls = np.concatenate(X_val_list)
y_val_cls = np.argmax(np.concatenate(y_val_list), axis=1)

print("Ekstrakcja cech HOG w toku...")
X_train_features = extract_advanced_features(X_train_cls)
X_val_features = extract_advanced_features(X_val_cls)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_features, y_train_cls)
knn_acc = knn_model.score(X_val_features, y_val_cls)
print(f"Dokładność KNN (z HOG): {knn_acc*100:.2f}%")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_features, y_train_cls)
rf_acc = rf_model.score(X_val_features, y_val_cls)
print(f"Dokładność Random Forest (z HOG): {rf_acc*100:.2f}%")

svm_model = SVC(kernel='linear')
svm_model.fit(X_train_features, y_train_cls)
svm_acc = svm_model.score(X_val_features, y_val_cls)
print(f"Dokładność SVM (z HOG): {svm_acc*100:.2f}%")

print("\n=== ETAP 3: Głębokie Sieci Neuronowe (Keras) ===")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
]


#te modele testowałem, ale odrzuciłem przez za słabe wyniki
"""
 EKSPERYMENT 1: Własna, prosta sieć CNN
 Powód odrzucenia: Zbyt prosta architektura dla tak skomplikowanego zbioru danych.
 cnn_model = Sequential([
     Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
     MaxPooling2D((2, 2)),
     Conv2D(64, (3, 3), activation='relu'),
     MaxPooling2D((2, 2)),
     Flatten(),
     Dense(128, activation='relu'),
     Dropout(0.5),
     Dense(6, activation='softmax')
 ])
 cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
 cnn_history = cnn_model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=callbacks)

 EKSPERYMENT 2: Transfer Learning (ResNet50)
 Powód odrzucenia: Architektura okazała się "przewymiarowana" dla obrazków 128x128. 
 Model utknął na skuteczności ~30% i zadziałał mechanizm Early Stopping.
 base_resnet = ResNet50(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
 base_resnet.trainable = False
 resnet_model = Sequential([
     base_resnet,
     GlobalAveragePooling2D(),
     Dense(128, activation='relu'),
     Dropout(0.3),
     Dense(6, activation='softmax')
 ])
 resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
 rn_history = resnet_model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=callbacks)
"""
print("\nTrenowanie MobileNetV2 (Zwycięski Model)...")
base_mobilenet = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_mobilenet.trainable = False

mobilenet_model = Sequential([
    base_mobilenet,
    GlobalAveragePooling2D(),       # Spłaszcza wizję do jednego konkretnego wniosku (wektora)
    Dense(128, activation='relu'),
    Dropout(0.3),                   # losowo wyłącza 30% neuronów aby model nie uczył się na pamięć
    Dense(6, activation='softmax')
])
mobilenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
mn_history = mobilenet_model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=callbacks, verbose=1)

print("\nETAP 4: Zapisywanie i Generowanie Raportów")
mn_val_acc = max(mn_history.history['val_accuracy'])
mobilenet_model.save(MODEL_SAVE_PATH)

plt.figure(figsize=(8, 5))
plt.plot(mn_history.history['accuracy'], label='Dokładność - Trening')
plt.plot(mn_history.history['val_accuracy'], label='Dokładność - Walidacja')
plt.title('Krzywa Uczenia (MobileNetV2)')
plt.ylabel('Dokładność')
plt.xlabel('Epoka')
plt.legend()
plt.savefig(os.path.join(FIGURES_DIR, 'learning_curve.png'))

print("Generowanie macierzy pomyłek...")
val_gen.reset()
predictions = mobilenet_model.predict(val_gen)
y_pred = np.argmax(predictions, axis=1)
y_true = val_gen.classes

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Macierz Pomyłek')
plt.ylabel('Prawdziwa Klasa')
plt.xlabel('Przewidziana Klasa')
plt.savefig(os.path.join(FIGURES_DIR, 'confusion_matrix.png'))

with open(os.path.join(REPORT_DIR, 'raport_badawczy.txt'), 'w', encoding='utf-8') as f:
    f.write("=== RAPORT Z EKSPERYMENTÓW ===\n\n")
    f.write("-MODELE KLASYCZNE (Inżynieria Cech HOG)-\n")
    f.write(f"1. Baseline KNN: {knn_acc*100:.2f}%\n")
    f.write(f"2. Baseline Random Forest: {rf_acc*100:.2f}%\n")
    f.write(f"3. Baseline SVM: {svm_acc*100:.2f}%\n\n")
    f.write("-GŁĘBOKIE SIECI NEURONOWE-\n")
    f.write(f"MobileNetV2 Dokładność: {mn_val_acc*100:.2f}%\n")
    f.write(f"* Modele CNN oraz ResNet50 odrzucono na etapie badań.\n\n")
    f.write("-SZCZEGÓŁOWY RAPORT-\n")
    f.write(classification_report(y_true, y_pred, target_names=class_names))

print("Zakończono! Zapisano ulepszony model, wykresy i raport.")