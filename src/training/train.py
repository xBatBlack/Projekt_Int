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

# --- KONFIGURACJA ŚCIEŻEK ---
DATASET_PATH = 'data/raw/dataset-resized'
MODEL_SAVE_PATH = 'models/best_trash_model.h5'
FIGURES_DIR = 'reports/figures'
REPORT_DIR = 'reports/global'

os.makedirs('models', exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

print("=== ETAP 1: Preprocessing i Augmentacja ===")
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='training')
val_gen = datagen.flow_from_directory(DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset='validation', shuffle=False)
class_names = list(train_gen.class_indices.keys())

# --- INŻYNIERIA CECH (FEATURE ENGINEERING) ---
def extract_features(images):
    """Wyciąga inteligentne cechy z obrazków dla klasycznych algorytmów sklearn"""
    flat_pixels = images.reshape(images.shape[0], -1) # Stare cechy: surowe piksele
    mean_colors = np.mean(images, axis=(1, 2))        # Nowa cecha 1: Średni kolor RGB
    std_colors = np.std(images, axis=(1, 2))          # Nowa cecha 2: Wariancja (tekstura)
    # Łączymy wszystko w jedną potężną tablicę cech
    return np.hstack([flat_pixels, mean_colors, std_colors])

print("\n=== ETAP 2: Modele Klasyczne (sklearn) z nowymi cechami ===")
X_batch, y_batch = next(train_gen)
X_val_batch, y_val_batch = next(val_gen)

X_features = extract_features(X_batch)
X_val_features = extract_features(X_val_batch)
y_flat, y_val_flat = np.argmax(y_batch, axis=1), np.argmax(y_val_batch, axis=1)

# 1. KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_features, y_flat)
knn_acc = knn_model.score(X_val_features, y_val_flat)
print(f"Dokładność KNN: {knn_acc*100:.2f}%")

# 2. Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_features, y_flat)
rf_acc = rf_model.score(X_val_features, y_val_flat)
print(f"Dokładność Random Forest: {rf_acc*100:.2f}%")

# 3. NOWY MODEL: Support Vector Machine (SVM)
svm_model = SVC(kernel='linear')
svm_model.fit(X_features, y_flat)
svm_acc = svm_model.score(X_val_features, y_val_flat)
print(f"Dokładność SVM: {svm_acc*100:.2f}%")


print("\n=== ETAP 3: Głębokie Sieci Neuronowe (Keras) ===")
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
]

# 4. Transfer Learning (MobileNetV2)
print("\nTrenowanie MobileNetV2...")
base_mobilenet = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_mobilenet.trainable = False

mobilenet_model = Sequential([
    base_mobilenet,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(6, activation='softmax')
])
mobilenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
mn_history = mobilenet_model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=callbacks, verbose=1)

# 5. NOWY MODEL: Transfer Learning (ResNet50)
print("\nTrenowanie ResNet50 (To może chwilę potrwać)...")
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
rn_history = resnet_model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=callbacks, verbose=1)

print("\n=== ETAP 4: Zapisywanie i Generowanie Raportów ===")
# Wybieramy lepszy model głęboki do zapisu dla aplikacji
mn_val_acc = max(mn_history.history['val_accuracy'])
rn_val_acc = max(rn_history.history['val_accuracy'])

if mn_val_acc >= rn_val_acc:
    print("MobileNetV2 wygrał! Zapisuję ten model.")
    best_model = mobilenet_model
    best_history = mn_history
else:
    print("ResNet50 wygrał! Zapisuję ten model.")
    best_model = resnet_model
    best_history = rn_history

best_model.save(MODEL_SAVE_PATH)

# Wykres uczenia (dla najlepszego modelu)
plt.figure(figsize=(8, 5))
plt.plot(best_history.history['accuracy'], label='Dokładność - Trening')
plt.plot(best_history.history['val_accuracy'], label='Dokładność - Walidacja')
plt.title('Krzywa Uczenia (Zwycięski Model)')
plt.ylabel('Dokładność')
plt.xlabel('Epoka')
plt.legend()
plt.savefig(os.path.join(FIGURES_DIR, 'learning_curve.png'))

print("Generowanie macierzy pomyłek dla najlepszego modelu...")
val_gen.reset()
predictions = best_model.predict(val_gen)
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
    f.write("RAPORT Z EKSPERYMENTÓW\n\n")
    f.write("MODELE KLASYCZNE (z inżynierią cech koloru)\n")
    f.write(f"1. Baseline KNN: {knn_acc*100:.2f}%\n")
    f.write(f"2. Baseline Random Forest: {rf_acc*100:.2f}%\n")
    f.write(f"3. Baseline SVM: {svm_acc*100:.2f}%\n\n")
    
    f.write("GŁĘBOKIE SIECI NEURONOWE\n")
    f.write(f"MobileNetV2 Najwyższa Dokładność: {mn_val_acc*100:.2f}%\n")
    f.write(f"ResNet50 Najwyższa Dokładność: {rn_val_acc*100:.2f}%\n\n")