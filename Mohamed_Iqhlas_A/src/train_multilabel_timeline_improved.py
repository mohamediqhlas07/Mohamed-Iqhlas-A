"""
IMPROVED Multilabel Timeline Training Script
Enhanced CNN architecture for better accuracy and generalization.
Includes data augmentation for unseen audio performance.
"""

import os
import json
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Add, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIGURATION =================
DATA_DIR = "data_multilabel/audio"
LABELS_CSV = "data_multilabel/labels.csv"
CLASS_INDEX_PATH = "class_indices.json"

# Audio processing
SR = 22050
WINDOW_SIZE = 3.0
HOP_SIZE = 1.0
IMG_SIZE = (128, 128)
N_MELS = 128

# Training - IMPROVED PARAMETERS
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Data augmentation
USE_AUGMENTATION = True
AUGMENTATION_FACTOR = 2  # How many augmented samples per original

# Model paths
MODEL_SAVE_PATH = "multilabel_timeline_model.keras"
CLASSES_JSON_PATH = "multilabel_classes.json"

# ================= LOAD CLASS MAPPING =================
print("=" * 70)
print("IMPROVED CNN-BASED MULTILABEL INSTRUMENT RECOGNITION SYSTEM")
print("=" * 70)
print("\nLoading class mapping...")

with open(CLASS_INDEX_PATH, 'r') as f:
    class_indices = json.load(f)

ALL_CLASSES = sorted(class_indices.keys())
NUM_CLASSES = len(ALL_CLASSES)
class_to_idx = {cls: idx for idx, cls in enumerate(ALL_CLASSES)}

print(f"✓ Total classes: {NUM_CLASSES}")
print(f"✓ Classes: {ALL_CLASSES}")

with open(CLASSES_JSON_PATH, 'w') as f:
    json.dump(ALL_CLASSES, f, indent=2)
print(f"✓ Class mapping saved to {CLASSES_JSON_PATH}")

# ================= LOAD LABELS =================
print("\n" + "=" * 70)
print("Loading labels from CSV...")
print("=" * 70)

if not os.path.exists(LABELS_CSV):
    raise FileNotFoundError(f"Labels CSV not found: {LABELS_CSV}")

df = pd.read_csv(LABELS_CSV)
print(f"✓ Loaded {len(df)} audio files from CSV")

# Validate instrument names
invalid_instruments = []
for _, row in df.iterrows():
    instruments = [inst.strip() for inst in str(row['instruments']).split(',')]
    for inst in instruments:
        if inst and inst not in ALL_CLASSES:
            invalid_instruments.append((row['file'], inst))

if invalid_instruments:
    print("\n⚠ WARNING: Invalid instrument names found:")
    for file, inst in invalid_instruments[:10]:
        print(f"  {file}: '{inst}'")

# ================= ENHANCED FEATURE EXTRACTION =================
def audio_to_mel_spectrogram(y, sr=SR, augment=False):
    """
    Enhanced mel-spectrogram extraction with optional augmentation.
    """
    # Generate mel-spectrogram with better parameters
    mel = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_mels=N_MELS,
        hop_length=512,
        n_fft=2048,
        fmin=0,
        fmax=sr//2
    )
    
    # Convert to dB
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Data augmentation (time stretching, pitch shifting simulation)
    if augment:
        # Time stretching (simulated by shifting)
        if np.random.random() > 0.5:
            shift = np.random.randint(-5, 6)
            mel_db = np.roll(mel_db, shift, axis=1)
        
        # Frequency masking (simulated)
        if np.random.random() > 0.5:
            mask_size = np.random.randint(5, 15)
            mask_start = np.random.randint(0, N_MELS - mask_size)
            mel_db[mask_start:mask_start+mask_size, :] *= 0.5
    
    # Normalize to [0, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-10)
    
    # Resize to IMG_SIZE
    mel_db = tf.image.resize(
        mel_db[..., np.newaxis], 
        IMG_SIZE,
        method='bilinear'
    )
    
    # Convert to 3-channel (repeat for RGB)
    mel_db = tf.repeat(mel_db, 3, axis=-1)
    
    return mel_db.numpy()

# ================= PREPARE DATASET =================
print("\n" + "=" * 70)
print("Preparing dataset with sliding windows...")
print("=" * 70)

X = []
Y = []
file_info = []

for idx, row in df.iterrows():
    file_path = os.path.join(DATA_DIR, row['file'])
    
    if not os.path.exists(file_path):
        print(f"⚠ Skipping {row['file']}: File not found")
        continue
    
    try:
        y, sr = librosa.load(file_path, sr=SR, mono=True)
    except Exception as e:
        print(f"⚠ Error loading {row['file']}: {e}")
        continue
    
    instruments_str = str(row['instruments']).strip()
    if not instruments_str:
        continue
    
    instruments = [inst.strip() for inst in instruments_str.split(',') if inst.strip()]
    
    label_vector = np.zeros(NUM_CLASSES, dtype=np.float32)
    valid_instruments = []
    
    for inst in instruments:
        if inst in class_to_idx:
            label_vector[class_to_idx[inst]] = 1.0
            valid_instruments.append(inst)
    
    if not valid_instruments:
        continue
    
    window_samples = int(WINDOW_SIZE * SR)
    hop_samples = int(HOP_SIZE * SR)
    num_windows = max(1, (len(y) - window_samples) // hop_samples + 1)
    
    for i in range(num_windows):
        start_idx = i * hop_samples
        end_idx = start_idx + window_samples
        
        if end_idx > len(y):
            clip = np.pad(y[start_idx:], (0, window_samples - (len(y) - start_idx)), 'constant')
        else:
            clip = y[start_idx:end_idx]
        
        try:
            mel_spec = audio_to_mel_spectrogram(clip, sr, augment=False)
            X.append(mel_spec)
            Y.append(label_vector)
            file_info.append({
                'file': row['file'],
                'window_idx': i,
                'start_time': start_idx / sr,
                'instruments': valid_instruments
            })
        except Exception as e:
            continue

X = np.array(X)
Y = np.array(Y)

print(f"\n✓ Dataset prepared:")
print(f"  Total samples: {len(X)}")
print(f"  Input shape: {X.shape}")
print(f"  Label shape: {Y.shape}")

# Data augmentation
if USE_AUGMENTATION and len(X) > 0:
    print("\n" + "=" * 70)
    print("Applying data augmentation for better generalization...")
    print("=" * 70)
    
    X_aug = []
    Y_aug = []
    
    for i in range(len(X)):
        X_aug.append(X[i])
        Y_aug.append(Y[i])
        
        # Create augmented versions
        for _ in range(AUGMENTATION_FACTOR):
            try:
                # Reconstruct audio from mel (approximate)
                mel_db = X[i][:, :, 0]
                mel_linear = librosa.db_to_power(mel_db)
                y_recon = librosa.feature.inverse.mel_to_audio(
                    mel_linear, sr=SR, hop_length=512, n_fft=2048
                )
                
                # Augment
                aug_mel = audio_to_mel_spectrogram(y_recon, sr, augment=True)
                X_aug.append(aug_mel)
                Y_aug.append(Y[i])
            except:
                # If augmentation fails, just add original
                X_aug.append(X[i])
                Y_aug.append(Y[i])
    
    X = np.array(X_aug)
    Y = np.array(Y_aug)
    print(f"✓ Augmented dataset: {len(X)} samples")

# Check label distribution
label_counts = Y.sum(axis=0)
print(f"\n✓ Label distribution:")
for idx, count in enumerate(label_counts):
    if count > 0:
        print(f"  {ALL_CLASSES[idx]}: {int(count)} samples ({count/len(Y)*100:.1f}%)")

# ================= TRAIN/VAL SPLIT =================
print("\n" + "=" * 70)
print("Splitting dataset...")
print("=" * 70)

X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, 
    test_size=VALIDATION_SPLIT,
    random_state=RANDOM_SEED,
    shuffle=True
)

print(f"✓ Train samples: {len(X_train)}")
print(f"✓ Validation samples: {len(X_val)}")

# ================= IMPROVED MODEL ARCHITECTURE =================
print("\n" + "=" * 70)
print("Building IMPROVED CNN model...")
print("=" * 70)

# Use EfficientNetB0 for better accuracy (or MobileNetV2 if preferred)
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Unfreeze some layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Build improved model
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Enhanced feature extraction layers
x = Dense(1024, activation='relu', name='fc1')(x)
x = BatchNormalization()(x)
x = Dropout(0.6)(x)

x = Dense(512, activation='relu', name='fc2')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Dense(256, activation='relu', name='fc3')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

# Output layer: SIGMOID for multilabel
output = Dense(NUM_CLASSES, activation='sigmoid', name='predictions')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile with improved optimizer settings
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

print("✓ Model architecture:")
model.summary()

# ================= CALLBACKS =================
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=7,
        min_lr=1e-8,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# ================= TRAINING =================
print("\n" + "=" * 70)
print("Starting training...")
print("=" * 70)
print("This may take 1-2 hours depending on dataset size.")
print("Model will automatically save best weights.\n")

history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# Save final model
model.save(MODEL_SAVE_PATH)
print(f"\n✓ Model saved to {MODEL_SAVE_PATH}")

# Save training history
import pickle
history_dict = {
    'loss': history.history['loss'],
    'accuracy': history.history['accuracy'],
    'val_loss': history.history['val_loss'],
    'val_accuracy': history.history['val_accuracy'],
    'precision': history.history.get('precision', []),
    'recall': history.history.get('recall', []),
    'auc': history.history.get('auc', [])
}
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history_dict, f)
print(f"✓ Training history saved to training_history.pkl")

# ================= EVALUATION =================
print("\n" + "=" * 70)
print("Evaluating model...")
print("=" * 70)

val_loss, val_acc, val_precision, val_recall, val_auc = model.evaluate(
    X_val, Y_val,
    batch_size=BATCH_SIZE,
    verbose=1
)

print(f"\n✓ Validation Results:")
print(f"  Loss: {val_loss:.4f}")
print(f"  Accuracy: {val_acc:.4f}")
print(f"  Precision: {val_precision:.4f}")
print(f"  Recall: {val_recall:.4f}")
print(f"  AUC: {val_auc:.4f}")

# Per-class metrics
print("\n✓ Per-class metrics (threshold=0.5):")
predictions = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
predicted_binary = (predictions >= 0.5).astype(int)

for idx in range(NUM_CLASSES):
    true_positives = np.sum((Y_val[:, idx] == 1) & (predicted_binary[:, idx] == 1))
    false_positives = np.sum((Y_val[:, idx] == 0) & (predicted_binary[:, idx] == 1))
    false_negatives = np.sum((Y_val[:, idx] == 1) & (predicted_binary[:, idx] == 0))
    
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    if Y_val[:, idx].sum() > 0:
        print(f"  {ALL_CLASSES[idx]:20s} | P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f} | Samples: {int(Y_val[:, idx].sum())}")

print("\n" + "=" * 70)
print("✅ IMPROVED MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 70)
print(f"\nModel saved: {MODEL_SAVE_PATH}")
print(f"Class mapping saved: {CLASSES_JSON_PATH}")
print("\nThis improved model should have:")
print("  ✓ Better accuracy on seen data")
print("  ✓ Better generalization to unseen audio")
print("  ✓ Improved multilabel detection")
print("\nNext steps:")
print("1. Test the model using inference_multilabel_timeline.py")
print("2. Use the model in Streamlit app for predictions")

