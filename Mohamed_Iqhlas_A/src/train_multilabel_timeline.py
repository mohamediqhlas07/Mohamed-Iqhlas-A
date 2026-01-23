"""
Complete Multilabel Timeline Training Script
Trains a CNN model for multilabel instrument detection with timeline support.
Uses weak labeling from CSV file.
"""

import os
import json
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
import pickle
from sklearn.model_selection import train_test_split

# ================= CONFIGURATION =================
DATA_DIR = "data_multilabel/audio"
LABELS_CSV = "data_multilabel/labels.csv"
CLASS_INDEX_PATH = "class_indices.json"

# Audio processing
SR = 22050
WINDOW_SIZE = 3.0  # seconds
HOP_SIZE = 1.0     # seconds
IMG_SIZE = (128, 128)
N_MELS = 128

# Training
EPOCHS = 60  # More epochs with early stopping
BATCH_SIZE = 32
INITIAL_LR = 0.001  # Higher initial learning rate
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Model paths
MODEL_SAVE_PATH = "multilabel_timeline_model.keras"
CLASSES_JSON_PATH = "multilabel_classes.json"

# ================= LOAD CLASS MAPPING =================
print("=" * 60)
print("Loading class mapping...")
print("=" * 60)

with open(CLASS_INDEX_PATH, 'r') as f:
    class_indices = json.load(f)

# All 28 instruments (fixed order)
ALL_CLASSES = sorted(class_indices.keys())
NUM_CLASSES = len(ALL_CLASSES)
class_to_idx = {cls: idx for idx, cls in enumerate(ALL_CLASSES)}

print(f"Total classes: {NUM_CLASSES}")
print(f"Classes: {ALL_CLASSES}")

# Save class mapping
with open(CLASSES_JSON_PATH, 'w') as f:
    json.dump(ALL_CLASSES, f, indent=2)
print(f"✓ Class mapping saved to {CLASSES_JSON_PATH}")

# ================= LOAD LABELS =================
print("\n" + "=" * 60)
print("Loading labels from CSV...")
print("=" * 60)

if not os.path.exists(LABELS_CSV):
    raise FileNotFoundError(
        f"Labels CSV not found: {LABELS_CSV}\n"
        f"Please create {LABELS_CSV} with format:\n"
        f"file,instruments\n"
        f"song1.mp3,Violin,Piano,Guitar\n"
        f"song2.mp3,Drums,Bass_Guitar\n"
    )

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
    if len(invalid_instruments) > 10:
        print(f"  ... and {len(invalid_instruments) - 10} more")
    print("\nThese will be ignored. Please check your CSV file.")

# ================= FEATURE EXTRACTION =================
def audio_to_mel_spectrogram(y, sr=SR):
    """
    Convert audio to mel-spectrogram image.
    
    Args:
        y: Audio time series
        sr: Sample rate
    
    Returns:
        Mel-spectrogram image (128, 128, 3)
    """
    # Generate mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_mels=N_MELS,
        hop_length=512,
        n_fft=2048
    )
    
    # Convert to dB
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
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
print("\n" + "=" * 60)
print("Preparing dataset with sliding windows...")
print("=" * 60)

X = []
Y = []
file_info = []  # Track which file each sample comes from

for idx, row in df.iterrows():
    file_path = os.path.join(DATA_DIR, row['file'])
    
    if not os.path.exists(file_path):
        print(f"⚠ Skipping {row['file']}: File not found")
        continue
    
    # Load audio
    try:
        y, sr = librosa.load(file_path, sr=SR, mono=True)
    except Exception as e:
        print(f"⚠ Error loading {row['file']}: {e}")
        continue
    
    # Parse instruments (handle empty strings)
    instruments_str = str(row['instruments']).strip()
    if not instruments_str:
        print(f"⚠ Skipping {row['file']}: No instruments listed")
        continue
    
    instruments = [inst.strip() for inst in instruments_str.split(',') if inst.strip()]
    
    # Create multilabel vector
    label_vector = np.zeros(NUM_CLASSES, dtype=np.float32)
    valid_instruments = []
    
    for inst in instruments:
        if inst in class_to_idx:
            label_vector[class_to_idx[inst]] = 1.0
            valid_instruments.append(inst)
        else:
            print(f"⚠ Unknown instrument '{inst}' in {row['file']}, skipping")
    
    if not valid_instruments:
        print(f"⚠ Skipping {row['file']}: No valid instruments")
        continue
    
    # Create sliding windows
    window_samples = int(WINDOW_SIZE * SR)
    hop_samples = int(HOP_SIZE * SR)
    
    num_windows = max(1, (len(y) - window_samples) // hop_samples + 1)
    
    for i in range(num_windows):
        start_idx = i * hop_samples
        end_idx = start_idx + window_samples
        
        if end_idx > len(y):
            # Pad last window if needed
            clip = np.pad(y[start_idx:], (0, window_samples - (len(y) - start_idx)), 'constant')
        else:
            clip = y[start_idx:end_idx]
        
        # Extract mel-spectrogram
        try:
            mel_spec = audio_to_mel_spectrogram(clip, sr)
            X.append(mel_spec)
            Y.append(label_vector)
            file_info.append({
                'file': row['file'],
                'window_idx': i,
                'start_time': start_idx / sr,
                'instruments': valid_instruments
            })
        except Exception as e:
            print(f"⚠ Error processing window {i} of {row['file']}: {e}")
            continue

X = np.array(X)
Y = np.array(Y)

print(f"\n✓ Dataset prepared:")
print(f"  Total samples: {len(X)}")
print(f"  Input shape: {X.shape}")
print(f"  Label shape: {Y.shape}")

# Check label distribution
label_counts = Y.sum(axis=0)
print(f"\n✓ Label distribution:")
for idx, count in enumerate(label_counts):
    if count > 0:
        print(f"  {ALL_CLASSES[idx]}: {int(count)} samples ({count/len(Y)*100:.1f}%)")

# ================= TRAIN/VAL SPLIT =================
print("\n" + "=" * 60)
print("Splitting dataset...")
print("=" * 60)

X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, 
    test_size=VALIDATION_SPLIT,
    random_state=RANDOM_SEED,
    shuffle=True
)

print(f"✓ Train samples: {len(X_train)}")
print(f"✓ Validation samples: {len(X_val)}")

# ================= MODEL ARCHITECTURE =================
print("\n" + "=" * 60)
print("Building model...")
print("=" * 60)

# Base model (EfficientNetB0 - better than MobileNetV2)
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Freeze base model initially
base_model.trainable = False

# Build model with better architecture
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu', name='fc1')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu', name='fc2')(x)
x = Dropout(0.4)(x)
x = BatchNormalization()(x)
x = Dense(128, activation='relu', name='fc3')(x)
x = Dropout(0.3)(x)

# Output layer: SIGMOID for multilabel (NOT softmax)
output = Dense(NUM_CLASSES, activation='sigmoid', name='predictions')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=INITIAL_LR),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

print("✓ Model architecture:")
model.summary()

# ================= CALLBACKS =================
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',  # Monitor accuracy instead of loss
        patience=12,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=6,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_SAVE_PATH.replace('.keras', '_best.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    CSVLogger('multilabel_training_log.csv', append=False)
]

# ================= PHASE 1: TRAINING WITH FROZEN BASE =================
print("\n" + "=" * 60)
print("PHASE 1: Training with frozen base model")
print("=" * 60)

history1 = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ================= PHASE 2: FINE-TUNING =================
print("\n" + "=" * 60)
print("PHASE 2: Fine-tuning (unfreezing top layers)")
print("=" * 60)

# Unfreeze top layers
base_model.trainable = True

# Freeze bottom layers, unfreeze top layers
for layer in base_model.layers[:-30]:  # Freeze all but last 30 layers
    layer.trainable = False

# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=INITIAL_LR * 0.1),  # 10x smaller LR
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

# Fine-tune callbacks
fine_tune_callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_SAVE_PATH.replace('.keras', '_best.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    CSVLogger('multilabel_training_log.csv', append=True)
]

history2 = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=40,  # Additional epochs for fine-tuning
    batch_size=BATCH_SIZE,
    callbacks=fine_tune_callbacks,
    verbose=1
)

# Combine histories
combined_history = {
    'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
    'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
    'loss': history1.history['loss'] + history2.history['loss'],
    'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
    'precision': history1.history['precision'] + history2.history['precision'],
    'val_precision': history1.history['val_precision'] + history2.history['val_precision'],
    'recall': history1.history['recall'] + history2.history['recall'],
    'val_recall': history1.history['val_recall'] + history2.history['val_recall']
}

# Save best model
if os.path.exists(MODEL_SAVE_PATH.replace('.keras', '_best.keras')):
    import shutil
    shutil.copy(MODEL_SAVE_PATH.replace('.keras', '_best.keras'), MODEL_SAVE_PATH)
    print(f"\n✓ Best model copied to {MODEL_SAVE_PATH}")
else:
    model.save(MODEL_SAVE_PATH)
    print(f"\n✓ Model saved to {MODEL_SAVE_PATH}")

# Save training history
with open('training_history.pkl', 'wb') as f:
    pickle.dump(combined_history, f)
print("✓ Training history saved to training_history.pkl")

# ================= EVALUATION =================
print("\n" + "=" * 60)
print("Evaluating model...")
print("=" * 60)

# Evaluate on validation set
val_loss, val_acc, val_precision, val_recall = model.evaluate(
    X_val, Y_val,
    batch_size=BATCH_SIZE,
    verbose=1
)

print(f"\n✓ Final Validation Results:")
print(f"  Loss: {val_loss:.4f}")
print(f"  Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"  Precision: {val_precision:.4f}")
print(f"  Recall: {val_recall:.4f}")
print(f"\n✓ Best Validation Accuracy: {max(combined_history['val_accuracy']):.4f} ({max(combined_history['val_accuracy'])*100:.2f}%)")

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
    
    if Y_val[:, idx].sum() > 0:  # Only show classes present in validation set
        print(f"  {ALL_CLASSES[idx]:20s} | P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f} | Samples: {int(Y_val[:, idx].sum())}")

print("\n" + "=" * 60)
print("✅ Training completed successfully!")
print("=" * 60)
print(f"\nModel saved: {MODEL_SAVE_PATH}")
print(f"Class mapping saved: {CLASSES_JSON_PATH}")
print("\nNext steps:")
print("1. Test the model using inference_multilabel_timeline.py")
print("2. Use the model in Streamlit app for predictions")
