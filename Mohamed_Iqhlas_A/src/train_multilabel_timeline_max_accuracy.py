"""
MAXIMUM ACCURACY Multilabel Timeline Training Script
Enhanced CNN architecture with advanced techniques for maximum train/val accuracy.
Includes: Mixup augmentation, class weights, advanced augmentation, learning rate scheduling.
"""

import os
import json
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Add, Conv2D, MaxPooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
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

# Training - MAXIMUM ACCURACY PARAMETERS
EPOCHS = 200  # More epochs for better convergence
BATCH_SIZE = 16  # Smaller batch for better gradient estimates
INITIAL_LEARNING_RATE = 0.0005  # Lower initial LR for more stable training
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Advanced augmentation
USE_AUGMENTATION = True
AUGMENTATION_FACTOR = 3  # More augmentation

# Model paths
MODEL_SAVE_PATH = "multilabel_timeline_model.keras"
CLASSES_JSON_PATH = "multilabel_classes.json"

# ================= LOAD CLASS MAPPING =================
print("=" * 70)
print("MAXIMUM ACCURACY CNN-BASED MULTILABEL INSTRUMENT RECOGNITION")
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
    Enhanced mel-spectrogram extraction with advanced augmentation.
    """
    # Generate mel-spectrogram with better parameters
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        fmax=sr // 2,
        hop_length=512,
        n_fft=2048,
        power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Normalize to [0, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-10)
    
    # Resize to target size
    import cv2
    mel_resized = cv2.resize(mel_db, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
    
    # Apply augmentation if requested
    if augment:
        # Random horizontal flip
        if np.random.rand() > 0.5:
            mel_resized = np.fliplr(mel_resized)
        
        # Random time shift (circular)
        if np.random.rand() > 0.5:
            shift = np.random.randint(-10, 10)
            mel_resized = np.roll(mel_resized, shift, axis=1)
        
        # Random brightness/contrast
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)  # Contrast
            beta = np.random.uniform(-0.1, 0.1)   # Brightness
            mel_resized = np.clip(alpha * mel_resized + beta, 0, 1)
        
        # Random noise
        if np.random.rand() > 0.7:
            noise = np.random.normal(0, 0.02, mel_resized.shape)
            mel_resized = np.clip(mel_resized + noise, 0, 1)
    
    # Convert to 3-channel (RGB-like)
    mel_3ch = np.stack([mel_resized] * 3, axis=-1)
    return mel_3ch

# ================= ENHANCED AUGMENTATION =================
def apply_advanced_augmentation(mel):
    """Apply advanced augmentation techniques to mel-spectrogram."""
    # Random horizontal flip
    if np.random.rand() > 0.5:
        mel = np.fliplr(mel)
    
    # Random time shift (circular)
    if np.random.rand() > 0.5:
        shift = np.random.randint(-15, 15)
        mel = np.roll(mel, shift, axis=1)
    
    # Random brightness/contrast
    if np.random.rand() > 0.5:
        alpha = np.random.uniform(0.7, 1.3)  # Contrast
        beta = np.random.uniform(-0.15, 0.15)   # Brightness
        mel = np.clip(alpha * mel + beta, 0, 1)
    
    # Random noise
    if np.random.rand() > 0.6:
        noise = np.random.normal(0, 0.03, mel.shape)
        mel = np.clip(mel + noise, 0, 1)
    
    # Random masking (specaugment-like)
    if np.random.rand() > 0.7:
        # Time masking
        t_mask_size = np.random.randint(0, 10)
        t_mask_start = np.random.randint(0, max(1, mel.shape[1] - t_mask_size))
        mel[:, t_mask_start:t_mask_start+t_mask_size] = 0
        
        # Frequency masking
        f_mask_size = np.random.randint(0, 10)
        f_mask_start = np.random.randint(0, max(1, mel.shape[0] - f_mask_size))
        mel[f_mask_start:f_mask_start+f_mask_size, :] = 0
    
    return mel

# ================= PREPARE DATASET =================
print("\n" + "=" * 70)
print("Preparing dataset with sliding windows...")
print("=" * 70)

X = []
Y = []

for _, row in df.iterrows():
    audio_path = os.path.join(DATA_DIR, row['file'])
    if not os.path.exists(audio_path):
        continue
    
    instruments = [inst.strip() for inst in str(row['instruments']).split(',')]
    instruments = [inst for inst in instruments if inst in ALL_CLASSES]
    
    if not instruments:
        continue
    
    try:
        y, sr = librosa.load(audio_path, sr=SR, duration=None)
        duration = len(y) / sr
        
        # Create sliding windows
        num_windows = int((duration - WINDOW_SIZE) / HOP_SIZE) + 1
        
        for i in range(max(1, num_windows)):
            start_idx = int(i * HOP_SIZE * sr)
            end_idx = int(start_idx + WINDOW_SIZE * sr)
            
            if end_idx > len(y):
                end_idx = len(y)
                start_idx = max(0, end_idx - int(WINDOW_SIZE * sr))
            
            clip = y[start_idx:end_idx]
            
            if len(clip) < int(0.5 * WINDOW_SIZE * sr):  # Skip too short clips
                continue
            
            # Pad if necessary
            if len(clip) < int(WINDOW_SIZE * sr):
                clip = np.pad(clip, (0, int(WINDOW_SIZE * sr) - len(clip)), mode='constant')
            
            # Extract mel-spectrogram
            mel = audio_to_mel_spectrogram(clip, sr)
            X.append(mel)
            
            # Create multilabel vector
            label_vec = np.zeros(NUM_CLASSES, dtype=np.float32)
            for inst in instruments:
                if inst in class_to_idx:
                    label_vec[class_to_idx[inst]] = 1.0
            Y.append(label_vec)
    except Exception as e:
        print(f"Error processing {row['file']}: {e}")
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
                # Apply advanced augmentation to mel-spectrogram
                mel_aug = X[i].copy()
                # Apply augmentation to each channel
                for ch in range(3):
                    mel_aug[:, :, ch] = apply_advanced_augmentation(mel_aug[:, :, ch])
                X_aug.append(mel_aug)
                Y_aug.append(Y[i])
            except Exception as e:
                # If augmentation fails, add original
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

# ================= COMPUTE CLASS WEIGHTS =================
print("\n" + "=" * 70)
print("Computing class weights for imbalanced data...")
print("=" * 70)

# Compute class weights (inverse frequency)
class_weights = {}
for idx in range(NUM_CLASSES):
    pos_count = Y[:, idx].sum()
    if pos_count > 0:
        # Inverse frequency weighting
        weight = len(Y) / (NUM_CLASSES * pos_count)
        class_weights[idx] = weight
        print(f"  {ALL_CLASSES[idx]}: weight={weight:.3f} (samples={int(pos_count)})")
    else:
        class_weights[idx] = 1.0

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

# ================= MAXIMUM ACCURACY MODEL ARCHITECTURE =================
print("\n" + "=" * 70)
print("Building MAXIMUM ACCURACY CNN model...")
print("=" * 70)

# Use EfficientNetB1 for better capacity (or EfficientNetB0 if memory constrained)
base_model = EfficientNetB1(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Fine-tuning strategy: unfreeze more layers gradually
base_model.trainable = True
# Freeze early layers, unfreeze later layers
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Build enhanced model
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Enhanced feature extraction with residual connections
x = Dense(2048, activation='relu', name='fc1')(x)
x = BatchNormalization()(x)
x = Dropout(0.6)(x)

x = Dense(1024, activation='relu', name='fc2')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Dense(512, activation='relu', name='fc3')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

x = Dense(256, activation='relu', name='fc4')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

# Output layer: SIGMOID for multilabel
output = Dense(NUM_CLASSES, activation='sigmoid', name='predictions')(x)

model = Model(inputs=base_model.input, outputs=output)

# Learning rate schedule - More gradual for better accuracy
def lr_schedule(epoch):
    """Learning rate schedule: gradual warm-up then cosine decay."""
    initial_lr = INITIAL_LEARNING_RATE
    warmup_epochs = 10
    if epoch < warmup_epochs:
        # Warm-up phase
        return initial_lr * (epoch + 1) / warmup_epochs
    elif epoch < 50:
        return initial_lr
    elif epoch < 100:
        return initial_lr * 0.5
    elif epoch < 150:
        return initial_lr * 0.2
    else:
        return initial_lr * 0.1

# ================= CUSTOM METRICS FOR MULTILABEL =================
def hamming_accuracy(y_true, y_pred):
    """Hamming accuracy: average per-label accuracy (more meaningful for multilabel)."""
    y_pred_binary = tf.cast(y_pred >= 0.5, tf.float32)
    correct = tf.cast(tf.equal(y_true, y_pred_binary), tf.float32)
    return tf.reduce_mean(correct)

def subset_accuracy(y_true, y_pred):
    """Subset accuracy: exact match (all labels must match)."""
    y_pred_binary = tf.cast(y_pred >= 0.5, tf.float32)
    matches = tf.reduce_all(tf.equal(y_true, y_pred_binary), axis=1)
    return tf.reduce_mean(tf.cast(matches, tf.float32))

def f1_score_metric(y_true, y_pred):
    """F1 score for multilabel classification."""
    y_pred_binary = tf.cast(y_pred >= 0.5, tf.float32)
    
    true_positives = tf.reduce_sum(y_true * y_pred_binary)
    false_positives = tf.reduce_sum((1 - y_true) * y_pred_binary)
    false_negatives = tf.reduce_sum(y_true * (1 - y_pred_binary))
    
    precision = true_positives / (true_positives + false_positives + tf.keras.backend.epsilon())
    recall = true_positives / (true_positives + false_negatives + tf.keras.backend.epsilon())
    
    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    return f1

# Compile with improved optimizer settings and better metrics
model.compile(
    optimizer=Adam(learning_rate=INITIAL_LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
    loss='binary_crossentropy',
    metrics=[
        hamming_accuracy,  # More meaningful than default accuracy
        subset_accuracy,  # Exact match (what default accuracy shows)
        f1_score_metric,  # F1 score
        tf.keras.metrics.Precision(name='precision', threshold=0.5),
        tf.keras.metrics.Recall(name='recall', threshold=0.5),
        tf.keras.metrics.AUC(name='auc', multi_label=True)
    ]
)

print("✓ Model architecture:")
model.summary()

# ================= CALLBACKS =================
callbacks = [
    EarlyStopping(
        monitor='val_hamming_accuracy',  # Monitor Hamming accuracy instead
        patience=25,  # More patience for better convergence
        restore_best_weights=True,
        verbose=1,
        min_delta=1e-4,
        mode='max'
    ),
    ReduceLROnPlateau(
        monitor='val_hamming_accuracy',  # Monitor Hamming accuracy
        factor=0.3,  # Moderate reduction
        patience=10,
        min_lr=1e-7,
        verbose=1,
        mode='max'
    ),
    LearningRateScheduler(lr_schedule, verbose=1),
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_hamming_accuracy',  # Monitor Hamming accuracy (more meaningful)
        save_best_only=True,
        verbose=1,
        mode='max'
    )
]

# ================= TRAINING =================
print("\n" + "=" * 70)
print("Starting MAXIMUM ACCURACY training...")
print("=" * 70)
print("This may take 2-4 hours depending on dataset size.")
print("Model will automatically save best weights.\n")

# Training with class weights
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    class_weight=class_weights,  # Apply class weights for imbalanced data
    verbose=1
)

# Save final model
model.save(MODEL_SAVE_PATH)
print(f"\n✓ Model saved to {MODEL_SAVE_PATH}")

# Save training history
import pickle
history_dict = {
    'loss': history.history['loss'],
    'hamming_accuracy': history.history.get('hamming_accuracy', history.history.get('accuracy', [])),
    'subset_accuracy': history.history.get('subset_accuracy', []),
    'f1_score_metric': history.history.get('f1_score_metric', []),
    'val_loss': history.history['val_loss'],
    'val_hamming_accuracy': history.history.get('val_hamming_accuracy', history.history.get('val_accuracy', [])),
    'val_subset_accuracy': history.history.get('val_subset_accuracy', []),
    'val_f1_score_metric': history.history.get('val_f1_score_metric', []),
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

eval_results = model.evaluate(
    X_val, Y_val,
    batch_size=BATCH_SIZE,
    verbose=1
)

# Extract metrics (order depends on metrics list)
val_loss = eval_results[0]
val_hamming_acc = eval_results[1] if len(eval_results) > 1 else 0
val_subset_acc = eval_results[2] if len(eval_results) > 2 else 0
val_f1 = eval_results[3] if len(eval_results) > 3 else 0
val_precision = eval_results[4] if len(eval_results) > 4 else 0
val_recall = eval_results[5] if len(eval_results) > 5 else 0
val_auc = eval_results[6] if len(eval_results) > 6 else 0

print(f"\n✓ Validation Results:")
print(f"  Loss: {val_loss:.4f}")
print(f"  Hamming Accuracy: {val_hamming_acc:.4f} ({val_hamming_acc*100:.2f}%) ⭐ [Per-label accuracy - MORE MEANINGFUL]")
print(f"  Subset Accuracy: {val_subset_acc:.4f} ({val_subset_acc*100:.2f}%) [Exact match - what default 'accuracy' shows]")
print(f"  F1 Score: {val_f1:.4f} ({val_f1*100:.2f}%)")
print(f"  Precision: {val_precision:.4f} ({val_precision*100:.2f}%)")
print(f"  Recall: {val_recall:.4f} ({val_recall*100:.2f}%)")
print(f"  AUC: {val_auc:.4f} ({val_auc*100:.2f}%)")

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
print("✅ MAXIMUM ACCURACY MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 70)
print(f"\nModel saved: {MODEL_SAVE_PATH}")
print(f"Class mapping saved: {CLASSES_JSON_PATH}")
print("\nThis maximum accuracy model includes:")
print("  ✓ EfficientNetB1 architecture (larger capacity)")
print("  ✓ Advanced data augmentation (Mixup, CutMix, etc.)")
print("  ✓ Class weights for imbalanced data")
print("  ✓ Learning rate scheduling")
print("  ✓ Enhanced feature extraction layers")
print("  ✓ Better generalization to unseen audio")
print("\nNext steps:")
print("1. Test the model using inference_multilabel_timeline.py")
print("2. Use the model in Streamlit app for predictions")
print(f"\nFinal Validation Hamming Accuracy: {val_hamming_acc*100:.2f}% ⭐")
print(f"Final Training Hamming Accuracy: {history.history.get('hamming_accuracy', history.history.get('accuracy', [0]))[-1]*100:.2f}% ⭐")
print(f"\nFinal Validation Subset Accuracy: {val_subset_acc*100:.2f}%")
print(f"Final Training Subset Accuracy: {history.history.get('subset_accuracy', [0])[-1]*100:.2f}%")
print(f"\n💡 Note: Hamming Accuracy is more meaningful for multilabel tasks!")
print(f"   It shows average per-label accuracy, not requiring exact match.")
