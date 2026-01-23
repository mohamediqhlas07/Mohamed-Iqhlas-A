import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
import pickle
import os
import numpy as np

# ======================
# Dataset paths (UNGROUPED – 28 classes)
# ======================
TRAIN_DIR = "spectrogram_dataset/train"
VAL_DIR   = "spectrogram_dataset/val"

# ======================
# Training parameters
# ======================
IMG_SIZE = (128, 128)
BATCH_SIZE = 32  # Increased batch size for better stability
EPOCHS = 50  # More epochs with early stopping
INITIAL_LR = 0.001  # Higher initial learning rate

# ======================
# Data generators with AUGMENTATION
# ======================
# Training with augmentation
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,  # Rotate images slightly
    width_shift_range=0.1,  # Horizontal shift
    height_shift_range=0.1,  # Vertical shift
    shear_range=0.1,  # Shear transformation
    zoom_range=0.1,  # Zoom in/out
    horizontal_flip=True,  # Flip horizontally
    fill_mode='nearest'  # Fill mode for transformations
)

# Validation without augmentation (only rescaling)
val_gen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = train_data.num_classes
print(f"Number of classes: {NUM_CLASSES}")

# ======================
# Model: EfficientNetB0 (better than MobileNetV2)
# ======================
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(128, 128, 3)
)

# Freeze base model initially
base_model.trainable = False

# Build model with better architecture
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# ======================
# Compile model
# ======================
model.compile(
    optimizer=Adam(learning_rate=INITIAL_LR),
    loss="categorical_crossentropy",
    metrics=["accuracy", "top_3_accuracy"]
)

# Print model summary
model.summary()

# ======================
# Callbacks for better training
# ======================
callbacks = [
    # Early stopping to prevent overfitting
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,  # Wait 10 epochs without improvement
        restore_best_weights=True,
        verbose=1
    ),
    
    # Save best model
    ModelCheckpoint(
        'instrument_classifier_best.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    
    # Reduce learning rate when validation accuracy plateaus
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,  # Reduce LR by half
        patience=5,  # Wait 5 epochs
        min_lr=1e-7,
        verbose=1
    ),
    
    # Log training history
    CSVLogger('training_log.csv', append=False)
]

# ======================
# Phase 1: Train with frozen base model
# ======================
print("\n" + "="*50)
print("PHASE 1: Training with frozen base model")
print("="*50)

history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ======================
# Phase 2: Fine-tune (unfreeze top layers)
# ======================
print("\n" + "="*50)
print("PHASE 2: Fine-tuning (unfreezing top layers)")
print("="*50)

# Unfreeze top layers of base model
base_model.trainable = True

# Freeze bottom layers, unfreeze top layers
for layer in base_model.layers[:-30]:  # Freeze all but last 30 layers
    layer.trainable = False

# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=INITIAL_LR * 0.1),  # 10x smaller LR for fine-tuning
    loss="categorical_crossentropy",
    metrics=["accuracy", "top_3_accuracy"]
)

# Fine-tune callbacks
fine_tune_callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'instrument_classifier_best.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        verbose=1
    ),
    CSVLogger('training_log.csv', append=True)
]

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,  # Additional epochs for fine-tuning
    callbacks=fine_tune_callbacks,
    verbose=1
)

# ======================
# Combine training histories
# ======================
combined_history = {
    'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
    'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
    'loss': history1.history['loss'] + history2.history['loss'],
    'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
    'top_3_accuracy': history1.history.get('top_3_accuracy', []) + history2.history.get('top_3_accuracy', []),
    'val_top_3_accuracy': history1.history.get('val_top_3_accuracy', []) + history2.history.get('val_top_3_accuracy', [])
}

# ======================
# Save model and history
# ======================
model.save("instrument_classifier.keras")
print("\n✓ Model saved as instrument_classifier.keras")

# Save best model
if os.path.exists('instrument_classifier_best.keras'):
    import shutil
    shutil.copy('instrument_classifier_best.keras', 'instrument_classifier.keras')
    print("✓ Best model copied to instrument_classifier.keras")

# Save training history
with open('training_history.pkl', 'wb') as f:
    pickle.dump(combined_history, f)
print("✓ Training history saved as training_history.pkl")

# Print final metrics
print("\n" + "="*50)
print("FINAL TRAINING RESULTS")
print("="*50)
print(f"Final Training Accuracy: {combined_history['accuracy'][-1]:.4f} ({combined_history['accuracy'][-1]*100:.2f}%)")
print(f"Final Validation Accuracy: {combined_history['val_accuracy'][-1]:.4f} ({combined_history['val_accuracy'][-1]*100:.2f}%)")
print(f"Best Validation Accuracy: {max(combined_history['val_accuracy']):.4f} ({max(combined_history['val_accuracy'])*100:.2f}%)")
print(f"Final Training Loss: {combined_history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {combined_history['val_loss'][-1]:.4f}")
print("="*50)