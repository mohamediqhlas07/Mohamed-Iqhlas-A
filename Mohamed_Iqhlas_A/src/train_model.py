import os
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ================= PATHS =================
TRAIN_DIR = "spectrogram_dataset/train"
VAL_DIR = "spectrogram_dataset/val"

# ================= CONFIG =================
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 25          

NUM_CLASSES = len(os.listdir(TRAIN_DIR))
print("Number of classes:", NUM_CLASSES)

# ================= DATA GENERATORS =================
train_gen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# ================= MODEL =================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(128, 128, 3)
)

base_model.trainable = False  # IMPORTANT

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ================= CALLBACKS =================
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3)
]

# ================= TRAIN =================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ================= SAVE MODEL =================
model.save("instrument_classifier.keras")

# ================= SAVE TRAINING HISTORY =================
with open("training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("✅ Training completed")
print("✅ Model saved as instrument_classifier.keras")
print("✅ Training history saved as training_history.pkl")
