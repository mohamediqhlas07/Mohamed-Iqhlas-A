import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ======================
# Dataset paths (UNGROUPED – 28 classes)
# ======================
TRAIN_DIR = "spectrogram_dataset/train"
VAL_DIR   = "spectrogram_dataset/val"

# ======================
# Training parameters
# ======================
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 30   # ✅ FINAL: 30 epochs

# ======================
# Data generators
# ======================
train_gen = ImageDataGenerator(rescale=1.0 / 255)
val_gen   = ImageDataGenerator(rescale=1.0 / 255)

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

NUM_CLASSES = train_data.num_classes

# ======================
# Model: MobileNetV2
# ======================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(128, 128, 3)
)

base_model.trainable = False  # transfer learning

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# ======================
# Compile model
# ======================
model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ======================
# Train model
# ======================
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ======================
# Save model
# ======================
model.save("instrument_classifier.keras")
