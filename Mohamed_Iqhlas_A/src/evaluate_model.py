import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODEL_PATH = "instrument_classifier.keras"
VAL_DIR = "spectrogram_dataset/val"
IMG_SIZE = (128, 128)
BATCH_SIZE = 16

model = tf.keras.models.load_model(MODEL_PATH)

val_gen = ImageDataGenerator(rescale=1./255)

val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

loss, accuracy = model.evaluate(val_data)

print("\n==============================")
print(f"Validation Accuracy : {accuracy*100:.2f}%")
print(f"Validation Loss     : {loss:.4f}")
print("==============================")
