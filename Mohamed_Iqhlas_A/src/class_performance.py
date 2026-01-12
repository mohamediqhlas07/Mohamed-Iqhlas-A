import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

MODEL_PATH = "instrument_classifier.keras"
VAL_DIR = "spectrogram_dataset/val"
IMG_SIZE = (128, 128)
BATCH_SIZE = 16

model = tf.keras.models.load_model(MODEL_PATH)

datagen = ImageDataGenerator(rescale=1./255)
val_data = datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

preds = model.predict(val_data)
y_pred = np.argmax(preds, axis=1)
y_true = val_data.classes
class_names = list(val_data.class_indices.keys())

report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

with open("classification_report.txt", "w") as f:
    f.write(report)
