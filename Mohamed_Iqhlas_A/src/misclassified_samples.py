import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODEL_PATH = "instrument_classifier.keras"
VAL_DIR = "spectrogram_dataset/val"
IMG_SIZE = (128, 128)
BATCH_SIZE = 16

model = tf.keras.models.load_model(MODEL_PATH)

datagen = ImageDataGenerator(rescale=1./255)
val_gen = datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode="categorical"
)

class_names = list(val_gen.class_indices.keys())

preds = model.predict(val_gen)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes

misclassified = np.where(y_pred != y_true)[0]

print("Showing first 10 misclassified samples:\n")
for idx in misclassified[:10]:
    print(
        f"File: {val_gen.filenames[idx]} | "
        f"True: {class_names[y_true[idx]]} | "
        f"Predicted: {class_names[y_pred[idx]]}"
    )
