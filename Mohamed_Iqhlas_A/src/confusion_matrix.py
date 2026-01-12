import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns

MODEL_PATH = "instrument_classifier.keras"
VAL_DIR = "spectrogram_dataset/val"

model = tf.keras.models.load_model(MODEL_PATH)

datagen = ImageDataGenerator(rescale=1./255)
val_data = datagen.flow_from_directory(
    VAL_DIR,
    target_size=(128, 128),
    batch_size=16,
    class_mode="categorical",
    shuffle=False
)

preds = model.predict(val_data)
y_pred = np.argmax(preds, axis=1)
y_true = val_data.classes
labels = list(val_data.class_indices.keys())

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, xticklabels=labels, yticklabels=labels, cmap="Blues", fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
