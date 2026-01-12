import numpy as np
import tensorflow as tf

THRESHOLD = 0.5

model = tf.keras.models.load_model(
    "instrument_classifier_multilabel.keras"
)


sample_output = np.array([0.82, 0.55, 0.12, 0.03, 0.61])

binary_prediction = (sample_output >= THRESHOLD).astype(int)

print("Raw probabilities:", sample_output)
print("Binary prediction:", binary_prediction)
