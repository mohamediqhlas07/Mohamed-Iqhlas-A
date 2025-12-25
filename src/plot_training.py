import pickle
import matplotlib.pyplot as plt
import os

HISTORY_PATH = "training_history.pkl"

with open(HISTORY_PATH, "rb") as f:
    history = pickle.load(f)

epochs = range(1, len(history["accuracy"]) + 1)

# -------- Accuracy Plot --------
plt.figure(figsize=(8, 6))
plt.plot(epochs, history["accuracy"], label="Training Accuracy")
plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_plot.png")
plt.close()

# -------- Loss Plot --------
plt.figure(figsize=(8, 6))
plt.plot(epochs, history["loss"], label="Training Loss")
plt.plot(epochs, history["val_loss"], label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_plot.png")
plt.close()

print("✅ accuracy_plot.png and loss_plot.png generated successfully")
