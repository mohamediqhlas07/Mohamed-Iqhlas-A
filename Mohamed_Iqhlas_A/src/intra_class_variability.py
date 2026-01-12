import os
import matplotlib.pyplot as plt
from PIL import Image

CLASS_NAME = "flute"  # change to any class
CLASS_DIR = f"spectrogram_dataset/train/{CLASS_NAME}"

files = os.listdir(CLASS_DIR)[:4]

plt.figure(figsize=(8, 6))
for i, file in enumerate(files):
    img = Image.open(os.path.join(CLASS_DIR, file))
    plt.subplot(2, 2, i+1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{CLASS_NAME} sample {i+1}") 

plt.tight_layout()
plt.savefig(f"{CLASS_NAME}_variability.png")
plt.show()
