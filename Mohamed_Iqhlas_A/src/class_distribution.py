import os
import matplotlib.pyplot as plt

TRAIN_DIR = "spectrogram_dataset/train"


class_counts = {}

for class_name in os.listdir(TRAIN_DIR):
    class_path = os.path.join(TRAIN_DIR, class_name)
    if os.path.isdir(class_path):
        class_counts[class_name] = len(os.listdir(class_path))

# Print counts
print("Class Distribution:")
for k, v in class_counts.items():
    print(f"{k}: {v}")

# Plot
plt.figure(figsize=(12, 5))
plt.bar(class_counts.keys(), class_counts.values())
plt.xticks(rotation=90)
plt.title("Class Distribution")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.show()
