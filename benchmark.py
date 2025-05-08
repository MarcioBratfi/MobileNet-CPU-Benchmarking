### benchmark.py

import os
import time
import csv
import random
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions

# === Settings ===
SEED = 42
N = 100  # Number of images to process
image_folder = "tiny-imagenet-200/val/images"
output_csv = f"mobilenet_results_seed{SEED}_{int(time.time())}.csv"

# === Set seeds for reproducibility ===
random.seed(SEED)
np.random.seed(SEED)

# === Load and shuffle image list ===
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.JPEG')]
random.shuffle(image_files)
image_files = image_files[:N]

# === Load MobileNet model ===
print("Loading MobileNet...")
model = MobileNet(weights="imagenet")

# === Warm-up ===
for warmup_path in image_files:
    try:
        img = Image.open(warmup_path).convert("RGB").resize((224, 224))
        x = preprocess_input(np.expand_dims(np.array(img), axis=0))
        model.predict(x)
        break
    except Exception:
        continue

print(f"\nRunning benchmark on {N} randomly selected images...")
print(f"Seed used: {SEED}")
print(f"Saving results to: {output_csv}\n")

# === Run Inference ===
successful_inferences = 0
results = []
start_time = time.time()

for img_path in image_files:
    try:
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        x = preprocess_input(np.expand_dims(np.array(img), axis=0))

        t0 = time.time()
        preds = model.predict(x, verbose=0)
        t1 = time.time()

        decoded = decode_predictions(preds, top=1)[0][0]

        results.append([
            os.path.basename(img_path),
            decoded[1],              # Predicted class
            round(decoded[2] * 100, 2),  # Confidence %
            round(t1 - t0, 4)        # Inference time in seconds
        ])

        print(f"{os.path.basename(img_path)} → {decoded[1]} ({decoded[2]*100:.2f}%) - {round(t1 - t0, 4)} sec")
        successful_inferences += 1

    except Exception as e:
        print(f"Skipped {img_path}: {e}")

end_time = time.time()

# === Save results to CSV ===
with open(output_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Seed Used", SEED])
    writer.writerow(["Image", "Predicted Class", "Confidence (%)", "Inference Time (s)"])
    writer.writerows(results)

# === Summary ===
print(f"\nBenchmark complete!")
print(f"Total time for {successful_inferences} inferences: {end_time - start_time:.2f} seconds")
print(f"Average time per inference: {(end_time - start_time)/successful_inferences:.4f} seconds")
print(f"Results saved to {output_csv}")
