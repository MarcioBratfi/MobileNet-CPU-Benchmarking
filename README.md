# MobileNet CPU Benchmarking

Welcome to the MobileNet CPU Benchmarking repository! This project was developed for **CS3339 (Spring 2025)** to evaluate AI inference performance on different CPU architectures (Intel, AMD, and ARM via AWS EC2).

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/9/93/Amazon_Web_Services_Logo.svg" width="120"/>
  <img src="https://upload.wikimedia.org/wikipedia/commons/7/7d/Intel_logo_%282006-2020%29.svg" width="120"/>
  <img src="https://upload.wikimedia.org/wikipedia/commons/7/7c/AMD_Logo.svg" width="100"/>
  <img src="https://upload.wikimedia.org/wikipedia/commons/7/77/Arm_logo_2017.svg" width="80"/>
</p>

---

## Repository Overview

This repository contains a self-contained Python script to benchmark image classification using [MobileNet](https://keras.io/api/applications/mobilenet/) on CPUs. It was developed for performance comparison across Intel, AMD, and ARM CPU architectures on AWS EC2.

---

## What This Code Does

- Loads **100 random JPEG images** from the Tiny ImageNet dataset.
- Preprocesses them for MobileNet input format (224×224 RGB).
- Runs inference using TensorFlow's pre-trained MobileNet model.
- Measures:
  - Inference time per image
  - Prediction confidence and label
- Saves results to a timestamped CSV.

---

## Code Overview

### Settings
<pre>
SEED = 42
N = 100
image_folder = "tiny-imagenet-200/val/images"
</pre>

- Uses a fixed seed for reproducibility
- Reads 100 random .JPEG images from the specified dataset path

### Load and Shuffle Images
<pre>
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.JPEG')]
random.shuffle(image_files)
image_files = image_files[:N]
</pre>

- Filters only .JPEG files
- Randomly selects a consistent subset using the seed

### Load The Model
<pre>
model = MobileNet(weights="imagenet")
</pre>

- Loads the standard MobileNet with pre-trained ImageNet weights

### Warm-Up
<pre>
img = Image.open(...).resize(...)
x = preprocess_input(np.expand_dims(np.array(img), axis=0))
model.predict(x)
</pre>

- Performs a warm-up prediction to initialize the model and TensorFlow runtime

### Inference Loop
<pre>
for img_path in image_files:
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    x = preprocess_input(np.expand_dims(np.array(img), axis=0))
    preds = model.predict(x)
</pre>
- For each image:
  - Opens and resizes to 224x224
  - Converts to NumPy array and preprocesses for MobileNet
  - Executes model.predict()

### Prediction Decoding
<pre>
decoded = decode_predictions(preds, top=1)[0][0]
</pre>

- Converts raw logits into human-readable labels with confidence scores

### Output CSV
<pre>
writer.writerow(["Image", "Predicted Class", "Confidence (%)", "Inference Time (s)"])
writer.writerows(results)
</pre>

- Stores each prediction in CSV:
  - Filename
  - Top-1 class label
  - Confidence (as %)
  - Inference time
 
### Running it
<pre>
python benchmark.py
</pre>
- Results are printed to the console and saved to a timestamped CSV file in the same directory
