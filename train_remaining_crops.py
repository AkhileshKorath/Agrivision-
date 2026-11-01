# ==============================================================================
# 🌿 FAST REMAINING CROPS CLASSIFIER (700 IMAGES/CLASS, BATCH 256)
# Crops: Wheat, Cotton, Sugarcane, Onion, Rose, Chickpea, Mustard
# ==============================================================================
!pip install -q tf2onnx onnx onnxruntime kagglehub

import os
import re
import json
import random
import shutil
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tf2onnx
import onnxruntime as ort
import kagglehub

random.seed(42)
np.random.seed(42)

print("1. TensorFlow:", tf.__version__)
print("   GPU Status:", tf.config.list_physical_devices('GPU'))

# ------------------------------------------------------------------------------
# 1. ATTACH THE 7 REMAINING DATASETS (Already in Kaggle cache)
# ------------------------------------------------------------------------------
REMAINING_DATASETS = [
    ("Wheat", "kushagra3204/wheat-plant-diseases"),
    ("Cotton", "nguynphancminh/cotton-leaf-diseases-dataset"),
    ("Sugarcane", "nirmalgaud/sugarcane-leaf-disease-dataset"),
    ("Onion", "tejasbargujepatil/onion-diseases"),
    ("Rose", "shuvokumarbasak4004/rose-leaf-disease-dataset"),
    ("Chickpea", "tejus0017/chickpea-diseases"),
    ("Indian_Mustard", "bintangpratomo/mustard-leaf-dataset"),
]

downloaded_dirs = []
print("\n2. Checking cached datasets...")
for crop, slug in REMAINING_DATASETS:
    try:
        p = kagglehub.dataset_download(slug)
        downloaded_dirs.append((crop, Path(p)))
    except Exception as e:
        print(f"   [Notice] {crop}: {e}")

if Path("/kaggle/input").exists():
    for p in Path("/kaggle/input").iterdir():
        if p.is_dir():
            downloaded_dirs.append(("Auto", p))

# ------------------------------------------------------------------------------
# 2. FILTER & BALANCE (MAX 700 IMAGES PER CLASS)
# ------------------------------------------------------------------------------
BASE_BALANCED = Path("/kaggle/working/remaining_crops_dataset")
if BASE_BALANCED.exists():
    shutil.rmtree(BASE_BALANCED)
BASE_BALANCED.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
GARBAGE_TERMS = {
    'annotation', 'pixel', 'example', 'belum', 'siap', 'aug_',
    '_test', '_valid', 'mask', 'image', 'dataset', 'data'
}

raw_class_images = {}
print("\n3. Collecting disease classes...")

for crop, root_path in downloaded_dirs:
    for img_file in root_path.rglob("*"):
        if img_file.is_file() and img_file.suffix.lower() in IMAGE_EXTS:
            parent_folder = img_file.parent.name
            clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', parent_folder).strip('_')
            
            if clean_name.lower() in {'train', 'test', 'val', 'valid', 'validation', 'images', 'data'}:
                clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', img_file.parent.parent.name).strip('_')
            
            if any(term in clean_name.lower() for term in GARBAGE_TERMS):
                continue
            
            class_label = clean_name if "___" in clean_name else f"{crop}___{clean_name}"
            
            if any(term in class_label.lower() for term in GARBAGE_TERMS):
                continue

            if class_label not in raw_class_images:
                raw_class_images[class_label] = []
            raw_class_images[class_label].append(img_file)

# MAX 700 IMAGES PER CLASS
MAX_PER_CLASS = 700
total_balanced = 0
valid_classes = []

for class_label, img_list in sorted(raw_class_images.items()):
    if len(img_list) < 5:
        continue
    
    target_dir = BASE_BALANCED / class_label
    target_dir.mkdir(exist_ok=True)
    valid_classes.append(class_label)
    
    selected_imgs = random.sample(img_list, min(len(img_list), MAX_PER_CLASS))
    for idx, img_file in enumerate(selected_imgs):
        dest = target_dir / f"{idx}_{img_file.name}"
        try:
            os.symlink(img_file, dest)
        except Exception:
            shutil.copy2(img_file, dest)
        total_balanced += 1

print(f"   [SUCCESS] Created Dataset (Max 700/class):")
print(f"   -> Total Classes: {len(valid_classes)}")
print(f"   -> Total Images:  {total_balanced:,}")

# Save matching remaining_classes.json
classes_dict = {str(i): c for i, c in enumerate(sorted(valid_classes))}
with open("/kaggle/working/remaining_classes.json", "w", encoding="utf-8") as f:
    json.dump(classes_dict, f, indent=2)
print("   -> Saved clean /kaggle/working/remaining_classes.json")

# ------------------------------------------------------------------------------
# 3. BUILD FAST DATA LOADERS (BATCH_SIZE = 256)
# ------------------------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 256

train_ds = keras.utils.image_dataset_from_directory(
    BASE_BALANCED,
    validation_split=0.2,
    subset='training',
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    follow_links=True
)

val_ds = keras.utils.image_dataset_from_directory(
    BASE_BALANCED,
    validation_split=0.2,
    subset='validation',
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    follow_links=True
)

NUM_CLASSES = len(train_ds.class_names)
print(f"   [OK] Active classes: {NUM_CLASSES}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ------------------------------------------------------------------------------
# 4. BUILD MOBILENETV2
# ------------------------------------------------------------------------------
train_inputs = keras.Input(shape=(224, 224, 3), name="train_input")

base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

x = keras.applications.mobilenet_v2.preprocess_input(train_inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
train_outputs = layers.Dense(NUM_CLASSES, name="output")(x)

model = keras.Model(inputs=train_inputs, outputs=train_outputs, name="remaining_crops_model")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1.5e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# ------------------------------------------------------------------------------
# 5. FAST GPU TRAINING (Total ~3-4 Minutes)
# ------------------------------------------------------------------------------
print("\n6. Training Phase 1 (Transfer Learning - 6 epochs)...")
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
]

history = model.fit(train_ds, validation_data=val_ds, epochs=6, callbacks=callbacks)

print("\n7. Training Phase 2 (Fine-tuning top 40 layers - 5 epochs)...")
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=2e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

history_fine = model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=callbacks)

# ------------------------------------------------------------------------------
# 6. EXPORT TO ONNX
# ------------------------------------------------------------------------------
print("\n8. Exporting clean remaining_crops.onnx...")
onnx_inputs = keras.Input(shape=(3, 224, 224), name="input")
x_perm = layers.Permute((2, 3, 1))(onnx_inputs)
onnx_outputs = model(x_perm)
export_model = keras.Model(inputs=onnx_inputs, outputs=onnx_outputs, name="export_remaining_crops")

output_onnx_path = "/kaggle/working/remaining_crops.onnx"
input_signature = [tf.TensorSpec([None, 3, 224, 224], tf.float32, name="input")]

onnx_model, _ = tf2onnx.convert.from_keras(
    export_model,
    input_signature=input_signature,
    opset=14,
    output_path=output_onnx_path
)

# Verify
session = ort.InferenceSession(output_onnx_path, providers=["CPUExecutionProvider"])
dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
out = session.run(None, {"input": dummy})
print("   [OK] Verification done! Output shape:", out[0].shape)

print("\n" + "=" * 80)
print("🎉 SUCCESS! REMAINING CROPS MODEL READY!")
print(f"1. Model file:   /kaggle/working/remaining_crops.onnx")
print(f"2. Classes file: /kaggle/working/remaining_classes.json")
print("=" * 80)
