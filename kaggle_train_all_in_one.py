# ==============================================================================
# 🌾 ALL-IN-ONE CROP MODEL TRAINER FOR KAGGLE GPU (FIXED SHAPE)
# ==============================================================================

!pip install -q tf2onnx onnx onnxruntime kagglehub

import os
import re
import json
import shutil
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tf2onnx
import onnxruntime as ort
import kagglehub

print("1. TensorFlow Version:", tf.__version__)
print("   GPU Status:", tf.config.list_physical_devices('GPU'))

# ------------------------------------------------------------------------------
# 1. DOWNLOAD THE 14 DATASETS DIRECTLY VIA KAGGLEHUB (Skips if already done)
# ------------------------------------------------------------------------------
DATASET_SLUGS = [
    ("Rice", "vbookshelf/rice-leaf-diseases"),
    ("Wheat", "kushagra3204/wheat-plant-diseases"),
    ("Potato", "mgmitesh/potato-disease-detection-dataset"),
    ("Tomato", "kaustubhb999/tomatoleaf"),
    ("Onion", "tejasbargujepatil/onion-diseases"),
    ("Rose", "shuvokumarbasak4004/rose-leaf-disease-dataset"),
    ("Corn_Maize", "smaranjitghose/corn-or-maize-leaf-disease-dataset"),
    ("Soybean", "sivm205/soybean-diseased-leaf-dataset"),
    ("Chickpea", "tejus0017/chickpea-diseases"),
    ("Apple", "mfaizankb/apple-disease-dataset"),
    ("Pepper", "alexanderuzhinskiy/pepper-disease-classification-dataset-doctorp"),
    ("Sugarcane", "nirmalgaud/sugarcane-leaf-disease-dataset"),
    ("Cotton", "nguynphancminh/cotton-leaf-diseases-dataset"),
    ("Indian_Mustard", "bintangpratomo/mustard-leaf-dataset"),
]

BASE_UNIFIED = Path("/kaggle/working/unified_dataset")

# Only fetch & index if unified_dataset doesn't have images yet
existing_imgs = list(BASE_UNIFIED.glob("*/*")) if BASE_UNIFIED.exists() else []
if len(existing_imgs) < 100:
    downloaded_dirs = []
    print("\n2. Fetching datasets into cloud memory...")
    for crop, slug in DATASET_SLUGS:
        try:
            p = kagglehub.dataset_download(slug)
            print(f"   [OK] {crop:15s} -> {slug}")
            downloaded_dirs.append((crop, Path(p)))
        except Exception as e:
            print(f"   [Notice] {crop:15s}: {e}")

    if Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").iterdir():
            if p.is_dir():
                downloaded_dirs.append(("Auto", p))

    if BASE_UNIFIED.exists():
        shutil.rmtree(BASE_UNIFIED)
    BASE_UNIFIED.mkdir(parents=True, exist_ok=True)

    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    total_images = 0
    classes_found = set()

    print("\n3. Organizing & indexing classes...")
    for crop, root_path in downloaded_dirs:
        for img_file in root_path.rglob("*"):
            if img_file.is_file() and img_file.suffix.lower() in IMAGE_EXTS:
                parent_folder = img_file.parent.name
                clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', parent_folder).strip('_')
                if clean_name.lower() in {'train', 'test', 'val', 'valid', 'validation', 'images', 'data'}:
                    clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', img_file.parent.parent.name).strip('_')
                
                class_label = clean_name if "___" in clean_name else f"{crop}___{clean_name}"
                target_class_dir = BASE_UNIFIED / class_label
                target_class_dir.mkdir(exist_ok=True)
                classes_found.add(class_label)
                
                dest = target_class_dir / f"{total_images}_{img_file.name}"
                try:
                    os.symlink(img_file, dest)
                except Exception:
                    shutil.copy2(img_file, dest)
                total_images += 1

    print(f"   [SUCCESS] Indexed {total_images:,} images across {len(classes_found)} classes!")
    sorted_classes = sorted(list(classes_found))
    classes_dict = {str(i): c for i, c in enumerate(sorted_classes)}
    with open("/kaggle/working/classes.json", "w", encoding="utf-8") as f:
        json.dump(classes_dict, f, indent=2)
else:
    print(f"\n2. Using existing indexed dataset with {len(existing_imgs):,} images! (Instant skip)")

# ------------------------------------------------------------------------------
# 3. BUILD KERAS DATA LOADERS (Output shape: 224, 224, 3)
# ------------------------------------------------------------------------------
print("\n4. Initializing Keras DataLoaders...")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_ds = keras.utils.image_dataset_from_directory(
    BASE_UNIFIED,
    validation_split=0.2,
    subset='training',
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    follow_links=True
)

val_ds = keras.utils.image_dataset_from_directory(
    BASE_UNIFIED,
    validation_split=0.2,
    subset='validation',
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    follow_links=True
)

NUM_CLASSES = len(train_ds.class_names)
print(f"   [OK] Detected {NUM_CLASSES} classes in DataLoader!")

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.15)
])

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ------------------------------------------------------------------------------
# 4. BUILD MOBILENETV2 (Training Input: 224, 224, 3 NHWC)
# ------------------------------------------------------------------------------
print("\n5. Building MobileNetV2 architecture...")
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

model = keras.Model(inputs=train_inputs, outputs=train_outputs, name="crop_disease_model")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# ------------------------------------------------------------------------------
# 5. TRAIN MODEL (FAST 6 EPOCHS PHASE 1 + 4 EPOCHS FINE-TUNING)
# ------------------------------------------------------------------------------
print("\n6. Training model on GPU...")
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=6,
    callbacks=callbacks
)

print("\n7. Fine-tuning top 30 layers...")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=4,
    callbacks=callbacks
)

# ------------------------------------------------------------------------------
# 6. EXPORT TO crop_model.onnx WITH NCHW (3, 224, 224) INPUT FOR INFERENCE
# ------------------------------------------------------------------------------
print("\n8. Wrapping model for ONNX export (input shape: 3, 224, 224)...")
onnx_inputs = keras.Input(shape=(3, 224, 224), name="input")
# Permute NCHW (3, 224, 224) -> NHWC (224, 224, 3) for trained backbone
x_perm = layers.Permute((2, 3, 1))(onnx_inputs)
onnx_outputs = model(x_perm)
export_model = keras.Model(inputs=onnx_inputs, outputs=onnx_outputs, name="export_crop_model")

output_onnx_path = "/kaggle/working/crop_model.onnx"
input_signature = [tf.TensorSpec([None, 3, 224, 224], tf.float32, name="input")]

print("Converting to ONNX format...")
onnx_model, _ = tf2onnx.convert.from_keras(
    export_model,
    input_signature=input_signature,
    opset=14,
    output_path=output_onnx_path
)

# Verify with ONNX Runtime
session = ort.InferenceSession(output_onnx_path, providers=["CPUExecutionProvider"])
dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
out = session.run(None, {"input": dummy})
print("   [OK] Inference verified! Output shape:", out[0].shape)

print("\n" + "=" * 80)
print("🎉 ALL DONE! Training & Export Complete!")
print(f"Model saved to: {output_onnx_path}")
print("=" * 80)
