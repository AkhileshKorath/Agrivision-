import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tf2onnx
import onnx
import onnxruntime as ort

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)

NUM_CLASSES = 78
IMG_SHAPE = (224, 224, 3)

# Build MobileNetV2 / MobileNetV3 backbone for the 16 crops
base_model = keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# Build model accepting NCHW or NHWC
# In PyTorch / ONNX convention: (batch, 3, 224, 224)
inputs = keras.Input(shape=(3, 224, 224), name="input")

# Permute (batch, 3, 224, 224) -> (batch, 224, 224, 3) to match Keras backbone
x = layers.Permute((2, 3, 1))(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, name="output")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="crop_classifier_16")
model.summary()

# Convert to ONNX
output_onnx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crop_model.onnx")

# Input signature: float32 [None, 3, 224, 224]
input_signature = [tf.TensorSpec([None, 3, 224, 224], tf.float32, name="input")]

onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    opset=14,
    output_path=output_onnx_path
)

print(f"Successfully saved ONNX model to: {output_onnx_path}")

# Verify with ONNX Runtime
session = ort.InferenceSession(output_onnx_path, providers=["CPUExecutionProvider"])
dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
out = session.run(None, {"input": dummy})
print("Inference output shape:", out[0].shape)
print("Inference verified successfully!")
