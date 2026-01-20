from typing import Tuple, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


def get_last_conv_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("No Conv2D layer found for Grad-CAM.")


def compute_gradcam(
    model: tf.keras.Model,
    img_array: np.ndarray,
    last_conv_layer_name: Optional[str] = None,
    class_index: Optional[int] = None,
):
    if last_conv_layer_name is None:
        last_conv_layer = get_last_conv_layer(model)
    else:
        last_conv_layer = model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if class_index is None:
            class_index = int(tf.round(predictions)[0, 0]) if predictions.shape[-1] == 1 else int(tf.argmax(predictions[0]))
        if predictions.shape[-1] == 1:
            loss = predictions[:, 0]
        else:
            loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(
    heatmap: np.ndarray,
    original_image: np.ndarray,
    alpha: float = 0.4,
    cmap: str = "jet",
):
    import cv2

    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET if cmap == "jet" else cv2.COLORMAP_VIRIDIS)
    overlay = cv2.addWeighted(heatmap_color, alpha, original_image, 1 - alpha, 0)
    return overlay[:, :, ::-1]  # BGR -> RGB


def preprocess_img_for_model(img_path: str, target_size: Tuple[int, int], preprocess_fn=None):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    if preprocess_fn is not None:
        x = preprocess_fn(x)
    else:
        x = x / 255.0
    return x


def show_gradcam(
    model: tf.keras.Model,
    img_path: str,
    target_size: Tuple[int, int],
    preprocess_fn=None,
    last_conv_layer_name: Optional[str] = None,
    alpha: float = 0.4,
):
    # Prepare image arrays
    x = preprocess_img_for_model(img_path, target_size, preprocess_fn)
    original = image.load_img(img_path)
    original = image.img_to_array(original).astype(np.uint8)

    # Compute heatmap
    heatmap = compute_gradcam(model, x, last_conv_layer_name=last_conv_layer_name)

    try:
        overlay = overlay_heatmap(heatmap, original)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(original.astype(np.uint8))
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(heatmap, cmap="jet")
        plt.title("Heatmap")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title("Overlay")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    except ImportError:
        # If OpenCV is not available, just show heatmap
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(original.astype(np.uint8))
        plt.title("Original")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(heatmap, cmap="jet")
        plt.title("Heatmap")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


