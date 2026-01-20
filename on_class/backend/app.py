import base64
import io
import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficient_preprocess
# Preprocess cho từng model
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

app = Flask(__name__)
CORS(app)

# Cấu hình model
MODEL_DIR = "./"
AVAILABLE_MODELS = {
    "cnn": "pneumonia_cnn_model.keras",
    "resnet": "pneumonia_resnet50.keras",
    "densenet": "pneumonia_densenet121_fixed.keras",
    "efficientnet": "EfficientNetB3.keras"
}

loaded_models = {}

# Load model có cache
def get_model(model_key):
    if model_key not in AVAILABLE_MODELS:
        return None, f"Model '{model_key}' không tồn tại."
    if model_key not in loaded_models:
        path = os.path.join(MODEL_DIR, AVAILABLE_MODELS[model_key])
        if not os.path.exists(path):
            return None, f"Không tìm thấy file model: {path}"
        print(f"Đang load model: {path}")
        loaded_models[model_key] = tf.keras.models.load_model(path, compile=False)
    return loaded_models[model_key], None


# Tiện ích
def ensure_4d_array(x):
    while isinstance(x, (list, tuple)):
        x = x[0]
    x = np.array(x)
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)
    return x.astype(np.float32)


# Grad-CAM chuẩn cho ResNet/DenseNet/EfficientNet
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    x = ensure_4d_array(img_array)

    def force_tensor(y):
        if isinstance(y, (list, tuple)):
            y = y[0]
        if not tf.is_tensor(y):
            y = tf.convert_to_tensor(y)
        return y

    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except Exception as e:
        raise ValueError(f"Không tìm thấy layer: {last_conv_layer_name} ({e})")

    model_input = model.inputs
    while isinstance(model_input, (list, tuple)):
        model_input = model_input[0]

    grad_model = tf.keras.models.Model(
        inputs=model_input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tf.convert_to_tensor(x))
        conv_outputs = force_tensor(conv_outputs)
        predictions = force_tensor(predictions)

        if predictions.shape[-1] == 1:
            class_output = predictions[:, 0]
        else:
            class_idx = tf.argmax(predictions[0])
            class_output = predictions[:, class_idx]

    grads = tape.gradient(class_output, conv_outputs)
    if grads is None:
        raise ValueError("Không tính được gradient – kiểm tra layer cuối cùng.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
    return heatmap.numpy()


# Grad-CAM riêng cho CNN
def make_gradcam_cnn(img_array, model):
    x = ensure_4d_array(img_array)
    x = tf.convert_to_tensor(x, dtype=tf.float32)

    # bóc lớp Sequential bên trong
    seq_model = None
    for l in model.layers:
        if isinstance(l, tf.keras.Sequential):
            seq_model = l
            break
    if seq_model is None:
        seq_model = model
    print(f"Dùng model con: {seq_model.name}")

    conv_layers = [l for l in seq_model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    if not conv_layers:
        raise ValueError(f"Không tìm thấy Conv2D trong model. Layers: {[l.name for l in seq_model.layers]}")
    first_conv, last_conv = conv_layers[0], conv_layers[-1]
    print(f"CNN tự phát hiện conv cuối: {last_conv.name}")

    in_ch_required = int(first_conv.kernel.shape[2])
    cur_ch = int(x.shape[-1])
    if in_ch_required == 1 and cur_ch == 3:
        x = tf.image.rgb_to_grayscale(x)
    elif in_ch_required == 3 and cur_ch == 1:
        x = tf.tile(x, [1, 1, 1, 3])

    H, W, C = int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
    input_tensor = tf.keras.Input(shape=(H, W, C), dtype=tf.float32, name="cnn_input")

    z = input_tensor
    conv_target_output = None
    for layer in seq_model.layers:
        z = layer(z)
        if layer.name == last_conv.name:
            conv_target_output = z
    final_output = z

    if conv_target_output is None:
        raise ValueError("Không lấy được output của layer conv cuối!")

    grad_model = tf.keras.models.Model(
        inputs=input_tensor,
        outputs=[conv_target_output, final_output],
        name="cnn_grad_manual"
    )

    # tính Grad-CAM
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x, training=False)
        if predictions.shape[-1] == 1:
            class_output = predictions[:, 0]
        else:
            class_idx = tf.argmax(predictions[0])
            class_output = predictions[:, class_idx]

    grads = tape.gradient(class_output, conv_outputs)
    if grads is None:
        raise ValueError("Không tính được gradient – kiểm tra layer cuối cùng.")
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
    return heatmap.numpy()


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Pneumonia Classification API is running!",
        "available_models": list(AVAILABLE_MODELS.keys())
    })


@app.route("/predict", methods=["POST"])
def predict():
    model_key = request.form.get("model")
    if not model_key:
        return jsonify({"error": "Thiếu tham số 'model'"}), 400

    model, err = get_model(model_key)
    if err:
        return jsonify({"error": err}), 400

    if "file" not in request.files:
        return jsonify({"error": "Không có file ảnh"}), 400

    file = request.files["file"]
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")

        # Tiền xử lý theo model
        if model_key == "cnn":
            size = (150, 150)
            x = np.expand_dims(np.array(img.resize(size)) / 255.0, axis=0)
        elif model_key == "resnet":
            size = (224, 224)
            x = np.expand_dims(np.array(img.resize(size)), axis=0)
            x = resnet_preprocess(x)
        elif model_key == "densenet":
            size = (224, 224)
            x = np.expand_dims(np.array(img.resize(size)), axis=0)
            x = densenet_preprocess(x)
        elif model_key == "efficientnet":
            size = (300, 300)
            x = np.expand_dims(np.array(img.resize(size)), axis=0)
            x = efficient_preprocess(x)
        else:
            return jsonify({"error": f"Model '{model_key}' không được hỗ trợ"}), 400

        preds = model(tf.convert_to_tensor(x), training=False)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        prob = float(preds[0][0])
        result = "Viêm phổi" if prob > 0.5 else "Bình thường"

        # Grad-CAM
        gradcam_base64 = None
        heatmap = None

        if model_key == "cnn":
            last_conv_layer_name = "conv2d_8"
            heatmap = make_gradcam_cnn(x, model)
        elif model_key == "resnet":
            last_conv_layer_name = "conv5_block3_out"
            heatmap = make_gradcam_heatmap(x, model, last_conv_layer_name)
        elif model_key == "densenet":
            last_conv_layer_name = "conv5_block16_concat"
            heatmap = make_gradcam_heatmap(x, model, last_conv_layer_name)
        elif model_key == "efficientnet":
            last_conv_layer_name = "top_conv"
            heatmap = make_gradcam_heatmap(x, model, last_conv_layer_name)

        if heatmap is not None:
            heatmap_resized = cv2.resize(heatmap, size)
            heatmap_colored = np.uint8(255 * heatmap_resized)
            heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
            original_cv = np.array(img.resize(size))[:, :, ::-1]
            superimposed = cv2.addWeighted(original_cv, 0.6, heatmap_colored, 0.4, 0)
            superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode(".jpg", superimposed_rgb)
            gradcam_base64 = base64.b64encode(buffer).decode("utf-8")

        response = {
            "prediction": result,
            "probability": round(prob, 4),
            "model_used": model_key
        }
        if gradcam_base64:
            response["gradcam"] = gradcam_base64

        return jsonify(response)

    except Exception as e:
        print("Lỗi xử lý ảnh:", e)
        return jsonify({"error": str(e)}), 500


# Run server
if __name__ == "__main__":
    print("🚀 Flask server đang chạy tại http://127.0.0.1:5000/")
    app.run(debug=True, host="0.0.0.0", port=5000)
