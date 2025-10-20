from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Import preprocess cho từng model
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficient_preprocess

app = Flask(__name__)
CORS(app)

# Cấu hình mô hình
MODEL_DIR = "./"
AVAILABLE_MODELS = {
    "cnn": "pneumonia_cnn_model.keras",
    "resnet": "pneumonia_resnet50.keras",
    "densenet": "pneumonia_densenet121.keras",
    "efficientnet": "EfficientNetB3.keras"
}

# Cache model
loaded_models = {}


def get_model(model_key):
    if model_key not in AVAILABLE_MODELS:
        return None, f"Model '{model_key}' không tồn tại. Hợp lệ: {list(AVAILABLE_MODELS.keys())}"

    if model_key not in loaded_models:
        model_path = os.path.join(MODEL_DIR, AVAILABLE_MODELS[model_key])
        if not os.path.exists(model_path):
            return None, f"Không tìm thấy file model: {model_path}"

        print(f"Đang load model: {model_path}")
        loaded_models[model_key] = tf.keras.models.load_model(model_path, compile=False)

    return loaded_models[model_key], None


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Pneumonia Classification is running!",
        "available_models": list(AVAILABLE_MODELS.keys())
    })


@app.route("/predict", methods=["POST"])
def predict():
    model_key = request.form.get("model")
    if not model_key:
        return jsonify({"error": "Thiếu tham số 'model' trong form-data"}), 400

    model, err = get_model(model_key)
    if err:
        return jsonify({"error": err}), 400

    if "file" not in request.files:
        return jsonify({"error": "Không có file ảnh được gửi"}), 400

    file = request.files["file"]
    try:
        img = Image.open(io.BytesIO(file.read()))
        # print("Ảnh gốc:", img.mode, img.size)

        # Xử lý ảnh theo model
        img = img.convert("RGB")

        if model_key == "cnn":
            target_size = (150, 150)
            img = img.resize(target_size)
            x = np.array(img) / 255.0
            x = np.expand_dims(x, axis=0)

        elif model_key == "resnet":
            target_size = (224, 224)
            img = img.resize(target_size)
            x = np.expand_dims(np.array(img), axis=0)
            x = resnet_preprocess(x)

        elif model_key == "densenet":
            target_size = (224, 224)
            img = img.resize(target_size)
            x = np.expand_dims(np.array(img), axis=0)
            x = densenet_preprocess(x)

        elif model_key == "efficientnet":
            target_size = (300, 300)
            img = img.resize(target_size)
            x = np.expand_dims(np.array(img), axis=0)
            x = efficient_preprocess(x)

        else:
            return jsonify({"error": f"Model '{model_key}' không được hỗ trợ"}), 400

        # print(f"Input shape vào model {model_key}: {x.shape}")

    except Exception as e:
        return jsonify({"error": f"Lỗi xử lý ảnh: {e}"}), 400

    # Dự đoán
    preds = model.predict(x)
    prob = float(preds[0][0])
    result = "Viêm phổi" if prob > 0.5 else "Bình thường"

    return jsonify({
        "prediction": result,
        "probability": prob,
        "model_used": model_key
    })

if __name__ == "__main__":
    print("🚀 Flask server đang chạy tại http://127.0.0.1:5000/")
    app.run(debug=True, host="0.0.0.0", port=5000)
