import React, { useState } from "react";
import {
  Upload,
  Button,
  Card,
  message,
  Image,
  Spin,
  Typography,
  Space,
  Select,
} from "antd";
import {
  UploadOutlined,
  ReloadOutlined,
  ExperimentOutlined,
} from "@ant-design/icons";
import axios from "axios";
import Footerapp from "./components/footer";

const { Title, Text } = Typography;

function App() {
  const [file, setFile] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [model, setModel] = useState("cnn");
  const [messageApi, contextHolder] = message.useMessage();

  const handleSelect = ({ file }) => {
    setFile(file);
    setImageUrl(URL.createObjectURL(file));
    setPrediction(null);
  };

  const handlePredict = async () => {
    if (!file) {
      messageApi.error("Vui lòng tải ảnh X-quang!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("model", model); // gửi loại model lên backend

    setLoading(true);
    try {
      const res = await axios.post("http://localhost:5000/predict", formData);
      setPrediction(res.data);
    } catch (err) {
      console.error(err);
      messageApi.error("Lỗi khi gửi ảnh đến server.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setImageUrl(null);
    setPrediction(null);
  };

  return (
    <>
      {contextHolder}
      <div
        style={{
          minHeight: "100px",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          background: "linear-gradient(135deg, #f0f5ff, #e6f7ff)",
        }}
      >
        <Card
          style={{
            width: 900,
            textAlign: "center",
            boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
            borderRadius: 16,
            padding: "40px 20px",
          }}
        >
          <Title level={2} style={{ color: "#1677ff", marginBottom: 12 }}>
            Dự đoán Viêm phổi từ Ảnh X-quang
          </Title>

          <Title level={4} style={{ marginBottom: 8, color: "#555" }}>
            Lựa chọn mô hình
          </Title>

          <Select
            value={model}
            style={{ width: 240, marginBottom: 24 }}
            onChange={setModel}
          >
            <Select.Option value="cnn">CNN</Select.Option>
            <Select.Option value="resnet">ResNet50</Select.Option>
            <Select.Option value="densenet">DenseNet121</Select.Option>
            <Select.Option value="efficientnet">EfficientNetB3</Select.Option>
          </Select>

          <Upload
            accept="image/*"
            showUploadList={false}
            customRequest={handleSelect}
          >
            <Button
              icon={<UploadOutlined />}
              type="dashed"
              block
              style={{ width: 240, margin: "0 auto" }}
            >
              Chọn ảnh X-quang
            </Button>
          </Upload>

          {imageUrl && (
            <div style={{ marginTop: 24 }}>
              <Image
                src={imageUrl}
                alt="X-ray"
                width={320}
                style={{
                  borderRadius: 12,
                  boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
                }}
              />
            </div>
          )}

          <Space style={{ marginTop: 30 }}>
            <Button
              type="primary"
              icon={<ExperimentOutlined />}
              onClick={handlePredict}
              loading={loading}
            >
              Dự đoán
            </Button>
            <Button icon={<ReloadOutlined />} onClick={handleReset}>
              Làm mới
            </Button>
          </Space>

          {loading && <Spin style={{ marginTop: 24 }} />}

          {prediction && (
            <div style={{ marginTop: 30 }}>
              <Text strong style={{ fontSize: 20, color: "#333" }}>
                {prediction.prediction}
              </Text>
              <p style={{ color: "#666", marginTop: 6 }}>
                Xác suất: {(prediction.probability * 100).toFixed(2)}%
              </p>
              <p style={{ color: "#888", fontSize: 13 }}>
                Mô hình sử dụng:{" "}
                <b style={{ textTransform: "capitalize" }}>
                  {prediction.model_used}
                </b>
              </p>
            </div>
          )}
        </Card>
      </div>

      <Footerapp />
    </>
  );
}

export default App;
