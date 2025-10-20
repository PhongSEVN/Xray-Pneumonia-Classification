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
  Row,
  Col,
  Badge,
  Divider,
} from "antd";
import {
  UploadOutlined,
  ReloadOutlined,
  ExperimentOutlined,
  CheckCircleFilled,
  RobotOutlined,
  FileImageOutlined,
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
    formData.append("model", model);

    setLoading(true);
    try {
      const res = await axios.post("http://localhost:5000/predict", formData);
      setPrediction(res.data);
      messageApi.success("Phân tích hoàn tất!");
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

  const modelInfo = {
    cnn: {
      name: "CNN",
      color: "#1890ff",
      desc: "Mạng nơ-ron tích chập cơ bản",
    },
    resnet: {
      name: "ResNet50",
      color: "#722ed1",
      desc: "Kiến trúc Residual Network",
    },
    densenet: {
      name: "DenseNet121",
      color: "#13c2c2",
      desc: "Dense Convolutional Network",
    },
  };

  return (
    <>
      {contextHolder}
      <div
        style={{
          minHeight: "100vh",
          background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
          padding: "60px 20px",
          position: "relative",
          overflow: "hidden",
        }}
      >
        {/* Decorative circles */}
        <div
          style={{
            position: "absolute",
            width: "400px",
            height: "400px",
            borderRadius: "50%",
            background: "rgba(255, 255, 255, 0.1)",
            top: "-100px",
            right: "-100px",
            filter: "blur(60px)",
          }}
        />
        <div
          style={{
            position: "absolute",
            width: "300px",
            height: "300px",
            borderRadius: "50%",
            background: "rgba(255, 255, 255, 0.1)",
            bottom: "-50px",
            left: "-50px",
            filter: "blur(60px)",
          }}
        />

        <div
          style={{
            maxWidth: "1000px",
            margin: "0 auto",
            position: "relative",
            zIndex: 1,
          }}
        >
          {/* Header Section */}
          <div style={{ textAlign: "center", marginBottom: "40px" }}>
            <Badge.Ribbon text="AI Powered" color="#52c41a">
              <div
                style={{
                  background: "rgba(255, 255, 255, 0.95)",
                  padding: "30px",
                  borderRadius: "20px",
                  boxShadow: "0 8px 32px rgba(0, 0, 0, 0.1)",
                  backdropFilter: "blur(10px)",
                }}
              >
                <RobotOutlined
                  style={{
                    fontSize: "48px",
                    color: "#667eea",
                    marginBottom: "16px",
                  }}
                />
                <Title
                  level={1}
                  style={{
                    color: "#667eea",
                    marginBottom: 8,
                    fontSize: "36px",
                  }}
                >
                  Phân Tích Viêm Phổi AI
                </Title>
                <Text style={{ color: "#666", fontSize: "16px" }}>
                  Phát hiện viêm phổi từ ảnh X-quang sử dụng Deep Learning
                </Text>
              </div>
            </Badge.Ribbon>
          </div>

          {/* Main Card */}
          <Card
            style={{
              borderRadius: "24px",
              boxShadow: "0 12px 48px rgba(0, 0, 0, 0.15)",
              border: "none",
              overflow: "hidden",
            }}
            bodyStyle={{ padding: "40px" }}
          >
            {/* Model Selection Section */}
            <div style={{ marginBottom: "32px" }}>
              <Space direction="vertical" size={12} style={{ width: "100%" }}>
                <Text strong style={{ fontSize: "16px", color: "#333" }}>
                  <ExperimentOutlined style={{ marginRight: "8px" }} />
                  Chọn Mô Hình AI
                </Text>
                <Select
                  value={model}
                  style={{ width: "100%" }}
                  size="large"
                  onChange={setModel}
                >
                  {Object.entries(modelInfo).map(([key, info]) => (
                    <Select.Option key={key} value={key}>
                      <Space>
                        <Badge color={info.color} />
                        <span style={{ fontWeight: 600 }}>{info.name}</span>
                        <Text type="secondary" style={{ fontSize: "12px" }}>
                          - {info.desc}
                        </Text>
                      </Space>
                    </Select.Option>
                  ))}
                </Select>
              </Space>
            </div>

            <Divider />

            {/* Upload Section */}
            <div style={{ textAlign: "center", marginBottom: "32px" }}>
              <Upload
                accept="image/*"
                showUploadList={false}
                customRequest={handleSelect}
              >
                <Button
                  icon={<UploadOutlined />}
                  size="large"
                  type={imageUrl ? "default" : "primary"}
                  style={{
                    height: "56px",
                    borderRadius: "12px",
                    fontSize: "16px",
                    fontWeight: 600,
                    minWidth: "280px",
                    background: imageUrl
                      ? "#fff"
                      : "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                    border: imageUrl ? "2px dashed #d9d9d9" : "none",
                  }}
                >
                  <FileImageOutlined
                    style={{ fontSize: "18px", marginRight: "8px" }}
                  />
                  {imageUrl ? "Chọn ảnh khác" : "Tải ảnh X-quang lên"}
                </Button>
              </Upload>
            </div>

            {/* Image Preview Section */}
            {imageUrl && (
              <Row gutter={24} style={{ marginTop: "32px" }}>
                <Col xs={24} md={prediction?.gradcam ? 12 : 24}>
                  <div style={{ textAlign: "center" }}>
                    <Text
                      strong
                      style={{
                        display: "block",
                        marginBottom: "16px",
                        fontSize: "15px",
                      }}
                    >
                      📷 Ảnh X-quang Gốc
                    </Text>
                    <div
                      style={{
                        background: "#f5f5f5",
                        padding: "20px",
                        borderRadius: "16px",
                        display: "inline-block",
                      }}
                    >
                      <Image
                        src={imageUrl}
                        alt="X-ray"
                        width={280}
                        style={{
                          borderRadius: "12px",
                          boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
                        }}
                      />
                    </div>
                  </div>
                </Col>

                {prediction?.gradcam && (
                  <Col xs={24} md={12}>
                    <div style={{ textAlign: "center" }}>
                      <Text
                        strong
                        style={{
                          display: "block",
                          marginBottom: "16px",
                          fontSize: "15px",
                        }}
                      >
                        🔍 Grad-CAM Visualization
                      </Text>
                      <div
                        style={{
                          background: "#f5f5f5",
                          padding: "20px",
                          borderRadius: "16px",
                          display: "inline-block",
                        }}
                      >
                        <Image
                          src={`data:image/jpeg;base64,${prediction.gradcam}`}
                          alt="Grad-CAM"
                          width={280}
                          style={{
                            borderRadius: "12px",
                            boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
                          }}
                        />
                      </div>
                    </div>
                  </Col>
                )}
              </Row>
            )}

            {/* Action Buttons */}
            <div style={{ textAlign: "center", marginTop: "32px" }}>
              <Space size="large">
                <Button
                  type="primary"
                  icon={<ExperimentOutlined />}
                  onClick={handlePredict}
                  loading={loading}
                  size="large"
                  style={{
                    height: "48px",
                    borderRadius: "12px",
                    fontSize: "15px",
                    fontWeight: 600,
                    minWidth: "160px",
                    background:
                      "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                    border: "none",
                  }}
                  disabled={!imageUrl}
                >
                  Phân Tích Ngay
                </Button>
                <Button
                  icon={<ReloadOutlined />}
                  onClick={handleReset}
                  size="large"
                  style={{
                    height: "48px",
                    borderRadius: "12px",
                    fontSize: "15px",
                    fontWeight: 600,
                    minWidth: "140px",
                  }}
                >
                  Làm Mới
                </Button>
              </Space>
            </div>

            {/* Loading State */}
            {loading && (
              <div style={{ textAlign: "center", marginTop: "32px" }}>
                <Spin size="large" />
                <Text
                  style={{ display: "block", marginTop: "16px", color: "#666" }}
                >
                  Đang phân tích ảnh X-quang...
                </Text>
              </div>
            )}

            {/* Prediction Results */}
            {prediction && !loading && (
              <div
                style={{
                  marginTop: "40px",
                  background:
                    "linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)",
                  padding: "32px",
                  borderRadius: "16px",
                  border: "2px solid #bae6fd",
                }}
              >
                <div style={{ textAlign: "center" }}>
                  <CheckCircleFilled
                    style={{
                      fontSize: "48px",
                      color: "#52c41a",
                      marginBottom: "16px",
                    }}
                  />
                  <Title
                    level={3}
                    style={{ color: "#0369a1", marginBottom: "8px" }}
                  >
                    Kết Quả Phân Tích
                  </Title>

                  <div
                    style={{
                      background: "#fff",
                      padding: "24px",
                      borderRadius: "12px",
                      marginTop: "20px",
                      boxShadow: "0 2px 8px rgba(0,0,0,0.05)",
                    }}
                  >
                    <Row gutter={[24, 16]}>
                      <Col xs={24} sm={8}>
                        <div>
                          <Text type="secondary" style={{ fontSize: "13px" }}>
                            Chẩn đoán
                          </Text>
                          <Title
                            level={4}
                            style={{ margin: "8px 0", color: "#1890ff" }}
                          >
                            {prediction.prediction}
                          </Title>
                        </div>
                      </Col>
                      <Col xs={24} sm={8}>
                        <div>
                          <Text type="secondary" style={{ fontSize: "13px" }}>
                            Độ tin cậy
                          </Text>
                          <Title
                            level={4}
                            style={{ margin: "8px 0", color: "#52c41a" }}
                          >
                            {(prediction.probability * 100).toFixed(2)}%
                          </Title>
                        </div>
                      </Col>
                      <Col xs={24} sm={8}>
                        <div>
                          <Text type="secondary" style={{ fontSize: "13px" }}>
                            Mô hình
                          </Text>
                          <Title
                            level={4}
                            style={{
                              margin: "8px 0",
                              color: modelInfo[model]?.color,
                            }}
                          >
                            {modelInfo[prediction.model_used]?.name ||
                              prediction.model_used}
                          </Title>
                        </div>
                      </Col>
                    </Row>
                  </div>
                </div>
              </div>
            )}
          </Card>
        </div>
      </div>

      <Footerapp />
    </>
  );
}

export default App;
