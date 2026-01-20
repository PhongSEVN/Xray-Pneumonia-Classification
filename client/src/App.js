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
        cnn: { name: "CNN", color: "#1890ff", desc: "Mạng nơ-ron tích chập" },
        resnet: { name: "ResNet50", color: "#722ed1", desc: "Residual Network" },
        densenet: { name: "DenseNet121", color: "#13c2c2", desc: "Dense Network" },
    };

    return (
        <>
            {contextHolder}
            <div
                style={{
                    height: "100vh",
                    display: "flex",
                    flexDirection: "column",
                    background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
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

                {/* Main Content */}
                <div
                    style={{
                        flex: 1,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        padding: "20px",
                        position: "relative",
                        zIndex: 1,
                        overflow: "auto",
                    }}
                >
                    <div style={{ maxWidth: "1200px", width: "100%" }}>
                        {/* Compact Header */}
                        <div style={{ textAlign: "center", marginBottom: "20px" }}>
                            <Badge.Ribbon color="#52c41a" style={{ fontSize: "11px" }}>
                                <div
                                    style={{
                                        background: "rgba(255, 255, 255, 0.95)",
                                        padding: "16px 24px",
                                        borderRadius: "16px",
                                        boxShadow: "0 8px 32px rgba(0, 0, 0, 0.1)",
                                        display: "inline-block",
                                    }}
                                >
                                    <Space size={12}>
                                        <RobotOutlined style={{ fontSize: "32px", color: "#667eea" }} />
                                        <div style={{ textAlign: "left" }}>
                                            <Title level={2} style={{ color: "#667eea", margin: 0, fontSize: "24px" }}>
                                                Phân Tích Viêm Phổi AI
                                            </Title>
                                            <Text style={{ color: "#666", fontSize: "13px" }}>
                                                Phát hiện viêm phổi từ X-quang
                                            </Text>
                                        </div>
                                    </Space>
                                </div>
                            </Badge.Ribbon>
                        </div>

                        {/* Compact Main Card */}
                        <Card
                            style={{
                                borderRadius: "20px",
                                boxShadow: "0 12px 48px rgba(0, 0, 0, 0.15)",
                                border: "none",
                            }}
                            bodyStyle={{ padding: "24px" }}
                        >
                            <Row gutter={24}>
                                {/* Left Column - Controls */}
                                <Col xs={24} lg={10}>
                                    <Space direction="vertical" size={16} style={{ width: "100%" }}>
                                        {/* Model Selection */}
                                        <div>
                                            <Text strong style={{ fontSize: "14px", color: "#333", marginBottom: "8px", display: "block" }}>
                                                <ExperimentOutlined /> Mô Hình AI
                                            </Text>
                                            <Select value={model} style={{ width: "100%" }} size="large" onChange={setModel}>
                                                {Object.entries(modelInfo).map(([key, info]) => (
                                                    <Select.Option key={key} value={key}>
                                                        <Space>
                                                            <Badge color={info.color} />
                                                            <span style={{ fontWeight: 600 }}>{info.name}</span>
                                                        </Space>
                                                    </Select.Option>
                                                ))}
                                            </Select>
                                        </div>

                                        {/* Upload Button */}
                                        <Upload accept="image/*" showUploadList={false} customRequest={handleSelect}>
                                            <Button
                                                icon={<UploadOutlined />}
                                                size="large"
                                                block
                                                type={imageUrl ? "default" : "primary"}
                                                style={{
                                                    height: "48px",
                                                    borderRadius: "10px",
                                                    fontSize: "15px",
                                                    fontWeight: 600,
                                                    background: imageUrl ? "#fff" : "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                                                    border: imageUrl ? "2px dashed #d9d9d9" : "none",
                                                }}
                                            >
                                                <FileImageOutlined style={{ fontSize: "16px" }} />
                                                {imageUrl ? "Chọn ảnh khác" : "Tải ảnh X-quang"}
                                            </Button>
                                        </Upload>

                                        {/* Action Buttons */}
                                        <Space style={{ width: "100%" }}>
                                            <Button
                                                type="primary"
                                                icon={<ExperimentOutlined />}
                                                onClick={handlePredict}
                                                loading={loading}
                                                size="large"
                                                block
                                                style={{
                                                    height: "44px",
                                                    borderRadius: "10px",
                                                    fontSize: "14px",
                                                    fontWeight: 600,
                                                    flex: 1,
                                                    background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                                                    border: "none",
                                                }}
                                                disabled={!imageUrl}
                                            >
                                                Phân Tích
                                            </Button>
                                            <Button
                                                icon={<ReloadOutlined />}
                                                onClick={handleReset}
                                                size="large"
                                                style={{
                                                    height: "44px",
                                                    borderRadius: "10px",
                                                    fontSize: "14px",
                                                    fontWeight: 600,
                                                    minWidth: "110px",
                                                }}
                                            >
                                                Làm Mới
                                            </Button>
                                        </Space>

                                        {/* Results - Compact */}
                                        {prediction && !loading && (
                                            <div
                                                style={{
                                                    background: "linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)",
                                                    padding: "16px",
                                                    borderRadius: "12px",
                                                    border: "2px solid #bae6fd",
                                                }}
                                            >
                                                <CheckCircleFilled style={{ fontSize: "28px", color: "#52c41a", marginBottom: "8px" }} />
                                                <div
                                                    style={{
                                                        background: "#fff",
                                                        padding: "12px",
                                                        borderRadius: "8px",
                                                        marginTop: "8px",
                                                    }}
                                                >
                                                    <Space direction="vertical" size={8} style={{ width: "100%" }}>
                                                        <div>
                                                            <Text type="secondary" style={{ fontSize: "11px" }}>Chẩn đoán</Text>
                                                            <Title level={5} style={{ margin: "4px 0", color: "#1890ff" }}>
                                                                {prediction.prediction}
                                                            </Title>
                                                        </div>
                                                        <Row gutter={12}>
                                                            <Col span={12}>
                                                                <Text type="secondary" style={{ fontSize: "11px" }}>Độ tin cậy</Text>
                                                                <div style={{ fontSize: "16px", fontWeight: 600, color: "#52c41a" }}>
                                                                    {(prediction.probability * 100).toFixed(1)}%
                                                                </div>
                                                            </Col>
                                                            <Col span={12}>
                                                                <Text type="secondary" style={{ fontSize: "11px" }}>Mô hình</Text>
                                                                <div style={{ fontSize: "16px", fontWeight: 600, color: modelInfo[model]?.color }}>
                                                                    {modelInfo[prediction.model_used]?.name}
                                                                </div>
                                                            </Col>
                                                        </Row>
                                                    </Space>
                                                </div>
                                            </div>
                                        )}

                                        {/* Loading */}
                                        {loading && (
                                            <div style={{ textAlign: "center", padding: "16px" }}>
                                                <Spin size="large" />
                                                <Text style={{ display: "block", marginTop: "12px", color: "#666", fontSize: "13px" }}>
                                                    Đang phân tích...
                                                </Text>
                                            </div>
                                        )}
                                    </Space>
                                </Col>

                                {/* Right Column - Images */}
                                <Col xs={24} lg={14}>
                                    {imageUrl ? (
                                        <Row gutter={16}>
                                            <Col xs={24} md={prediction?.gradcam ? 12 : 24}>
                                                <div style={{ textAlign: "center" }}>
                                                    <Text strong style={{ display: "block", marginBottom: "12px", fontSize: "13px" }}>
                                                        📷 Ảnh Gốc
                                                    </Text>
                                                    <div
                                                        style={{
                                                            background: "#f5f5f5",
                                                            padding: "12px",
                                                            borderRadius: "12px",
                                                            display: "inline-block",
                                                        }}
                                                    >
                                                        <Image
                                                            src={imageUrl}
                                                            alt="X-ray"
                                                            style={{
                                                                maxWidth: "100%",
                                                                height: "auto",
                                                                maxHeight: "400px",
                                                                borderRadius: "8px",
                                                                boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
                                                            }}
                                                        />
                                                    </div>
                                                </div>
                                            </Col>

                                            {prediction?.gradcam && (
                                                <Col xs={24} md={12}>
                                                    <div style={{ textAlign: "center" }}>
                                                        <Text strong style={{ display: "block", marginBottom: "12px", fontSize: "13px" }}>
                                                            🔍 Grad-CAM
                                                        </Text>
                                                        <div
                                                            style={{
                                                                background: "#f5f5f5",
                                                                padding: "12px",
                                                                borderRadius: "12px",
                                                                display: "inline-block",
                                                            }}
                                                        >
                                                            <Image
                                                                src={`data:image/jpeg;base64,${prediction.gradcam}`}
                                                                alt="Grad-CAM"
                                                                style={{
                                                                    maxWidth: "100%",
                                                                    height: "auto",
                                                                    maxHeight: "400px",
                                                                    borderRadius: "8px",
                                                                    boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
                                                                }}
                                                            />
                                                        </div>
                                                    </div>
                                                </Col>
                                            )}
                                        </Row>
                                    ) : (
                                        <div
                                            style={{
                                                height: "100%",
                                                minHeight: "400px",
                                                display: "flex",
                                                alignItems: "center",
                                                justifyContent: "center",
                                                background: "#f5f5f5",
                                                borderRadius: "12px",
                                                border: "2px dashed #d9d9d9",
                                            }}
                                        >
                                            <div style={{ textAlign: "center", color: "#999" }}>
                                                <FileImageOutlined style={{ fontSize: "64px", marginBottom: "16px" }} />
                                                <Text style={{ display: "block", fontSize: "15px" }}>
                                                    Chưa có ảnh nào được tải lên
                                                </Text>
                                            </div>
                                        </div>
                                    )}
                                </Col>
                            </Row>
                        </Card>
                    </div>
                </div>

                {/* Footer */}
                <Footerapp />
            </div>
        </>
    );
}

export default App;
