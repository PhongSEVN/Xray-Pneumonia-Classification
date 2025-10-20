import { Layout, Row, Col, Typography, Space } from "antd";
import {
  GithubOutlined,
  CopyrightOutlined,
  TeamOutlined,
  HeartFilled,
} from "@ant-design/icons";

const { Footer } = Layout;
const { Text, Link } = Typography;

function Footerapp() {
  return (
    <Footer
      style={{
        background: "linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)",
        color: "#ffffff",
        padding: "40px 50px 24px",
        position: "relative",
        overflow: "hidden",
      }}
    >
      {/* Decorative background elements */}
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          height: "3px",
          background: "linear-gradient(90deg, #1890ff, #52c41a, #faad14)",
        }}
      />

      <Row gutter={[32, 24]} justify="space-between" align="middle">
        {/* Left Section - Project Info */}
        <Col xs={24} md={12} lg={10}>
          <Space direction="vertical" size="small" style={{ width: "100%" }}>
            <Text
              strong
              style={{
                color: "#ffffff",
                fontSize: "18px",
                display: "block",
                marginBottom: "8px",
              }}
            >
              Xray Pneumonia Classification
            </Text>
            <Space
              align="center"
              style={{ color: "rgba(255, 255, 255, 0.85)" }}
            >
              <TeamOutlined style={{ fontSize: "16px" }} />
              <Text style={{ color: "rgba(255, 255, 255, 0.85)" }}>
                Nguyễn Văn Phong • Trần Đình Khải
              </Text>
            </Space>
          </Space>
        </Col>

        {/* Right Section - GitHub Link */}
        <Col xs={24} md={12} lg={8} style={{ textAlign: "right" }}>
          <Link
            href="https://github.com/PhongSEVN/Xray-Pneumonia-Classification"
            target="_blank"
            style={{
              color: "#ffffff",
              fontSize: "15px",
              padding: "8px 20px",
              background: "rgba(255, 255, 255, 0.1)",
              borderRadius: "20px",
              display: "inline-flex",
              alignItems: "center",
              gap: "8px",
              transition: "all 0.3s ease",
              border: "1px solid rgba(255, 255, 255, 0.2)",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = "rgba(255, 255, 255, 0.2)";
              e.currentTarget.style.transform = "translateY(-2px)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = "rgba(255, 255, 255, 0.1)";
              e.currentTarget.style.transform = "translateY(0)";
            }}
          >
            <GithubOutlined style={{ fontSize: "18px" }} />
            <span>View on GitHub</span>
          </Link>
        </Col>
      </Row>

      {/* Bottom Copyright Section */}
      <div
        style={{
          marginTop: "32px",
          paddingTop: "20px",
          borderTop: "1px solid rgba(255, 255, 255, 0.15)",
          textAlign: "center",
        }}
      >
        <Space size={4} align="center">
          <CopyrightOutlined style={{ fontSize: "14px" }} />
          <Text
            style={{ color: "rgba(255, 255, 255, 0.75)", fontSize: "14px" }}
          >
            2025 Nguyễn Văn Phong. Made with
          </Text>
          <HeartFilled style={{ color: "#ff4d4f", fontSize: "14px" }} />
          <Text
            style={{ color: "rgba(255, 255, 255, 0.75)", fontSize: "14px" }}
          >
            in Vietnam
          </Text>
        </Space>
      </div>
    </Footer>
  );
}

export default Footerapp;
