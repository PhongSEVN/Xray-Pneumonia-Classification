import { Layout, Space, Typography, Divider } from 'antd';
import { GithubOutlined, CopyrightOutlined } from '@ant-design/icons';

const { Footer } = Layout;
const { Text, Link } = Typography;

function Footerapp() {
    return (
        <Footer style={{
            backgroundColor: '#001529',
            color: 'rgba(255, 255, 255, 0.85)',
            textAlign: 'center',
            padding: '24px 50px'
        }}>
            <Space direction="vertical" size="small" style={{ width: '100%' }}>
                <Text style={{ color: 'rgba(255, 255, 255, 0.85)' }}>
                    <CopyrightOutlined /> Copyright by @Nguyễn Văn Phong
                </Text>

                <Divider style={{
                    margin: '12px 0',
                    borderColor: 'rgba(255, 255, 255, 0.2)'
                }} />

                <Space>
                    <GithubOutlined style={{ fontSize: '16px' }} />
                    <Link
                        href="https://github.com/PhongSEVN/Xray-Pneumonia-Classification"
                        target="_blank"
                        style={{ color: '#1890ff' }}
                    >
                        View Project on GitHub
                    </Link>
                </Space>
            </Space>
        </Footer>
    );
}

export default Footerapp;
