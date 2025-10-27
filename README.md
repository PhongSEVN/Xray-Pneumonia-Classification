# Phân loại bệnh viêm phổi

# Bộ dữ liệu sử dụng

Dự án sử dụng bộ dữ liệu [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).  

---

## 1. Tổng quan

### Kiến trúc hệ thống
```
┌────────────────────────────┐
│        Frontend (React)    │
│  - Upload ảnh X-ray        │
│  - Gửi request đến API     │
│  - Hiển thị kết quả dự đoán│
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│       Backend (FastAPI)    │
│  - Nhận ảnh từ frontend    │
│  - Load model .keras       │
│  - Dự đoán & trả JSON      │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│       Model                 │
│  - Trained bằng TensorFlow  │
│  - Lưu tại ./models/        │
└────────────────────────────┘
```

---

## 2. Chạy Frontend

### Cài đặt
```bash
cd client
npm install
```

Trong `src/App.js`, backend api là http://localhost:5000/predict


### Chạy app
```bash
npm start
```

Ứng dụng mở tại: [http://localhost:3000](http://localhost:3000)

Tính năng:
- Upload ảnh X-ray (.jpg/.png)  
- Tùy chọn mô hình muốn sử dụng
- Nhấn **Phân tích**  
- Nhận kết quả và độ tin cậy

---

## 3. Chạy Backend

Tải các thư viện cần thiết sau đó chạy file start.bat


