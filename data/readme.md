```markdown
# 🩻 Chest X-Ray Dataset (Normal vs Pneumonia)

## 📘 Giới thiệu

Thư mục này chứa **tập dữ liệu X-quang ngực (Chest X-ray Images)** được sử dụng để huấn luyện và đánh giá mô hình phát hiện **viêm phổi (Pneumonia)**.  
Bộ dữ liệu gồm **hai lớp chính**:

- 🟢 **NORMAL** – Ảnh phổi bình thường  
- 🔴 **PNEUMONIA** – Ảnh phổi bị viêm (bao gồm cả do vi khuẩn và virus)

---

## 🔗 Nguồn dữ liệu

Bộ dữ liệu được công khai trên **Kaggle**, được đóng góp và duy trì bởi người dùng *Ghost5612*.

> 📎 Link tải dataset:  
> [https://www.kaggle.com/datasets/ghost5612/chest-x-ray-images-normal-and-pneumonia](https://www.kaggle.com/datasets/ghost5612/chest-x-ray-images-normal-and-pneumonia)

---

## 📂 Cấu trúc thư mục sau khi tải về

Sau khi tải và giải nén, hãy đặt dữ liệu vào thư mục `data/` trong dự án này, theo đúng cấu trúc sau:

```

data/chest_xray
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── test/
├── NORMAL/
└── PNEUMONIA/

````

> ⚠️ Nếu dataset không có thư mục `val/`, bạn có thể tự tách từ tập `train/` (ví dụ 80% train – 20% val).

---

## 🧾 Thông tin thêm

* Bộ dữ liệu bao gồm ảnh X-ray ngực của **người lớn và trẻ em**.
* Các ảnh được phân loại và kiểm định thủ công bởi bác sĩ chuyên khoa.
* Mục tiêu chính là huấn luyện mô hình nhận diện phổi viêm và phổi bình thường từ ảnh y tế.

---

## 🪪 Bản quyền và sử dụng

* Tập dữ liệu được phát hành công khai trên **Kaggle** và thuộc quyền sở hữu của người đóng góp (*Ghost5612*).
* Khi sử dụng cho mục đích học tập, nghiên cứu hoặc báo cáo, vui lòng **trích dẫn nguồn Kaggle** theo đúng quy định.

---

🧠 *“Data is the foundation of every intelligent system.”*
— NIH & Kaggle Open Health Initiative

```
```
