import os
import shutil

# Thư mục gốc chứa ảnh Pneumonia
root_dir = r"C:\Users\PC\Desktop\test\PNEUMONIA"

# Tạo hai thư mục con nếu chưa có
virus_dir = os.path.join(root_dir, "virus")
bacteria_dir = os.path.join(root_dir, "bacteria")

os.makedirs(virus_dir, exist_ok=True)
os.makedirs(bacteria_dir, exist_ok=True)

# Duyệt qua tất cả file trong thư mục Pneumonia
for filename in os.listdir(root_dir):
    filepath = os.path.join(root_dir, filename)

    # Bỏ qua nếu là thư mục con (virus, bacteria)
    if os.path.isdir(filepath):
        continue

    # Phân loại dựa trên tên file
    name_lower = filename.lower()
    if "virus" in name_lower:
        shutil.move(filepath, os.path.join(virus_dir, filename))
    elif "bacteria" in name_lower:
        shutil.move(filepath, os.path.join(bacteria_dir, filename))

print("Xong. Đã chia ảnh vào thư mục con virus/ và bacteria/")
