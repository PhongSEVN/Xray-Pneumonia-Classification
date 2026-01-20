import os
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def count_images_by_class(train_dir: str, test_dir: str, val_dir: str) -> pd.DataFrame:
    def count_in_dir(d: str) -> Dict[str, int]:
        return {
            "NORMAL": len(os.listdir(os.path.join(d, "NORMAL"))) if os.path.exists(os.path.join(d, "NORMAL")) else 0,
            "PNEUMONIA": len(os.listdir(os.path.join(d, "PNEUMONIA"))) if os.path.exists(os.path.join(d, "PNEUMONIA")) else 0,
        }

    train_counts = count_in_dir(train_dir)
    test_counts = count_in_dir(test_dir)
    val_counts = count_in_dir(val_dir)

    df_counts = pd.DataFrame(
        {
            "Tập Dữ Liệu": [
                "Train",
                "Train",
                "Test",
                "Test",
                "Validation",
                "Validation",
            ],
            "Lớp": [
                "PNEUMONIA",
                "NORMAL",
                "PNEUMONIA",
                "NORMAL",
                "PNEUMONIA",
                "NORMAL",
            ],
            "Số Lượng": [
                train_counts["PNEUMONIA"],
                train_counts["NORMAL"],
                test_counts["PNEUMONIA"],
                test_counts["NORMAL"],
                val_counts["PNEUMONIA"],
                val_counts["NORMAL"],
            ],
        }
    )

    return df_counts


def plot_counts(df_counts: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Tập Dữ Liệu", y="Số Lượng", hue="Lớp", data=df_counts)
    plt.title("Phân Bổ Số Lượng Ảnh Trong Từng Tập Dữ Liệu", fontsize=14)
    plt.xlabel("Tập Dữ Liệu", fontsize=12)
    plt.ylabel("Số Lượng Ảnh", fontsize=12)
    plt.legend(title="Loại Ảnh")
    plt.tight_layout()
    plt.show()


def sample_preview(train_dir: str, num_per_class: int = 5):
    import matplotlib.image as mpimg

    normal_images = os.listdir(os.path.join(train_dir, "NORMAL"))[:num_per_class]
    pneumonia_images = os.listdir(os.path.join(train_dir, "PNEUMONIA"))[:num_per_class]

    # NORMAL
    fig, axes = plt.subplots(1, len(normal_images), figsize=(4 * len(normal_images), 4))
    for i, img_name in enumerate(normal_images):
        img_path = os.path.join(train_dir, "NORMAL", img_name)
        img = mpimg.imread(img_path)
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title("NORMAL")
        axes[i].axis("off")
    plt.show()

    # PNEUMONIA
    fig, axes = plt.subplots(1, len(pneumonia_images), figsize=(4 * len(pneumonia_images), 4))
    for i, img_name in enumerate(pneumonia_images):
        img_path = os.path.join(train_dir, "PNEUMONIA", img_name)
        img = mpimg.imread(img_path)
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title("PNEUMONIA")
        axes[i].axis("off")
    plt.show()


