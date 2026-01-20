import os
from typing import Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_datagen(
    rescale: Optional[float] = 1.0 / 255,
    preprocessing_function=None,
    rotation_range: int = 0,
    width_shift_range: float = 0.0,
    height_shift_range: float = 0.0,
    shear_range: float = 0.0,
    zoom_range: float = 0.0,
    horizontal_flip: bool = False,
    brightness_range: Optional[Tuple[float, float]] = None,
):
    """Create an ImageDataGenerator with common augmentation options."""
    return ImageDataGenerator(
        rescale=rescale,
        preprocessing_function=preprocessing_function,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        brightness_range=brightness_range,
    )


def flow_from_directory(
    datagen: ImageDataGenerator,
    directory: str,
    target_size: Tuple[int, int],
    batch_size: int,
    class_mode: str = "binary",
    shuffle: bool = True,
):
    return datagen.flow_from_directory(
        directory=directory,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=shuffle,
    )


def split_train_val_from_train_dir(
    train_dir: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Create train/val dataframes by scanning a train directory with subfolders as labels."""
    normal_dir = os.path.join(train_dir, "NORMAL")
    pneumonia_dir = os.path.join(train_dir, "PNEUMONIA")

    normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir)]
    pneumonia_files = [os.path.join(pneumonia_dir, f) for f in os.listdir(pneumonia_dir)]

    all_filepaths = normal_files + pneumonia_files
    all_labels = ["NORMAL"] * len(normal_files) + ["PNEUMONIA"] * len(pneumonia_files)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_filepaths,
        all_labels,
        test_size=test_size,
        stratify=all_labels,
        random_state=random_state,
    )

    train_df = pd.DataFrame({"filepath": train_paths, "label": train_labels})
    val_df = pd.DataFrame({"filepath": val_paths, "label": val_labels})
    return train_df, val_df


def dataframe_from_directory(directory: str) -> pd.DataFrame:
    """Create a dataframe with columns filepath and label from a directory tree."""
    rows = []
    for root, _, files in os.walk(directory):
        label = os.path.basename(root)
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append({"filepath": os.path.join(root, file), "label": label})
    return pd.DataFrame(rows)


def flow_from_dataframe(
    datagen: ImageDataGenerator,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    target_size: Tuple[int, int],
    batch_size: int,
    class_mode: str = "binary",
    shuffle: bool = True,
):
    return datagen.flow_from_dataframe(
        dataframe=df,
        x_col=x_col,
        y_col=y_col,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=shuffle,
    )


