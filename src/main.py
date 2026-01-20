import os

from preprocessing import (
    build_datagen,
    flow_from_directory,
    split_train_val_from_train_dir,
    flow_from_dataframe,
)
from eda import count_images_by_class, plot_counts, sample_preview
from model_training import build_simple_cnn, compile_and_train, build_transfer_model, stage_train_transfer
from evaluation import evaluate_and_report


def run_simple_cnn(url_train: str, url_val: str, url_test: str):
    img_size = (150, 150)
    batch_size = 32

    train_datagen = build_datagen(rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = build_datagen(rescale=1.0 / 255)

    training_set = flow_from_directory(train_datagen, url_train, img_size, batch_size)
    validation_set = flow_from_directory(test_datagen, url_val, img_size, batch_size)
    test_set = flow_from_directory(test_datagen, url_test, img_size, batch_size, shuffle=False)

    model = build_simple_cnn((img_size[0], img_size[1], 3))
    history = compile_and_train(model, training_set, validation_set, epochs=10)
    evaluate_and_report(model, test_set)


def run_transfer_learning(url_train: str, url_test: str, base: str):
    if base.lower() == "densenet121":
        img_size = (224, 224)
        batch_size = 16
        epochs1, epochs2 = 20, 20
        fine_tune_substr = "conv5_block"
    elif base.lower() == "resnet50":
        img_size = (224, 224)
        batch_size = 32
        epochs1, epochs2 = 10, 30
        fine_tune_substr = "conv5_block"
    elif base.lower() == "efficientnetb3":
        img_size = (300, 300)
        batch_size = 16
        epochs1, epochs2 = 8, 25
        fine_tune_substr = None  # handled differently typically
    else:
        raise ValueError("Unsupported base model")

    # Split new train/val from original train dir
    train_df, val_df = split_train_val_from_train_dir(url_train, test_size=0.2, random_state=42)

    # Datagens
    if base.lower() == "densenet121":
        from tensorflow.keras.applications.densenet import preprocess_input
    elif base.lower() == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
    else:
        from tensorflow.keras.applications.efficientnet import preprocess_input

    train_datagen = build_datagen(preprocessing_function=preprocess_input, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, horizontal_flip=True, brightness_range=(0.8, 1.2) if base.lower()=="densenet121" else None)
    val_datagen = build_datagen(preprocessing_function=preprocess_input)

    train_gen = flow_from_dataframe(train_datagen, train_df, "filepath", "label", img_size, batch_size)
    val_gen = flow_from_dataframe(val_datagen, val_df, "filepath", "label", img_size, batch_size)

    # Test generator
    from preprocessing import dataframe_from_directory

    test_df = dataframe_from_directory(url_test)
    test_gen = flow_from_dataframe(val_datagen, test_df, "filepath", "label", img_size, batch_size, shuffle=False)

    # Build and train
    model, base_model = build_transfer_model(base, (img_size[0], img_size[1], 3))
    history1, history2 = stage_train_transfer(
        model,
        base_model,
        train_gen,
        val_gen,
        epochs_stage1=epochs1,
        epochs_stage2=epochs2,
        fine_tune_from_layer_name_substr=fine_tune_substr,
    )

    evaluate_and_report(model, test_gen)


if __name__ == "__main__":
    # Update these paths as needed
    url_train = "/kaggle/input/chest-x-ray-images-normal-and-pneumonia/chest_xray/train"
    url_test = "/kaggle/input/chest-x-ray-images-normal-and-pneumonia/chest_xray/test"
    url_val = "/kaggle/input/chest-x-ray-images-normal-and-pneumonia/chest_xray/val"

    # EDA example
    df_counts = count_images_by_class(url_train, url_test, url_val)
    plot_counts(df_counts)
    sample_preview(url_train, num_per_class=5)

    # Simple CNN
    # run_simple_cnn(url_train, url_val, url_test)

    # Transfer learning example
    # run_transfer_learning(url_train, url_test, base="densenet121")
    # run_transfer_learning(url_train, url_test, base="resnet50")
    # run_transfer_learning(url_train, url_test, base="efficientnetb3")


