from typing import Tuple, Optional

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    GlobalAveragePooling2D,
    Dropout,
)


def build_simple_cnn(input_shape: Tuple[int, int, int] = (150, 150, 3)) -> Sequential:
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return model


def compile_and_train(
    model: models.Model,
    train_data,
    val_data,
    epochs: int = 10,
    optimizer: str | tf.keras.optimizers.Optimizer = "adam",
    loss: str = "binary_crossentropy",
    metrics: Optional[list] = None,
):
    if metrics is None:
        metrics = ["accuracy"]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    history = model.fit(train_data, epochs=epochs, validation_data=val_data)
    return history


def build_transfer_model(
    base: str,
    input_shape: Tuple[int, int, int],
    dense_units: int = 256,
    dropout: float = 0.5,
    weights: str = "imagenet",
    include_top: bool = False,
):
    base_model: Model
    if base.lower() == "densenet121":
        from tensorflow.keras.applications import DenseNet121

        base_model = DenseNet121(weights=weights, include_top=include_top, input_shape=input_shape)
    elif base.lower() == "resnet50":
        from tensorflow.keras.applications import ResNet50

        base_model = ResNet50(weights=weights, include_top=include_top, input_shape=input_shape)
    elif base.lower() == "efficientnetb3":
        from tensorflow.keras.applications import EfficientNetB3

        base_model = EfficientNetB3(weights=weights, include_top=include_top, input_shape=input_shape)
    else:
        raise ValueError("Unsupported base model")

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(dropout)(x)
    out = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=out)
    return model, base_model


def stage_train_transfer(
    model: Model,
    base_model: Model,
    train_data,
    val_data,
    epochs_stage1: int,
    epochs_stage2: int,
    fine_tune_from_layer_name_substr: Optional[str] = None,
    learning_rate_stage2: float = 1e-5,
):
    # Stage 1: freeze base
    base_model.trainable = False
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    history1 = model.fit(train_data, epochs=epochs_stage1, validation_data=val_data)

    # Stage 2: partial unfreeze
    base_model.trainable = True
    if fine_tune_from_layer_name_substr:
        for layer in base_model.layers:
            if fine_tune_from_layer_name_substr not in layer.name:
                layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_stage2),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    history2 = model.fit(
        train_data,
        epochs=history1.epoch[-1] + 1 + epochs_stage2,
        initial_epoch=history1.epoch[-1] + 1,
        validation_data=val_data,
    )
    return history1, history2


