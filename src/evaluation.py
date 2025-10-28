from typing import Sequence

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def evaluate_and_report(model, data, class_names: Sequence[str] | None = None):
    loss, acc = model.evaluate(data)
    print(f"Test accuracy: {acc * 100:.2f}%")
    print(f"Test loss: {loss:.4f}")

    y_true = data.classes
    y_prob = model.predict(data)
    y_pred = (y_prob > 0.5).astype(int).flatten()

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names if class_names is not None else None,
        yticklabels=class_names if class_names is not None else None,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    if class_names is None and hasattr(data, "class_indices"):
        class_names = list(data.class_indices.keys())
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))


