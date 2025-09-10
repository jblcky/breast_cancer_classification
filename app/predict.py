from typing import Union, List
import numpy as np
import tensorflow as tf


def predict_image(model: tf.keras.Model,
            image_tensor: tf.Tensor,
            threshold: float = 0.004) -> tuple[List[str], np.ndarray]:
    """
    Predict binary class label and confidence for a single image.

    Args:
        model: Trained Keras model.
        image_tensor: Preprocessed image tensor (1, 224, 224, 3)
        threshold: Probability threshold for malignant class.

    Returns:
        class_label: 'benign' or 'malignant'
        confidence_percent: float, confidence in percentage
    """
    preds = model.predict(image_tensor)  # shape: (1, 1)
    prob = float(preds[0][0])  # scalar probability

    class_label = "malignant" if prob >= threshold else "benign"
    confidence_percent = prob * 100 if class_label == "malignant" else (1 - prob) * 100

    return class_label, confidence_percent
