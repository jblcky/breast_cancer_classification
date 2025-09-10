import numpy as np
from typing import Union, List, Tuple
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf

def preprocess_images_resnet50_bytes(
    file,
    target_size: tuple = (224, 224)
):
    """
    Preprocess a single UploadFile into a batch tensor for ResNet50.

    Args:
        file: UploadFile.file object (raw bytes)

    Returns:
        tf.Tensor of shape (1, 224, 224, 3)
    """

    contents = file.read()  # Read raw bytes
    img = tf.image.decode_image(contents, channels=3, expand_animations=False)
    img = tf.image.resize(img, target_size)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)  # ResNet50 preprocessing
    img = tf.expand_dims(img, axis=0)  # add batch dimension

    return img
