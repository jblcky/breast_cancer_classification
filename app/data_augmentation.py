from tensorflow.keras import layers
import tensorflow as tf

def get_data_augmentation():
    """
    Data augmentation pipeline.
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),      # safe, GPU-accelerated
        layers.RandomRotation(0.05),          # ±5% only, medically safe
        layers.RandomZoom(0.05),              # small zoom, GPU fast
        # Skip contrast/brightness (slow + risky for medical)
    ], name="data_augmentation")
