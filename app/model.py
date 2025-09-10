from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, regularizers, models
import tensorflow as tf
from data_augmentation import get_data_augmentation


def initialize_resnet_model(input_shape=(224,224,3)):

    # Input layer
    inputs = layers.Input(shape=input_shape)

    # Data augmentation
    x = get_data_augmentation()(inputs)

    # Load Resnet50 base model with ImageNet weights
    base_model = ResNet50(
    include_top=False,
    weights="imagenet",
    input_tensor=x,  # use x directly instead
    input_shape=input_shape,
    pooling=None
)
    base_model.trainable = False   # freeze base model

    # Pass augmented images through ResNet50
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)   # helps stabilize training
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)

    return model


MODEL_PATH = 'saved_model/resnet_cnn_model_version2.weights.h5'
model = initialize_resnet_model()
model.load_weights(MODEL_PATH)
