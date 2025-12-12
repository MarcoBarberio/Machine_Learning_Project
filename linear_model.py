import tensorflow as tf
import numpy as np
def linear_model(is_regression,learning_rate):
    if(is_regression):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(4, activation='linear', input_shape=(12,))
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    model= tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(6,))
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
