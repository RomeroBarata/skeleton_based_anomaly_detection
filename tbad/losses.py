import numpy as np
import tensorflow as tf
import keras.backend as K


def mean_squared_error(original_trajectory, reconstructed_trajectory):
    mask = original_trajectory != 0.0
    a = np.square(original_trajectory - reconstructed_trajectory) * mask
    b = np.sum(mask, axis=1, keepdims=True)
    b = np.where(b > 0, b, 1)
    return np.sum((a * mask) / b, axis=1)


def balanced_mean_squared_error(original_trajectory, reconstructed_trajectory):
    weights = np.array([8.5] * 4 + [1.0] * 34, dtype=np.float32)
    mask = original_trajectory != 0.0
    a = np.square(original_trajectory - reconstructed_trajectory) * mask
    b = np.sum(mask, axis=1, keepdims=True)
    b = np.where(b > 0, b, 1)
    return np.sum((a * mask * weights) / b, axis=1)


def mean_absolute_error(original_trajectory, reconstructed_trajectory):
    eps = 1e-8
    mask = original_trajectory != 0.0
    a = np.sqrt(eps + np.square(original_trajectory - reconstructed_trajectory)) * mask
    b = np.sum(mask, axis=1, keepdims=True)
    b = np.where(b > 0, b, 1)
    return np.sum((a * mask) / b, axis=1)


def balanced_mean_absolute_error(original_trajectory, reconstructed_trajectory):
    weights = np.array([8.5] * 4 + [1.0] * 34, dtype=np.float32)
    eps = 1e-8
    mask = original_trajectory != 0.0
    a = np.sqrt(eps + np.square(original_trajectory - reconstructed_trajectory)) * mask
    b = np.sum(mask, axis=1, keepdims=True)
    b = np.where(b > 0, b, 1)
    return np.sum((a * mask * weights) / b, axis=1)


def binary_crossentropy(original_trajectory, reconstructed_trajectory):
    mask = original_trajectory != 0.0
    a = -original_trajectory * np.log(reconstructed_trajectory) - \
        (1 - original_trajectory) * np.log(1 - reconstructed_trajectory)
    b = np.sum(mask, axis=1, keepdims=True)
    b = np.where(b > 0, b, 1)
    return np.sum((a * mask) / b, axis=1)


def modified_mean_squared_error(y_true, y_pred):
    mask = tf.not_equal(y_true, 0.0)
    mask = tf.cast(mask, dtype=tf.float32)
    return K.mean(K.square(y_pred - y_true) * mask, axis=-1)


def modified_mean_squared_error_2(y_true, y_pred):
    mask = tf.not_equal(y_true, 0.0)
    mask = tf.cast(mask, dtype=tf.float32)
    a = (y_pred - y_true) ** 2
    b = K.sum(mask, axis=-1, keepdims=True)
    c = tf.ones_like(b)
    b = K.tf.where(b > 0.0, b, c)
    return K.sum((a * mask) / b, axis=-1)


# Balanced MSE
def modified_mean_squared_error_3(y_true, y_pred):
    weights = tf.constant([8.5] * 4 + [1.0] * 34)
    mask = tf.not_equal(y_true, 0.0)
    mask = tf.cast(mask, dtype=tf.float32)
    a = (y_pred - y_true) ** 2
    b = K.sum(mask, axis=-1, keepdims=True)
    c = tf.ones_like(b)
    b = K.tf.where(b > 0.0, b, c)
    return K.sum((a * mask * weights) / b, axis=-1)


def modified_mean_absolute_error(y_true, y_pred):
    eps = 1e-8
    mask = tf.not_equal(y_true, 0.0)
    mask = tf.cast(mask, dtype=tf.float32)
    a = K.sqrt(eps + (y_pred - y_true) ** 2)
    b = K.sum(mask, axis=-1, keepdims=True)
    c = tf.ones_like(b)
    b = K.tf.where(b > 0.0, b, c)
    return K.sum((a * mask) / b, axis=-1)


def modified_balanced_mean_absolute_error(y_true, y_pred):
    weights = tf.constant([8.5] * 4 + [1.0] * 34)
    eps = 1e-8
    mask = tf.not_equal(y_true, 0.0)
    mask = tf.cast(mask, dtype=tf.float32)
    a = K.sqrt(eps + (y_pred - y_true) ** 2)
    b = K.sum(mask, axis=2, keepdims=True)
    c = tf.ones_like(b)
    b = K.tf.where(b > 0.0, b, c)
    return K.sum((a * mask * weights) / b, axis=-1)


def modified_binary_crossentropy(y_true, y_pred):
    mask = tf.not_equal(y_true, 0.0)
    mask = tf.cast(mask, dtype=tf.float32)
    return K.mean(K.binary_crossentropy(y_true * mask, y_pred * mask), axis=-1)


def modified_binary_crossentropy_2(y_true, y_pred):
    mask = tf.not_equal(y_true, 0.0)
    mask = tf.cast(mask, dtype=tf.float32)
    a = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
    b = K.sum(mask, axis=2, keepdims=True)
    c = tf.ones_like(b)
    b = K.tf.where(b > 0.0, b, c)
    return K.sum((a * mask) / b, axis=-1)
