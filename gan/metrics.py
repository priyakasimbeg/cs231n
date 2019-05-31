import tensorflow as tf
import numpy as np
from skimage.measure import compare_ssim as sk_ssim
from gan.utils import tensor_to_array, array_to_tensor
import scipy.signal as signal
import matplotlib.pyplot as plt


def mse(x, y):
    """
    Computes the MSE error between two images
    """
    
    try:
        N, D = tf.shape(x)
    except ValueError:  
        try:
            N, C, D = tf.shape(x)
        except ValueError:
            try:
                D = tf.shape(x)
            except ValueError as e:
                raise e           
        
    error = tf.cast(tf.square(tf.reduce_sum(x - y)), tf.float32) / tf.cast((D), tf.float32)
    
    return error


def psnr(x, y):
    """
    Computes the Peak Signal to Noise Ratio (PSNR) between two images.
    x, y: Tensors of shape (1, H, W, 1)
    """
    return float(tf.image.psnr(x, y, 1))


def ssim(x, y):
    """
    Computes the structural similarity index between two images.
    """
    x = tf.image.convert_image_dtype(x, tf.float32)
    y = tf.image.convert_image_dtype(y, tf.float32)
    return float(tf.image.ssim(x, y, 1))


def wiener(x):
    """
    Applies Wiener filter to x.
    x: Tensor of shape (1, H, W, 1)
    """
    x_array = tensor_to_array(x)
    x_filtered = signal.wiener(x_array)
    
    return array_to_tensor(x_filtered)
    
    