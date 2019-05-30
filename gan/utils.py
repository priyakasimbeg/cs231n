import scipy
from PIL import Image, ImageFilter

import tensorflow as tf
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DIM = 28
NOISE_DIM = DIM * DIM

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# A bunch of utility functions
def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    return

def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

def rel_error(x,y):
    return np.max(tf.abs(x - y) / (tf.maximum(float32(1e-8), tf.abs(x) + tf.abs(y))))

def count_params(model):
    """Count the number of parameters in the current TensorFlow graph """
    param_count = np.sum([np.prod(p.shape) for p in model.weights])
    return param_count

def sample_noise(batch_size, dim):
    """Generate random uniform noise from -1 to 1.
    
    Inputs:
    - batch_size: integer giving the batch size of noise to generate
    - dim: integer giving the dimension of the noise to generate
    
    Returns:
    TensorFlow Tensor containing uniform noise in [-1, 1] with shape [batch_size, dim]
    """

    return tf.random.uniform([batch_size, dim], minval=-1, maxval=1)

    
    
## Data
def load_data(data_dir, blur = False, noise=False):
    image_paths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
    image_count = len(image_paths)
    
    X = np.zeros((image_count, DIM * DIM))
    
    for i in range(0, image_count):
        arr = (scipy.ndimage.imread(image_paths[i], flatten=True))
        im = Image.fromarray(arr).resize((DIM, DIM))

        if blur:
            im = im.convert("RGBA")
            im = im.filter(ImageFilter.GaussianBlur(1))
            im = im.convert("L")
            
        im = np.array(im).flatten()
        
        if noise:
            im = im +  10 * np.random.randn(im.shape[0])
            
        X[i] = im
    return X


class EPID(object):
    def __init__(self, batch_size, shuffle=False):
        data_dir = "data/train"
        X = load_data(data_dir)
        X = X.astype(np.float32)/255
        X = X.reshape((X.shape[0], -1))
        self.X = X
        self.batch_size, self.shuffle = batch_size, shuffle
        
    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i+B]) for i in range(0, N, B)) 

class PHANTOM(object):
    def __init__(self, batch_size, shuffle=False, fake=False):
        
        if fake:
            data_dir = "data/train"
            X = load_data(data_dir, blur=True, noise=True)
        else:
            data_dir = "data/noise"
            X = load_data(data_dir)
            
        X = X.astype(np.float32)/255
        X = X.reshape((X.shape[0], -1))
        self.X = X
        self.batch_size, self.shuffle = batch_size, shuffle
        
    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i+B]) for i in range(0, N, B))
    
  