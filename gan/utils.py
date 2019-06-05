import scipy
from PIL import Image, ImageFilter

import tensorflow as tf
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DIM = 32
NOISE_DIM = DIM * DIM

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# A bunch of utility functions

# Plotting functions
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

def show_images_set(image_set, labels = ['(a)', '(b)', '(c)']):
    
    assert(len(image_set) == len(labels))
    
    images = image_set[0]
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))
    
    fig = plt.figure(figsize=(len(image_set) * sqrtn, sqrtn))
    outer_grid = gridspec.GridSpec(1, len(image_set)) # gridspec with two adjacent horizontal cells
    outer_grid.update(wspace=0.15, hspace=0.05)

    for j, images in enumerate(image_set):
        cell = outer_grid[0,j] # the subplotspec within outer grid
        gs = gridspec.GridSpecFromSubplotSpec(sqrtn, sqrtn, cell)
        
        images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
        
        
        for i, img in enumerate(images):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            if i == 0:
                ax.set_title(labels[j], fontsize=16)
            plt.imshow(img.reshape([sqrtimg, sqrtimg]))
            
    return fig

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
            im = im +  5 * np.random.randn(im.shape[0])
            
        X[i] = im
    return X


class EPID(object):
    def __init__(self, batch_size, shuffle=False, data_base_dir='data'):
        data_dir = os.path.join(data_base_dir, 'train')
        X = load_data(data_dir)
        X = X.astype(np.float32)/255
        X = X.reshape((X.shape[0], -1))
        self.X = X
        self.train = X[:200]
        self.val = X[200:250]
        self.test = X[250:300]
        self.batch_size, self.shuffle = batch_size, shuffle
        
    def __iter__(self):
        N, B = self.train.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.train[i:i+B]) for i in range(0, N, B)) 
    
    def get_tensor(self, index):
        image = self.X[index]
        image = np.reshape(image, (1, DIM, DIM, 1))
        return tf.multiply(image, 1)
    
    def get_2Darray(self, index):
        image = self.X[index]
        image = np.reshape(image, (DIM, DIM))
        return image
                     

class PHANTOM(object):
    def __init__(self, batch_size, shuffle=False, fake=False, data_base_dir='data'):
        if fake:
            data_dir = os.path.join(data_base_dir, 'train')
            X = load_data(data_dir, blur=True, noise=True)
        else:
            data_dir = os.path.join(data_base_dir, 'noise')
            X = load_data(data_dir)
            
        X = X.astype(np.float32)/255
        X = X.reshape((X.shape[0], -1))
        self.X = X
        self.train = X[:200]
        self.val = X[200:250]
        self.test = X[250:300]
        self.batch_size, self.shuffle = batch_size, shuffle
        
    def __iter__(self):
        N, B = self.train.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.train[i:i+B]) for i in range(0, N, B))
    
    def get_tensor(self, index):
        image = self.X[index]
        image = np.reshape(image, (1, DIM, DIM, 1))
        return tf.multiply(image, 1)
    
    def get_2Darray(self, index):
        image = self.X[index]
        image = np.reshape(image, (DIM, DIM))
        return image
    
def array_to_tensor(x):
    H, W = np.shape(x)
    x = np.reshape(x, (1, H, W, 1))
    return tf.multiply(x, 1)

def tensor_to_array(x):
    _, H, W, _ = x.shape.as_list()      
    x = np.reshape(x, (H, W))
    return x 

def array_batch_to_tensor(x):
    N, D = np.shape(x)
    H = int(np.sqrt(D))
    x = np.reshape(x, (N, H, H, 1))
    
    return tf.multiply(x, 1)

def plot_tensor(x):
    """
    Plots a tensor representing an image.
    """
    plt.imshow(tensor_to_array(x))
    plt.show()
                
  