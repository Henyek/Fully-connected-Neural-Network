from PIL import Image

try:
    import cupy as np
except: 
    import numpy as np

from network import *

def show_image_from_array(array):
    array = np.clip(array, 0, 255).astype(np.uint8)
    new_image = Image.fromarray(array, 'RGB')
    new_image.show()

def reLu(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return( (np.tanh(x) + 1) / 2 )

GEOMETRY = [3072, 256, 256, 3072] # GEOMETRY is an array containing a 
                              # geometry of our neural network where
                              # first element is the number of inputs
                              # and last element is the number of outputs

ACTIVATION_FUNCTIONS = [reLu, reLu, sigmoid]


if __name__ == "__main__":
    inputs = np.random.rand(32,32, 3)

    nn = Network(GEOMETRY, ACTIVATION_FUNCTIONS)

    inputs = nn.fProp(inputs)
    show_image_from_array(inputs.reshape(32, 32, 3) * 255)