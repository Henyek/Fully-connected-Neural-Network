from PIL import Image

from network import *

import numpy as np
from icecream import ic


def show_image_from_array(array):
    array = np.clip(array, 0, 255).astype(np.uint8)
    new_image = Image.fromarray(array, 'RGB')
    new_image.show()

def reLu(x):
    return np.maximum(x, 0)

def reLu_derivative(x):
    return x > 0

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return np.multiply(sigmoid(x), (1-sigmoid(x)))

def tanh(x):
    return( (np.tanh(x) + 1) / 2 )

def tanh_derivative(x):
    return((np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)))

GEOMETRY = [4, 2, 2, 4] # GEOMETRY is an array containing a 
                              # geometry of our neural network where
                              # first element is the number of inputs
                              # and last element is the number of outputs

ACTIVATION_FUNCTIONS = [reLu, reLu, sigmoid]
DERIVATIVES = [reLu_derivative, reLu_derivative, sigmoid_derivative]


if __name__ == "__main__":
    inputs = np.array([1,-1,1,-1])

    nn = Network(GEOMETRY, ACTIVATION_FUNCTIONS, DERIVATIVES)
    inputs = nn.fProp(inputs)
    nn.bProp(inputs, np.array([2, 3, 5, 2]))
    # show_image_from_array(inputs.reshape(32, 32, 3) * 255)