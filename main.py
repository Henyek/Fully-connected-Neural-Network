from ctypes import Array
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

from pyparsing import original_text_for

from network import *

import numpy as np
from icecream import ic

# figure, axis = plt.subplot(2, 2)

def reLu(x):
    return np.maximum(x, 0)

def reLu_derivative(x):
    return x > 0

def leaky_reLu(x):
    return np.where(x > 0, x, x * 0.01)  
    
def leaky_reLu_derivative(x):
    return np.where(x > 0, 1, 0.01)  

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return np.multiply(sigmoid(x), (1-sigmoid(x)))

def tanh(x):
    return( (np.tanh(x) + 1) / 2 )

def tanh_derivative(x):
    return((np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)))

def sin(x):
    return np.sin(x)

def sin_derivative(x):
    return np.cos(x)

def no_func(x):
    return x

GEOMETRY = [2, 8, 16, 16, 8, 1] # GEOMETRY is an array containing a 
                              # geometry of our neural network where
                              # first element is the number of inputs
                              # and last element is the number of outputs
                

# ACTIVATION_FUNCTIONS = [leaky_reLu, leaky_reLu, leaky_reLu, leaky_reLu]
# DERIVATIVES = [leaky_reLu_derivative, leaky_reLu_derivative, leaky_reLu_derivative, leaky_reLu_derivative]
ACTIVATION_FUNCTIONS = [sin, sin, sin, sin, sin]
DERIVATIVES = [sin_derivative, sin_derivative, sin_derivative, sin_derivative, sin_derivative]
EPOCHS = 10000

PROBING_AREA = 5
POINTS_IN_ROW = 40
ROWS_TO_PROBE = 40

def probe_points() -> Tuple[List[List[Tuple[float, float]]], List[List[float]]]:
    points: Tuple[List[List[Tuple[float, float]]], List[List[float]]]
    coords = []
    vals = []
    for i in range(ROWS_TO_PROBE):
        row = []
        row_val = []
        for j in range(POINTS_IN_ROW):
            coordinates = (round(i * (PROBING_AREA / POINTS_IN_ROW), 5), round(j * (PROBING_AREA / ROWS_TO_PROBE), 5))
            val = round(origin_function(coordinates), 5)
            row.append(coordinates)
            row_val.append(val)
        coords.append(row)
        vals.append(row_val)
    points = (coords, vals)            
    return points
        

def origin_function(x) -> float:
    # return x[0]**2 + x[1]**2
    # return np.log(x[0] + 1) - x[1] / 10
    return np.sin(x[0]) * np.cos(x[1])
    # return x[0] + x[1]


def show_graph_from_function(func):
    t = np.linspace(0, PROBING_AREA)
    X, Y = np.meshgrid(t, t)
    data2d = func([X,Y])

    fig, ax = plt.subplots()
    im = ax.imshow(data2d)
    
    for i in range(POINTS_IN_ROW):
        for j in range(ROWS_TO_PROBE):
            xy = (i * (PROBING_AREA / POINTS_IN_ROW), j * (PROBING_AREA / ROWS_TO_PROBE))
            plt.plot(xy[0] * 10, xy[1] * 10, 'ro', markersize=2)
            plt.annotate('%.2f' % origin_function((xy[0], xy[1])), xy=(xy[0] * 10, xy[1] * 10), fontsize=8)
    
    fig.colorbar(im, ax=ax)

    plt.show()
    
def show_graph_nn(nn):
    t = np.linspace(0, PROBING_AREA)
    X, Y = np.meshgrid(t, t)
    data2d = nn.fProp(np.dstack([X,Y]))

    fig, ax = plt.subplots()
    im = ax.imshow(data2d)
    
    for i in range(POINTS_IN_ROW):
        for j in range(ROWS_TO_PROBE):
            xy = (i * (PROBING_AREA / POINTS_IN_ROW), j * (PROBING_AREA / ROWS_TO_PROBE))
            plt.plot(xy[0] * 10, xy[1] * 10, 'ro', markersize=2)
            plt.annotate('%.2f' % origin_function((xy[0], xy[1])), xy=(xy[0] * 10, xy[1] * 10), fontsize=8)
    
    fig.colorbar(im, ax=ax)

    plt.show()

if __name__ == "__main__":
    points = probe_points()
    inputs = points[0]
    targets = points[1]


    nn = Network(GEOMETRY, ACTIVATION_FUNCTIONS, DERIVATIVES, 0.0001)
    show_graph_from_function(origin_function)
    show_graph_nn(nn)
    for i in range(EPOCHS):
        outputs = nn.fProp(inputs)
        if i%500==0:
            nn.cost_function(np.array(outputs).flatten(), np.array(targets).flatten())
        nn.bProp(inputs, targets)
    nn.cost_function(np.array(outputs).flatten(), np.array(targets).flatten())
    outputs = nn.fProp(inputs)
    show_graph_from_function(origin_function)
    show_graph_nn(nn)