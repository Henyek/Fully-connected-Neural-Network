import matplotlib.pyplot as plt
from typing import Tuple, List

from network import *

import numpy as np

from tqdm import trange

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
    return np.tanh(x)

def tanh_derivative(x):
    return 1-tanh(x)**2

def sin(x):
    return np.sin(x)

def sin_derivative(x):
    return np.cos(x)

def no_func(x):
    return x

def no_func_derivative(x):
    return 1

GEOMETRY = [2, 64, 128, 128, 1] # GEOMETRY is an array containing a 
                              # geometry of our neural network where
                              # first element is the number of inputs
                              # and last element is the number of outputs
                
LEARNING_FACTOR = 0.00001
COST_UPDATE = 10

ACTIVATION_FUNCTIONS = [tanh, tanh, reLu, no_func]
DERIVATIVES = [tanh_derivative, tanh_derivative, reLu_derivative, no_func_derivative]


EPOCHS = 10000
CENTER_SHIFT = 2
PROBING_AREA = 4
POINTS_IN_ROW = 30
ROWS_TO_PROBE = 30

def probe_points() -> Tuple[List[List[Tuple[float, float]]], List[List[float]]]:
    points: Tuple[List[List[Tuple[float, float]]], List[List[float]]]
    coords = []
    vals = []
    for i in range(ROWS_TO_PROBE):
        row = []
        row_val = []
        for j in range(POINTS_IN_ROW):
            coordinates = (round(i * (PROBING_AREA / POINTS_IN_ROW) - CENTER_SHIFT, 5), round(j * (PROBING_AREA / ROWS_TO_PROBE) - CENTER_SHIFT, 5))
            val = round(origin_function(coordinates), 5) + np.random.uniform(high=1, low=-1)
            row.append(coordinates)
            row_val.append(val)
        coords.append(row)
        vals.append(row_val)
    points = (coords, vals)            
    return points
        

def origin_function(x) -> float:
    return (1 - x[0])**2+100*(x[1]-x[0]**2)**2
    # return x[0]**2 + x[1]**2
    # return np.sin(x[0]) * np.cos(x[1])


def show_graph_from_function(func):
    t = np.linspace(-CENTER_SHIFT, PROBING_AREA - CENTER_SHIFT)
    X, Y = np.meshgrid(t, t)
    data2d = func([X,Y])

    fig, ax = plt.subplots()
    im = ax.imshow(data2d)

    fig.colorbar(im, ax=ax)

    plt.show()
    
def show_graph_nn(nn):
    t = np.linspace(-CENTER_SHIFT, PROBING_AREA - CENTER_SHIFT)
    X, Y = np.meshgrid(t, t)
    data2d = nn.fProp(np.dstack([X,Y])/PROBING_AREA)
    fig, ax = plt.subplots()
    im = ax.imshow(data2d)

    
    fig.colorbar(im, ax=ax)

    plt.show()
    
def show_graph_from_array(arr):
    plt.rcParams["figure.figsize"] = [5, 3]
    plt.rcParams["figure.autolayout"] = True

    x = np.array([i[0] for i in arr])
    y = np.array([i[1] for i in arr])

    plt.title("Cost through time")
    plt.plot(x, y, color="red")

    plt.show()

def show_error_graph(outputs, targets):
    error = np.subtract(np.array(targets).T, np.array(outputs).T)
    fig, ax = plt.subplots()
    im = ax.imshow(error)
    fig.colorbar(im, ax=ax)

    plt.show()
    
def probe_validation(perc):
    points: Tuple[List[Tuple[float, float]], List[float]]
    point_num = int(POINTS_IN_ROW * ROWS_TO_PROBE * perc)
    coords = []
    vals = []
    for i in range(point_num):
        temp_coords = np.random.random_sample(2) * PROBING_AREA - CENTER_SHIFT
        coords.append(tuple(temp_coords))
        vals.append(origin_function(temp_coords))
    points = (coords, vals)
    return points


if __name__ == "__main__":
    points = probe_points()
    inputs = np.array(points[0])/PROBING_AREA
    targets = points[1]
    
    validation_points = probe_validation(.15)
    validation_inputs = np.array(validation_points[0])/PROBING_AREA
    validation_targets = validation_points[1]
    
    test_points = probe_validation(.05)
    test_inputs = np.array(validation_points[0])/PROBING_AREA
    test_targets = validation_points[1] 
    cost_through_time = []
    
    nn = Network(GEOMETRY, ACTIVATION_FUNCTIONS, DERIVATIVES, LEARNING_FACTOR)
    t = trange(EPOCHS, desc='Bar desc', leave=True)
    cost = 0
    for i in t:
        outputs = nn.fProp(inputs)
        nn.bProp(inputs, targets)
        temp_cost = nn.cost_function(np.array(nn.fProp([validation_inputs])).flatten(), np.array(validation_targets).flatten())
        temp_arr = (i, temp_cost)
        if i%COST_UPDATE==0:
            nn.learning_ratio *= 0.99
            t.set_description("cost: %f " % temp_cost)
            t.refresh()
        cost_through_time.append(temp_arr)
        nn.clear_net()
        cost = temp_cost
    final_cost = nn.cost_function(np.array(nn.fProp([test_inputs])).flatten(), np.array(test_targets).flatten())
    show_graph_from_function(origin_function)
    show_graph_nn(nn)
    show_error_graph(outputs, targets)
    show_graph_from_array(cost_through_time)