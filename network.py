import numpy as np
from icecream import ic

class Network:
    def __init__(self, GEOMETRY, activationFunctions, derivatives):
        if len(GEOMETRY) - 1 == len(activationFunctions):
            self.weights = []
            for id, layer in enumerate(GEOMETRY[1:]):
                W = np.array([np.random.uniform(low = -1, high = 1, size = (GEOMETRY[id], layer))])
                self.weights.append(W)
            self.activationFunctions = activationFunctions
            self.derivatives = derivatives
            self.weightGradient = []
            self.layerOutputs = []
            self.nodeOutputs = []
        else:
            print("wronk")
            return False


    def cost_function(self, values, target):
        return np.subtract(target, values)**2


    def fProp(self, inputs):
        inputs = inputs.flatten()
        for id, layer in enumerate(self.weights):
            layer = np.array(layer).reshape(np.shape(layer[-1]))
            inputs = np.dot(inputs, layer)
            np.sum(inputs)
            self.layerOutputs.append(inputs)
            inputs = self.activationFunctions[id](inputs)
            self.nodeOutputs.append(inputs)
        return inputs
    
    def bProp(self, inputs, target):
        baseGradient = np.multiply(2*(np.subtract(target, self.nodeOutputs[-1])),self.derivatives[-1](self.layerOutputs[-1]))
        layerGradient = []
        for x in self.nodeOutputs[-2]:
            layerGradient.append(baseGradient * x)
        self.weightGradient.insert(0, layerGradient)# base gradient is the first part of the equation before multiplying by the values of the layer before

        #choose weight and iterate over all nodes in the layer before
