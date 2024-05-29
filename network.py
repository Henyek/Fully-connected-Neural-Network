try:
    import cupy as np
except: 
    import numpy as np



class Network:
    def __init__(self, GEOMETRY, activationFunctions):
        if len(GEOMETRY) - 1 == len(activationFunctions):
            self.layers = []
            for id, layer in enumerate(GEOMETRY[1:]):
                W = np.array([np.random.uniform(low = -1, high = 1, size = (GEOMETRY[id], layer))])
                self.layers.append(W)
            self.activationFunctions = activationFunctions
        else:
            print("wronk")
            return False


    def fProp(self, inputs):
        inputs = inputs.flatten()
        for id, layer in enumerate(self.layers):
            layer = np.array(layer).reshape(np.shape(layer[-1]))
            inputs = np.dot(inputs, layer)
            np.sum(inputs)
            inputs = self.activationFunctions[id](inputs)
        return inputs