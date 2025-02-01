import numpy as np
from icecream import ic

class Network:
    def __init__(self, GEOMETRY, activationFunctions, derivatives, learning_ratio):
        if len(GEOMETRY) - 1 == len(activationFunctions):
            self.weights = []
            for id, layer in enumerate(GEOMETRY[1:]):
                # Initialize weights with biases included
                W = np.random.uniform(low=-1, high=1, size=(GEOMETRY[id] + 1, layer))
                self.weights.append(W)
            self.GEOMETRY = GEOMETRY
            self.activationFunctions = activationFunctions
            self.derivatives = derivatives
            self.learning_ratio = learning_ratio
            self.layer_outputs = []  # Raw layer outputs (pre-activation)
            self.layer_after_activation = []  # Activated layer outputs
            self.nn_outputs = []  # Final network outputs
        else:
            raise ValueError("Mismatch between layers and activation functions.")

    def cost_function(self, values, targets):
        cost = np.mean((targets - values) ** 2)
        ic(cost)
        return cost

    def cost_derivative(self, values, targets):
        # Derivative of Mean Squared Error: 2 * (output - target)
        return 2 * (values - targets)

    def fProp(self, inputs):
        temp_inputs = np.array(inputs).reshape(-1, np.array(inputs).shape[-1])
        self.layer_outputs = []
        self.layer_after_activation = []

        network_output = []
        for input_set in temp_inputs:
            temp_activation = []
            temp_outputs = []
            # Input with bias
            temp_input = np.insert(np.array(input_set), 0, [1])
            for id, layer in enumerate(self.weights):
                outputs = np.dot(temp_input, layer)
                temp_outputs.append(outputs)

                activated_output = self.activationFunctions[id](outputs)
                temp_activation.append(activated_output)

                # Add bias for the next layer
                temp_input = np.insert(activated_output, 0, [1])

            self.layer_outputs.append(temp_outputs)
            self.layer_after_activation.append(temp_activation)

            network_output.append(temp_activation[-1])
        self.nn_outputs = network_output
        return np.array(network_output).reshape(np.array(inputs).shape[0], np.array(inputs).shape[1])

    def adjust_network(self, gradients):
        # Update weights using computed gradients
        for id, gradient in enumerate(gradients):
            self.weights[id] -= self.learning_ratio * gradient

        # Reset stored outputs after adjustment
        self.layer_outputs = []
        self.layer_after_activation = []
        self.nn_outputs = []

    def bProp(self, inputs, targets):
        temp_inputs = np.dstack(inputs).reshape(-1, np.array(inputs).shape[-1])
        batch_size = len(temp_inputs)
        gradients = [np.zeros_like(w) for w in self.weights]
        temp_targets = np.array(targets).flatten()
        for i in range(batch_size):
            # Forward pass already performed; use stored outputs
            output_error = self.cost_derivative(self.nn_outputs[i], temp_targets[i])

            # Delta for the output layer
            delta = output_error * self.derivatives[-1](self.layer_outputs[i][-1])
            
            # Backpropagate through layers
            for l in range(len(self.GEOMETRY) - 2, -1, -1):
                # Add bias to the previous layer's activations
                if l > 0:
                    prev_activation = np.insert(self.layer_after_activation[i][l - 1], 0, [1])
                else:
                    prev_activation = np.insert(temp_inputs[i], 0, [1])
                
                # Compute weight gradients
                gradients[l] += np.outer(prev_activation, delta)

                # Compute delta for the previous layer (skip if input layer)
                if l > 0:
                    delta = (np.dot(self.weights[l][1:], delta)) * self.derivatives[l - 1](self.layer_outputs[i][l - 1])

        # Average gradients over the batch
        gradients = [g / batch_size for g in gradients]

        self.adjust_network(gradients)