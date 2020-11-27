import numpy as np

class MLP(object):
    """A Multilayer Perception Class"""

    def __init__(self, num_inputs=3, hidden_layers=[3,3], num_outputs=2):
        """Constructor for the MLP. Takes the number of inputs, a variable number
        of hidden layers, and number of outputs
        Args:
            num_inputs (int) : Number of inputs
            hidden_layer (list) : A list of ints for the hidden layers
            num_outputs (int) : Number of outputs
            """
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # Create a generic representation of the layers
        layers = [num_inputs] + hidden_layers + [num_outputs]

        # Create random connection weights for the layers
        weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights

    def forward_propagate(self, inputs):
        """ Computes forward propagation of the network based on input signals.
        Args:
            inputs (ndarray): Input signals
        Returns :
            activations(ndarray): Output values
            """
        # the input layer activation is just the input itself
        activations = inputs

        # iterate through the network layer
        for w in self.weights:
            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)

            # apply sigmoid activation function
            activations = self._sigmoid(net_inputs)

        # return output layer activation
        return activations

    def _sigmoid(self, x):
        """Sigmoid activation function
            Args:
                 x (float) : Value to be processed
            Returns :
                y (float) : output
        """

        y = 1.0 / (1 + np.exp(-x))
        return y
if __name__ == "__main__":
    # create a Multilayer Perception
    mlp = MLP()

    # set random values for network's input
    inputs = np.random.rand(mlp.num_inputs)

    # perform forward propagation
    output = mlp.forward_propagate(inputs)
    print(f"inputs : {inputs}")
    print(f"output : {output}")
    print("Network activation : {}". format(output))
