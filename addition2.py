import numpy as np
from random import random




#initial weights, derivatives and actions
# Im items [0.95624786 0.41699756]
# [0.39875302]
# 2 5
# 5 1
# im weights [array([[0.13368545, 0.87449214, 0.564836  , 0.83269763, 0.76456232],
#        [0.7464612 , 0.03174006, 0.16076652, 0.16659563, 0.09792046]]), array([[0.75003682],
#        [0.4287732 ],
#        [0.77202862],
#        [0.4175419 ],
#        [0.06944695]])]
# im derivatives [array([[0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0.]]), array([[0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.]])]
# im activations [array([0., 0.]), array([0., 0., 0., 0., 0.]), array([0.])] [2, 5, 1]



class MLP(object):
    """A Multilayer Perceptron class.
    """

    def __init__(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=2):
        """Constructor for the MLP. Takes the number of inputs,
            a variable number of hidden layers, and number of outputs
        Args:
            num_inputs (int): Number of inputs
            hidden_layers (list): A list of ints for the hidden layers
            num_outputs (int): Number of outputs
        """

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # create a generic representation of the layers
        layers = [num_inputs] + hidden_layers + [num_outputs]

        # create random connection weights for the layers
        weights = []
        for i in range(len(layers) - 1):
            print(layers[i], layers[i + 1])
            w = np.random.rand(layers[i], layers[i + 1]) ## basically the weights rae between layers. SO previous layer * next layer is the numbers of weights required for fwd prop
            weights.append(w)
        self.weights = weights
        print('im weights',  self.weights)
        

        # save derivatives per layer
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            # print('im derivatives', i, d)

            derivatives.append(d)
        self.derivatives = derivatives
        print('im derivatives',  self.derivatives)


        # save activations per layer
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations
        print('im activations', self.activations, layers)


    def forward_propagate(self, inputs):
        """Computes forward propagation of the network based on input signals.
        Args:
            inputs (ndarray): Input signals
        Returns:
            activations (ndarray): Output values
        """

        # the input layer activation is just the input itself
        activations = inputs

        # save the activations for backpropogation
        self.activations[0] = activations

        # iterate through the network layers
        for i, w in enumerate(self.weights):
            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)

            # apply sigmoid activation function
            activations = self._sigmoid(net_inputs)

            # save the activations for backpropogation
            self.activations[i + 1] = activations

        # return output layer activation
        return activations


    def back_propagate(self, error):
        """Backpropogates an error signal.
        dE/dW_yi = (y-a[i+1])) s'(h_[i+1])) a_i
        s'(h_[i+1]) = s(h_[i+1]) (1-s(h_[i+1]))
        s(h_[i+1]) = a_[i+1]

        dE/dW_[i-1] = (y-a[i+1]) s'(h[i+1])) W_i s'(h_i) a[i-1]

        Args:
            error (ndarray): The error to backprop.
        Returns:
            error (ndarray): The final error of the input
        """

        # iterate backwards through the network layers
        for i in reversed(range(len(self.derivatives))):

            # get activation for previous layer
            activations = self.activations[i+1]

            # apply sigmoid derivative function
            delta = error * self._sigmoid_derivative(activations)
            # print('im delta',delta)
            # reshape delta as to have it as a 2d array
            delta_re = delta.reshape(delta.shape[0], -1).T ##numpy trickery tom move from ndarray([0.1,0.2]) --> ndarray([0.1], [0.2])
            # print('im delta_re',delta_re)

            # get activations for current layer
            current_activations = self.activations[i]

            # reshape activations as to have them as a 2d column matrix
            current_activations = current_activations.reshape(current_activations.shape[0],-1) ##numpy trickery tom move from ndarray([0.1,0.2]) --> ndarray([0.1], [0.2])

            # save derivative after applying matrix multiplication
            self.derivatives[i] = np.dot(current_activations, delta_re)

            # backpropogate the next error
            error = np.dot(delta, self.weights[i].T) # (y-a[i+1]) s'(h[i+1])) W_i


    def train(self, inputs, targets, epochs, learning_rate):
        """Trains model running forward prop and backprop
        Args:
            inputs (ndarray): X
            targets (ndarray): Y
            epochs (int): Num. epochs we want to train the network for
            learning_rate (float): Step to apply to gradient descent
        """
        # now enter the training loop
        for i in range(epochs):
            sum_errors = 0

            # iterate through all the training data
            for j, input in enumerate(inputs):
                target = targets[j]

                # activate the network!
                output = self.forward_propagate(input)

                error = target - output

                self.back_propagate(error)

                # now perform gradient descent on the derivatives
                # (this will update the weights
                self.gradient_descent(learning_rate)

                # keep track of the MSE for reporting later
                sum_errors += self._mse(target, output)

            # Epoch complete, report the training error
            print("Error: {} at epoch {}".format(sum_errors / len(items), i+1))

        print("Training complete!")
        print("=====")


    def gradient_descent(self, learningRate=1):
        """Learns by descending the gradient
        Args:
            learningRate (float): How fast to learn.
        """
        # update the weights by stepping down the gradient
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learningRate


    def _sigmoid(self, x):
        """Sigmoid activation function
        Args:
            x (float): Value to be processed
        Returns:
            y (float): Output
        """

        y = 1.0 / (1 + np.exp(-x))
        return y


    def _sigmoid_derivative(self, x):
        """Sigmoid derivative function
        Args:
            x (float): Value to be processed
        Returns:
            y (float): Output
        """
        return x * (1.0 - x)


    def _mse(self, target, output):
        """Mean Squared Error loss function
        Args:
            target (ndarray): The ground trut
            output (ndarray): The predicted values
        Returns:
            (float): Output
        """
        return np.average((target - output) ** 2)


if __name__ == "__main__":

    # create a dataset to train a network for the sum operation
    items = np.array([[random() for _ in range(2)] for _ in range(10000)])
    targets = np.array([[i[0] * i[1]] for i in items])
    import random


    # X =[]
    # y =[]
    # for i in range(1000):
    #     X.append([random(), random()])
    #     y.append(sum(X[i]))

    # print('Im X', X[0])
    # items = np.array(X)
    print('Im items', items[0])

    # targets = np.array(y).reshape(-1,1)
    # print('Im targets', targets[0])

    # print(items)

    print(targets[0])

    # create a Multilayer Perceptron with one hidden layer
    mlp = MLP(2, [5], 1)

    # train network
    mlp.train(items, targets, 50, 0.1)

    # create dummy data
    # input = np.array([0.3, 0.1])
    input = np.array([.3, .41])

    target = np.array([0.4])

    # get a prediction
    output = mlp.forward_propagate(input)

    # print()
    print("Our prediction for {} * {} is equal to {}".format(input[0], input[1], output[0]))