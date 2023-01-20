from numpy import *
import numpy as np
from Layers.Base import BaseLayer
# from Base import BaseLayer


class FullyConnected(BaseLayer):

    def __init__(self, input_size, output_size):
        self.input_tensor_new = None  # Important to access it in backward pass!
        super().__init__()  # inherits the base layer that we implemented earlier
        self.trainable = True
        self.weights = random.rand(input_size + 1, output_size)
        self.input_size = input_size
        self.output_size = output_size
        self.weights_size = (input_size, output_size)  # SAVE THE WEIGHT SIZE BUT BEFORE APPENDING BIAS!!!
        self.input_tensor_new = None
        self._optimizer = None
        self.output_tensor = None

    def forward(self, input_tensor):
        batch_size = np.shape(input_tensor)[0]
        bias_size = np.ones((batch_size, 1))
        # for create a same size  with bias, to successfully combine input tensor
        self.input_tensor_new = np.concatenate((input_tensor, bias_size), axis=1)
        self.output_tensor = np.dot(self.input_tensor_new, self.weights)
        return self.output_tensor
        # ˆy 激活函数的输入
        #  矩阵乘法

    # Defining the getter and setter of the optimizer attribute
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def backward(self, error_tensor):
        previous_error_tensor = error_tensor @ self.weights.T  # X'*W' = Y'
        self.gradient_weight = self.input_tensor_new.T @ error_tensor  # Compute the gradient of the weights

        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weight)
            # Compute the new weights

        return previous_error_tensor[:, :-1]
        # Return the Y' but without the last column (row in our memory layout) 最后一列

    @property
    def gradient_weights(self):
        return self.gradient_weight

    def initialize(self, weights_initializer, bias_initializer):
        # Append initialized weights to our current weights
        self.weights[:-1, :] = weights_initializer.initialize(self.weights_size, self.input_size, self.output_size)
        # Append the initialized bias to the weights matrix
        self.weights[-1, :] = bias_initializer.initialize(self.output_size, self.input_size, self.output_size)

