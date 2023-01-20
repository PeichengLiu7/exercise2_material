from Layers.Base import BaseLayer
import numpy as np


# to avoid Vanishing Gradient
class ReLU(BaseLayer):
    def __init__(self):
        self.input_tensor = None
        super().__init__
        self.trainable = False

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        output_tensor = np.where(input_tensor > 0, input_tensor, 0)

        return output_tensor

    def backward(self, error_tensor):
        pre_tensor = np.where(self.input_tensor > 0, error_tensor, 0)

        return pre_tensor
