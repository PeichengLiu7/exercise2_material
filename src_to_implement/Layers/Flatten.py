#  Team-Mitglieder : Huang, Jin [an46ykim]; Liu, Peicheng [ha46tika]
import numpy as np
from Layers.Base import BaseLayer


class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.output_tensor = None
        self.input_tensor_shape = None
        self.error_tensor = None

    def forward(self, input_tensor):
        self.input_tensor_shape = np.shape(input_tensor)
        input_tensor = np.ravel(input_tensor).reshape(self.input_tensor_shape[0], -1)
        # Calculate the number of columns
        # self.input_tensor_shape[0]: The amount of extracted batch_size
        # input_tensor = self.input_tensor_shape.ravel()
        # np.ravel(input_tensor).reshape(self.input_tensor_shape[0], -1)
        return input_tensor

    def backward(self, error_tensor):
        error_tensor = error_tensor.reshape(self.input_tensor_shape)
        return error_tensor
