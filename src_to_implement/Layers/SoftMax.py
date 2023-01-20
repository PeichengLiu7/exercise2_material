#  Team-Mitglieder : Huang, Jin [an46ykim]; Liu, Peicheng [ha46tika]
import numpy as np
from Layers.Base import BaseLayer


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.output_tensor = None
        self.t = None

    def forward(self, input_tensor):
        xk_stability = input_tensor - np.max(input_tensor)  # shift xk to increase stability. for overflow to prevent
        # und This leaves the scores unchanged!
        yk_up = np.exp(xk_stability)
        yk_down = np.sum(np.exp(xk_stability), axis=1, keepdims=True)  # axis=1 sums across the rows
        self.output_tensor = yk_up / yk_down

        return self.output_tensor

    def backward(self, error_tensor):
        For_previouslaye = self.output_tensor * (error_tensor - np.sum(np.multiply(error_tensor, self.output_tensor), axis=1, keepdims=True))
        return  For_previouslaye