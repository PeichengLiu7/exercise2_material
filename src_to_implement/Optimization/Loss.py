#  Team-Mitglieder : Huang, Jin [an46ykim]; Liu, Peicheng [ha46tika]
import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        return -np.sum(label_tensor * np.log(prediction_tensor + np.finfo(float).eps))

    # 􏰈 −ln(ˆyk +ε)where =1 (15)
    # ε represents the smallest representable number. Take a look into np.finfo.eps

    def backward(self, label_tensor):
        return -label_tensor / (self.prediction_tensor + np.finfo(float).eps)
