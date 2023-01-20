from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.ReLU import ReLU
from Layers.SoftMax import SoftMax
from Optimization.Loss import CrossEntropyLoss
from Optimization.Optimizers import Sgd
import copy


class NeuralNetwork:
    def __init__(self,optimizer,weights_initializer,bias_initializer):
        self._optimizer=optimizer
        self.loss=list()
        self.layers=list()
        self.data_layer = None
        self.loss_layer = None

        self.input_tensor = None
        self.label_tensor = None

        self.weights_initializer=weights_initializer
        self.bias_initializer=bias_initializer


    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self,AnyOptimizer):
        self._optimizer=AnyOptimizer

    def forward(self):
        self.input_tensor,self.label_tensor=self.data_layer.next()

        for singleLayer in self.layers:
            self.input_tensor=singleLayer.forward(self.input_tensor)

        loss=self.loss_layer.forward(self.input_tensor,self.label_tensor)
        return loss

    def backward(self):
        backPropagation=self.loss_layer.backward(self.label_tensor)

        for oneLayer in reversed(self.layers):
            backPropagation=oneLayer.backward(backPropagation)


    def append_layer(self,layer):
        if layer.trainable:
            newOptimizer=copy.deepcopy(self.optimizer)
            layer.optimizer=newOptimizer

            layer.initialize(self.weights_initializer,self.bias_initializer)


        self.layers.append(layer)

        return self.layers

    def train(self,iterations):
        for i in range(iterations):
            oneTimeLoss=self.forward()
            self.loss.append(oneTimeLoss)
            self.backward()

    def test(self,input_tensor):
        self.input_tensor=input_tensor.copy()
        for singleLayer in self.layers:
            self.input_tensor=singleLayer.forward(self.input_tensor)

        prediction=self.input_tensor.copy()

        return prediction