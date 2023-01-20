from Layers.Base import BaseLayer
import numpy as np
import copy
import math

from scipy.signal import convolve, correlate


class Conv(BaseLayer):

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.conv_shape = convolution_shape
        self.num_kernels = num_kernels

        self.weights = np.random.uniform(0, 1, (self.num_kernels, *self.conv_shape))
        self.bias = np.random.uniform(0, 1, self.num_kernels)
        self.input_tensor = None

        self._gradient_weights = None
        self._gradient_bias = None
        self._optimizer = None
        self.one = np.random.rand()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        # durch stride um 1D oder 2D zu entscheiden
        stride_len = len(self.stride_shape)
        if stride_len > 1:  # multiply dimension convolution
            SH, SW = self.stride_shape
            N, C, H, W = self.input_tensor.shape
            FN, _, FH, FW = self.weights.shape
            OH = math.ceil(H / SH)
            OW = math.ceil(W / SW)

            convResult = np.zeros([N, FN, H, W])
            intermedium1 = np.zeros([C, H, W])
            for n in range(N):
                for fn in range(FN):
                    for c in range(C):
                        intermedium1[c] = correlate(self.input_tensor[n][c], self.weights[fn][c], "same")
                    convResult[n, fn] = np.sum(intermedium1, axis=0)
                    convResult[n, fn] += self.bias[fn]

            if SH > 1 or SW > 1:
                medium = np.zeros([N, FN, OH, OW])
                for row in range(OH):
                    for col in range(OW):
                        medium[:, :, row, col] = convResult[:, :, row * SH, col * SW]
                output = medium
            else:
                output = convResult
        else:
            SW = self.stride_shape[0]
            N, C, W = self.input_tensor.shape
            FN = self.weights.shape[0]
            OW = math.ceil(W / SW)
            convResult = np.zeros([N, FN, W])
            intermedium1 = np.zeros([C, W])
            for n in range(N):
                for fn in range(FN):
                    for c in range(C):
                        intermedium1[c] = correlate(self.input_tensor[n][c], self.weights[fn][c], "same")
                    convResult[n, fn] = np.sum(intermedium1, axis=0)
                    convResult[n, fn] += self.bias[fn]

            if OW > 1:
                medium = np.zeros([N, FN, OW])
                for col in range(OW):
                    medium[:, :, col] = convResult[:, :, col * SW]
                output = medium
            else:
                output = convResult

        return output

    def backward(self, error_tensor):

        stride_len = len(self.stride_shape)
        pass_error_tensor = np.zeros_like(self.input_tensor)
        self.gradient_weights = np.zeros_like(self.weights)
        if stride_len > 1:  # multiply dimension convolution
            SH, SW = self.stride_shape
            N, _, H, W = self.input_tensor.shape
            FN, C, FH, FW = self.weights.shape
            _, _, OH, OW = error_tensor.shape

            intermedium = np.zeros([N, FN, H, W])
            for row in range(OH):
                for col in range(OW):
                    intermedium[:, :, row * SH, col * SW] = error_tensor[:, :, row, col]
                    #还原回去

            # get gradient weights
            input_tensor_pad = np.zeros([N, C, H + FH - 1, W + FW - 1])
            padH = FH // 2
            padW = FW // 2

            for n in range(N):
                for c in range(C):
                    input_tensor_pad[n, c, padH:H + padH, padW:padW + W] = self.input_tensor[n, c]

        else:  # 1D convolution
            SW = self.stride_shape[0]
            N, C, W = self.input_tensor.shape
            FN, OW = self.weights.shape[0], error_tensor.shape[2]
            FW = self.weights.shape[2]

            intermedium = np.zeros([N, FN, W])
            for col in range(OW):
                intermedium[:, :, col * SW] = error_tensor[:, :, col]

            # calculate gradients
            input_tensor_pad = np.zeros([N, C, W + FW - 1])
            padW = FW // 2
            for n in range(N):
                for c in range(C):
                    input_tensor_pad[n, c, padW:padW + W] = self.input_tensor[n, c]

        # use convolution to get a backpassed error tensor
        for n in range(N):
            for c in range(C):
                for fn in range(FN):
                    pass_error_tensor[n, c] += convolve(intermedium[n, fn], self.weights[fn, c], "same")

        # calculate bias gradient
        FN, N = error_tensor.shape[1], error_tensor.shape[0]
        self.gradient_bias = np.zeros(FN)
        for n in range(N):
            for fn in range(FN):
                self.gradient_bias[fn] += np.sum(error_tensor[n, fn])

        FN = self.weights.shape[0]
        # calculate weight gradient
        for fn in range(FN):
            for n in range(N):
                for c in range(C):
                    self.gradient_weights[fn, c] += correlate(input_tensor_pad[n, c], intermedium[n, fn], "valid")

        # update weights and bias
        if self.optimizer is not None:
            optForWeights, optForBias = copy.deepcopy(self.optimizer), copy.deepcopy(self.optimizer)
            self.weights = optForWeights.calculate_update(self.weights, self.gradient_weights)
            self.bias = optForBias.calculate_update(self.bias, self.gradient_bias)

        return pass_error_tensor

    def initialize(self, weights_initializer, bias_initializer):
        weight_shape = (self.num_kernels, *self.conv_shape)
        fan_in = self.conv_shape[0] * self.conv_shape[1] * self.conv_shape[2]
        fan_out = self.num_kernels * self.conv_shape[1] * self.conv_shape[2]

        self.weights = weights_initializer.initialize(weight_shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, someWeights):
        self._gradient_weights = someWeights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, someBias):
        self._gradient_bias = someBias

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, someOptimizer):
        self._optimizer = someOptimizer
