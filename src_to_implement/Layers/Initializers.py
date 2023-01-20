import numpy as np

class Constant:

    def __init__(self,constant=0.1):
        self.constant=constant

    def initialize(self,weight_shape,fan_in,fan_out):

        return np.ones(weight_shape)*self.constant

class UniformRandom:

    def __init__(self,lower=0,upper=1):
        self.lower=lower
        self.upper=upper

    def initialize(self, weight_shape, fan_in, fan_out):
        return np.random.uniform(self.lower,self.upper,weight_shape)

class Xavier:

    def __init__(self,mean=0):
        self.mean=mean
        self.sigma=0

    def initialize(self, weight_shape, fan_in, fan_out):
        self.sigma=np.sqrt(2/(fan_out+fan_in))

        return np.random.normal(loc=self.mean,scale=self.sigma,size=weight_shape)

class He:

    def __init__(self):
        self.sigma=0

    def initialize(self, weight_shape, fan_in, fan_out):
        self.sigma=np.sqrt(2/fan_in)

        return np.random.normal(loc=0,scale=self.sigma,size=weight_shape)




# from Layers.Base import BaseLayer
# import numpy as np
#
#
# class Constant(BaseLayer):
#
#     def __init__(self, constant_default=0.1):
#         super().__init__()
#         self.initial_tensor = None
#         self.con = constant_default
#
#     def initialize(self, weights_shape, fan_in, fan_out):
#         # self.initial_tensor = np.full(shape=weights_shape, fill_value=self.con)
#         return self.initial_tensor
#
#
# class UniformRandom(BaseLayer):
#
#     # def __init__(self):
#     #     super().__init__()
#     #     self.initial_tensor = None
#
#     def __init__(self):
#         super().__init__()
#         self.initial_tensor = None
#
#     def initialize(self, weights_shape, fan_in, fan_out):
#         self.initial_tensor = np.random.uniform(size=weights_shape)
#         return self.initial_tensor
#
#
# class Xavier(BaseLayer):
#
#     # def __init__(self):
#     #     super().__init__()
#
#     def __init__(self):
#         super().__init__()
#         self.initial_tensor = None
#
#     def initialize(self, weights_shape, fan_in, fan_out):
#         sigma = np.sqrt(2 / (fan_in + fan_out))
#         self.initial_tensor = np.random.normal(loc=0, scale=sigma, size=weights_shape)
#         return self.initial_tensor
#
#
# class He(BaseLayer):
#
#     # def __init__(self):
#     #     super().__init__()
#
#     def __init__(self):
#         super().__init__()
#         self.initial_tensor = None
#
#     def initialize(self, weights_shape, fan_in, fan_out):
#         sigma = np.sqrt(2 / fan_in)
#         self.initial_tensor = np.random.normal(loc=0, scale=sigma, size=weights_shape)
#         return self.initial_tensor
