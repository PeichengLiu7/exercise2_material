import array

import numpy as np

# batch_size = 9
# input_shape = (3, 4, 11)
# input_tensor = np.array(range(int(np.prod(input_shape) * batch_size)), dtype=float)
# input_tensor = input_tensor.reshape(batch_size, *input_shape)
# input_tensor2 = np.ravel(input_tensor).reshape(np.shape(input_tensor)[0], -1)

# a = np.zeros((1, 2, 3, 4))
#
# print(a)
# print(input_tensor)
def f(x):
    if x > 0:
        return x + f(x-1)
    # else:
    #     return 0
print(f(100))



# print(input_tensor2)
