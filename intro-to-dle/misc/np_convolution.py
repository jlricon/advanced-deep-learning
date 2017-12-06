import numpy as np
from scipy.signal import convolve2d
my_array = np.array([[0, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]])

kernel = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]])

convolved = convolve2d(my_array, kernel, mode="same")
print(convolved)
