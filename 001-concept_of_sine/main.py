import math
import numpy as np
import matplotlib.pyplot as plt

# Based on: "Can Machine Learn the Concept of Sine" by Ying Xie
# Medium Link: https://towardsdatascience.com/can-machine-learn-the-concept-of-sine-4047dced3f11
# GitHub: https://github.com/looselyconnected/ml-examples

# Goal: create ANN to learn generalized model 'y = sin(A*x)'

INPUT_COUNT = 40
N = 100

x_array = np.array(range(0, N))
sine_array = lambda a, N: np.array([math.sin(a*k) for k in range(0, N)])

y = sine_array(a=0.06, N=N)

fig = plt.figure()
plt.plot(x_array, y)




plt.show()

