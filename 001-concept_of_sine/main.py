import math
import numpy as np

# Based on: "Can Machine Learn the Concept of Sine" by Ying Xie
# Medium Link: https://towardsdatascience.com/can-machine-learn-the-concept-of-sine-4047dced3f11
# GitHub: https://github.com/looselyconnected/ml-examples

# Goal: create ANN to learn generalized model 'y = sin(A*x)'

INPUT_COUNT = 40

sine_array = lambda a, N: np.array([math.sin(a*k) for k in range(0, N)])

y = sine_array(a=0.1, N=10)

print(type(y))
print(y)


