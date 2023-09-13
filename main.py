
import numpy as np
from customfunction.functions import Function
from customfunction.advancedfunctions import *

# Input and Target data
x = Variable(np.array([[1.0, 0.5]]), name="Input")
t = Variable(np.array([[1.0, 0.5]]), name="Target")

# First layer weights and biases
W1 = Variable(np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]), name="W1")
b1 = Variable(np.array([0.1, 0.2, 0.3]), name="b1")

# Second layer weights and biases
W2 = Variable(np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]), name="W2")
b2 = Variable(np.array([0.1, 0.2]), name="b2")

# Forward pass
h1 = linear(x, W1, b1)
a1 = relu(h1)
h2 = linear(a1, W2, b2)
loss = mean_squared_error(h2, t)

# Backward pass
print("Starting backward pass...\n")
loss.backward()
