## Objective

Implement a simple two-layer feedforward neural network using the provided pseudo-codes. This network incorporates linear transformations, ReLU activations, and computes the mean squared error as its loss.

## Neural Network Overview

The network structure you'll be building consists of:

### 1. Input Layer
- This is where we feed in our data, represented by $x$.

### 2. First Hidden Layer

#### Linear Transformation
- The input data is multiplied by a weight matrix $W_1$ and added to a bias vector $b_1$.
- Formula: $h_1 = xW_1 + b_1$

#### ReLU Activation
- The output of the linear transformation, $h_1$, is passed through a Rectified Linear Unit (ReLU) activation function to introduce non-linearity.
- Formula: $a_1(i) = \max(0, h_1(i))$
- Here, $i$ denotes the $i^{th}$ element of the vector.

### 3. Second Hidden Layer

#### Linear Transformation
- The output of the first layer, $a_1$, is multiplied by a second weight matrix $W_2$ and added to another bias vector $b_2$.
- Formula: $h_2 = a_1W_2 + b_2$

### 4. Loss Calculation

#### Mean Squared Error (MSE)
- The difference between the predicted output, $h_2$, and the target, $t$, is computed using the Mean Squared Error (MSE) to evaluate the performance of the model.
- Formula: $L = \frac{1}{N} \sum_{i=1}^{N} (h_2(i) - t(i))^2$
- Here, $N$ is the number of elements in $h_2$ or $t$.

## Backpropagation

After computing the loss, the next step is to adjust the weights and biases in a manner that minimizes this loss. This is done using the backpropagation algorithm, which computes the gradient of the loss with respect to each weight and bias by applying the chain rule of calculus.

## Instructions

### `variable.py`:
- Implement the `Variable` class. Variables are used to store data, gradients, and other properties needed for backpropagation.

### `functions.py`:
- Implement the base `Function` class. This class serves as a parent for specific operations like Linear transformation and ReLU.

### `advancedfunctions.py`:
- Implement the `Linear`, `ReLU`, and `MeanSquaredError` classes. These represent the core functionalities of our neural network.

After building the necessary classes, use them to construct and run the neural network in `main.py`. Ensure the forward pass computes the correct predictions and that the backward pass updates the weights and biases based on the computed gradients.
