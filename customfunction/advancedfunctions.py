Class Linear (Inherits from Function):
    Forward(x, W, b):
        Compute y = xW + b
        Return y

    Backward(gy):
        Compute gradients for x, W, and b
        Return gx, gW, gb

Function linear(x, W, b):
    Call the Linear class with x, W, b

Class ReLU (Inherits from Function):
    Forward(x):
        Compute y = max(0, x)
        Return y

    Backward(gy):
        Compute gradient for x using gy and x
        Return gx

Function relu(x):
    Call the ReLU class with x

Class MeanSquaredError (Inherits from Function):
    Forward(y, t):
        Compute mean squared error between y and t
        Return error

    Backward(gy):
        Compute gradient for y and t using gy, y, and t
        Return gy and -gy

Function mean_squared_error(y, t):
    Call the MeanSquaredError class with y and t
