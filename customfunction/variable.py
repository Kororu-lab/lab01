Class Variable:
    Initialize(data, name=None):
        Set self.data to data
        Initialize grad, creator, and generation to None, None, and 0 respectively
        Set self.name to name

    Function set_creator(func):
        Set self.creator to func
        Set self.generation to func's generation + 1

    Function cleargrad():
        Set grad to None

    Function backward():
        If grad is None:
            Initialize grad with ones of same shape as data
        Create an empty list "funcs"
        Add creator to "funcs"
        While funcs is not empty:
            Pop a function from funcs
            Compute the backward gradients for function's outputs
            Update the gradients for function's inputs
            If input has a creator, add it to funcs
            Print gradient if input has a name
