Function as_array(x):
    If x is scalar:
        Return x as an array
    Else:
        Return x

Class Function:
    Call(*inputs):
        Extract data from inputs
        Compute forward results
        For each input:
            Set its creator as this function
        Store inputs and outputs in the function
        Return outputs

    Forward(xs):
        Placeholder for forward computation

    Backward(gys):
        Placeholder for backward computation
