"""Utility functions da do not fit in naturally elsewhere."""
import numpy as np

# SGD should be used when calculating the gradient, to reduce the computations.
# We should at least test and see how much it affects the convergence. 
# This does not draw without replacement however, does it?
def stochastic_elements(Z_0, C, I, chunk): 
    """Pick out a fixed amount of elements from Z."""        
    start = np.random.randint(I-chunk)
    Z0_chunk = Z_0[:,start:start+chunk] 
    C_chunk = C[start:start+chunk]
    return Z0_chunk, C_chunk 
# Not sure where to put this utility-function yet. 
