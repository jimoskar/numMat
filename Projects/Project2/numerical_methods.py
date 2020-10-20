"""Implement the numerical methods here, since one needs one neural network per gradient in the formulas."""
from project2vs import *

def symplectic_euler_step(y, h, grad_T, grad_V): 
    """Step in symplectic Euler. Used in run_symplectic_euler to take each step."""
    # Assume that the input is passed as a vector y.shape(d, I), 
    # to match the dimension of the gradients.

    # Know that we have two coordinates (q and p). (assuming that index 0 = q)
    q1 = y[0, :] + h*grad_T[0, :]
    p1 = y[1, :] - h*grad_V[0, ] # Where in the gradient, which has shape (d, I) can we use indexing of q and p values?
                                 # This is of course needed in the numerical methods!     

    return np.array([q1, p1]) # Returns the solution as a np.array. 

def run_symplectic_euler(values, NNT, NNV, steps, stepsize):
    """Symplectic Euler; first order method for integrating functions numerically."""
    solution = np.zeros((steps, 2))
    # Calculate the gradients for NNT and NNV.
    for t in range(steps):
        # Or perhaps the gradients should be calculated here in each new step, based on the two NNs. 
        solution[t, :] = symplectic_euler_step(values, stepsize, grad_T, grad_V)
    return solution 

def stormer_verlet_step(y, delta_t, grad_T, grad_V):
    """Step in Størmer-Verlet. Used in run_stormer_verlet to take each step."""
    # Assume that the input is passed as a vector y = [q, p]. 
    # Maybe implement Euler correctly first, then use the same principles here!
    p12 = y[1, :] - delta_t/2*grad_V[] # indexing somehow.
    q1 = y[0, :] + delta_t*grad_T[] # same as above.
    p1 = p12 - delta_t/2*grad_V[] # same as above!

    return np.array([q1, p1]) # Returns the solution as a np.array. 

def run_stormer_verlet(values, NNT, NNV, steps, stepsize):
    """Størmer-Verlet; second order method for integrating functions numerically."""
    solution = np.zeros((steps, 2))
    # Calculate the gradients for NNT and NNV.
    for t in range(steps):
        # Or perhaps the gradients should be calculated here in each new step, based on the two NNs. 
        solution[t, :] = stormer_verlet_step(values, stepsize, grad_T, grad_V)
    return solution 
    