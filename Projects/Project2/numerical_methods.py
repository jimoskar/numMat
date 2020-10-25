"""Implement the numerical methods here, since one needs one neural network per gradient in the formulas."""
from network_algo import *

def symplectic_euler_NN(NNT, NNV, q0, p0, times,d0):
    """Symplectic Euler; first order method for integrating functions numerically.
    
    Two trained Neural Networks are input. values is a dictionary. 
    """

    solution = np.zeros((2*d0, len(times)))
    solution[:, 0] = np.concatenate((q0[0:d0], p0[0:d0]))

    grad_V_list = np.zeros(len(times))

    for n in range(len(times)-1):
        t1 = times[n+1]
        t0 = times[n]
        stepsize = t1-t0 # In case the stepsize is not constant.

        grad_T = calculate_gradient(NNT, solution[d0:, n], solution[:d0, n])
        q_new = solution[:d0, n] + stepsize*grad_T[:d0,:]

        grad_V = calculate_gradient(NNV, q_new, q_new)
        grad_V_list[n+1] = grad_V[:d0]
        p_new = solution[d0:, n] - stepsize*grad_V[:d0,:]

        solution[:d0, n+1] = q_new
        solution[d0:, n+1] = p_new
    return solution, times, grad_V_list# Using the indices from 0 in the solution, not times as indices. 

def symplectic_euler_exact(q0, p0, times, grad_T, grad_V,d0):
    solution = np.zeros((2*d0,len(times)))
    solution[:, 0] = np.concatenate((q0[0:d0], p0[0:d0]))
    for n in range(len(times)-1):
        t1 = times[n+1]
        t0 = times[n]
        stepsize = t1 -t0

        q_new = solution[:d0,n] + stepsize*grad_T(solution[d0:,n])
        p_new = solution[d0:,n] - stepsize*grad_V(q_new)
    
        solution[:d0, n+1] = q_new
        solution[d0:, n+1] = p_new

    return solution, times

def stormer_verlet_step(y, delta_t, grad_T, grad_V):
    """Step in Størmer-Verlet. Used in run_stormer_verlet to take each step."""
    # Assume that the input is passed as a vector y = [q, p]. 
    # Maybe implement Euler correctly first, then use the same principles here!
    p12 = y[1, :] - delta_t/2*grad_V[j] # indexing somehow.
    q1 = y[0, :] + delta_t*grad_T[1] # same as above.
    p1 = p12 - delta_t/2*grad_V[1] # same as above!
    # This is not true. 

    return np.array([q1, p1]) # Returns the solution as a np.array. 

def stormer_verlet(values, NNT, NNV, steps, stepsize):
    """Størmer-Verlet; second order method for integrating functions numerically.

    Two trained Neural Networks are input. values is a dictionary. 
    """
    P = values["P"]
    T = values["T"]
    Q = values["Q"]
    V = values["V"]
    times = values["t"]
    solution = np.zeros((len(times), V.shape[0], V.shape[1])) # Unsure about the dimensions (also when adding to the solution below)
    solution[0, :, :] = np.array([])
    for n in range(len(times)-1):
        t1 = times[n+1]
        t0 = times[n]
        stepsize = times[t1]-times[t0] # In case the stepsize is not constant. 
        grad_T = calculate_gradient(NNT, P[:, t0], T[:, t0])
        q1 = Q[:, t0] + stepsize*grad_T
        grad_V = calculate_gradient(NNV, q1, V[:, t1])
        p1 = P[:, t0] - stepsize*grad_V
        solution[n+1, :, :] = np.array([q1, p1])
    return solution # Using the indices from 0 in the solution, not times as indices. 
    
def calculate_gradient(NN, abscissae, ordinates):
    """Calculate gradient in specific point from network."""
    # Embed new data. 
    NN.embed_test_input(abscissae, ordinates)

    # Run forward function.
    NN.forward_function()

    # Calculate gradient.
    grad = NN.Hamiltonian_gradient()

    return grad

def embed_data(inp, d):
    result = np.zeros((d,len(inp)))
    result[0,:] = inp
    return result
