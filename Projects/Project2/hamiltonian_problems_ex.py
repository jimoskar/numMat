"""The supplied examples of separable Hamiltonian problems tested."""
import numpy as np

# Nonlinear pendulum.
def H(p,q):
    """Maybe split into T and V for all these problems."""
    g = 9.81
    m = 1
    theta = np.pi/6
    return 0.5*p**2+m*g*(1-np.cos(theta))

# Kepler two-body problem.
