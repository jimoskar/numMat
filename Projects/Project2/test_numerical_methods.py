"""Test the numerical methods."""
from numerical_methods import *
from import_data import *
from hamiltonian_problems_ex import *

domain_T = domain_V = [[2, 2], [2, 2]]
kepler = Kepler(domain_T, domain_V)
d0 = 2
I = 500
d = 4
K = 15
h = 0.2
iterations = 2000
tau = alpha = beta = None
scaling = False
chunk = int(I/10)

# Train the neural networks. 
NNT = algorithm(I, d, d0, K, h, iterations, tau, chunk, kepler.T, kepler.domain_T, scaling, alpha, beta)
NNV = algorithm(I, d, d0, K, h, iterations, tau, chunk, kepler.V, kepler.domain_V, scaling, alpha, beta)

input_T = generate_input(kepler.T, kepler.domain_T, d0, I, d)
input_V = generate_input(kepler.V, kepler.domain_V, d0, I, d)

# Not needed right now. 
values = {
    "Q": input_V, 
    "P": input_T, 
    "V": get_solution(kepler.V, input_V, d, I, d0), 
    "T": get_solution(kepler.T, input_T, d, I, d0), 
    "t": np.linspace(0, 10, 100)
}

solution_T = symplectic_euler(NNT, NNV, input_V[0], input_T[0], values["t"])
