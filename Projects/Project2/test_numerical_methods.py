"""Test the numerical methods."""
from numerical_methods import *
from import_data import *
from hamiltonian_problems_ex import *


#====================#
# Nonlinear Pendelum #
#====================#

domain_T = domain_V = [-2, 2]
pendelum = Pendelum(domain_T, domain_V)
d0 = 1
I = 1000
d = 2
K = 23
h = 0.2
iterations = 2000
tau = alpha = beta = None
scaling = False
chunk = int(I/10)

# Train the neural networks. 
NNT = algorithm(I, d, d0, K, h, iterations, tau, chunk, pendelum.T, pendelum.domain_T, scaling, alpha, beta)
NNV = algorithm(I, d, d0, K, h, iterations, tau, chunk, pendelum.V, pendelum.domain_V, scaling, alpha, beta)

#input_T = generate_input(pendelum.T, pendelum.domain_T, d0, I, d)
#input_V = generate_input(pendelum.V, pendelum.domain_V, d0, I, d)

q0 = p0 = np.array([0.2])
times = np.linspace(0, 10, 1000)
network_sol, times, grad_V_list = symplectic_euler_NN(NNT, NNV, p0, q0, times,d0)



exact_sol, times= symplectic_euler_exact(p0, q0, times, pendelum.grad_T, pendelum.grad_V, d0)


plt.plot(times,grad_V_list, label = "network", linewidth = 0.5)
plt.plot(times,pendelum.grad_V(network_sol[0,:]), linestyle = "dashed", label = "Exact")
plt.plot(times, pendelum.grad_V(exact_sol[0,:]))


#plt.plot(times, network_sol[0,:], linewidth = 0.5, label = "network")
#plt.plot(times, exact_sol[0,:], linewidth = 0.5, label = "exact")
plt.legend()
plt.show()

#=========================#
# Kepler Two-vody Problem #
#=========================#

"""
domain_T = domain_V = [[-2, 2], [-2, 2]]
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

"""