"""Test the numerical methods."""
from numerical_methods import *
from import_data import *
from hamiltonian_problems_ex import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


#====================#
# Nonlinear Pendulum #
#====================#
"""

domain_T = domain_V = [-1, 1]
pendulum = Pendulum(domain_T, domain_V)
d0 = 1
I = 1000
d = 3
K = 23
h = 0.2
iterations = 3000
tau = alpha = beta = None
scaling = False
#alpha = 0.2
#beta = 0.8
#scaling = True
chunk = int(I/10)

# Train the neural networks. 
NNT = algorithm_sgd(I, d, d0, K, h, iterations, tau, chunk, pendulum.T, pendulum.domain_T, scaling, alpha, beta)
NNV = algorithm_sgd(I, d, d0, K, h, iterations, tau, chunk, pendulum.V, pendulum.domain_V, scaling, alpha, beta)


#Initial position and momentum
q0 = np.array([0.2])
p0 = np.array([0.2])
times = np.linspace(0, 10, 1000)


#Network and exact solution. 'Exact' refers to that the gradient can be computed exactly.
network_sol, times = symplectic_euler_network(NNT, NNV, p0, q0, times,d0)
exact_sol, times= symplectic_euler_exact(p0, q0, times, pendulum.grad_T, pendulum.grad_V, d0)


#Plotting the exact position/impulse and the network position/impulse resulting from symplectic Euler
plt.plot(times, network_sol[0,:], linewidth = 0.5, label = "network")
plt.plot(times, exact_sol[0,:], linewidth = 0.5, label = "exact")
plt.legend()
plt.show()


#Plotting the hamiltonian. It should be constant along the exact solution.
q_data = embed_data(network_sol[0,:],d)
p_data = embed_data(network_sol[1,:],d)
network_ham = NNT.calculate_output(p_data) + NNV.calculate_output(q_data)
print(p_data.shape)
print(NNT.calculate_output(p_data).shape)
print(network_ham.shape)
plt.plot(times, network_ham, label = "network")

exact_ham = pendulum.T(exact_sol[1,:])+pendulum.V(exact_sol[0,:])
#plt.plot(times,pendulum.T(network_sol[1,:])+pendulum.V(network_sol[0,:]), label = "exact")
#Hamilton = pendulum.T(exact_sol[1,:]) + pendulum.V(exact_sol[0,:])
plt.plot(times, exact_ham, label = "exact")
plt.axhline(y = pendulum.T(p0)+pendulum.V(q0))
plt.title("Hamiltonian function")
plt.legend()
plt.show()
"""
#=========================#
# Kepler Two-body Problem #
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

q0  = p0 = np.array([0.2,0.2])
times = np.linspace(0, 10, 1000)

network_sol, times = symplectic_euler_network(NNT, NNV, q0, p0, times, d0)

exact_sol, times = symplectic_euler_exact(q0, p0, times, kepler.grad_T, kepler.grad_V, d0)


#plt.plot(times, exact_sol[0,:])
plt.plot(times, exact_sol[0,:])
plt.show()


ax = plt.axes(projection="3d")# Data for a three-dimensional line
zline = times #time
xline = exact_sol[0,:]
yline = exact_sol[1,:]
ax.plot3D(xline, yline, zline, "gray")# Data for three-dimensional 
plt.show()



"""
#======================#
# Henon-Heiles Problem #
#======================#


domain_T = domain_V = [[-2, 2], [-2, 2]]
HH = Henon_Heiles(domain_T, domain_V)
d0 = 2
I = 1000
d = 3
K = 23
h = 0.1
iterations = 5000
tau = alpha = beta = None
scaling = False
chunk = int(I/10)

# Train the neural networks. 
NNT = algorithm(I, d, d0, K, h, iterations, tau, chunk, HH.T, HH.domain_T, scaling, alpha, beta)
NNV = algorithm(I, d, d0, K, h, iterations, tau, chunk, HH.V, HH.domain_V, scaling, alpha, beta)

q0  = p0 = np.array([0.2,0.2])
times = np.linspace(0, 10, 1000)

network_sol, times = symplectic_euler_network(NNT, NNV, q0, p0, times, d0)
exact_sol, times = symplectic_euler_exact(q0, p0, times, HH.grad_T, HH.grad_V, d0)

plt.plot(times, exact_sol[0,:])
plt.plot(times, network_sol[0,:])
plt.show()

