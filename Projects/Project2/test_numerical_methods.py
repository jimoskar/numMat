"""Test the numerical methods on the supplied examples of separable Hamiltonian problems."""
from numerical_methods import *
from import_data import *
from hamiltonian_problems_ex import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#====================#
# Nonlinear Pendulum #
#====================#

def nonlinear_pendulum_test():
    """Train networks, calculate numerical solutions and plot figures for the nonlinear pendulum problem."""
    domain_T = domain_V = [-1, 1]
    pendulum = Pendulum(domain_T, domain_V)
    d0 = 1
    I = 1000
    d = 4
    K = 30
    h = 0.1
    iterations = 5000
    tau = alpha = beta = None
    scaling = False
    chunk = int(I/10)

    # Train the neural networks. 
    NNT = algorithm_sgd(I, d, d0, K, h, iterations, tau, chunk, pendulum.T, pendulum.domain_T, scaling, alpha, beta)
    NNV = algorithm_sgd(I, d, d0, K, h, iterations, tau, chunk, pendulum.V, pendulum.domain_V, scaling, alpha, beta)

    # Initial position and momentum.
    q0 = np.array([0.2])
    p0 = np.array([0.2])
    times = np.linspace(0, 10, 1000)

    # Network and exact solution. 'Exact' refers to that the gradient can be computed exactly.
    network_sol_euler = symplectic_euler_network(NNT, NNV, q0, p0, times,d0)
    exact_sol_euler = symplectic_euler_exact(q0, p0, times, pendulum.grad_T, pendulum.grad_V, d0)
    network_sol_SV = stormer_verlet_network(NNT, NNV, q0, p0, times,d0)
    exact_sol_SV = stormer_verlet_exact(q0, p0, times, pendulum.grad_T, pendulum.grad_V, d0)

    # Plot the exact position and the network position resulting from symplectic Euler and Størmer-Verlet.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(times, network_sol_euler[0,:], linewidth = 0.5, label = "Network (Euler)")
    plt.plot(times, exact_sol_euler[0,:], linewidth = 0.5, label = "Exact (Euler)")
    plt.plot(times, network_sol_SV[0,:], linewidth = 0.5, label = "Network (SV)")
    plt.plot(times, exact_sol_SV[0,:], linewidth = 0.5, label = "Exact (SV)")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("q")
    fig.suptitle("Position against Time", fontweight = "bold")
    plt.savefig("PendulumPosTime.pdf", bbox_inches='tight')
    plt.show()

    # Plot the exact momentum and the network momentum resulting from symplectic Euler and Størmer-Verlet.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(times, network_sol_euler[1,:], linewidth = 0.5, label = "Network (Euler)")
    plt.plot(times, exact_sol_euler[1,:], linewidth = 0.5, label = "Exact (Euler)")
    plt.plot(times, network_sol_SV[1,:], linewidth = 0.5, label = "Network (SV)")
    plt.plot(times, exact_sol_SV[1,:], linewidth = 0.5, label = "Exact (SV)")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("p")
    fig.suptitle("Momentum against Time", fontweight = "bold")
    plt.savefig("PendulumMomentumTime.pdf", bbox_inches='tight')
    plt.show()

    # Plotting the hamiltonian. It should be constant along the exact solution.
    #q_data = embed_data(network_sol[0,:],d)
    #p_data = embed_data(network_sol[1,:],d)
    network_ham_euler = NNT.calculate_output(network_sol_euler[1,:].reshape(1,1000)) + NNV.calculate_output(network_sol_euler[0,:].reshape(1,1000))
    network_ham_SV = NNT.calculate_output(network_sol_SV[1,:].reshape(1,1000)) + NNV.calculate_output(network_sol_SV[0,:].reshape(1,1000))
    #print(p_data.shape)
    #print(NNT.calculate_output(p_data).shape)
    #print(network_ham.shape)

    # Plot Hamiltonian with symplectic Euler.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(times, network_ham_euler, label = "Network (Euler)", color = "red", linestyle = "dashed")
    exact_ham_euler = pendulum.T(exact_sol_euler[1,:])+pendulum.V(exact_sol_euler[0,:])
    #plt.plot(times,pendulum.T(network_sol[1,:])+pendulum.V(network_sol[0,:]), label = "exact")
    plt.plot(times, exact_ham_euler, color = "blue", label = "Num method (Euler)")
    plt.axhline(y = pendulum.T(p0)+pendulum.V(q0), color = "yellow", label = "Correct")
    fig.suptitle("Hamiltonian Functions", fontweight = "bold")
    plt.xlabel("Time")
    plt.ylabel("Hamiltonian")
    plt.legend()
    plt.savefig("PendulumHamiltonianEuler.pdf", bbox_inches='tight')
    plt.show()

    # Plot Hamiltonian with Størmer-Verlet. 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(times, network_ham_SV, label = "Network (SV)", color = "red", linestyle = "dashed")
    exact_ham_SV = pendulum.T(exact_sol_SV[1,:])+pendulum.V(exact_sol_SV[0,:])
    plt.plot(times, exact_ham_SV, color = "blue", label = "Num method (SV)")
    plt.axhline(y = pendulum.T(p0)+pendulum.V(q0), color = "yellow", label = "Correct")
    fig.suptitle("Hamiltonian Functions", fontweight = "bold")
    plt.xlabel("Time")
    plt.ylabel("Hamiltonian")
    plt.legend()
    plt.savefig("PendulumHamiltonianSV.pdf", bbox_inches='tight')
    plt.show()

#=========================#
# Kepler Two-body Problem #
#=========================#
"""
domain_T = domain_V = [[-2, 2], [-2, 2]]
kepler = Kepler(domain_T, domain_V)
d0 = 2
I = 1000
d = 4
K = 30
h = 0.1
iterations = 2000
tau = alpha = beta = None
scaling = False
chunk = int(I/10)


# Train the neural networks. 
NNT = algorithm_sgd(I, d, d0, K, h, iterations, tau, chunk, kepler.T, kepler.domain_T, scaling, alpha, beta)
NNV = algorithm_sgd(I, d, d0, K, h, iterations, tau, chunk, kepler.V, kepler.domain_V, scaling, alpha, beta)

input_T = generate_input(kepler.T, kepler.domain_T, d0, I, d)
input_V = generate_input(kepler.V, kepler.domain_V, d0, I, d)

q0  = np.array([0.2,0.2])
p0 = np.array([0,0])
times = np.linspace(0, 40, 1000)

network_sol, times = stormer_verlet_network(NNT, NNV, q0, p0, times, d0)

exact_sol, times = stormer_verlet_exact(q0, p0, times, kepler.grad_T, kepler.grad_V, d0)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(times, network_sol[0,:], label = "Network")
plt.plot(times, exact_sol[0,:], label = "Exact")
plt.legend()
plt.xlabel("Time")
plt.ylabel("First Coord")
fig.suptitle("Position against Time", fontweight = "bold")
plt.savefig("KeplerFirstPosCoordTime.pdf", bbox_inches='tight')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")
zline = times 
xline = exact_sol[0,:]
yline = exact_sol[1,:]
ax.plot3D(xline, yline, zline, "gray", label = "Exact")
xnet = network_sol[0,:]
ynet = network_sol[1,:]
ax.plot3D(xnet, ynet, zline, "red", label = "Network") 
fig.suptitle("Position in Time", fontweight = "bold")
ax.set_xlabel("Coord 1")
ax.set_ylabel("Coord 2")
ax.set_zlabel("Time")
ax.axes.yaxis.set_ticklabels([])
ax.axes.xaxis.set_ticklabels([])
ax.axes.zaxis.set_ticklabels([])
plt.legend()
plt.savefig("Kepler3d.pdf", bbox_inches='tight')
plt.show()

# Plot Hamiltionian.
exact_ham = kepler.T(exact_sol[d0:,:])+kepler.V(exact_sol[:d0,:])
plt.plot(times, exact_ham, color = "blue", label = "num method (sympl euler)")
plt.axhline(y = kepler.T(p0)+kepler.V(q0), color = "yellow", label = "anal")
plt.title("Hamiltonian function")
plt.legend()
plt.show()
"""
#======================#
# Henon-Heiles Problem #
#======================#

def hehon_heiles_test():
    """Train networks, calculate numerical solutions and plot figures for the Henon-Heiles problem."""
    domain_T = domain_V = [[-2, 2], [-2, 2]]
    HH = Henon_Heiles(domain_T, domain_V)
    d0 = 2
    I = 1000
    d = 4
    K = 30
    h = 0.1
    iterations = 5000
    tau = alpha = beta = None
    scaling = False
    chunk = int(I/10)

    # Train the neural networks. 
    NNT = algorithm_sgd(I, d, d0, K, h, iterations, tau, chunk, HH.T, HH.domain_T, scaling, alpha, beta)
    NNV = algorithm_sgd(I, d, d0, K, h, iterations, tau, chunk, HH.V, HH.domain_V, scaling, alpha, beta)

    q0  = p0 = np.array([0.2,0.2])
    times = np.linspace(0, 10, 1000)

    # Network and exact solution. 'Exact' refers to that the gradient can be computed exactly.
    network_sol_euler = symplectic_euler_network(NNT, NNV, q0, p0, times, d0)
    exact_sol_euler = symplectic_euler_exact(q0, p0, times, HH.grad_T, HH.grad_V, d0)
    network_sol_SV = stormer_verlet_network(NNT, NNV, q0, p0, times, d0)
    exact_sol_SV = stormer_verlet_exact(q0, p0, times, HH.grad_T, HH.grad_V, d0)

    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(times, exact_sol[0,:], label="Exact")
    plt.plot(times, network_sol[0,:], label="Network")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("First Coord")
    fig.suptitle("Position against Time", fontweight = "bold")
    plt.savefig("HenonFirstPosCoordTime.pdf", bbox_inches='tight')
    plt.show()
    """

    # Plot the exact position and the network position resulting from symplectic Euler and Størmer-Verlet.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = "3d")
    zline = times #time
    xline = exact_sol_euler[0,:]
    yline = exact_sol_euler[1,:]
    ax.plot3D(xline, yline, zline, "gray", label = "Exact (Euler)")
    xline = exact_sol_SV[0,:]
    yline = exact_sol_SV[1,:]
    ax.plot3D(xline, yline, zline, "gray", label = "Exact (SV)")
    xnet = network_sol_euler[0,:]
    ynet = network_sol_euler[1,:]
    ax.plot3D(xnet, ynet, zline, "red", label = "Network (Euler)") 
    xnet = network_sol_SV[0,:]
    ynet = network_sol_SV[1,:]
    ax.plot3D(xnet, ynet, zline, "red", label = "Network (SV)") 
    plt.legend()
    fig.suptitle("Position in Time Space", fontweight = "bold")
    ax.set_xlabel("q1")
    ax.set_ylabel("q2")
    ax.set_zlabel("Time")
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    plt.savefig("HenonPosition.pdf", bbox_inches='tight')
    plt.show()

    # Plot the exact momentum and the network momentum resulting from symplectic Euler and Størmer-Verlet.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = "3d")
    zline = times #time
    xline = exact_sol_euler[2,:]
    yline = exact_sol_euler[3,:]
    ax.plot3D(xline, yline, zline, "gray", label = "Exact (Euler)")
    xline = exact_sol_SV[2,:]
    yline = exact_sol_SV[3,:]
    ax.plot3D(xline, yline, zline, "gray", label = "Exact (SV)")
    xnet = network_sol_euler[2,:]
    ynet = network_sol_euler[3,:]
    ax.plot3D(xnet, ynet, zline, "red", label = "Network (Euler)") 
    xnet = network_sol_SV[2,:]
    ynet = network_sol_SV[3,:]
    ax.plot3D(xnet, ynet, zline, "red", label = "Network (SV)") 
    plt.legend()
    fig.suptitle("Momentum in Time Space", fontweight = "bold")
    ax.set_xlabel("p1")
    ax.set_ylabel("p2")
    ax.set_zlabel("Time")
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    plt.savefig("HenonMomentum.pdf", bbox_inches='tight')
    plt.show()

    # Plot Hamiltonian with symplectic Euler.

    # Plot Hamiltonian with Størmer-Verlet.

hehon_heiles_test()
