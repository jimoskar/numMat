"""Train and test networks on the given data (from unknown Hamiltonian function)."""
from network_algo import *
from numerical_methods import *
from import_data import generate_data, concatenate

def embed_input(inp,d):
    """Embed the input into the dimension of the hidden layers."""
    d0 = inp.shape[0]
    I = inp.shape[1]
    while d0 < d:
        zero_row = np.zeros(I)
        inp = np.vstack((inp,zero_row))
        d0 += 1
    return inp


# Could just get rid of this!
def algorithm_input_and_sol_semi_sgd(inp, sol, method, I, d, d0, K, h, iterations, tau, chunk, scaling, alpha, beta, plot = False, savename = ""):
    """Train a network based on prespecified input and solution. Returns the trained network."""

    # This semi-sgd function takes waaay shorter time then the correct SGD!
    # With equal amount of iterations. 
    # It gives a bad result however. 

    if scaling:
        inp, a1, b1 = scale_data(alpha,beta,inp)
        output, a2, b2 = scale_data(alpha,beta,sol)


    Z_0 = inp[:,0:chunk]
    c_0 = sol[0:chunk]
    NN = Network(K,d,chunk,h,Z_0,c_0)
    
    # For plotting J. 
    J_list = np.zeros(iterations)
    it = np.zeros(iterations)

    counter = 0
    for j in range(1,iterations+1):
        print(j)
        NN.forward_function()
        gradient = NN.back_propagation()
        NN.theta.update_parameters(gradient,method,tau,j)

        if counter < I/chunk - 1:
            NN.Z_list[0,:,:] = inp[:,chunk*(counter+1):chunk*(counter+2)]
            NN.c = sol[chunk*(counter + 1):chunk*(counter + 2)]
        else:
            counter = 0

        J_list[j-1] = NN.J()
        it[j-1] = j
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(it,J_list)
        fig.suptitle("Objective Function J as a Function of Iterations.", fontweight = "bold")
        plt.yscale("log")
        ax.set_ylabel("J")
        ax.set_xlabel("Iteration")
        plt.text(0.5, 0.5, "Value of J at iteration "+str(iterations)+": "+str(round(J_list[-1], 6)), 
                horizontalalignment="center", verticalalignment="center", 
                transform=ax.transAxes, fontsize = 16)
        if savename != "": 
            plt.savefig(savename, bbox_inches='tight')
        plt.show()
    
    return NN

def algorithm_input_and_sol_sgd(inp, sol, method, I, d, d0, K, h, iterations, tau, chunk, scaling, alpha, beta, plot = False, savename = ""):
    """Train a network based on prespecified input and solution. Returns the trained network. Uses SGD."""

    if scaling:
        inp, a1, b1 = scale_data(alpha,beta,inp)
        sol, a2, b2 = scale_data(alpha,beta,sol)
    
    index_list = [i for i in range(I)]
    
    Z_0, c_0, index_list = get_random_sample(inp,sol,index_list,chunk,d)
    NN = Network(K,d,chunk,h,Z_0,c_0)


    # For plotting J. 
    J_list = np.zeros(iterations)
    it = np.zeros(iterations)

    counter = 0
    for j in range(1,iterations+1):
        print(j)

        NN.forward_function()
        gradient = NN.back_propagation()
        NN.theta.update_parameters(gradient,method,tau,j)

        # For plotting.
        J_list[j-1] = NN.J()
        it[j-1] = j

        if counter < I/chunk - 1:
            Z, c, index_list = get_random_sample(inp,sol,index_list,chunk,d)
            NN.Z_list[0,:,:] = Z
            NN.c = c
            counter += 1
        else:
            # All data has been sifted through.
            counter = 0
            index_list = [i for i in range(I)]
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(it,J_list)
        plt.yscale("log")
        fig.suptitle("Objective Function J as a Function of Iterations.", fontweight = "bold")
        ax.set_ylabel("J")
        ax.set_xlabel("Iteration")
        plt.text(0.5, 0.5, "Value of J at iteration "+str(iterations)+": "+str(round(J_list[-1], 6)), 
                horizontalalignment="center", verticalalignment="center", 
                transform=ax.transAxes, fontsize = 16)
        if savename != "": 
            plt.savefig(savename+".pdf", bbox_inches='tight')
        plt.show()
    
    return NN


def find_ratio(NN, tol = 0.005):
    """Find amount of correctly classified points, within some tolerance."""
    # Find amount of correctly classified points (within some tolerance).
    placeholder = np.array([NNV.Y, NN.c])
    diff = np.diff(placeholder, axis = 0)
    ratio = len(diff[abs(diff)<tol])/len(diff[0])
    return ratio 

training_data = concatenate(batchmax=30)
inp_Q = embed_input(training_data['Q'],d)
inp_P = embed_input(training_data['P'],d)
sol_Q = training_data['V']
sol_P = training_data['T']

I = inp_Q.shape[1] # Amount of points ran through the network at once. 
K = 30 # Amount of hidden layers in the network.
d0 = 3 # Input dimensions. 
d = 4 # Dimensions of the networks's hidden layers.  
h = 0.1 # Scaling of the activation function application in algorithm.  
iterations = 3000 # Number of iterations in the Algorithm.
method = "adams"
tau = 0.1 # For the Vanilla Gradient method.
chunk = int(I/128) 

# For scaling.
scaling = False
alpha = 0.2
beta = 0.8 

# Training.
NNV = algorithm_input_and_sol_sgd(inp_Q, sol_Q, method, I, d, d0, K, h, iterations, tau, chunk, scaling, alpha, beta, plot = True, savename = "JDataGivenNNV")
NNT = algorithm_input_and_sol_sgd(inp_P, sol_P, method, I, d, d0, K, h, iterations, tau, chunk, scaling, alpha, beta, plot = True, savename = "JDataGivenNNT")

# Regular algorithm is very slow. Semi-sgd (not stochastic) is very fast, but gives bad results. 
# Sgd is a bit faster than regular algo, and hopefully gives good results. 

print("Ratios from training: (tol = 0.05, batch = I/256)")
ratio_NNV = find_ratio(NNV, tol = 0.05)
ratio_NNT = find_ratio(NNT, tol = 0.05)
print("V:", ratio_NNV)
print("T:", ratio_NNT)

# Testing.
test_batch = generate_data(batch = 39)
Q_data = test_batch['Q']
P_data = test_batch['P']
times = test_batch['t']

def plot3d(xex, yex, zex, xnet, ynet, znet, suptitle, xlabel, ylabel, zlabel, savename = ""):
    """Plot the given x, y and z values for net and exact in a 3d plot."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = "3d")
    ax.plot3D(xex, yex, zex, "gray", label = "exact")
    fig.suptitle(suptitle, fontweight = "bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    ax.plot3D(xnet, ynet, znet, "red", label = "network") 
    plt.legend()
    if savename != "":
        plt.savefig(savename+".pdf", bbox_inches='tight')
    plt.show()

network_sol, times = stormer_verlet_network(NNT, NNV, Q_data[:,0],P_data[:,0], times, d0)

# Plot the two first positional coord with time on third axis.
plot3d(Q_data[0,:], Q_data[1,:], times, network_sol[0,:], network_sol[1,:], times, 
        "Position in Time", "Coord 1", "Coord 2", "Time", "givenDataAndNetwork3dTimePos")

# Plot all three positional coord in space.
plot3d(Q_data[0,:], Q_data[1,:], Q_data[2,:], network_sol[0,:], network_sol[1,:], network_sol[2,:], 
        "Position in Space", "Coord 1", "Coord 2", "Coord 3", "givenDataAndNetwork3dPos")

# Plot two first momentum coord with time on the third axis.
plot3d(P_data[0,:], P_data[1,:], times, network_sol[3,:], network_sol[4,:], times, 
        "Momentum in Time", "Coord 1", "Coord 2", "Time", "givenDataAndNetwork3dTimeMomentum")

# Plot all three momentum coord in space.
plot3d(P_data[0,:], P_data[1,:], P_data[2,:], network_sol[3,:], network_sol[4,:], network_sol[5,:], 
        "Momentum in Space", "Coord 1", "Coord 2", "Coord 3", "givenDataAndNetwork3dMomentum")


# For reference (to check if some of the coordinates actually are any good!)

# Plot the impulse in 2d (for reference)
fig = plt.figure()
plt.subplot(111)
plt.suptitle("impulse Against Time", fontweight = "bold")
plt.xlabel("Time")
plt.ylabel("First Impulse Coordinate")
plt.plot(times, P_data[0,:], label = "data")
plt.plot(times, network_sol[0,:], label = "network")
plt.legend()
plt.savefig("2dImpulse4Reference.pdf", bbox_inches='tight')
plt.show()

# Plot the pos in 2d (for reference)
fig = plt.figure()
plt.subplot(111)
plt.suptitle("Position Against Time", fontweight = "bold")
plt.xlabel("Time")
plt.ylabel("First Position Coordinate")
plt.plot(times, Q_data[0,:], label = "data")
plt.plot(times, network_sol[0,:], label = "network")
plt.legend()
plt.savefig("2dPosition4Reference.pdf", bbox_inches='tight')
plt.show()


# Første tre er q og de neste 3 er p!
print("Norm of difference between network's solution and Størmer-Verlet (position)")
print(la.norm(network_sol[:3, :] - Q_data)) 

print("Norm of difference between network's solution and Størmer-Verlet (momentum)")
print(la.norm(network_sol[3:, :] - Q_data)) 
