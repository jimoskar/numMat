"""Train and test networks on the given data (from unknown Hamiltonian function)."""
from network_algo import *
from import_data import generate_data, concatenate

# Run training and tests on the given data here! (from import_data)


def embed_input(inp,d):
    d0 = inp.shape[0]
    I = inp.shape[1]
    while d0 < d:
        zero_row = np.zeros(I)
        inp = np.vstack((inp,zero_row))
        d0 += 1
    return inp


def algorithm_input_and_sol(inp, sol, method, I, d, d0, K, h, iterations, tau, chunk, scaling, alpha, beta, plot = False, savename = ""):
    """Trains a network based on prespecified input and solution. Returns the trained network"""

    if scaling:
        inp, a1, b1 = scale_data(alpha,beta,inp)
        sol, a2, b2 = scale_data(alpha,beta,sol)

    Z_0 = inp[:,0:chunk]
    c_0 = sol[0:chunk]
    NN = Network(K,d,chunk,h,Z_0,c_0)
    
  # For plotting J. 
    J_list = np.zeros(iterations)
    it = np.zeros(iterations)

    counter = 0
    for j in range(1,iterations+1):
        
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
        ax.set_ylabel("J")
        ax.set_xlabel("Iteration")
        plt.text(0.5, 0.5, "Value of J at iteration "+str(iterations)+": "+str(round(J_list[-1], 4)), 
                horizontalalignment="center", verticalalignment="center", 
                transform=ax.transAxes, fontsize = 16)
        if savename != "": 
            plt.savefig(savename, bbox_inches='tight')
        plt.show()
    
    return NN



d = 4
data = concatenate(batchmax=3)
inp = embed_input(data['Q'],d)
sol = data['V']

I = inp.shape[1] # Amount of points ran through the network at once. 
K = 20 # Amount of hidden layers in the network.
d0 = 3
h = 0.05 # Scaling of the activation function application in algorithm.  
iterations = 2000 #Number of iterations in the Algorithm 
method = "adams"
tau = 0.1 #For the Vanilla Gradient method
chunk = int(I/256)

#For scaling
scaling = False
alpha = 0.2
beta = 0.8 


NNV = algorithm_input_and_sol(inp, sol, method, I, d, d0, K, h, iterations, tau, chunk, scaling, alpha, beta, plot = True)








