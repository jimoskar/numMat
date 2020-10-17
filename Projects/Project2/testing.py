import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import numpy.linalg as la
from mpl_toolkits import mplot3d

plt.style.use('seaborn')

# Add parameters for plotting. 
mpl.rcParams['figure.titleweight'] = "bold"
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['axes.titleweight'] = "bold"

def testing(Network, test_input, function, domain, d0, d, I):
    """Testing the Neural Network with new random data. 
    
    The parameters found from the training of the Neural Network are employed.
    """
    c = get_solution(function, test_input, d, I, d0)
    Z_list = np.zeros((Network.K+1,d,I))
    Z_list[0,:,:] = test_input
    Network.Z_list = Z_list
    output = Network.forward_function()
    
    return output 


def generate_input(function,domain,d0,I,d):
    """Generate and embed input in d-dimensional space.
        Done by drawing from a uniform distribution on the domain.
    """
    result = np.zeros((d,I))
    if d0 == 1:
        for i in range(I):
            num = np.random.uniform(domain[0],domain[1])
            result[:,i] = np.repeat(num,d)
    if d0 == 2:
        for i in range(I):
            num = np.random.uniform(domain[0][0],domain[0][1])
            result[:int(d/2),i] = np.repeat(num,d/2)
        for i in range(I):
            num = np.random.uniform(domain[1][0],domain[1][1])
            result[int(d/2):,i] = np.repeat(num,d/2)

    return result


def get_solution(function,input_values,d,I,d0):
    """Generate points from the test function on the given domain.
    """
    result = np.zeros(I)
    if d0 == 1:
        for i in range(I):
            result[i] = function(input_values[0,i])
        
    if d0 == 2:
        for i in range(I):
            result[i] = function(input_values[0,i],input_values[int(d/2),i])
    return result
    

def plot_graph_and_output(output,input,function,domain,d0,d):
    """
    Plot results from testing the network together with the analytical graph
    """
    if d0 == 1:
        # Plotting the output from the network.
        x = input[0,:]

        fig, ax = plt.subplots()
        ax.scatter(x,output, color="orange", label="Network")
        fig.suptitle("Analytical Solution Compared to Output From Network", fontweight = "bold")
        ax.set_xlabel("Domain [y]")
        ax.set_ylabel("F(y)")

        # Plotting the analytical solution
        x = np.linspace(domain[0],domain[1])
        ax.plot(x,function(x), color="blue", label="Function")
        ax.legend()
        plt.savefig("compTest1Pic2.pdf", bbox_inches='tight')
        plt.show()

    elif d0 == 2:
        # Plotting output
        ax = plt.axes(projection='3d')
        zdata = output
        xdata = input[0,:]
        ydata = input[int(d/2),:]
        ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Reds')
        
        # Plotting analytical graph
        x = np.linspace(domain[0][0],domain[0][1], 30)
        y = np.linspace(domain[1][0], domain[1][1], 30)
        
        X, Y = np.meshgrid(x, y)
        Z = function(X, Y)

        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='Greys', edgecolor='none', alpha = 0.5)
        plt.show()

def get_residue(true_vals, network_vals):
    """Find difference between function values and values from network."""
    return la.norm(true_vals - network_vals)
