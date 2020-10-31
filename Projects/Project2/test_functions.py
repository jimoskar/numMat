"""Train networks and run tests (manually) on the four provided test functions.

This is used when wanting to run simple tests manually on the four suggested functions, 
separately from all the systematic tests in the larger testing framework. 
"""
from network_algo import *
from test_model import *

I = 1000 # Amount of points ran through the network at once. 
K = 23 # Amount of hidden layers in the network.
d = 4 # Dimension of the hidden layers in the network. 
h = 0.05 # Scaling of the activation function application in algorithm.  
iterations = 2000 # Number of iterations in the Algorithm.
tau = 0.1 # For the Vanilla Gradient method.

# For scaling.
scaling = False
alpha = 0.2
beta = 0.8 

#================#
#Test function 1 #
#================#

def run_test_on_test_function1():
    """Run test (training + testing with new, unused data) of test function 1."""
    d0 = 1 # Dimension of the input layer. 
    domain = [-2,2]
    chunk = int(I/10)
    def test_function1(x):
        return 0.5*x**2

    NN = algorithm_sgd(I,d,d0, K,h,iterations, tau, chunk, test_function1,domain,scaling, alpha, beta, plot = True, savename = "ONEJBestTesting")
    test_input = generate_input(test_function1,domain,d0,I,d)

    # The a's and b's are for potential scaling fo the data.
    output, a1, b1, a2, b2 = testing(NN, test_input, test_function1, domain, d0, d, I, scaling, alpha, beta)
    plot_graph_and_output(output, test_input, test_function1, domain, d0,d, scaling, alpha, beta, a1, b1, a2, b2, savename = "ONEGraphBestTesting")

#================#
#Test function 2 #
#================#

def run_test_on_test_function2():
    """Run test (training + testing with new, unused data) of test function 2."""
    d0 = 1 # Dimension of the input layer. 
    domain = [-2,2]
    chunk = int(I/10)
    def test_function2(x):
        return 1 - np.cos(x)

    NN = algorithm_sgd(I,d,d0, K,h,iterations,tau,chunk, test_function2,domain, scaling, alpha, beta, plot = True, savename = "TWOJBestTesting")
    test_input = generate_input(test_function2,domain,d0,I,d)

    # The a's and b's are for potential scaling fo the data.
    output, a1, b1, a2, b2 = testing(NN, test_input, test_function2, domain, d0, d, I, scaling, alpha, beta)
    plot_graph_and_output(output, test_input, test_function2, domain, d0,d, scaling, alpha, beta, a1, b1, a2, b2, savename = "TWOGraphBestTesting")


#================#
#Test function 3 #
#================#

def run_test_on_test_function3():
    """Run test (training + testing with new, unused data) of test function 3."""
    d0 = 2
    d = 4
    domain = [[-2,2],[-2,2]]
    chunk = int(I/10)
    def test_function3(x):
        return 0.5*(x[0]**2 + x[1]**2)

    NN = algorithm_sgd(I,d,d0, K,h,iterations, tau, chunk, test_function3,domain,scaling,alpha,beta, plot = True, savename = "THREEJBestTesting")
    test_input = generate_input(test_function3,domain,d0,I,d)

    # The a's and b's are for potential scaling fo the data.
    output, a1, b1, a2, b2 = testing(NN, test_input, test_function3, domain, d0, d, I, scaling, alpha, beta)
    plot_graph_and_output(output, test_input, test_function3, domain, d0,d, scaling, alpha, beta, a1, b1, a2, b2, savename = "THREEGraphBestTesting")


#================#
#Test function 4 #
#================#

def run_test_on_test_function4():
    """Run test (training + testing with new, unused data) of test function 4."""
    #iterations = 10000 # Show that more iterations makes convergence better in the last iteration obviously.
    d0 = 2
    d = 4
    domain = [[-2,2],[-2,2]]
    chunk = int(I/10)
    def test_function4(x):
        return -1/np.sqrt(x[0]**2 + x[1]**2)

    NN = algorithm_sgd(I,d,d0,K,h,iterations, tau, chunk, test_function4,domain,scaling,alpha,beta, plot = True, savename = "FOURJBestTesting")
    test_input = generate_input(test_function4,domain,d0,I,d)

    # The a's and b's are for potential scaling fo the data.
    output, a1, b1, a2, b2 = testing(NN, test_input, test_function4, domain, d0, d, I, scaling, alpha, beta)
    plot_graph_and_output(output, test_input, test_function4, domain, d0,d, scaling, alpha, beta, a1, b1, a2, b2, savename = "FOURGraphBestTesting")
