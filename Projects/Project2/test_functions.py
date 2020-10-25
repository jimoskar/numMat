"""Train networks and run tests on the four provided test functions."""
from network_algo import *
from test_model import *

I = 1000 # Amount of points ran through the network at once. 
K = 20 # Amount of hidden layers in the network.
d = 2 # Dimension of the hidden layers in the network. 
h = 0.05 # Scaling of the activation function application in algorithm.  
iterations = 10000 #Number of iterations in the Algorithm 
tau = 0.1 #For the Vanilla Gradient method

#For scaling
scaling = False
alpha = 0.2
beta = 0.8 

#================#
#Test function 1 #
#================#


d0 = 1 # Dimension of the input layer. 
domain = [-2,2]
chunk = int(I/10)
def test_function1(x):
    return 0.5*x**2

NN = algorithm(I,d,d0, K,h,iterations, tau, chunk, test_function1,domain,scaling, alpha, beta)
test_input = generate_input(test_function1,domain,d0,I,d)

#The a's and b's are for potential scaling fo the data
output, a1, b1, a2, b2 = testing(NN, test_input, test_function1, domain, d0, d, I, scaling, alpha, beta)
plot_graph_and_output(output, test_input, test_function1, domain, d0,d, scaling, alpha, beta, a1, b1, a2, b2)

#================#
#Test function 2 #
#================#
"""
d0 = 1 # Dimensin of the input layer. 
domain = [-2,2]
chunk = int(I/10)
def test_function2(x):
    return 1 - np.cos(x)

NN = algorithm(I,d,K,h,iterations,tau,chunk, test_function2,domain, scaling, alpha, beta)
test_input = generate_input(test_function2,domain,d0,I,d)

#The a's and b's are for potential scaling fo the data
output, a1, b1, a2, b2 = testing(NN, test_input, test_function2, domain, d0, d, I, scaling, alpha, beta)
plot_graph_and_output(output, test_input, test_function2, domain, d0,d, scaling, alpha, beta, a1, b1, a2, b2)
"""

#================#
#Test function 3 #
#================#
"""
d0 = 2
d = 4
domain = [[-2,2],[-2,2]]
chunk = int(I/10)
def test_function3(x,y):
    return 0.5*(x**2 + y**2)

NN = algorithm(I,d,K,h,iterations, tau, chunk, test_function3,domain,scaling,alpha,beta)
test_input = generate_input(test_function3,domain,d0,I,d)

#The a's and b's are for potential scaling fo the data
output, a1, b1, a2, b2 = testing(NN, test_input, test_function3, domain, d0, d, I, scaling, alpha, beta)
plot_graph_and_output(output, test_input, test_function3, domain, d0,d, scaling, alpha, beta, a1, b1, a2, b2)

"""
#================#
#Test function 4 #
#================#
"""

d0 = 2
d = 4
domain = [[-2,2],[-2,2]]
chunk = int(I/10)
def test_function4(x,y):
    return -1/np.sqrt(x**2 + y**2)

NN = algorithm(I,d,K,h,iterations, tau, chunk, test_function4,domain,scaling,alpha,beta)
test_input = generate_input(test_function4,domain,d0,I,d)

#The a's and b's are for potential scaling fo the data
output, a1, b1, a2, b2 = testing(NN, test_input, test_function4, domain, d0, d, I, scaling, alpha, beta)
plot_graph_and_output(output, test_input, test_function4, domain, d0,d, scaling, alpha, beta, a1, b1, a2, b2)

"""
