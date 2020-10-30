"""Need to write a sweet description here also!"""
from network_algo import *
from test_model import *


I = 500 # Amount of points ran through the network at once. 
K = 23 # Amount of hidden layers in the network.
d = 2 # Dimension of the hidden layers in the network. 
h = 0.1 # Scaling of the activation function application in algorithm.  
iterations = 2000 #Number of iterations in the Algorithm 
tau = 0.01 #For the Vanilla Gradient method

#For scaling
scaling = True
alpha = 0.2
beta = 0.8 


d0 = 1 # Dimension of the input layer. 
domain = [-2,2]
chunk = int(I/10)
def test_function1(x):
    return 0.5*x**2

method = "vanilla"
scaling = True
NN1, a1, b1, a2, b2, J_list1, it = algorithm_scaling(I,d,d0, K,h,iterations, tau, chunk, method, test_function1,domain,scaling, alpha, beta, plot = False)
method = "adams"
scaling = True
NN2, a1, b1, a2, b2, J_list2, it = algorithm_scaling(I,d,d0, K,h,iterations, tau, chunk, method, test_function1,domain,scaling, alpha, beta, plot = False)
method = "adams"
scaling = False
NN3, a1, b1, a2, b2, J_list3, it = algorithm_scaling(I,d,d0, K,h,iterations, tau, chunk, method, test_function1,domain,scaling, alpha, beta, plot = False)
#plotting convergence:
plt.plot(it, J_list1, label = "Vanilla and scaling")
plt.plot(it, J_list2, label = "ADAM and scaling")
plt.plot(it, J_list3, label = "ADAM without scaling")
plt.yscale("log")
plt.legend()
plt.show()
#test_input = generate_input(test_function1,domain,d0,I,d)



#The a's and b's are for potential scaling fo the data
#output, a1, b1, a2, b2 = testing(NN, test_input, test_function1, domain, d0, d, I, scaling, alpha, beta)
#plot_graph_and_output(output, test_input, test_function1, domain, d0,d, scaling, alpha, beta, a1, b1, a2, b2)
