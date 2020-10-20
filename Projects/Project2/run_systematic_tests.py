"""Testing framework for running systematic tests of the neural network.

This is in order to fint he optimal values of the parameters in the network, 
by method of experimentation (systematic tests).
"""

"""
Parameters to test for:
- K
- tau
- d 
- h
- Optimization algorithm: Adam or Vanilla (+SGD).
- I
Think we should use a fixed amount of iterations (perhaps not as many as we have done thus far, bc of runtime)
Also a fixed amount of images probably, fewer than the amount used now!

We need to construct NN for each of the configurations and save the objective function 
(perhaps only the last value) somewhere. From these values we can find max, min, argmax and argmin :)

"""

from project2vs import *

# For the one-dimensional test function 1.
K_values = [10, 15, 20, 23, 30]
d_values = [2, 3, 4, 5, 6]
h_values = [0.05, 0.1, 0.2, 0.3, 0.4]
iterations = 2000
d0 = 1 # Dimension of the input layer. 
#domain = [-2,2]
I = 1000
chunk = int(I/1)
scaling = False
alpha = 0.2
beta = 0.8 
def test_function1(x):
    return 0.5*x**2

d0 = 2
d = 4
domain = [[-2,2],[-2,2]]
chunk = int(I/10)

def test_function3(x,y):
    return 0.5*(x**2 + y**2)


config = {}
i = 1
for K in K_values:
    config[K] = {}
    for d in d_values:
        config[K][d] = {}
        for h in h_values:
            """
            # Test function 1:
            # Only training. 
            NN = algorithm(I,d,d0,K,h,iterations,tau, chunk, test_function1, domain, scaling, alpha, beta) # This only tests convergence when training. 
            # The below adds the J_list from training. 
            #config[K][d][h] = NN.J_last # add the value of J in the last iteration to the hash table.
            
            # Training and testing (uncomment this also if both)
            test_input = generate_input(test_function1,domain,d0,I,d)
            testing(NN, test_input, test_function1, domain, d0, d, I, scaling, alpha, beta)
            config[K][d][h] = NN.J()
            """
            
            # Test function 2:
            NN = algorithm(I,d,d0,K,h,iterations,tau, chunk, test_function3, domain, scaling, alpha, beta) # This only tests convergence when training. 
            # The below adds the J_list from training. 
            #config[K][d][h] = NN.J_last # add the value of J in the last iteration to the hash table.
            
            # Training and testing (uncomment this also if both)
            test_input = generate_input(test_function3,domain,d0,I,d)
            testing(NN, test_input, test_function3, domain, d0, d, I, scaling, alpha, beta)
            config[K][d][h] = NN.J()
            
            # Etc. 
            
            print(i)
            del NN # Safety measure to avoid leak. 
            i += 1

# Save data to file. 
import pickle
filename = 'testing_testfunc3'
outfile = open(filename,'wb')
pickle.dump(config, outfile)
outfile.close()
