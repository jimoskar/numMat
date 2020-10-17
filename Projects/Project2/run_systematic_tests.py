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
K_values = [10, 15, 20, 23]
d_values = [2, 3, 4]
h_values = [0.05, 0.1, 0.2, 0.3, 0.4]
iterations = 2000

config = {}
i = 1
for K in K_values:
    config[K] = {}
    for d in d_values:
        config[K][d] = {}
        for h in h_values:

            # Test function 1:
            # Only training. 
            NN = algorithm(I,d,K,h,iterations,test_function1,domain) # This only tests convergence when training. 
            # The below adds the J_list from training. 
            #config[K][d][h] = NN.J_list # add the value of J in the last iteration to the hash table.

            # Training and testing (uncomment this also if both)
            test_input = generate_input(test_function1,domain,d0,I,d)
            output = testing(NN, test_input, test_function1, domain, d0, d, I)
            config[K][d][h] = NN.J(output, test_input)

            """
            # Test function 2:
            # Only training. 
            NN = algorithm(I,d,K,h,iterations,test_function2,domain) # This only tests convergence when training. 
            # The below adds the J_list from training. 
            #config[K][d][h] = NN.J_list # add the value of J in the last iteration to the hash table.

            # Training and testing (uncomment this also if both)
            test_input = generate_input(test_function2,domain,d0,I,d)
            output = testing(NN, test_input, test_function2, domain, d0, d, I)
            config[K][d][h] = NN.J(output, test_input)
            """
            # Etc. 
            
            print(i)
            del NN # Safety measure to avoid leak. 
            i += 1

# Save data to file. 
import pickle
filename = 'testing_testfunc1'
outfile = open(filename,'wb')
pickle.dump(config, outfile)
outfile.close()
