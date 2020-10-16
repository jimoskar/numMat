"""Testing framework for running systematic tests of the neural network.

This is in order to fint he optimal values of the parameters in the network, 
by method of experimentation (systematic tests).
"""

# Python's unittest module? 
# Or write tests ourselves. Google a bit!

#https://towardsdatascience.com/how-to-rapidly-test-dozens-of-deep-learning-models-in-python-cb839b518531

# This needs to be thought through in detail before doing!
# Maybe run the code on Markow since it will probably take a very long time!

# Perhaps make a hash table (dictionary) of the different parameters. 

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

# I am going to write some easy tests to begin with, to test (the tests ;))

from project2vs import *

# For the one-dimensional test function 1.
K_values = [10, 15, 20, 23]
d_values = [2, 3, 4]
h_values = [0.05, 0.1, 0.3, 0.6, 0.8]

config = {}
i = 1
for K in K_values:
    config[K] = {}
    for d in d_values:
        config[K][d] = {}
        for h in h_values:
            NN = algorithm(I,d,K,h,iterations,test_function1,domain) # This only tests convergence when training. 
            # Could also test with random data, via testing!
            config[K][d][h] = NN.J_list # add the value of J in the last iteration to the hash table.
            print(i)
            del NN # Safety measure to avoid leak. 
            i += 1

# Now, find max and argmax from the hash-table config. 
import pickle
filename = 'cats'
outfile = open(filename,'wb')
pickle.dump(config, outfile)
outfile.close()
#print(config)
