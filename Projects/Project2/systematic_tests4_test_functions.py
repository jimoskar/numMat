"""Testing framework for running systematic tests of the neural network.

This is in order to find the optimal values of the parameters in the network, 
by method of experimentation (systematic tests).

Parameters tested systematically:
- K
- tau
- d 
- h

Used a fixed amount of iterations and images in each test (see report for further justification)
"""
from network_algo import *
import pickle

def training_test_function1(filename):
    """Only training the test function 1."""
    K_values = [10, 15, 20, 23]
    d_values = [2, 3, 4]
    h_values = [0.05, 0.1, 0.2, 0.3, 0.4]
    iterations = 2000
    d0 = 1 # Dimension of the input layer. 
    domain = [-2,2]
    I = 1000
    chunk = int(I/1)
    scaling = False
    alpha = 0.2
    beta = 0.8 
    tau = 1
    def test_function1(x):
        return 0.5*x**2

    config = {}
    i = 1
    for K in K_values:
        config[K] = {}
        for d in d_values:
            config[K][d] = {}
            for h in h_values:
                NN = algorithm(I,d,d0,K,h,iterations,tau, chunk, test_function1, domain, scaling, alpha, beta)
                config[K][d][h] = NN.J_last # add the value of J in the last iteration to the hash table.
                               
                print(i)
                del NN # Safety measure to avoid leak. 
                i += 1

    # Save data to file.
    filenm = 'data/'+filename
    outfile = open(filenm,'wb')
    pickle.dump(config, outfile)
    outfile.close()


def testing_test_function1(filename, tol = 0.05):
    """Training and testing random data with test function 1."""
    K_values = [10, 15, 20, 23]
    d_values = [2, 3, 4]
    h_values = [0.05, 0.1, 0.2, 0.3, 0.4]
    iterations = 2000
    d0 = 1 # Dimension of the input layer. 
    domain = [-2,2]
    I = 1000
    chunk = int(I/1)
    scaling = False
    alpha = 0.2
    beta = 0.8 
    tau = 1
    def test_function1(x):
        return 0.5*x**2

    config = {}
    i = 1
    for K in K_values:
        config[K] = {}
        for d in d_values:
            config[K][d] = {}
            for h in h_values:
                NN = algorithm(I,d,d0,K,h,iterations,tau, chunk, test_function1, domain, scaling, alpha, beta) 
                test_input = generate_input(test_function1,domain,d0,I,d)
                testing(NN, test_input, test_function1, domain, d0, d, I, scaling, alpha, beta)

                # Find amount of correctly classified points (within some tolerance).
                placeholder = np.array([NN.Y, NN.c])
                diff = np.diff(placeholder, axis = 0)
                ratio = len(diff[abs(diff)<tol])/len(diff[0])

                # Add the convergence (last value) of the objective function and the ratio of correctly classified points to testing output.
                config[K][d][h] = [NN.J(), ratio]
                               
                print(i)
                del NN # Safety measure to avoid leak. 
                i += 1

    # Save data to file. 
    filenm = 'data/'+filename
    outfile = open(filenm,'wb')
    pickle.dump(config, outfile)
    outfile.close()


def training_test_function3(filename):
    K_values = [10, 15, 20, 23, 30]
    d_values = [2, 3, 4, 5, 6]
    h_values = [0.05, 0.1, 0.2, 0.3, 0.4]
    d0 = 2
    d = 4
    domain = [[-2,2],[-2,2]]
    chunk = int(I/10)
    scaling = False
    alpha = 0.2
    beta = 0.8 
    I = 1000

    def test_function3(x,y):
        return 0.5*(x**2 + y**2)

    config = {}
    i = 1
    for K in K_values:
        config[K] = {}
        for d in d_values:
            config[K][d] = {}
            for h in h_values:
                NN = algorithm(I,d,d0,K,h,iterations,tau, chunk, test_function3, domain, scaling, alpha, beta) # This only tests convergence when training. 
                # The below adds the J_list from training. 
                config[K][d][h] = NN.J_last # add the value of J in the last iteration to the hash table.
                                
                print(i)
                del NN # Safety measure to avoid leak. 
                i += 1

    # Save data to file. 
    filenm = 'data/'+filename
    outfile = open(filenm,'wb')
    pickle.dump(config, outfile)
    outfile.close()

def testing_test_function3(filename, tol):
    K_values = [10, 15, 20, 23, 30]
    d_values = [2, 3, 4, 5, 6]
    h_values = [0.05, 0.1, 0.2, 0.3, 0.4]
    d0 = 2
    d = 4
    domain = [[-2,2],[-2,2]]
    chunk = int(I/10)
    scaling = False
    alpha = 0.2
    beta = 0.8 
    I = 1000

    def test_function3(x,y):
        return 0.5*(x**2 + y**2)

    config = {}
    i = 1
    for K in K_values:
        config[K] = {}
        for d in d_values:
            config[K][d] = {}
            for h in h_values:
                NN = algorithm(I,d,d0,K,h,iterations,tau, chunk, test_function3, domain, scaling, alpha, beta) 
                test_input = generate_input(test_function3,domain,d0,I,d)
                testing(NN, test_input, test_function3, domain, d0, d, I, scaling, alpha, beta)

                # Find amount of correctly classified points (within some tolerance).
                placeholder = np.array([NN.Y, NN.c])
                diff = np.diff(placeholder, axis = 0)
                ratio = len(diff[abs(diff)<tol])/len(diff[0])
            
                # Add the convergence (last value) of the objective function and the ratio of correctly classified points to testing output.
                config[K][d][h] = [NN.J(), correct/len(NN.c)]
                                
                print(i)
                del NN # Safety measure to avoid leak. 
                i += 1

    # Save data to file. 
    filenm = 'data/'+filename
    outfile = open(filenm,'wb')
    pickle.dump(config, outfile)
    outfile.close()
