import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.linalg as la
from mpl_toolkits import mplot3d
from import_data import generate_data, concatenate
from testing import *
from utilities import stochastic_elements


class Parameters:
    """Class for the parameters in the Neural Network."""
    def __init__(self,K,d,I,iterations):
        self.W_k = np.random.randn(K,d,d)
        self.b_k = np.random.randn(K,d,1)
        self.b_k_I = np.zeros((K,d,I))
        for i in range(K):
            self.b_k_I[i,:,:] = self.b_k[i,:,:]

        self.w = np.random.randn(d)  
        self.my = np.random.rand()
        self.K = K
        
        # For Adam descent.
        self.beta_1 = 0.9; self.beta_2 = 0.999; self.alpha = 0.01; self.epsilon = 10E-8
        self.v = [0]; self.m = [0]
        
    # Perhaps this class should only contain the parameters 
    # and the update + root functions should be part of the Network?
    # Perhaps this makes some sense when seeing the algorithm-function after all. 
    def update_parameters(self,gradient,method,tau,j):
        """Update the parameters via Vanilla Gradient Method or Adam Descent."""
        if(method == "vanilla"):
            self.W_k = self.W_k - tau*gradient[0]
            self.b_k = self.b_k - tau*gradient[1]
            for i in range(self.K):
                self.b_k_I[i,:,:] = self.b_k[i,:,:]
            self.w = self.w - tau*gradient[2]
            self.my = self.my - tau*gradient[3]
            
        elif(method == "adams"):
            g = gradient 
            self.m.append( self.beta_1*self.m[j-1] + (1-self.beta_1)*g)
            self.v.append(self.beta_2*self.v[j-1]+(1-self.beta_2)*(g*g))
            m_hat = self.m[j]/(1-self.beta_1**j)
            v_hat = self.v[j]/(1-self.beta_2**j)
            subtract = self.alpha*m_hat/(self.root(self.v[j])+self.epsilon)
            
            self.W_k = self.W_k - subtract[0]
            self.b_k = self.b_k - subtract[1]
            for i in range(self.K):
                self.b_k_I[i,:,:] = self.b_k[i,:,:]
            self.w = self.w - subtract[2]
            self.my = self.my - subtract[3]      
                      
        else:
            print("No method found") 
    
    def root(self, v_j):
        """Calculate the root of v_j componentwise for Adam descent."""
        for i in range(4):
            v_j[i] = np.sqrt(v_j[i])
        return v_j


class Network:
    """Class for the Neural Network (ResNet)."""
    def __init__(self,K,d,I,h,Z_0,c,iterations):
        self.theta = Parameters(K,d,I,iterations)
        self.Z_list = np.zeros((K+1,d,I))
        self.Z_list[0,:,:] = Z_0
        self.h = h
        self.K = K
        self.I = I
        self.d = d
        self.Y = None
    
    def J(self,Y,c): 
        """Objective function."""
        return 0.5 * la.norm(Y - c)**2
    
    def sigma(self, x):
        """Sigmoid activation function."""
        return np.tanh(x)

    def sigma_der(self, x):
        """Derivative of the sigmoid activation function."""
        val = 1-np.tanh(x)**2
        return val


    def eta(self, x):
        """Hypothesis function. Appears in the final layer of the neural network."""
        return x


    def eta_der(self, x):
        """The derivative of the hypothesis function."""
        return 1

    def forward_function(self): 
        """Calculate and return Z."""
        for i in range(self.K):
            #jmf. formel (1) i heftet
            self.Z_list[i+1,:,:] = self.Z_list[i,:,:] + \
                self.h*self.sigma(self.theta.W_k[i,:,:] @ \
                    self.Z_list[i,:,:] + self.theta.b_k_I[i,:,:])
            
        ZT_K = np.transpose(self.Z_list[-1,:,:]) #Enklere notasjon
        one_vec = np.ones(self.I)
        '''
        print("comp1")
        print(ZT_K @ self.theta.w)
        print("comp2")
        print(self.theta.my*one_vec)
        print("input Z")
        print(ZT_K @ self.theta.w + self.theta.my*one_vec)
        '''
        Y = self.eta(ZT_K @ self.theta.w + self.theta.my*one_vec)
        self.Y = Y
        return Y  #Y er en Ix1 vektor
        # Why return Y here, when it is saved as an attribute of the class? 
        # Could just take it from the object when needed instead. 
        
    
    def back_propagation(self,Y,c):
        """Calculate and return the gradient of the objective function."""
        ZT_K = np.transpose(self.Z_list[-1,:,:])
        one_vec = np.ones(self.I)
        J_der_my =  np.transpose(one_vec) @ (Y-c)
        #eta_der(ZT_K @ self.theta.w + self.theta.my*one_vec) @ (Y-c)#Blir en skalar  
        J_der_omega = self.Z_list[-1,:,:] @ ((Y-c) * \
                            self.eta_der(ZT_K @ self.theta.w + self.theta.my*one_vec))
        
        P_K = np.outer(self.theta.w,(Y-c)*self.eta_der(ZT_K @ \
                                            self.theta.w + self.theta.my*one_vec)) #Blir en dxI matrise
        
        
        P_list = np.zeros((self.K,self.d,self.I)) #K matriser, skal ikke ha med P_0
        P_list[-1,:,:] = P_K      #Legger P_K bakerst i P_list
        for i in range(self.K-1,0,-1):  #Starter på P_k(=indeks K-1) og helt til og med P_1(=indeks 0)
            P_list[i-1,:,:] = P_list[i,:,:] + self.h*np.transpose(self.theta.W_k[i-1,:,:]) @ \
            (self.sigma_der(self.theta.W_k[i-1,:,:] @ self.Z_list[i-1,:,:] + \
                            self.theta.b_k_I[i-1,:,:]) * P_list[i,:,:])

        J_der_W = np.zeros((self.K,self.d,self.d))
        J_der_b = np.zeros((self.K,self.d,1))
        one_vec = np.ones((self.I,1))  #Må gjøre vec_I til en matrise av en eller annen grunn
        #P_Kk går fra P_1(=indeks 0) til P_K(=indeks K-1)
        
        for i in range(self.K):
            val = P_list[i,:,:] * self.sigma_der(self.theta.W_k[i,:,:] @ \
                                                 self.Z_list[i,:,:] + self.theta.b_k_I[i,:,:])
            J_der_W[i,:,:] = self.h*(val @ np.transpose(self.Z_list[i,:,:]))
            J_der_b[i,:,:] = self.h*(val @ one_vec)
        
        gradient = np.array((J_der_W,J_der_b,J_der_omega,J_der_my))
        
        return gradient


    # Perhaps there should be one class per part of the Hamiltonian and
    # this hould be a method of those classes? (or an abstract class 
    # for the complete Hamiltonian?)
    # Also fine to leave it here, since it uses all the attributes from the object.  
    def Hamiltonian_gradient(self):
        """Calculate the gradient of F, according to the theoretical derivation.
        
        In general it is used for the entire Hamiltonian F, but since we are 
        working with separable Hamiltonians, it can be used on both T and V.
        """

        # Here I have assumed that Y_list is Z^(k) from the report. 
        one_vec = np.ones((self.I,1)) 
        gradient = self.theta.w*np.transpose((self.eta_der( \
            np.transpose(self.Z_list[self.K,:,:])+self.theta.my*one_vec)))
        for i in range(self.K-1, -1, -1):
            gradient *= (np.identity(self.d) + self.theta.W_k[i,:,:]*self.h* \
                np.transpose(self.sigma_der(self.theta.W_k[i,:,:]*self.Z_list[i,:,:] \
                    + self.theta.b_k_I[i,:,:])))
        
        # Could either set the gradient as an attribute of the object
        self.hamiltonian_gradient = gradient
        # or just return the gradient. 
        return gradient

    
def algorithm(I,d,K,h,iterations,function,domain):
    """Main training algorithm."""
    tau = 0.1
      
    Z_0 = generate_input(function,domain,d0,I,d)
    c = get_solution(function,Z_0,d,I,d0)

    # Try with SGD: Pick out only 1/10 of the points at a time. 
    chunk = I//10
    Z_0, c = stochastic_elements(Z_0, c, I, chunk)
    
    NN = Network(K,d,chunk,h,Z_0,c,iterations)
    
    # For plotting J. 
    J_list = np.zeros(iterations)
    it = np.zeros(iterations)
    
    for j in range(1,iterations+1):
        
        Z = NN.forward_function() # See comment in forward function about returning Z.
        
        gradient = NN.back_propagation(Z,c)
        NN.theta.update_parameters(gradient,"adams",tau,j)
        
        J_list[j-1] = NN.J(Z,c)
        it[j-1] = j
        
    plt.figure()
    plt.plot(it,J_list)
    plt.ylabel("J")
    plt.xlabel("iteration")
    plt.show()

    print("Objective function in iteration", str(iterations) + ": " + str(J_list[-1]))
    
    return NN


""" 
Below, we train and test the neural network with the provided test functions.
"""

I = 500 # Amount of points ran through the network at once. 
K = 20 # Amount of hidden layers in the network.
d = 2 # Dimension of the hidden layers in the network. 
h = 0.1 # Scaling of the activation function application in algorithm.  
iterations = 1000

#================#
#Test function 1 #
#================#

d0 = 1 # Dimension of the input layer. 
domain = [-2,2]
def test_function1(x):
    return 0.5*x**2

NN = algorithm(I,d,K,h,iterations,test_function1,domain)
test_input = generate_input(test_function1,domain,d0,I,d)
output = testing(NN, test_input, test_function1, domain, d0, d, I)
plot_graph_and_output(output, test_input, test_function1, domain, d0,d)

#================#
#Test function 2 #
#================#
""""
d0 = 1 # Dimensin of the input layer. 
domain = [-2,2]
def test_function2(x):
    return 1 - np.cos(x)

NN = algorithm(I,d,K,h,iterations,test_function2,domain)
test_input = generate_input(test_function2,domain,d0,I,d)
output = testing(NN, test_input, test_function2, domain, d0, d, I)
plot_graph_and_output(output,test_input, test_function2, domain, d0,d)
"""
#================#
#Test function 3 #
#================#
"""
d0 = 2
d = 4
domain = [[-2,2],[-2,2]]
def test_function3(x,y):
    return 0.5*(x**2 + y**2)

NN = algorithm(I,d,K,h,iterations,test_function3,domain)

test_input = generate_input(test_function3,domain,d0,I,d)
output = testing(NN, test_input, test_function3, domain, d0, d, I)
plot_graph_and_output(output, test_input, test_function3, domain, d0,d)
"""

#================#
#Test function 4 #
#================#
"""
d0 = 2
d = 4
domain = [[-2,2],[-2,2]]
def test_function4(x,y):
    return -1/np.sqrt(x**2 + y**2)

NN = algorithm(I,d,K,h,iterations,test_function4,domain)

test_input = generate_input(test_function4,domain,d0,I,d)
output = testing(NN, test_input, test_function4, domain, d0, d,  I)
plot_graph_and_output(output,test_input, test_function4, domain, d0,d)

"""
