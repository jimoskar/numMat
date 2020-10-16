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
