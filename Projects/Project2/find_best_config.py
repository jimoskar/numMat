"""Run wanted test and find (manually) the best configurations from the data."""
from systematic_tests4_test_functions import *
import json
filename = 'testing_testfunc1GD'

# Run test.
testing_test_function1(filename)

# Find best config from the generated data. 
filenm = 'data/'+filename
infile = open(filenm, 'rb')
config = pickle.load(infile)
infile.close()

# Print data in nice format, to be able to read it more easily. 
print(json.dumps(config, indent = 5)) 
