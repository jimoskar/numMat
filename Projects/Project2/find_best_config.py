"""Run wanted test and find (manually) the best configurations from the data."""
from systematic_tests4_test_functions import *
import sys
import json

# Get filename and function name from command line.
filename = sys.argv[2]
function_name = sys.argv[1]

function_dict = {
    "training_test_function1": training_test_function1,
    "testing_test_function1": testing_test_function1,
    "training_test_function2": training_test_function2,
    "testing_test_function2": testing_test_function2,
    "training_test_function3": training_test_function3,
    "testing_test_function3": testing_test_function3, 
    "training_test_function4": training_test_function4,
    "testing_test_function4": testing_test_function4,

}

function_dict[function_name](filename)

"""
# Find best config from the generated data. 
filenm = 'data/'+filename
infile = open(filenm, 'rb')
config = pickle.load(infile)
infile.close()

# Print data in nice format, to be able to read it more easily. 
#print(json.dumps(config, indent = 5)) 
"""
