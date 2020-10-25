"""Run wanted test and find (manually) the best configurations from the data."""
from systematic_tests4_test_functions import *
import sys
import json

"""
# Get filename and function name from command line.
filename = sys.argv[2]
function_name = sys.argv[1]
sgd_or_not = sys.argv[3]

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

if sgd_or_not == "t":
    sgd = True
else:
    sgd = False

function_dict[function_name](filename, sgd = sgd)
"""

# Find best config from the generated data. 
# Make a loop to run through all the files in the given directory! This way it goes automatically.

infile = open("data/markov/tol0point05/test_func_1_Reg", 'rb')
config = pickle.load(infile)
infile.close()
#print(config)
# Print data in nice format, to be able to read it more easily. 
#print(json.dumps(config, indent = 5)) 
import csv
fields = [ 'K', 'd', 'h', 'J', 'Ratio of Correctness' ]
K_values = [10, 15, 20, 23, 30]
d_values = [2, 3, 4]
h_values = [0.05, 0.1, 0.2, 0.3, 0.4]
print(config.items())
with open("data/markov/tol0point05/test_output.csv", "w") as f:
    w = csv.writer(f)#DictWriter(f, fields)
    w.writerow(fields)
    for K in K_values:
        for d in d_values:
            for h in h_values:
                w.writerow([K, d, h, config[K][d][h][0], config[K][d][h][1]])
                