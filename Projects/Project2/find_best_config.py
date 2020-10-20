import pickle
filename = 'testing_testfunc3'
infile = open(filename,'rb')
config = pickle.load(infile)
infile.close()

import json
print(json.dumps(config, indent = 4)) 
# Find the value manually from printing this (pretty easy)
