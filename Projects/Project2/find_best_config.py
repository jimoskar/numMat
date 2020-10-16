import pickle
filename = 'dogs'
infile = open(filename,'rb')
config = pickle.load(infile)
infile.close()

# Now find argmax, max, argmin and min from the config"
#print(max(dict["City"].items(), key=lambda x: x[1]['n_trips'])[0]) # Inspirert av dette!?
# This should give argmax! Also need argmin (similarly) 
# and max/min. Should be similar as well I imagine.  
print(config)
#min_K = list(config.keys())[0]
#print(min_K)
#print(config.values())
"""
for key, value in config.items():
    #print(key, "|", value)
    min_K = key
    for key2, value2 in value.items():
        min_d = key2
        print(key2, "|", value2)
"""
# I have not been able to find the max yet!?
# Perhaps store the simulated values in some other way is easier?
