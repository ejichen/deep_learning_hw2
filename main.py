import numpy as np
from utils import *

#loading data
# use the train_load to load the data, the path indicate the location of the data
# the parts indicate which .npz should be loaded, for example {1} loads only 1.processed.npz and {1,2,tes}
# loads the 1, 2 and test .processed.npz
path  = "/Users/eden/CMU/11785_Deep_Learning/handout3/hw2p2-fall18/hw2p2_A"

features, speakers, nspeakers = train_load(path, {1})


print("success")