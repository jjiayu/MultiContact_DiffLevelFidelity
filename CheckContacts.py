import pickle
import numpy as np

import matplotlib.pyplot as plt #Matplotlib
from mpl_toolkits.mplot3d import Axes3D

filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/result_with_wrong_terrain_tangent_and_norm/RefMotions/10x_TerminalCost/flat_ref/" + "10LookAhead_Trial0.p"

with open(filename, 'rb') as f:
    data = pickle.load(f)


print("PL inits: ",data["PL_init_fullres"])
print("PR inits: ",data["PR_init_fullres"])