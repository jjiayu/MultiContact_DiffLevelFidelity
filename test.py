# creating contact points
import numpy as np

for Px in np.arange(0,0.5,0.05):
    for Py in np.arange(-0.1,-0.5,0.05):
        print([Px,Py,0])