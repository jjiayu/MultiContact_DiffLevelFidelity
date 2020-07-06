from numpy import array, dot, stack, vstack, hstack, asmatrix, identity, cross, concatenate
from numpy.linalg import norm
import numpy as np
from sl1m.constants_and_tools import *
from PlotResult import *
import matplotlib.pyplot as plt #Matplotlib
from mpl_toolkits.mplot3d import Axes3D

step1 = array([[0.16, 1., 0], [-1.8, 1., 0], [-1.8, -1., 0.], [0.16, -1., 0.]])

fig=plt.figure()
ax = Axes3D(fig)

PlotSurface(Surface = step1, ax = ax)

ax.set_xlim3d(0-0.2, 1+0.35)
ax.set_ylim3d(-0.5,0.5)
ax.set_zlim3d(0,0.8)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

step1 = step1.T

surfhalf = convert_surface_to_inequality(step1)

#print(np.concatenate(surfhalf,axis=None))