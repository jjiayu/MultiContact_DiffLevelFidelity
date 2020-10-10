
import mpl_toolkits.mplot3d as a3
import pylab as pl
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt #Matplotlib
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

def plotPolyhedron(hull, ax, color = "r", alpha = 0.1):
    for s in hull.simplices:
        tri = a3.art3d.Poly3DCollection(hull.points[s], alpha = alpha)
        tri.set_color(color)
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)
    return ax

def plotPoint(point,ax, color = "g", alpha = 1):
    ax.scatter(point[0], point[1], point[2], c=color, marker='o', linewidth = 10) 

#Load  
lf_in_rf_points = np.array([
[-0.296233,0.274978,0.141514],
[-0.287472,0.142944,0.137646],
[0.293593,0.260494,0.157628],
[-0.281891,0.456295,0.155254],
[-0.292398,0.499051,-0.153242],
[-0.285717,0.143971,-0.132349],
[0.266554,0.451861,-0.155800],
[0.295180,0.231727,-0.143239],
[0.284648,0.133310,0.041392],
[0.282648,0.453987,0.147154],
[0.293738,0.503263,-0.050931],
[-0.185346,0.489137,0.146593]
])

rf_in_lf_points = np.array([[0.289139,-0.166685,-0.150686],
[-0.030917,-0.499853,-0.160519],
[-0.299581,-0.180950,-0.158220],
[-0.299640,-0.140275,0.152504],
[0.299235,-0.166577,0.154751],
[-0.290924,-0.491739,0.047988],
[-0.286598,-0.475018,0.155675],
[0.294880,-0.492564,-0.134272],
[0.274790,-0.419324,0.146844],
[0.178476,-0.493711,0.126348],
[-0.250186,-0.465852,-0.132670]])

com_in_lf = np.array([[0.038395,-0.290074,0.551716],
[-0.149505,-0.203483,0.568595],
[-0.044942,0.003901,0.606689],
[-0.215187,-0.613909,0.300059],
[0.079828,-0.163866,0.587590],
[0.341034,-0.081221,0.520543],
[0.257744,-0.284187,0.499712],
[-0.319260,-0.224750,0.503677],
[0.064146,-0.377686,0.504203],
[0.516027,0.006606,0.300065],
[0.515678,-0.328989,0.299747],
[0.274168,-0.551651,0.300000],
[-0.645475,0.009988,0.299978],
[0.241920,0.007852,0.550602],
[0.457366,0.003112,0.396014],
[-0.161147,-0.300031,0.529137],
[-0.570543,-0.300646,0.300039],
[-0.254403,0.005235,0.562349]])

com_in_rf = np.array([[-0.153951,0.255618,0.553467],
[-0.044942,-0.003901,0.606689],
[-0.215187,0.613909,0.300059],
[0.079829,0.163865,0.587590],
[0.341034,0.081221,0.520543],
[0.257743,0.284187,0.499712],
[-0.319260,0.224750,0.503677],
[0.056900,0.335274,0.532251],
[0.475328,0.443365,0.302048],
[-0.645475,-0.009988,0.299978],
[0.532470,-0.003379,0.302202],
[0.241920,-0.007852,0.550602],
[-0.570543,0.300646,0.300039],
[-0.254403,-0.005235,0.562349]])

#hull_lf_in_rf = ConvexHull(lf_in_rf_points)
#hull_rf_in_lf = ConvexHull(rf_in_lf_points)
#hull_com_in_lf = ConvexHull(com_in_lf)
#hull_com_in_rf = ConvexHull(com_in_rf)

fig=plt.figure()
ax = Axes3D(fig)

#plotPolyhedron(hull_lf_in_rf, ax, color = "r", alpha = 0.1)
#plotPolyhedron(hull_rf_in_lf, ax, color = "b", alpha = 0.1)
#plotPolyhedron(hull_com_in_lf, ax, color = "m", alpha = 0.1)
#plotPolyhedron(hull_com_in_rf, ax, color = "y", alpha = 0.1)

#load constraints
from sl1m.problem_definition import *
from sl1m.planner_scenarios.talos.constraints import *

K_CoM_Right,k_CoM_Right = right_foot_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]))
K_CoM_Left,k_CoM_Left = left_foot_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]))

Q_rf_in_lf,q_rf_in_lf = right_foot_in_lf_frame_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]))
Q_lf_in_rf,q_lf_in_rf = left_foot_in_rf_frame_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]))

#Lfoot_pos = np.array([0,0.2,0])
#Rfoot_pos = np.array([0,-0.2,0])

#-----------------------------------
##Sample points
##randompoints = np.random.rand(10, 3) 
#xtick = np.linspace(-1,1,40)
#ytick = np.linspace(-1,1,40)
#ztick = np.linspace(-1,1,40)
##xtick = np.array([0])
##ytick = np.array([0])
##ztick = np.array([0.8])
#np.set_printoptions(precision=4)
#
##print(randompoints)
#
#for xpoint in xtick:
#    for ypoint in ytick:
#        for zpoint in ztick:
#            pointTemp = np.concatenate((xpoint,ypoint,zpoint),axis=None)
#            indicators = K_CoM_Left@pointTemp<=k_CoM_Left
#            lampedindicator = True
#            for indicator in indicators:
#                if indicator == False:
#                    lampedindicator = False
#            if lampedindicator == True:
#                plotPoint(pointTemp,ax, color = "g", alpha = 1)
##for randpoint in randompoints:
##    x_temp = -1 + randpoint[0]*2
##    y_temp = -1 + randpoint[1]*2
##    z_temp = 0 + randpoint[1]*1.5
#
##    scalepoint = np.concatenate((x_temp,y_temp,z_temp),axis=None)
#
##    #print(scalepoint)
#------------------------------------------

#Shift the polytopes
#Lfoot_pos = np.array([0.8918,0.1136,0])
#Rfoot_pos = np.array([0.6,-0.1,0])
#CoM_pos = np.array([0.7744,9.4532e-03,0.5267])

Lfoot_pos = np.array([0,0,0])
Rfoot_pos = np.array([0,0,0])
CoM_pos = np.array([0.,0,0.])

lf_in_rf_points = lf_in_rf_points + Rfoot_pos
rf_in_lf_points = rf_in_lf_points + Lfoot_pos
com_in_lf = com_in_lf + Lfoot_pos
com_in_rf = com_in_rf + Rfoot_pos

hull_lf_in_rf = ConvexHull(lf_in_rf_points)
hull_rf_in_lf = ConvexHull(rf_in_lf_points)
hull_com_in_lf = ConvexHull(com_in_lf)
hull_com_in_rf = ConvexHull(com_in_rf)

plotPolyhedron(hull_lf_in_rf, ax, color = "r", alpha = 0.1)
plotPolyhedron(hull_rf_in_lf, ax, color = "b", alpha = 0.1)
plotPolyhedron(hull_com_in_lf, ax, color = "m", alpha = 0.1)
plotPolyhedron(hull_com_in_rf, ax, color = "y", alpha = 0.1)


plotPoint(CoM_pos,ax, color = "g", alpha = 1)
plotPoint(Lfoot_pos,ax, color = "r", alpha = 1)
plotPoint(Rfoot_pos,ax, color = "b", alpha = 1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()