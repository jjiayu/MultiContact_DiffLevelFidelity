import numpy as np

from sl1m.constants_and_tools import *
from sl1m.planner import *
from constraints import * 

#   Set Decimal Printing Precision
np.set_printoptions(precision=4)

#Kinematics Constraint for Talos
kinematicConstraints = genKinematicConstraints(left_foot_constraints,right_foot_constraints)

Al = kinematicConstraints[0][0]
bl = kinematicConstraints[0][1]
Ar = kinematicConstraints[1][0]
br = kinematicConstraints[1][1]

#Relative Foot Constraint matrices
relativeConstraints = genFootRelativeConstraints(right_foot_in_lf_frame_constraints,left_foot_in_rf_frame_constraints)

Rl = relativeConstraints[0][0] #named lf in rf, but representing rf in lf
rl = relativeConstraints[0][1] #named lf in rf, but representing rf in lf
Rr = relativeConstraints[1][0] #named rf in lf, but representing lf in rf
rr = relativeConstraints[1][1] #named rf in lf, but representing lf in rf
#
#
#LFoot = np.array([0.8892,0.0815,0])
#RFoot = np.array([1.1913,-0.0954,0])
#CoMpos = np.array([1.0174,0.0,0.5689])

LFoot = np.array([0.8892,0.0815,0])
RFoot = np.array([1.1913,-0.0954,0])
CoMpos = np.array([1.0174,0.0,0.5689])

print("Lfoot")
print(Al@(CoMpos-LFoot)<=bl)

print("Rfoot")
print(Ar@(CoMpos-RFoot)<=br)

print("R foot in L foot")
print(Rr@(RFoot-LFoot)<=rr)
