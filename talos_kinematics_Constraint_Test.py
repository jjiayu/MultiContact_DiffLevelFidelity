import numpy as np

from sl1m.problem_definition import *

from sl1m.planner_scenarios.talos.constraints import *

#   Set Decimal Printing Precision
np.set_printoptions(precision=4)

Ar,br = right_foot_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]))

Al,bl = left_foot_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]))

Rr,rr = right_foot_in_lf_frame_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]))

Rl,rl = left_foot_in_rf_frame_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]))

#
#LFoot = np.array([1.2499,0.2287,0])
#RFoot = np.array([0.9231,-0.0046,0])
#CoMpos = np.array([1.0837,0.0734,0.8698])

LFoot = np.array([0.9268,0.1333,0])
RFoot = np.array([0.6,-0.1,0])
CoMpos = np.array([1.0837,0.0734,0.8698])

print("Lfoot")
print(Al@(CoMpos-LFoot)<=bl)

print("Rfoot")
print(Ar@(CoMpos-RFoot)<=br)

print("R foot in L foot")
print(Rr@(RFoot-LFoot)<=rr)

