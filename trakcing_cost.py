import pickle
import numpy as np

import matplotlib.pyplot as plt #Matplotlib
from mpl_toolkits.mplot3d import Axes3D

from NLP_Ref_Traj_Constructor import *

import casadi as ca

NumofLookAhead = 2
roundNum = 2

x_traj_ref, y_traj_ref, z_traj_ref, xdot_traj_ref, ydot_traj_ref, zdot_traj_ref, FLx_traj_ref, FLy_traj_ref, FLz_traj_ref, FRx_traj_ref, FRy_traj_ref, FRz_traj_ref, Px_seq_ref, Py_seq_ref, Pz_seq_ref, SwitchingTimeVec_ref = NLP_ref_trajectory_construction(StartStepNum = roundNum, LookAheadSteps = NumofLookAhead)


z_vars = ca.SX.sym('z',len(z_traj_ref))
z_vars_lb = [0.7]*len(z_traj_ref)
z_vars_ub = [0.8]*len(z_traj_ref)

z_vars_init = [0]*len(z_traj_ref)

J = 0

for idx in range(len(z_traj_ref)):

    J = J + (z_vars[idx]-z_traj_ref[idx])**2

prob = {'x': z_vars, 'f': J, 'g': []}
#solver = ca.nlpsol('solver', 'knitro', prob,opts)
solver = ca.nlpsol('solver', 'knitro', prob)

res = solver(x0=z_vars_init, lbx = z_vars_lb, ubx = z_vars_ub, lbg = [], ubg = [])

x_opt = res["x"]
x_opt = x_opt.full().flatten()

plt.plot(x_opt,label='SecondLevel')
plt.plot(z_traj_ref,label = "ref_traj")
#plt.ylim([0.7,0.82])
plt.title("z traj for " + str(NumofLookAhead) + 'Lookahead ' + str(roundNum) + "th Round")
plt.legend(loc="upper right")
plt.show()