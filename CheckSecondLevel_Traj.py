import pickle
import numpy as np

import matplotlib.pyplot as plt #Matplotlib
from mpl_toolkits.mplot3d import Axes3D

from NLP_Ref_Traj_Constructor import *

#filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/antfarm_CoM_No_TerminalCost_AntfarmRef/5LookAhead_Trial0.p"

NumLookAhead = 5
RoundNum = 5

filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/antfarm_CoM_Tracking_SecondLevel/"+str(NumLookAhead)+"LookAhead_Trial0.p"

#filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/Result_10x_TerminalCost/Uneven_Tracking_Individual_Ones/antfarm_CoM_antfarm_ref_005Height/5LookAhead_Trial0.p"

with open(filename, 'rb') as f:
    data = pickle.load(f)

Level1_VarIndex = data["VarIdx_of_All_Levels"]["Level1_Var_Index"]
Level2_VarIndex = data["VarIdx_of_All_Levels"]["Level2_Var_Index"]

#print(Level2_VarIndex)

Trajectories = data["Trajectory_of_All_Rounds"]

traj = Trajectories[RoundNum]

traj_secondlevel = traj[Level1_VarIndex["Ts"][1]+1:]

#print(len(traj))

x_secondlevel = traj_secondlevel[Level2_VarIndex["x"][0]:Level2_VarIndex["x"][1]+1]
#print(x_secondlevel)
y_secondlevel = traj_secondlevel[Level2_VarIndex["y"][0]:Level2_VarIndex["y"][1]+1]

singlevar = traj_secondlevel[Level2_VarIndex["xdot"][0]:Level2_VarIndex["xdot"][1]+1]

#plt.plot(x_secondlevel,y_secondlevel,label='SecondLevel')
plt.plot(singlevar,label='SecondLevel')

#-------------
#Get ref traj
x_traj, y_traj, z_traj, xdot_traj, ydot_traj, zdot_traj, FLx_traj, FLy_traj, FLz_traj, FRx_traj, FRy_traj, FRz_traj, Px_seq, Py_seq, Pz_seq, SwitchingTimeVec = NLP_ref_trajectory_construction(StartStepNum = RoundNum, LookAheadSteps = NumLookAhead)
#x_traj, y_traj, z_traj, xdot_traj, ydot_traj, zdot_traj, FLx_traj, FLy_traj, FLz_traj, FRx_traj, FRy_traj, FRz_traj, Px_seq, Py_seq, Pz_seq, SwitchingTimeVec = NLP_ref_trajectory_from_SecondLevel(StartStepNum = RoundNum, LookAheadSteps = NumLookAhead)


#plt.plot(x_traj,y_traj,label = "ref_traj")
plt.plot(xdot_traj,label = "ref traj")
plt.title("y dot traj for " + str(NumLookAhead) + 'Lookahead ' + str(RoundNum) + "th Round")


plt.legend(loc="upper right")

plt.show()
