import pickle
import numpy as np

import matplotlib.pyplot as plt #Matplotlib
from mpl_toolkits.mplot3d import Axes3D

from NLP_Ref_Traj_Constructor import *

#filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/antfarm_CoM_No_TerminalCost_AntfarmRef/5LookAhead_Trial0.p"

NumLookAhead = 10
RoundNum = 0

#filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/antfarm_CoM_Tracking_SecondLevel/"+str(NumLookAhead)+"LookAhead_Trial0.p"
filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/Barrule_fixed/track_antfarm005_10steps_uneven_007/darpa_like_left_first_CoM_previous/"+str(NumLookAhead)+"LookAhead_Trial0.p"
#filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/TrackInvestigation_AntFarm/Tracking_CostOnly_Decoupled_from_The_FirstLevel_Integrator_Constraint(Dynamics)/" +str(NumLookAhead)+"LookAhead_Trial0.p"
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

#-------------
#Get ref traj
x_traj, y_traj, z_traj, xdot_traj, ydot_traj, zdot_traj, FLx_traj, FLy_traj, FLz_traj, FRx_traj, FRy_traj, FRz_traj, Px_seq, Py_seq, Pz_seq, SwitchingTimeVec,px_init_ref, py_init_ref, pz_init_ref = NLP_ref_trajectory_construction(StartStepNum = RoundNum, LookAheadSteps = NumLookAhead)
#x_traj, y_traj, z_traj, xdot_traj, ydot_traj, zdot_traj, FLx_traj, FLy_traj, FLz_traj, FRx_traj, FRy_traj, FRz_traj, Px_seq, Py_seq, Pz_seq, SwitchingTimeVec = NLP_ref_trajectory_from_SecondLevel(StartStepNum = RoundNum, LookAheadSteps = NumLookAhead)


#x-y plot
x_secondlevel = traj_secondlevel[Level2_VarIndex["x"][0]:Level2_VarIndex["x"][1]+1]
#print(x_secondlevel)
y_secondlevel = traj_secondlevel[Level2_VarIndex["y"][0]:Level2_VarIndex["y"][1]+1]

z_secondlevel = traj_secondlevel[Level2_VarIndex["z"][0]:Level2_VarIndex["z"][1]+1]

xdot_secondlevel = traj_secondlevel[Level2_VarIndex["xdot"][0]:Level2_VarIndex["xdot"][1]+1]
ydot_secondlevel = traj_secondlevel[Level2_VarIndex["ydot"][0]:Level2_VarIndex["ydot"][1]+1]
zdot_secondlevel = traj_secondlevel[Level2_VarIndex["zdot"][0]:Level2_VarIndex["zdot"][1]+1]

FLx_secondlevel = traj_secondlevel[Level2_VarIndex["FLx"][0]:Level2_VarIndex["FLx"][1]+1]
FLy_secondlevel = traj_secondlevel[Level2_VarIndex["FLy"][0]:Level2_VarIndex["FLy"][1]+1]
FLz_secondlevel = traj_secondlevel[Level2_VarIndex["FLz"][0]:Level2_VarIndex["FLz"][1]+1]

FRx_secondlevel = traj_secondlevel[Level2_VarIndex["FRx"][0]:Level2_VarIndex["FRx"][1]+1]
FRy_secondlevel = traj_secondlevel[Level2_VarIndex["FRy"][0]:Level2_VarIndex["FRy"][1]+1]
FRz_secondlevel = traj_secondlevel[Level2_VarIndex["FRz"][0]:Level2_VarIndex["FRz"][1]+1]

print(SwitchingTimeVec)

print(np.sum(x_secondlevel-x_traj))
print(np.sum(y_secondlevel-y_traj))
print(np.sum(z_secondlevel-z_traj))
print(np.sum(xdot_secondlevel-xdot_traj))
print(np.sum(ydot_secondlevel-ydot_traj))
print(np.sum(zdot_secondlevel-zdot_traj))

print(np.sum(FLx_secondlevel-FLx_traj))
print(np.sum(FLy_secondlevel-FLy_traj))
print(np.sum(FLz_secondlevel-FLz_traj))

print(np.sum(FRx_secondlevel-FRx_traj))
print(np.sum(FRy_secondlevel-FRy_traj))
print(np.sum(FRz_secondlevel-FRz_traj))

plt.plot(x_secondlevel,y_secondlevel,label='SecondLevel')
plt.plot(x_traj,y_traj,label = "ref_traj")
plt.title("x-y traj for " + str(NumLookAhead) + 'Lookahead ' + str(RoundNum) + "th Round")
plt.legend(loc="upper right")
plt.show()

plt.plot(z_secondlevel,label='SecondLevel')
plt.plot(z_traj,label = "ref_traj")
plt.title("z traj for " + str(NumLookAhead) + 'Lookahead ' + str(RoundNum) + "th Round")
plt.legend(loc="upper right")
plt.show()

plt.plot(xdot_secondlevel,label='SecondLevel')
plt.plot(xdot_traj,label = "ref_traj")
plt.title("xdot for " + str(NumLookAhead) + 'Lookahead ' + str(RoundNum) + "th Round")
plt.legend(loc="upper right")
plt.show()

plt.plot(ydot_secondlevel,label='SecondLevel')
plt.plot(ydot_traj,label = "ref_traj")
plt.title("ydot for " + str(NumLookAhead) + 'Lookahead ' + str(RoundNum) + "th Round")
plt.legend(loc="upper right")
plt.show()

plt.plot(zdot_secondlevel,label='SecondLevel')
plt.plot(zdot_traj,label = "ref_traj")
plt.title("zdot for " + str(NumLookAhead) + 'Lookahead ' + str(RoundNum) + "th Round")
plt.legend(loc="upper right")
plt.show()

plt.plot(FLx_secondlevel,label='SecondLevel')
plt.plot(FLx_traj,label = "ref_traj")
plt.title("FLx for " + str(NumLookAhead) + 'Lookahead ' + str(RoundNum) + "th Round")
plt.legend(loc="upper right")
plt.show()

plt.plot(FLy_secondlevel,label='SecondLevel')
plt.plot(FLy_traj,label = "ref_traj")
plt.title("FLy for " + str(NumLookAhead) + 'Lookahead ' + str(RoundNum) + "th Round")
plt.legend(loc="upper right")
plt.show()

plt.plot(FLz_secondlevel,label='SecondLevel')
plt.plot(FLz_traj,label = "ref_traj")
plt.title("FLz for " + str(NumLookAhead) + 'Lookahead ' + str(RoundNum) + "th Round")
plt.legend(loc="upper right")
plt.show()

plt.plot(FRx_secondlevel,label='SecondLevel')
plt.plot(FRx_traj,label = "ref_traj")
plt.title("FRx for " + str(NumLookAhead) + 'Lookahead ' + str(RoundNum) + "th Round")
plt.legend(loc="upper right")
plt.show()

plt.plot(FRy_secondlevel,label='SecondLevel')
plt.plot(FRy_traj,label = "ref_traj")
plt.title("FRy for " + str(NumLookAhead) + 'Lookahead ' + str(RoundNum) + "th Round")
plt.legend(loc="upper right")
plt.show()

plt.plot(FRz_secondlevel,label='SecondLevel')
plt.plot(FRz_traj,label = "ref_traj")
plt.title("FRz for " + str(NumLookAhead) + 'Lookahead ' + str(RoundNum) + "th Round")
plt.legend(loc="upper right")
plt.show()


plt.plot(FLx_secondlevel + FRx_secondlevel,label='SecondLevel')
plt.plot(FLx_traj + FRx_traj,label = "ref_traj")
plt.title("Fx for " + str(NumLookAhead) + 'Lookahead ' + str(RoundNum) + "th Round")
plt.legend(loc="upper right")
plt.show()

plt.plot(FLy_secondlevel + FRy_secondlevel,label='SecondLevel')
plt.plot(FLy_traj + FRy_traj,label = "ref_traj")
plt.title("Fy for " + str(NumLookAhead) + 'Lookahead ' + str(RoundNum) + "th Round")
plt.legend(loc="upper right")
plt.show()

plt.plot(FLz_secondlevel + FRz_secondlevel,label='SecondLevel')
plt.plot(FLz_traj + FRz_traj,label = "ref_traj")
plt.title("Fz for " + str(NumLookAhead) + 'Lookahead ' + str(RoundNum) + "th Round")
plt.legend(loc="upper right")
plt.show()



