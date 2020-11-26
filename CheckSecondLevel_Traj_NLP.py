import pickle
import numpy as np

import matplotlib.pyplot as plt #Matplotlib
from mpl_toolkits.mplot3d import Axes3D

from NLP_Ref_Traj_Constructor import *

#filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/antfarm_CoM_No_TerminalCost_AntfarmRef/5LookAhead_Trial0.p"

NumLookAhead = 2
RoundNum = 1

#filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/antfarm_CoM_Tracking_SecondLevel/"+str(NumLookAhead)+"LookAhead_Trial0.p"
filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/antfarm_firstLevel_left_start_NLP_previous/"+str(NumLookAhead)+"LookAhead_Trial0.p"
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
x_traj, y_traj, z_traj, xdot_traj, ydot_traj, zdot_traj, Lx_ref, Ly_ref, Lz_ref, Ldotx_ref, Ldoty_ref,Ldotz_ref, FL1x_traj, FL1y_traj, FL1z_traj, FL2x_traj, FL2y_traj, FL2z_traj, FL3x_traj, FL3y_traj, FL3z_traj, FL4x_traj, FL4y_traj, FL4z_traj, FR1x_traj, FR1y_traj, FR1z_traj, FR2x_traj, FR2y_traj, FR2z_traj, FR3x_traj, FR3y_traj, FR3z_traj, FR4x_traj, FR4y_traj, FR4z_traj, Px_seq, Py_seq, Pz_seq, SwitchingTimeVec,px_init_ref, py_init_ref, pz_init_ref = ref_trajectory_construction_for_NLPSecondLevel(StartStepNum = RoundNum, LookAheadSteps = NumLookAhead)
#x_traj, y_traj, z_traj, xdot_traj, ydot_traj, zdot_traj, FLx_traj, FLy_traj, FLz_traj, FRx_traj, FRy_traj, FRz_traj, Px_seq, Py_seq, Pz_seq, SwitchingTimeVec = NLP_ref_trajectory_from_SecondLevel(StartStepNum = RoundNum, LookAheadSteps = NumLookAhead)


#x-y plot
x_secondlevel = traj_secondlevel[Level2_VarIndex["x"][0]:Level2_VarIndex["x"][1]+1]
#print(x_secondlevel)
y_secondlevel = traj_secondlevel[Level2_VarIndex["y"][0]:Level2_VarIndex["y"][1]+1]

z_secondlevel = traj_secondlevel[Level2_VarIndex["z"][0]:Level2_VarIndex["z"][1]+1]

xdot_secondlevel = traj_secondlevel[Level2_VarIndex["xdot"][0]:Level2_VarIndex["xdot"][1]+1]
ydot_secondlevel = traj_secondlevel[Level2_VarIndex["ydot"][0]:Level2_VarIndex["ydot"][1]+1]
zdot_secondlevel = traj_secondlevel[Level2_VarIndex["zdot"][0]:Level2_VarIndex["zdot"][1]+1]

px_secondLevel = traj_secondlevel[Level2_VarIndex["px"][0]:Level2_VarIndex["px"][1]+1]
py_secondLevel = traj_secondlevel[Level2_VarIndex["py"][0]:Level2_VarIndex["py"][1]+1]
pz_secondLevel = traj_secondlevel[Level2_VarIndex["pz"][0]:Level2_VarIndex["pz"][1]+1]

px_init_secondLevel = traj_secondlevel[Level2_VarIndex["px_init"][0]:Level2_VarIndex["px_init"][1]+1]
py_init_secondLevel = traj_secondlevel[Level2_VarIndex["py_init"][0]:Level2_VarIndex["py_init"][1]+1]
pz_init_secondLevel = traj_secondlevel[Level2_VarIndex["pz_init"][0]:Level2_VarIndex["pz_init"][1]+1]

FL1x_secondlevel = traj_secondlevel[Level2_VarIndex["FL1x"][0]:Level2_VarIndex["FL1x"][1]+1]
FL1y_secondlevel = traj_secondlevel[Level2_VarIndex["FL1y"][0]:Level2_VarIndex["FL1y"][1]+1]
FL1z_secondlevel = traj_secondlevel[Level2_VarIndex["FL1z"][0]:Level2_VarIndex["FL1z"][1]+1]

FR1x_secondlevel = traj_secondlevel[Level2_VarIndex["FR1x"][0]:Level2_VarIndex["FR1x"][1]+1]
FR1y_secondlevel = traj_secondlevel[Level2_VarIndex["FR1y"][0]:Level2_VarIndex["FR1y"][1]+1]
FR1z_secondlevel = traj_secondlevel[Level2_VarIndex["FR1z"][0]:Level2_VarIndex["FR1z"][1]+1]

print("Switching Time Vec: ",SwitchingTimeVec)

print(np.sum(x_secondlevel-x_traj))
print(np.sum(y_secondlevel-y_traj))
print(np.sum(z_secondlevel-z_traj))
print(np.sum(xdot_secondlevel-xdot_traj))
print(np.sum(ydot_secondlevel-ydot_traj))
print(np.sum(zdot_secondlevel-zdot_traj))

print(np.sum(FL1x_secondlevel-FL1x_traj))
print(np.sum(FL1y_secondlevel-FL1y_traj))
print(np.sum(FL1z_secondlevel-FL1z_traj))

print(np.sum(FR1x_secondlevel-FR1x_traj))
print(np.sum(FR1y_secondlevel-FR1y_traj))
print(np.sum(FR1z_secondlevel-FR1z_traj))

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

plt.plot(px_secondLevel,label='SecondLevel')
plt.plot(Px_seq,label = "ref_traj")
plt.title("px for " + str(NumLookAhead) + 'Lookahead ' + str(RoundNum) + "th Round")
plt.legend(loc="upper right")
plt.show()

print("Px result: ", px_secondLevel)
print("Px ref: ", Px_seq)

plt.plot(FL1x_secondlevel,label='SecondLevel')
plt.plot(FL1x_traj,label = "ref_traj")
plt.title("FL1x for " + str(NumLookAhead) + 'Lookahead ' + str(RoundNum) + "th Round")
plt.legend(loc="upper right")
plt.show()

plt.plot(FL1y_secondlevel,label='SecondLevel')
plt.plot(FL1y_traj,label = "ref_traj")
plt.title("FL1y for " + str(NumLookAhead) + 'Lookahead ' + str(RoundNum) + "th Round")
plt.legend(loc="upper right")
plt.show()

plt.plot(FL1z_secondlevel,label='SecondLevel')
plt.plot(FL1z_traj,label = "ref_traj")
plt.title("FL1z for " + str(NumLookAhead) + 'Lookahead ' + str(RoundNum) + "th Round")
plt.legend(loc="upper right")
plt.show()

plt.plot(FR1x_secondlevel,label='SecondLevel')
plt.plot(FR1x_traj,label = "ref_traj")
plt.title("FR1x for " + str(NumLookAhead) + 'Lookahead ' + str(RoundNum) + "th Round")
plt.legend(loc="upper right")
plt.show()

plt.plot(FR1y_secondlevel,label='SecondLevel')
plt.plot(FR1y_traj,label = "ref_traj")
plt.title("FR1y for " + str(NumLookAhead) + 'Lookahead ' + str(RoundNum) + "th Round")
plt.legend(loc="upper right")
plt.show()

plt.plot(FR1z_secondlevel,label='SecondLevel')
plt.plot(FR1z_traj,label = "ref_traj")
plt.title("FR1z for " + str(NumLookAhead) + 'Lookahead ' + str(RoundNum) + "th Round")
plt.legend(loc="upper right")
plt.show()


