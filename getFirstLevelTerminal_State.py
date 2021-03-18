import numpy as np
import pickle

import matplotlib.pyplot as plt #Matplotlib
from mpl_toolkits.mplot3d import Axes3D

def GetFirstLevelTerminalState(RoundNum = 0, NumofLookAhead = 6):
    filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotion/darpa_like_left_first_NLP_previous/" + str(NumofLookAhead) + "LookAhead_Trial0.p"
    #filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/antfarm_firstLevel_left_start_NLP_previous/" + str(NumofLookAhead) + "LookAhead_Trial0.p"

    #filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotion/Knitro/flat_ref/" + str(NumofLookAhead) + "LookAhead_Trial0.p"

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    #Get Result Data
    Level1_VarIndex = data["VarIdx_of_All_Levels"]["Level1_Var_Index"]
    Trajectories = data["Trajectory_of_All_Rounds"]
    Px_result = list(np.concatenate(data["Px_fullres"],axis=None))
    Py_result = list(np.concatenate(data["Py_fullres"],axis=None))
    Pz_result = list(np.concatenate(data["Pz_fullres"],axis=None))

    x_traj = Trajectories[RoundNum][Level1_VarIndex["x"][0]:Level1_VarIndex["x"][1]+1]
    y_traj = Trajectories[RoundNum][Level1_VarIndex["y"][0]:Level1_VarIndex["y"][1]+1]
    z_traj = Trajectories[RoundNum][Level1_VarIndex["z"][0]:Level1_VarIndex["z"][1]+1]

    xdot_traj = Trajectories[RoundNum][Level1_VarIndex["xdot"][0]:Level1_VarIndex["xdot"][1]+1]
    ydot_traj = Trajectories[RoundNum][Level1_VarIndex["ydot"][0]:Level1_VarIndex["ydot"][1]+1]
    zdot_traj = Trajectories[RoundNum][Level1_VarIndex["zdot"][0]:Level1_VarIndex["zdot"][1]+1]

    Lx_traj = Trajectories[RoundNum][Level1_VarIndex["Lx"][0]:Level1_VarIndex["Lx"][1]+1]
    Ly_traj = Trajectories[RoundNum][Level1_VarIndex["Ly"][0]:Level1_VarIndex["Ly"][1]+1]
    Lz_traj = Trajectories[RoundNum][Level1_VarIndex["Lz"][0]:Level1_VarIndex["Lz"][1]+1]

    px_Level1 = Trajectories[RoundNum][Level1_VarIndex["px"][0]:Level1_VarIndex["px"][1]+1]
    py_Level1 = Trajectories[RoundNum][Level1_VarIndex["py"][0]:Level1_VarIndex["py"][1]+1]
    pz_Level1 = Trajectories[RoundNum][Level1_VarIndex["pz"][0]:Level1_VarIndex["pz"][1]+1]

    x_end_Level1 = x_traj[-1]
    y_end_Level1 = y_traj[-1]
    z_end_Level1 = z_traj[-1]

    xdot_end_Level1 = xdot_traj[-1]
    ydot_end_Level1 = ydot_traj[-1]
    zdot_end_Level1 = zdot_traj[-1]

    Lx_end_Level1 = Lx_traj[-1]
    Ly_end_Level1 = Ly_traj[-1]
    Lz_end_Level1 = Lz_traj[-1]

    return x_end_Level1,y_end_Level1,z_end_Level1,xdot_end_Level1,ydot_end_Level1,zdot_end_Level1,px_Level1,py_Level1,pz_Level1,Lx_end_Level1,Ly_end_Level1,Lz_end_Level1

def GetFirstLevelTrajectory(RoundNum = 0, NumofLookAhead = 6):
    filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotion/darpa_like_left_first_NLP_previous/" + str(NumofLookAhead) + "LookAhead_Trial0.p"
    #filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/antfarm_firstLevel_left_start_NLP_previous/" + str(NumofLookAhead) + "LookAhead_Trial0.p"

    #filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotion/Knitro/flat_ref/" + str(NumofLookAhead) + "LookAhead_Trial0.p"

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    Level1_VarIndex = data["VarIdx_of_All_Levels"]["Level1_Var_Index"]
    Trajectories = data["Trajectory_of_All_Rounds"]

    FirstLevelTraj = Trajectories[RoundNum][0:Level1_VarIndex["Ts"][1]+1]

    # plt.plot(FirstLevelTraj)
    # plt.plot(Trajectories[RoundNum])
    # plt.show()

    print(FirstLevelTraj[-1],FirstLevelTraj[-2],FirstLevelTraj[-3])

    return FirstLevelTraj