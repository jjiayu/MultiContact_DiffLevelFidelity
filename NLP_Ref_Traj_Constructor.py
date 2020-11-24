import numpy as np
import pickle

import matplotlib.pyplot as plt #Matplotlib
from mpl_toolkits.mplot3d import Axes3D

def NLP_ref_trajectory_construction(StartStepNum = None, LookAheadSteps = None):

    #filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotions/Flat_Fixed_PhaseDurationBoth/4LookAhead_Trial0.p"
    #filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotions/Flat_varPhaseBoth_NoTerminalCostWight/10LookAhead_Trial0.p"
    #filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotions/UnevenTerrain_Ref/10LookAhead_Trial0.p"
    #filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotions/antfarm_ref/10LookAhead_Trial0.p"
    #filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotions/10x_TerminalCost/flat_ref/10LookAhead_Trial0.p"
    #filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotions/10x_TerminalCost/Uneven_Flat_Ref/10LookAhead_Trial0.p"

    #filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotion/Knitro/antfarm_ref/10LookAhead_Trial0.p"
    filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotion/Knitro/Uneven_005_ref/antfarm_ref/" + '10LookAhead_Trial0.p'

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    #Get Result Data
    Level1_VarIndex = data["VarIdx_of_All_Levels"]["Level1_Var_Index"]
    Level2_VarIndex = data["VarIdx_of_All_Levels"]["Level2_Var_Index"]
    Trajectories = data["Trajectory_of_All_Rounds"]
    Px_result = list(np.concatenate(data["Px_fullres"],axis=None))
    Py_result = list(np.concatenate(data["Py_fullres"],axis=None))
    Pz_result = list(np.concatenate(data["Pz_fullres"],axis=None))

    #Define we start from which step (which round --- for the first level) --- Start from 0
    #StartStepNum = 2
    #Define how many lookahead steps we have 
    #LookAheadSteps = 5

    #Get Step Index for the reference trajectory
    stepIndex = range(StartStepNum+1,StartStepNum+LookAheadSteps)
    #for elem in stepIndex:
    #    print(elem)

    #Build Reference Trajectory
    x_traj = []
    y_traj = []
    z_traj = []

    xdot_traj = []
    ydot_traj = []
    zdot_traj = []

    FLx_traj = []
    FLy_traj = []
    FLz_traj = []

    FRx_traj = []
    FRy_traj = []
    FRz_traj = []

    Px_seq = []
    Py_seq = []
    Pz_seq = []

    SwitchingTimeVec = []

    for cnt in range(len(stepIndex)):

        idx = stepIndex[cnt]
        #print("step Idx", idx)

        x_temp = list(Trajectories[idx][Level1_VarIndex["x"][0]:Level1_VarIndex["x"][1]+1])
        y_temp = list(Trajectories[idx][Level1_VarIndex["y"][0]:Level1_VarIndex["y"][1]+1])
        z_temp = list(Trajectories[idx][Level1_VarIndex["z"][0]:Level1_VarIndex["z"][1]+1])

        xdot_temp = list(Trajectories[idx][Level1_VarIndex["xdot"][0]:Level1_VarIndex["xdot"][1]+1])
        ydot_temp = list(Trajectories[idx][Level1_VarIndex["ydot"][0]:Level1_VarIndex["ydot"][1]+1])
        zdot_temp = list(Trajectories[idx][Level1_VarIndex["zdot"][0]:Level1_VarIndex["zdot"][1]+1])

        FLx_temp = list(Trajectories[idx][Level1_VarIndex["FL1x"][0]:Level1_VarIndex["FL1x"][1]+1] + Trajectories[idx][Level1_VarIndex["FL2x"][0]:Level1_VarIndex["FL2x"][1]+1] + Trajectories[idx][Level1_VarIndex["FL3x"][0]:Level1_VarIndex["FL3x"][1]+1] + Trajectories[idx][Level1_VarIndex["FL4x"][0]:Level1_VarIndex["FL4x"][1]+1])
        FLy_temp = list(Trajectories[idx][Level1_VarIndex["FL1y"][0]:Level1_VarIndex["FL1y"][1]+1] + Trajectories[idx][Level1_VarIndex["FL2y"][0]:Level1_VarIndex["FL2y"][1]+1] + Trajectories[idx][Level1_VarIndex["FL3y"][0]:Level1_VarIndex["FL3y"][1]+1] + Trajectories[idx][Level1_VarIndex["FL4y"][0]:Level1_VarIndex["FL4y"][1]+1])
        FLz_temp = list(Trajectories[idx][Level1_VarIndex["FL1z"][0]:Level1_VarIndex["FL1z"][1]+1] + Trajectories[idx][Level1_VarIndex["FL2z"][0]:Level1_VarIndex["FL2z"][1]+1] + Trajectories[idx][Level1_VarIndex["FL3z"][0]:Level1_VarIndex["FL3z"][1]+1] + Trajectories[idx][Level1_VarIndex["FL4z"][0]:Level1_VarIndex["FL4z"][1]+1])

        FRx_temp = list(Trajectories[idx][Level1_VarIndex["FR1x"][0]:Level1_VarIndex["FR1x"][1]+1] + Trajectories[idx][Level1_VarIndex["FR2x"][0]:Level1_VarIndex["FR2x"][1]+1] + Trajectories[idx][Level1_VarIndex["FR3x"][0]:Level1_VarIndex["FR3x"][1]+1] + Trajectories[idx][Level1_VarIndex["FR4x"][0]:Level1_VarIndex["FR4x"][1]+1])
        FRy_temp = list(Trajectories[idx][Level1_VarIndex["FR1y"][0]:Level1_VarIndex["FR1y"][1]+1] + Trajectories[idx][Level1_VarIndex["FR2y"][0]:Level1_VarIndex["FR2y"][1]+1] + Trajectories[idx][Level1_VarIndex["FR3y"][0]:Level1_VarIndex["FR3y"][1]+1] + Trajectories[idx][Level1_VarIndex["FR4y"][0]:Level1_VarIndex["FR4y"][1]+1])
        FRz_temp = list(Trajectories[idx][Level1_VarIndex["FR1z"][0]:Level1_VarIndex["FR1z"][1]+1] + Trajectories[idx][Level1_VarIndex["FR2z"][0]:Level1_VarIndex["FR2z"][1]+1] + Trajectories[idx][Level1_VarIndex["FR3z"][0]:Level1_VarIndex["FR3z"][1]+1] + Trajectories[idx][Level1_VarIndex["FR4z"][0]:Level1_VarIndex["FR4z"][1]+1])

        Px_temp = [Px_result[idx]]
        Py_temp = [Py_result[idx]]
        Pz_temp = [Pz_result[idx]]

        PhaseDuration_res = list(Trajectories[idx][Level1_VarIndex["Ts"][0]:Level1_VarIndex["Ts"][1]+1])
        PhaseDuration_temp = list(np.array([PhaseDuration_res[0],PhaseDuration_res[1]-PhaseDuration_res[0],PhaseDuration_res[2]-PhaseDuration_res[1]]))

        if cnt == len(stepIndex) - 1:
            x_traj = x_traj + x_temp
            y_traj = y_traj + y_temp
            z_traj = z_traj + z_temp

            xdot_traj = xdot_traj + xdot_temp
            ydot_traj = ydot_traj + ydot_temp
            zdot_traj = zdot_traj + zdot_temp

            FLx_traj = FLx_traj + FLx_temp
            FLy_traj = FLy_traj + FLy_temp
            FLz_traj = FLz_traj + FLz_temp

            FRx_traj = FRx_traj + FRx_temp
            FRy_traj = FRy_traj + FRy_temp
            FRz_traj = FRz_traj + FRz_temp

        else: 
            x_traj = x_traj + x_temp[0:-1]
            y_traj = y_traj + y_temp[0:-1]
            z_traj = z_traj + z_temp[0:-1]

            xdot_traj = xdot_traj + xdot_temp[0:-1]
            ydot_traj = ydot_traj + ydot_temp[0:-1]
            zdot_traj = zdot_traj + zdot_temp[0:-1]

            FLx_traj = FLx_traj + FLx_temp[0:-1]
            FLy_traj = FLy_traj + FLy_temp[0:-1]
            FLz_traj = FLz_traj + FLz_temp[0:-1]

            FRx_traj = FRx_traj + FRx_temp[0:-1]
            FRy_traj = FRy_traj + FRy_temp[0:-1]
            FRz_traj = FRz_traj + FRz_temp[0:-1]

        # if cnt == 0: #start the reference trajectory
        #     x_traj = x_traj + x_temp[0:-1]
        #     y_traj = y_traj + y_temp[0:-1]
        #     z_traj = z_traj + z_temp[0:-1]

        #     xdot_traj = xdot_traj + xdot_temp[0:-1]
        #     ydot_traj = ydot_traj + ydot_temp[0:-1]
        #     zdot_traj = zdot_traj + zdot_temp[0:-1]

        #     FLx_traj = FLx_traj + FLx_temp[0:-1]
        #     FLy_traj = FLy_traj + FLy_temp[0:-1]
        #     FLz_traj = FLz_traj + FLz_temp[0:-1]

        #     FRx_traj = FRx_traj + FRx_temp[0:-1]
        #     FRy_traj = FRy_traj + FRy_temp[0:-1]
        #     FRz_traj = FRz_traj + FRz_temp[0:-1]

        # else:
        #     x_traj = x_traj + x_temp[1:]
        #     y_traj = y_traj + y_temp[1:]
        #     z_traj = z_traj + z_temp[1:]

        #     xdot_traj = xdot_traj + xdot_temp[1:]
        #     ydot_traj = ydot_traj + ydot_temp[1:]
        #     zdot_traj = zdot_traj + zdot_temp[1:]

        #     FLx_traj = FLx_traj + FLx_temp[0:-1]
        #     FLy_traj = FLy_traj + FLy_temp[0:-1]
        #     FLz_traj = FLz_traj + FLz_temp[0:-1]

        #     FRx_traj = FRx_traj + FRx_temp[0:-1]
        #     FRy_traj = FRy_traj + FRy_temp[0:-1]
        #     FRz_traj = FRz_traj + FRz_temp[0:-1]

        SwitchingTimeVec = SwitchingTimeVec + PhaseDuration_temp
        Px_seq = Px_seq + Px_temp
        Py_seq = Py_seq + Py_temp
        Pz_seq = Pz_seq + Pz_temp

    x_traj = np.array(x_traj)
    y_traj = np.array(y_traj)
    z_traj = np.array(z_traj)
    xdot_traj = np.array(xdot_traj)
    ydot_traj = np.array(ydot_traj)
    zdot_traj = np.array(zdot_traj)
    FLx_traj = np.array(FLx_traj)
    FLy_traj = np.array(FLy_traj)
    FLz_traj = np.array(FLz_traj)
    FRx_traj = np.array(FRx_traj)
    FRy_traj = np.array(FRy_traj)
    FRz_traj = np.array(FRz_traj)
    Px_seq = np.array(Px_seq)
    Py_seq = np.array(Py_seq)
    Pz_seq = np.array(Pz_seq)
    SwitchingTimeVec = np.array(SwitchingTimeVec)

    #For p_init in the second level
    #Get the second level first
    traj_secondlevel = Trajectories[StartStepNum][Level1_VarIndex["Ts"][1]+1:]

    px_init_seq = traj_secondlevel[Level2_VarIndex["px_init"][0]:Level2_VarIndex["px_init"][1]+1]
    py_init_seq = traj_secondlevel[Level2_VarIndex["py_init"][0]:Level2_VarIndex["py_init"][1]+1]
    pz_init_seq = traj_secondlevel[Level2_VarIndex["pz_init"][0]:Level2_VarIndex["pz_init"][1]+1]

    #Convert to Numpy Array

    #plt.plot(zdot_traj)

    # print(len(x_traj))
    # print(len(y_traj))
    # print(len(z_traj))


    # print(len(xdot_traj))
    # print(len(ydot_traj))
    # print(len(zdot_traj))

    # print(len(FLx_traj))
    # print(len(FLy_traj))
    # print(len(FLz_traj))

    # print(len(FRx_traj))
    # print(len(FRy_traj))
    # print(len(FRz_traj))

    # print(len(Px_seq))
    # print(len(Py_seq))
    # print(len(Pz_seq))

    # print(len(SwitchingTimeVec))

    #plt.show()

    return  x_traj, y_traj, z_traj, xdot_traj, ydot_traj, zdot_traj, FLx_traj, FLy_traj, FLz_traj, FRx_traj, FRy_traj, FRz_traj, Px_seq, Py_seq, Pz_seq, SwitchingTimeVec, px_init_seq, py_init_seq, pz_init_seq


def NLP_ref_trajectory_from_SecondLevel(StartStepNum = None, LookAheadSteps = None):

    #filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotions/Flat_Fixed_PhaseDurationBoth/4LookAhead_Trial0.p"
    #filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotions/Flat_varPhaseBoth_NoTerminalCostWight/10LookAhead_Trial0.p"
    #filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotions/UnevenTerrain_Ref/10LookAhead_Trial0.p"
    #filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotions/antfarm_ref/10LookAhead_Trial0.p"
    #filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotions/10x_TerminalCost/flat_ref/10LookAhead_Trial0.p"
    #filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotions/10x_TerminalCost/Uneven_Flat_Ref/10LookAhead_Trial0.p"

    filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotion/Knitro/antfarm_ref/" + str(LookAheadSteps)+"LookAhead_Trial0.p"
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    #Get Result Data
    Level1_VarIndex = data["VarIdx_of_All_Levels"]["Level1_Var_Index"]
    Level2_VarIndex = data["VarIdx_of_All_Levels"]["Level2_Var_Index"]

    Trajectories = data["Trajectory_of_All_Rounds"]

    traj = Trajectories[StartStepNum]

    traj_secondlevel = traj[Level1_VarIndex["Ts"][1]+1:]

    x_traj = traj_secondlevel[Level2_VarIndex["x"][0]:Level2_VarIndex["x"][1]+1]
    y_traj = traj_secondlevel[Level2_VarIndex["y"][0]:Level2_VarIndex["y"][1]+1]
    z_traj = traj_secondlevel[Level2_VarIndex["z"][0]:Level2_VarIndex["z"][1]+1]

    xdot_traj = traj_secondlevel[Level2_VarIndex["xdot"][0]:Level2_VarIndex["xdot"][1]+1]
    ydot_traj = traj_secondlevel[Level2_VarIndex["ydot"][0]:Level2_VarIndex["ydot"][1]+1]
    zdot_traj = traj_secondlevel[Level2_VarIndex["zdot"][0]:Level2_VarIndex["zdot"][1]+1]


    FLx_traj = traj_secondlevel[Level2_VarIndex["FL1x"][0]:Level2_VarIndex["FL1x"][1]+1] + traj_secondlevel[Level2_VarIndex["FL2x"][0]:Level2_VarIndex["FL2x"][1]+1] + traj_secondlevel[Level2_VarIndex["FL3x"][0]:Level2_VarIndex["FL3x"][1]+1] + traj_secondlevel[Level2_VarIndex["FL4x"][0]:Level2_VarIndex["FL4x"][1]+1]
    FLy_traj = traj_secondlevel[Level2_VarIndex["FL1y"][0]:Level2_VarIndex["FL1y"][1]+1] + traj_secondlevel[Level2_VarIndex["FL2y"][0]:Level2_VarIndex["FL2y"][1]+1] + traj_secondlevel[Level2_VarIndex["FL3y"][0]:Level2_VarIndex["FL3y"][1]+1] + traj_secondlevel[Level2_VarIndex["FL4y"][0]:Level2_VarIndex["FL4y"][1]+1]
    FLz_traj = traj_secondlevel[Level2_VarIndex["FL1z"][0]:Level2_VarIndex["FL1z"][1]+1] + traj_secondlevel[Level2_VarIndex["FL2z"][0]:Level2_VarIndex["FL2z"][1]+1] + traj_secondlevel[Level2_VarIndex["FL3z"][0]:Level2_VarIndex["FL3z"][1]+1] + traj_secondlevel[Level2_VarIndex["FL4z"][0]:Level2_VarIndex["FL4z"][1]+1]

    FRx_traj = traj_secondlevel[Level2_VarIndex["FR1x"][0]:Level2_VarIndex["FR1x"][1]+1] + traj_secondlevel[Level2_VarIndex["FR2x"][0]:Level2_VarIndex["FR2x"][1]+1] + traj_secondlevel[Level2_VarIndex["FR3x"][0]:Level2_VarIndex["FR3x"][1]+1] + traj_secondlevel[Level2_VarIndex["FR4x"][0]:Level2_VarIndex["FR4x"][1]+1]
    FRy_traj = traj_secondlevel[Level2_VarIndex["FR1y"][0]:Level2_VarIndex["FR1y"][1]+1] + traj_secondlevel[Level2_VarIndex["FR2y"][0]:Level2_VarIndex["FR2y"][1]+1] + traj_secondlevel[Level2_VarIndex["FR3y"][0]:Level2_VarIndex["FR3y"][1]+1] + traj_secondlevel[Level2_VarIndex["FR4y"][0]:Level2_VarIndex["FR4y"][1]+1]
    FRz_traj = traj_secondlevel[Level2_VarIndex["FR1z"][0]:Level2_VarIndex["FR1z"][1]+1] + traj_secondlevel[Level2_VarIndex["FR2z"][0]:Level2_VarIndex["FR2z"][1]+1] + traj_secondlevel[Level2_VarIndex["FR3z"][0]:Level2_VarIndex["FR3z"][1]+1] + traj_secondlevel[Level2_VarIndex["FR4z"][0]:Level2_VarIndex["FR4z"][1]+1]

    Px_seq = traj_secondlevel[Level2_VarIndex["px"][0]:Level2_VarIndex["px"][1]+1]
    Py_seq = traj_secondlevel[Level2_VarIndex["py"][0]:Level2_VarIndex["py"][1]+1]
    Pz_seq = traj_secondlevel[Level2_VarIndex["pz"][0]:Level2_VarIndex["pz"][1]+1]

    SwitchingTimeVec = traj_secondlevel[Level2_VarIndex["Ts"][0]:Level2_VarIndex["Ts"][1]+1]

    return  x_traj, y_traj, z_traj, xdot_traj, ydot_traj, zdot_traj, FLx_traj, FLy_traj, FLz_traj, FRx_traj, FRy_traj, FRz_traj, Px_seq, Py_seq, Pz_seq, SwitchingTimeVec