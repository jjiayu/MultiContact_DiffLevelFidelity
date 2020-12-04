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
    #filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotion/Knitro/flat_patches/" + '10LookAhead_Trial0.p'
    filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotion/Knitro/DeformedTraj/" + '10LookAhead_Trial0.p'
    #filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotion/Knitro/flat_ref/" + '10LookAhead_Trial0.p'
    
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


def ref_trajectory_construction_for_NLPSecondLevel(StartStepNum = None, LookAheadSteps = None):

    #filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotions/Flat_Fixed_PhaseDurationBoth/4LookAhead_Trial0.p"
    #filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotions/Flat_varPhaseBoth_NoTerminalCostWight/10LookAhead_Trial0.p"
    #filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotions/UnevenTerrain_Ref/10LookAhead_Trial0.p"
    #filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotions/antfarm_ref/10LookAhead_Trial0.p"
    #filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotions/10x_TerminalCost/flat_ref/10LookAhead_Trial0.p"
    #filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotions/10x_TerminalCost/Uneven_Flat_Ref/10LookAhead_Trial0.p"

    #filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotion/Knitro/antfarm_ref/10LookAhead_Trial0.p"
    filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotion/Knitro/Uneven_005_ref/antfarm_ref/" + '10LookAhead_Trial0.p'
    #filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotion/Knitro/flat_ref/" + '10LookAhead_Trial0.p'
    
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
    if StartStepNum > 10:
        StartStepNum = 10

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

    Lx_traj  = []
    Ly_traj = []
    Lz_traj = []

    Ldotx_traj = []
    Ldoty_traj = []
    Ldotz_traj = []

    FL1x_traj = []
    FL1y_traj = []
    FL1z_traj = []

    FL2x_traj = []
    FL2y_traj = []
    FL2z_traj = []

    FL3x_traj = []
    FL3y_traj = []
    FL3z_traj = []

    FL4x_traj = []
    FL4y_traj = []
    FL4z_traj = []

    FR1x_traj = []
    FR1y_traj = []
    FR1z_traj = []

    FR2x_traj = []
    FR2y_traj = []
    FR2z_traj = []

    FR3x_traj = []
    FR3y_traj = []
    FR3z_traj = []

    FR4x_traj = []
    FR4y_traj = []
    FR4z_traj = []

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

        Lx_temp = list(Trajectories[idx][Level1_VarIndex["Lx"][0]:Level1_VarIndex["Lx"][1]+1])
        Ly_temp = list(Trajectories[idx][Level1_VarIndex["Ly"][0]:Level1_VarIndex["Ly"][1]+1])
        Lz_temp = list(Trajectories[idx][Level1_VarIndex["Lz"][0]:Level1_VarIndex["Lz"][1]+1])

        Ldotx_temp = list(Trajectories[idx][Level1_VarIndex["Ldotx"][0]:Level1_VarIndex["Ldotx"][1]+1])
        Ldoty_temp = list(Trajectories[idx][Level1_VarIndex["Ldoty"][0]:Level1_VarIndex["Ldoty"][1]+1])
        Ldotz_temp = list(Trajectories[idx][Level1_VarIndex["Ldotz"][0]:Level1_VarIndex["Ldotz"][1]+1])

        FL1x_temp = list(Trajectories[idx][Level1_VarIndex["FL1x"][0]:Level1_VarIndex["FL1x"][1]+1])
        FL1y_temp = list(Trajectories[idx][Level1_VarIndex["FL1y"][0]:Level1_VarIndex["FL1y"][1]+1])
        FL1z_temp = list(Trajectories[idx][Level1_VarIndex["FL1z"][0]:Level1_VarIndex["FL1z"][1]+1])

        FL2x_temp = list(Trajectories[idx][Level1_VarIndex["FL2x"][0]:Level1_VarIndex["FL2x"][1]+1])
        FL2y_temp = list(Trajectories[idx][Level1_VarIndex["FL2y"][0]:Level1_VarIndex["FL2y"][1]+1])
        FL2z_temp = list(Trajectories[idx][Level1_VarIndex["FL2z"][0]:Level1_VarIndex["FL2z"][1]+1])

        FL3x_temp = list(Trajectories[idx][Level1_VarIndex["FL3x"][0]:Level1_VarIndex["FL3x"][1]+1])
        FL3y_temp = list(Trajectories[idx][Level1_VarIndex["FL3y"][0]:Level1_VarIndex["FL3y"][1]+1])
        FL3z_temp = list(Trajectories[idx][Level1_VarIndex["FL3z"][0]:Level1_VarIndex["FL3z"][1]+1])

        FL4x_temp = list(Trajectories[idx][Level1_VarIndex["FL4x"][0]:Level1_VarIndex["FL4x"][1]+1])
        FL4y_temp = list(Trajectories[idx][Level1_VarIndex["FL4y"][0]:Level1_VarIndex["FL4y"][1]+1])
        FL4z_temp = list(Trajectories[idx][Level1_VarIndex["FL4z"][0]:Level1_VarIndex["FL4z"][1]+1])

        FR1x_temp = list(Trajectories[idx][Level1_VarIndex["FR1x"][0]:Level1_VarIndex["FR1x"][1]+1])
        FR1y_temp = list(Trajectories[idx][Level1_VarIndex["FR1y"][0]:Level1_VarIndex["FR1y"][1]+1])
        FR1z_temp = list(Trajectories[idx][Level1_VarIndex["FR1z"][0]:Level1_VarIndex["FR1z"][1]+1])

        FR2x_temp = list(Trajectories[idx][Level1_VarIndex["FR2x"][0]:Level1_VarIndex["FR2x"][1]+1])
        FR2y_temp = list(Trajectories[idx][Level1_VarIndex["FR2y"][0]:Level1_VarIndex["FR2y"][1]+1])
        FR2z_temp = list(Trajectories[idx][Level1_VarIndex["FR2z"][0]:Level1_VarIndex["FR2z"][1]+1])

        FR3x_temp = list(Trajectories[idx][Level1_VarIndex["FR3x"][0]:Level1_VarIndex["FR3x"][1]+1])
        FR3y_temp = list(Trajectories[idx][Level1_VarIndex["FR3y"][0]:Level1_VarIndex["FR3y"][1]+1])
        FR3z_temp = list(Trajectories[idx][Level1_VarIndex["FR3z"][0]:Level1_VarIndex["FR3z"][1]+1])

        FR4x_temp = list(Trajectories[idx][Level1_VarIndex["FR4x"][0]:Level1_VarIndex["FR4x"][1]+1])
        FR4y_temp = list(Trajectories[idx][Level1_VarIndex["FR4y"][0]:Level1_VarIndex["FR4y"][1]+1])
        FR4z_temp = list(Trajectories[idx][Level1_VarIndex["FR4z"][0]:Level1_VarIndex["FR4z"][1]+1])

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

            Lx_traj = Lx_traj + Lx_temp
            Ly_traj = Ly_traj + Ly_temp
            Lz_traj = Lz_traj + Lz_temp

            Ldotx_traj = Ldotx_traj + Ldotx_temp
            Ldoty_traj = Ldoty_traj + Ldoty_temp
            Ldotz_traj = Ldotz_traj + Ldotz_temp

            FL1x_traj = FL1x_traj + FL1x_temp
            FL1y_traj = FL1y_traj + FL1y_temp
            FL1z_traj = FL1z_traj + FL1z_temp

            FL2x_traj = FL2x_traj + FL2x_temp
            FL2y_traj = FL2y_traj + FL2y_temp
            FL2z_traj = FL2z_traj + FL2z_temp

            FL3x_traj = FL3x_traj + FL3x_temp
            FL3y_traj = FL3y_traj + FL3y_temp
            FL3z_traj = FL3z_traj + FL3z_temp

            FL4x_traj = FL4x_traj + FL4x_temp
            FL4y_traj = FL4y_traj + FL4y_temp
            FL4z_traj = FL4z_traj + FL4z_temp

            FR1x_traj = FR1x_traj + FR1x_temp
            FR1y_traj = FR1y_traj + FR1y_temp
            FR1z_traj = FR1z_traj + FR1z_temp

            FR2x_traj = FR2x_traj + FR2x_temp
            FR2y_traj = FR2y_traj + FR2y_temp
            FR2z_traj = FR2z_traj + FR2z_temp

            FR3x_traj = FR3x_traj + FR3x_temp
            FR3y_traj = FR3y_traj + FR3y_temp
            FR3z_traj = FR3z_traj + FR3z_temp

            FR4x_traj = FR4x_traj + FR4x_temp
            FR4y_traj = FR4y_traj + FR4y_temp
            FR4z_traj = FR4z_traj + FR4z_temp

        else: 
            x_traj = x_traj + x_temp[0:-1]
            y_traj = y_traj + y_temp[0:-1]
            z_traj = z_traj + z_temp[0:-1]

            xdot_traj = xdot_traj + xdot_temp[0:-1]
            ydot_traj = ydot_traj + ydot_temp[0:-1]
            zdot_traj = zdot_traj + zdot_temp[0:-1]

            Lx_traj = Lx_traj + Lx_temp[0:-1]
            Ly_traj = Ly_traj + Ly_temp[0:-1]
            Lz_traj = Lz_traj + Lz_temp[0:-1]

            Ldotx_traj = Ldotx_traj + Ldotx_temp[0:-1]
            Ldoty_traj = Ldoty_traj + Ldoty_temp[0:-1]
            Ldotz_traj = Ldotz_traj + Ldotz_temp[0:-1]

            FL1x_traj = FL1x_traj + FL1x_temp[0:-1]
            FL1y_traj = FL1y_traj + FL1y_temp[0:-1]
            FL1z_traj = FL1z_traj + FL1z_temp[0:-1]

            FL2x_traj = FL2x_traj + FL2x_temp[0:-1]
            FL2y_traj = FL2y_traj + FL2y_temp[0:-1]
            FL2z_traj = FL2z_traj + FL2z_temp[0:-1]

            FL3x_traj = FL3x_traj + FL3x_temp[0:-1]
            FL3y_traj = FL3y_traj + FL3y_temp[0:-1]
            FL3z_traj = FL3z_traj + FL3z_temp[0:-1]

            FL4x_traj = FL4x_traj + FL4x_temp[0:-1]
            FL4y_traj = FL4y_traj + FL4y_temp[0:-1]
            FL4z_traj = FL4z_traj + FL4z_temp[0:-1]

            FR1x_traj = FR1x_traj + FR1x_temp[0:-1]
            FR1y_traj = FR1y_traj + FR1y_temp[0:-1]
            FR1z_traj = FR1z_traj + FR1z_temp[0:-1]

            FR2x_traj = FR2x_traj + FR2x_temp[0:-1]
            FR2y_traj = FR2y_traj + FR2y_temp[0:-1]
            FR2z_traj = FR2z_traj + FR2z_temp[0:-1]

            FR3x_traj = FR3x_traj + FR3x_temp[0:-1]
            FR3y_traj = FR3y_traj + FR3y_temp[0:-1]
            FR3z_traj = FR3z_traj + FR3z_temp[0:-1]

            FR4x_traj = FR4x_traj + FR4x_temp[0:-1]
            FR4y_traj = FR4y_traj + FR4y_temp[0:-1]
            FR4z_traj = FR4z_traj + FR4z_temp[0:-1]

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
    Lx_traj = np.array(Lx_traj)
    Ly_traj = np.array(Ly_traj)
    Lz_traj = np.array(Lz_traj)
    Ldotx_traj = np.array(Ldotx_traj)
    Ldoty_traj = np.array(Ldoty_traj)
    Ldotz_traj = np.array(Ldotz_traj)
    FL1x_traj = np.array(FL1x_traj)
    FL1y_traj = np.array(FL1y_traj)
    FL1z_traj = np.array(FL1z_traj)
    FL2x_traj = np.array(FL2x_traj)
    FL2y_traj = np.array(FL2y_traj)
    FL2z_traj = np.array(FL2z_traj)
    FL3x_traj = np.array(FL3x_traj)
    FL3y_traj = np.array(FL3y_traj)
    FL3z_traj = np.array(FL3z_traj)
    FL4x_traj = np.array(FL4x_traj)
    FL4y_traj = np.array(FL4y_traj)
    FL4z_traj = np.array(FL4z_traj)
    FR1x_traj = np.array(FR1x_traj)
    FR1y_traj = np.array(FR1y_traj)
    FR1z_traj = np.array(FR1z_traj)
    FR2x_traj = np.array(FR2x_traj)
    FR2y_traj = np.array(FR2y_traj)
    FR2z_traj = np.array(FR2z_traj)
    FR3x_traj = np.array(FR3x_traj)
    FR3y_traj = np.array(FR3y_traj)
    FR3z_traj = np.array(FR3z_traj)
    FR4x_traj = np.array(FR4x_traj)
    FR4y_traj = np.array(FR4y_traj)
    FR4z_traj = np.array(FR4z_traj)
    Px_seq = np.array(Px_seq)
    Py_seq = np.array(Py_seq)
    Pz_seq = np.array(Pz_seq)

    print("Px_seq: ",Px_seq)
    print("Py_seq: ",Py_seq)
    print("Pz_seq: ",Pz_seq)

    SwitchingTimeVec = np.array(SwitchingTimeVec)
    print("SwitchingTimeVec: ",SwitchingTimeVec)

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

    return  x_traj, y_traj, z_traj, xdot_traj, ydot_traj, zdot_traj, Lx_traj, Ly_traj, Lz_traj, Ldotx_traj, Ldoty_traj, Ldotz_traj, FL1x_traj, FL1y_traj, FL1z_traj, FL2x_traj, FL2y_traj, FL2z_traj, FL3x_traj, FL3y_traj, FL3z_traj, FL4x_traj, FL4y_traj, FL4z_traj, FR1x_traj, FR1y_traj, FR1z_traj, FR2x_traj, FR2y_traj, FR2z_traj, FR3x_traj, FR3y_traj, FR3z_traj, FR4x_traj, FR4y_traj, FR4z_traj, Px_seq, Py_seq, Pz_seq, SwitchingTimeVec, px_init_seq, py_init_seq, pz_init_seq


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

# x_nlplevel2_ref, y_nlplevel2_ref, z_nlplevel2_ref, xdot_nlplevel2_ref, ydot_nlplevel2_ref, zdot_nlplevel2_ref, Lx_nlplevel2_ref, Ly_nlplevel2_ref, Lz_nlplevel2_ref, Ldotx_nlplevel2_ref, Ldoty_nlplevel2_ref, Ldotz_nlplevel2_ref, FL1x_nlplevel2_ref, FL1y_nlplevel2_ref, FL1z_nlplevel2_ref, FL2x_nlplevel2_ref, FL2y_nlplevel2_ref, FL2z_nlplevel2_ref, FL3x_nlplevel2_ref, FL3y_nlplevel2_ref, FL3z_nlplevel2_ref, FL4x_nlplevel2_ref, FL4y_nlplevel2_ref, FL4z_nlplevel2_ref, FR1x_nlplevel2_ref, FR1y_nlplevel2_ref, FR1z_nlplevel2_ref, FR2x_nlplevel2_ref, FR2y_nlplevel2_ref, FR2z_nlplevel2_ref, FR3x_nlplevel2_ref, FR3y_nlplevel2_ref, FR3z_nlplevel2_ref, FR4x_nlplevel2_ref, FR4y_nlplevel2_ref, FR4z_nlplevel2_ref, Px_seq_nlplevel2_ref, Py_seq_nlplevel2_ref, Pz_seq_nlplevel2_ref, SwitchingTimeVec_nlplevel2_ref, px_init_seq_nlplevel2_ref, py_init_seq_nlplevel2_ref, pz_init_seq_nlplevel2_ref = ref_trajectory_construction_for_NLPSecondLevel(StartStepNum = 0, LookAheadSteps = 10)

# plt.plot(FL3z_nlplevel2_ref)
# plt.show()

