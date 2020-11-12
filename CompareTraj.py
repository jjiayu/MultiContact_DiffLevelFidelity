import pickle
import numpy as np

import matplotlib.pyplot as plt #Matplotlib
from mpl_toolkits.mplot3d import Axes3D

#filename = '/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/TSID_Almost_Symmetric_TestMotions/flat_NLP_previous/5LookAhead_Trial0.p'
#filename = '/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/flat_CoM_previous_regu_GoodSet_with567/5LookAhead_Trial0.p'
#filename = '/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/flat_NLP_previous/5LookAhead_Trial0.p'

#NLP result retrieval
#filename = '/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/GoodFlat_1000y_Regu/flat_NLP_previous/5LookAhead_Trial0.p'

#folderpath = '/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/GoodFlat_1000y_Regu/'
#folderpath = '/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/'
#folderpath = '/home/jiayu/Desktop/ResultData/All_with_hrp2_kinematics/TSID_Almost_Symmetric_TestMotions/'
filename = '5LookAhead_Trial0.p'
#fullpath = folderpath + 'up_and_down_left_first_NLP_previous/' + filename
#fullpath = folderpath + 'flat_NLP_previous/' + filename

#fullpath = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/Result_Ref_Traj_Tracking/full_NLP_Solutions/up_and_down_left_first_NLP_previous/" + filename
fullpath = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/Result_10x_TerminalCost/Uneven_Tracking_Uneven_Flat_Ref/antfarm_Full_NLP/" + filename

#fullpath = "/home/jiayu/Desktop/Ponton_Result/RFCoMPolyLFRelaPloy/antfarm_firstLevel_left_start_NLP_previous/" + filename


#fullpath = folderpath + 'up_and_down_left_first_NLP_previous/' + filename
query_traj = "Lx_result"

startStepNum = 0
EndStepNum = 13


with open(fullpath, 'rb') as f:
    data = pickle.load(f)

#Make a result container
TISD_Trajectories = []

Level1_VarIndex = data["VarIdx_of_All_Levels"]["Level1_Var_Index"]
Level2_VarIndex = data["VarIdx_of_All_Levels"]["Level2_Var_Index"]

Trajectories = data["Trajectory_of_All_Rounds"]
CasadiParameters = data["CasadiParameters"]

x_result = []
y_result = []
z_result = []
xdot_result = []
ydot_result = []
zdot_result = []
Lx_result = []
Ly_result = []
Lz_result = []
Ldotx_result = []
Ldoty_result = []
Ldotz_result = []

FL1x_res = []
FL2x_res = []
FL3x_res = []
FL4x_res= []

FL1y_res = []
FL2y_res = []
FL3y_res = []
FL4y_res= []

FL1z_res = []
FL2z_res = []
FL3z_res = []
FL4z_res= []

FR1x_res = []
FR2x_res  = []
FR3x_res = []
FR4x_res = []

FR1y_res = []
FR2y_res  = []
FR3y_res = []
FR4y_res = []

FR1z_res = []
FR2z_res  = []
FR3z_res = []
FR4z_res = []

timeseries = []

time_offset = 0

for roundIdx in range(len(Trajectories)):

    #print(roundIdx)

    traj = Trajectories[roundIdx]
    casadiParams = CasadiParameters[roundIdx]

    #Get raw data
    x_traj = traj[Level1_VarIndex["x"][0]:Level1_VarIndex["x"][1]+1]
    y_traj = traj[Level1_VarIndex["y"][0]:Level1_VarIndex["y"][1]+1]
    z_traj = traj[Level1_VarIndex["z"][0]:Level1_VarIndex["z"][1]+1]
    xdot_traj = traj[Level1_VarIndex["xdot"][0]:Level1_VarIndex["xdot"][1]+1]
    ydot_traj = traj[Level1_VarIndex["ydot"][0]:Level1_VarIndex["ydot"][1]+1]
    zdot_traj = traj[Level1_VarIndex["zdot"][0]:Level1_VarIndex["zdot"][1]+1]

    Lx_traj = traj[Level1_VarIndex["Lx"][0]:Level1_VarIndex["Lx"][1]+1]
    Ly_traj = traj[Level1_VarIndex["Ly"][0]:Level1_VarIndex["Ly"][1]+1]
    Lz_traj = traj[Level1_VarIndex["Lz"][0]:Level1_VarIndex["Lz"][1]+1]
    Ldotx_traj = traj[Level1_VarIndex["Ldotx"][0]:Level1_VarIndex["Ldotx"][1]+1]
    Ldoty_traj = traj[Level1_VarIndex["Ldoty"][0]:Level1_VarIndex["Ldoty"][1]+1]
    Ldotz_traj = traj[Level1_VarIndex["Ldotz"][0]:Level1_VarIndex["Ldotz"][1]+1]

    FL1y_traj = traj[Level1_VarIndex["FL1y"][0]:Level1_VarIndex["FL1y"][1]+1]
    FL2y_traj = traj[Level1_VarIndex["FL2y"][0]:Level1_VarIndex["FL2y"][1]+1]
    FL3y_traj = traj[Level1_VarIndex["FL3y"][0]:Level1_VarIndex["FL3y"][1]+1]
    FL4y_traj = traj[Level1_VarIndex["FL4y"][0]:Level1_VarIndex["FL4y"][1]+1]

    FR1y_traj = traj[Level1_VarIndex["FR1y"][0]:Level1_VarIndex["FR1y"][1]+1]
    FR2y_traj = traj[Level1_VarIndex["FR2y"][0]:Level1_VarIndex["FR2y"][1]+1]
    FR3y_traj = traj[Level1_VarIndex["FR3y"][0]:Level1_VarIndex["FR3y"][1]+1]
    FR4y_traj = traj[Level1_VarIndex["FR4y"][0]:Level1_VarIndex["FR4y"][1]+1]

    px_res = traj[Level1_VarIndex["px"][0]:Level1_VarIndex["px"][1]+1]
    py_res = traj[Level1_VarIndex["py"][0]:Level1_VarIndex["py"][1]+1]
    pz_res = traj[Level1_VarIndex["pz"][0]:Level1_VarIndex["pz"][1]+1]

    Ts_res = traj[Level1_VarIndex["Ts"][0]:Level1_VarIndex["Ts"][1]+1]
    #Ts_level2_res = traj[Level2_VarIndex["Ts"][0]:Level2_VarIndex["Ts"][1]+1]

    PLx_init = casadiParams[14]
    PLy_init = casadiParams[15]
    PLz_init = casadiParams[16]

    PRx_init = casadiParams[17]
    PRy_init = casadiParams[18]
    PRz_init = casadiParams[19]

    LeftSwingFlag = casadiParams[0]
    RightSwingFlag = casadiParams[1]
    #get traj for each phase

    Phase1_TimeSeries = np.linspace(0,Ts_res[0],8)
    Phase2_TimeSeries = np.linspace(Ts_res[0],Ts_res[1],8)
    Phase3_TimeSeries = np.linspace(Ts_res[1],Ts_res[2],8)
    timeline = np.concatenate((time_offset+Phase1_TimeSeries,time_offset+Phase2_TimeSeries[1:],time_offset+Phase3_TimeSeries[1:]),axis=None)
    time_offset = time_offset+Phase3_TimeSeries[-1]

    Phase1_x = x_traj[0:8]
    Phase2_x = x_traj[7:15]
    Phase3_x = x_traj[14:]

    Phase1_y = y_traj[0:8]
    Phase2_y = y_traj[7:15]
    Phase3_y = y_traj[14:]

    Phase1_z = z_traj[0:8]
    Phase2_z = z_traj[7:15]
    Phase3_z = z_traj[14:]

    Phase1_xdot = xdot_traj[0:8]
    Phase2_xdot = xdot_traj[7:15]
    Phase3_xdot = xdot_traj[14:]

    Phase1_ydot = ydot_traj[0:8]
    Phase2_ydot = ydot_traj[7:15]
    Phase3_ydot = ydot_traj[14:]

    Phase1_zdot = zdot_traj[0:8]
    Phase2_zdot = zdot_traj[7:15]
    Phase3_zdot = zdot_traj[14:]

    Phase1_Lx = Lx_traj[0:8]
    Phase2_Lx = Lx_traj[7:15]
    Phase3_Lx = Lx_traj[14:]

    Phase1_Ly = Ly_traj[0:8]
    Phase2_Ly = Ly_traj[7:15]
    Phase3_Ly = Ly_traj[14:]

    Phase1_Lz = Lz_traj[0:8]
    Phase2_Lz = Lz_traj[7:15]
    Phase3_Lz = Lz_traj[14:]

    Phase1_Ldotx = Ldotx_traj[0:8]
    Phase2_Ldotx = Ldotx_traj[7:15]
    Phase3_Ldotx = Ldotx_traj[14:]

    Phase1_Ldoty = Ldoty_traj[0:8]
    Phase2_Ldoty = Ldoty_traj[7:15]
    Phase3_Ldoty = Ldoty_traj[14:]

    Phase1_Ldotz = Ldotz_traj[0:8]
    Phase2_Ldotz = Ldotz_traj[7:15]
    Phase3_Ldotz = Ldotz_traj[14:]

    TSIDTrajectory = {}
    
    #Init Double Phase
    TSIDTrajectory["InitDouble_TimeSeries"]=Phase1_TimeSeries
    TSIDTrajectory["InitDouble_x"]=Phase1_x
    TSIDTrajectory["InitDouble_y"]=Phase1_y
    TSIDTrajectory["InitDouble_z"]=Phase1_z
    TSIDTrajectory["InitDouble_Lx"]=Phase1_Lx
    TSIDTrajectory["InitDouble_Ly"]=Phase1_Ly
    TSIDTrajectory["InitDouble_Lz"]=Phase1_Lz
    TSIDTrajectory["InitDouble_xdot"]=Phase1_xdot
    TSIDTrajectory["InitDouble_ydot"]=Phase1_ydot
    TSIDTrajectory["InitDouble_zdot"]=Phase1_zdot
    TSIDTrajectory["InitDouble_Ldotx"]=Phase1_Ldotx
    TSIDTrajectory["InitDouble_Ldoty"]=Phase1_Ldoty
    TSIDTrajectory["InitDouble_Ldotz"]=Phase1_Ldotz

    #Swing Phase
    TSIDTrajectory["Swing_TimeSeries"]=Phase2_TimeSeries
    TSIDTrajectory["Swing_x"]=Phase2_x
    TSIDTrajectory["Swing_y"]=Phase2_y
    TSIDTrajectory["Swing_z"]=Phase2_z
    TSIDTrajectory["Swing_Lx"]=Phase2_Lx
    TSIDTrajectory["Swing_Ly"]=Phase2_Ly
    TSIDTrajectory["Swing_Lz"]=Phase2_Lz
    TSIDTrajectory["Swing_xdot"]=Phase2_xdot
    TSIDTrajectory["Swing_ydot"]=Phase2_ydot
    TSIDTrajectory["Swing_zdot"]=Phase2_zdot
    TSIDTrajectory["Swing_Ldotx"]=Phase2_Ldotx
    TSIDTrajectory["Swing_Ldoty"]=Phase2_Ldoty
    TSIDTrajectory["Swing_Ldotz"]=Phase2_Ldotz

    #DoubleSupport Phase
    TSIDTrajectory["DoubleSupport_TimeSeries"]=Phase3_TimeSeries
    TSIDTrajectory["DoubleSupport_x"]=Phase3_x
    TSIDTrajectory["DoubleSupport_y"]=Phase3_y
    TSIDTrajectory["DoubleSupport_z"]=Phase3_z
    TSIDTrajectory["DoubleSupport_Lx"]=Phase3_Lx
    TSIDTrajectory["DoubleSupport_Ly"]=Phase3_Ly
    TSIDTrajectory["DoubleSupport_Lz"]=Phase3_Lz
    TSIDTrajectory["DoubleSupport_xdot"]=Phase3_xdot
    TSIDTrajectory["DoubleSupport_ydot"]=Phase3_ydot
    TSIDTrajectory["DoubleSupport_zdot"]=Phase3_zdot
    TSIDTrajectory["DoubleSupport_Ldotx"]=Phase3_Ldotx
    TSIDTrajectory["DoubleSupport_Ldoty"]=Phase3_Ldoty
    TSIDTrajectory["DoubleSupport_Ldotz"]=Phase3_Ldotz

    #Contact config
    TSIDTrajectory["Init_PL"]=[PLx_init,PLy_init,PLz_init]
    TSIDTrajectory["Init_PR"]=[PRx_init,PRy_init,PRz_init]
    TSIDTrajectory["Landing_P"] = list(np.concatenate((px_res,py_res,pz_res),axis=None))
    TSIDTrajectory["LeftSwingFlag"]=LeftSwingFlag
    TSIDTrajectory["RightSwingFlag"]=RightSwingFlag

    TISD_Trajectories.append(TSIDTrajectory)

    #print(Ts_level2_res)
    #print("y motion:", np.max(y_traj) - np.min(y_traj))

    if roundIdx == 0:
        x_result.append(x_traj)
        y_result.append(y_traj)
        z_result.append(z_traj)
        xdot_result.append(xdot_traj)
        ydot_result.append(ydot_traj)
        zdot_result.append(zdot_traj)
        Lx_result.append(Lx_traj)
        Ly_result.append(Ly_traj)
        Lz_result.append(Lz_traj)
        Ldotx_result.append(Ldotx_traj)
        Ldoty_result.append(Ldoty_traj)
        Ldotz_result.append(Ldotz_traj)

        timeseries.append(timeline)
    else:
        x_result.append(x_traj[1:])
        y_result.append(y_traj[1:])
        z_result.append(z_traj[1:])
        xdot_result.append(xdot_traj[1:])
        ydot_result.append(ydot_traj[1:])
        zdot_result.append(zdot_traj[1:])
        Lx_result.append(Lx_traj[1:])
        Ly_result.append(Ly_traj[1:])
        Lz_result.append(Lz_traj[1:])
        Ldotx_result.append(Ldotx_traj[1:])
        Ldoty_result.append(Ldoty_traj[1:])
        Ldotz_result.append(Ldotz_traj[1:])
        timeseries.append(timeline[1:])

    #print("z_traj",z_traj)

#x_result = np.concatenate(x_result[startStepNum:EndStepNum+1])
#y_result = np.concatenate(y_result[startStepNum:EndStepNum+1]) 
#z_result = np.concatenate(z_result[startStepNum:EndStepNum+1]) 
#xdot_result = np.concatenate(xdot_result[startStepNum:EndStepNum+1])         
#ydot_result = np.concatenate(ydot_result[startStepNum:EndStepNum+1])    
#zdot_result = np.concatenate(zdot_result[startStepNum:EndStepNum+1])    
#Lx_result = np.concatenate(Lx_result[startStepNum:EndStepNum+1])    
#Ly_result = np.concatenate(Ly_result[startStepNum:EndStepNum+1])         
#Lz_result = np.concatenate(Lz_result[startStepNum:EndStepNum+1])   
#Ldotx_result = np.concatenate(Ldotx_result[startStepNum:EndStepNum+1])    
#Ldoty_result = np.concatenate(Ldoty_result[startStepNum:EndStepNum+1])         
#Ldotz_result = np.concatenate(Ldotz_result[startStepNum:EndStepNum+1])         
#timeseries = np.concatenate(timeseries[startStepNum:EndStepNum+1])

#CoM result
NLP_Traj = {"timeseries":timeseries,
            "x_result":x_result,
            "y_result":y_result,
            "z_result":z_result,
            "xdot_result":xdot_result,
            "ydot_result":ydot_result,
            "zdot_result":zdot_result,
            "Lx_result":Lx_result,
            "Ly_result":Ly_result,
            "Lz_result":Lz_result,
            "Ldotx_result":Ldotx_result,
            "Ldoty_result":Ldoty_result,
            "Ldotz_result":Ldotz_result,
}


#CoM result retrival
#fullpath = folderpath + 'up_and_down_left_first_CoM_previous/' + filename
#fullpath = folderpath + 'flat_CoM_previous/' + filename
#fullpath = folderpath + 'flat_CoM_previous/' + filename
#fullpath = folderpath + 'flat_Mix_previous/' + filename
#fullpath = folderpath + 'up_and_down_left_first_CoM_previous/' + filename

fullpath = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/antfarm_firstLevel_left_start_CoM_previous/" + filename

#fullpath = "/home/jiayu/Desktop/Ponton_Result/RFCoMPolyLFRelaPloy/antfarm_firstLevel_left_start_CoM_previous/" + filename

#fullpath = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/antfarm_firstLevel_left_start_CoM_previous/" + filename

with open(fullpath, 'rb') as f:
    data = pickle.load(f)

#Make a result container
TISD_Trajectories = []

Level1_VarIndex = data["VarIdx_of_All_Levels"]["Level1_Var_Index"]
Level2_VarIndex = data["VarIdx_of_All_Levels"]["Level2_Var_Index"]

Trajectories = data["Trajectory_of_All_Rounds"]
CasadiParameters = data["CasadiParameters"]

x_result = []
y_result = [] 
z_result = []
xdot_result = []  
ydot_result = []
zdot_result = []
Lx_result = []
Ly_result = []
Lz_result = []
Ldotx_result = []
Ldoty_result = []
Ldotz_result = []
timeseries = []

FL1x_res = []
FL2x_res = []
FL3x_res = []
FL4x_res= []

FL1y_res = []
FL2y_res = []
FL3y_res = []
FL4y_res= []

FL1z_res = []
FL2z_res = []
FL3z_res = []
FL4z_res= []

FR1x_res = []
FR2x_res  = []
FR3x_res = []
FR4x_res = []

FR1y_res = []
FR2y_res  = []
FR3y_res = []
FR4y_res = []

FR1z_res = []
FR2z_res  = []
FR3z_res = []
FR4z_res = []

time_offset = 0

for roundIdx in range(len(Trajectories)):

    #print(roundIdx)

    traj = Trajectories[roundIdx]
    casadiParams = CasadiParameters[roundIdx]

    #Get raw data
    x_traj = traj[Level1_VarIndex["x"][0]:Level1_VarIndex["x"][1]+1]
    y_traj = traj[Level1_VarIndex["y"][0]:Level1_VarIndex["y"][1]+1]
    z_traj = traj[Level1_VarIndex["z"][0]:Level1_VarIndex["z"][1]+1]
    xdot_traj = traj[Level1_VarIndex["xdot"][0]:Level1_VarIndex["xdot"][1]+1]
    ydot_traj = traj[Level1_VarIndex["ydot"][0]:Level1_VarIndex["ydot"][1]+1]
    zdot_traj = traj[Level1_VarIndex["zdot"][0]:Level1_VarIndex["zdot"][1]+1]

    Lx_traj = traj[Level1_VarIndex["Lx"][0]:Level1_VarIndex["Lx"][1]+1]
    Ly_traj = traj[Level1_VarIndex["Ly"][0]:Level1_VarIndex["Ly"][1]+1]
    Lz_traj = traj[Level1_VarIndex["Lz"][0]:Level1_VarIndex["Lz"][1]+1]
    Ldotx_traj = traj[Level1_VarIndex["Ldotx"][0]:Level1_VarIndex["Ldotx"][1]+1]
    Ldoty_traj = traj[Level1_VarIndex["Ldoty"][0]:Level1_VarIndex["Ldoty"][1]+1]
    Ldotz_traj = traj[Level1_VarIndex["Ldotz"][0]:Level1_VarIndex["Ldotz"][1]+1]

    FL1y_traj = traj[Level1_VarIndex["FL1y"][0]:Level1_VarIndex["FL1y"][1]+1]
    FL2y_traj = traj[Level1_VarIndex["FL2y"][0]:Level1_VarIndex["FL2y"][1]+1]
    FL3y_traj = traj[Level1_VarIndex["FL3y"][0]:Level1_VarIndex["FL3y"][1]+1]
    FL4y_traj = traj[Level1_VarIndex["FL4y"][0]:Level1_VarIndex["FL4y"][1]+1]

    FR1y_traj = traj[Level1_VarIndex["FR1y"][0]:Level1_VarIndex["FR1y"][1]+1]
    FR2y_traj = traj[Level1_VarIndex["FR2y"][0]:Level1_VarIndex["FR2y"][1]+1]
    FR3y_traj = traj[Level1_VarIndex["FR3y"][0]:Level1_VarIndex["FR3y"][1]+1]
    FR4y_traj = traj[Level1_VarIndex["FR4y"][0]:Level1_VarIndex["FR4y"][1]+1]

    px_res = traj[Level1_VarIndex["px"][0]:Level1_VarIndex["px"][1]+1]
    py_res = traj[Level1_VarIndex["py"][0]:Level1_VarIndex["py"][1]+1]
    pz_res = traj[Level1_VarIndex["pz"][0]:Level1_VarIndex["pz"][1]+1]

    Ts_res = traj[Level1_VarIndex["Ts"][0]:Level1_VarIndex["Ts"][1]+1]
    #Ts_level2_res = traj[Level2_VarIndex["Ts"][0]:Level2_VarIndex["Ts"][1]+1]

    PLx_init = casadiParams[14]
    PLy_init = casadiParams[15]
    PLz_init = casadiParams[16]

    PRx_init = casadiParams[17]
    PRy_init = casadiParams[18]
    PRz_init = casadiParams[19]

    LeftSwingFlag = casadiParams[0]
    RightSwingFlag = casadiParams[1]
    #get traj for each phase

    Phase1_TimeSeries = np.linspace(0,Ts_res[0],8)
    Phase2_TimeSeries = np.linspace(Ts_res[0],Ts_res[1],8)
    Phase3_TimeSeries = np.linspace(Ts_res[1],Ts_res[2],8)
    timeline = np.concatenate((time_offset+Phase1_TimeSeries,time_offset+Phase2_TimeSeries[1:],time_offset+Phase3_TimeSeries[1:]),axis=None)
    time_offset = time_offset+Phase3_TimeSeries[-1]

    Phase1_x = x_traj[0:8]
    Phase2_x = x_traj[7:15]
    Phase3_x = x_traj[14:]

    Phase1_y = y_traj[0:8]
    Phase2_y = y_traj[7:15]
    Phase3_y = y_traj[14:]

    Phase1_z = z_traj[0:8]
    Phase2_z = z_traj[7:15]
    Phase3_z = z_traj[14:]

    Phase1_xdot = xdot_traj[0:8]
    Phase2_xdot = xdot_traj[7:15]
    Phase3_xdot = xdot_traj[14:]

    Phase1_ydot = ydot_traj[0:8]
    Phase2_ydot = ydot_traj[7:15]
    Phase3_ydot = ydot_traj[14:]

    Phase1_zdot = zdot_traj[0:8]
    Phase2_zdot = zdot_traj[7:15]
    Phase3_zdot = zdot_traj[14:]

    Phase1_Lx = Lx_traj[0:8]
    Phase2_Lx = Lx_traj[7:15]
    Phase3_Lx = Lx_traj[14:]

    Phase1_Ly = Ly_traj[0:8]
    Phase2_Ly = Ly_traj[7:15]
    Phase3_Ly = Ly_traj[14:]

    Phase1_Lz = Lz_traj[0:8]
    Phase2_Lz = Lz_traj[7:15]
    Phase3_Lz = Lz_traj[14:]

    Phase1_Ldotx = Ldotx_traj[0:8]
    Phase2_Ldotx = Ldotx_traj[7:15]
    Phase3_Ldotx = Ldotx_traj[14:]

    Phase1_Ldoty = Ldoty_traj[0:8]
    Phase2_Ldoty = Ldoty_traj[7:15]
    Phase3_Ldoty = Ldoty_traj[14:]

    Phase1_Ldotz = Ldotz_traj[0:8]
    Phase2_Ldotz = Ldotz_traj[7:15]
    Phase3_Ldotz = Ldotz_traj[14:]

    TSIDTrajectory = {}
    
    #Init Double Phase
    TSIDTrajectory["InitDouble_TimeSeries"]=Phase1_TimeSeries
    TSIDTrajectory["InitDouble_x"]=Phase1_x
    TSIDTrajectory["InitDouble_y"]=Phase1_y
    TSIDTrajectory["InitDouble_z"]=Phase1_z
    TSIDTrajectory["InitDouble_Lx"]=Phase1_Lx
    TSIDTrajectory["InitDouble_Ly"]=Phase1_Ly
    TSIDTrajectory["InitDouble_Lz"]=Phase1_Lz
    TSIDTrajectory["InitDouble_xdot"]=Phase1_xdot
    TSIDTrajectory["InitDouble_ydot"]=Phase1_ydot
    TSIDTrajectory["InitDouble_zdot"]=Phase1_zdot
    TSIDTrajectory["InitDouble_Ldotx"]=Phase1_Ldotx
    TSIDTrajectory["InitDouble_Ldoty"]=Phase1_Ldoty
    TSIDTrajectory["InitDouble_Ldotz"]=Phase1_Ldotz

    #Swing Phase
    TSIDTrajectory["Swing_TimeSeries"]=Phase2_TimeSeries
    TSIDTrajectory["Swing_x"]=Phase2_x
    TSIDTrajectory["Swing_y"]=Phase2_y
    TSIDTrajectory["Swing_z"]=Phase2_z
    TSIDTrajectory["Swing_Lx"]=Phase2_Lx
    TSIDTrajectory["Swing_Ly"]=Phase2_Ly
    TSIDTrajectory["Swing_Lz"]=Phase2_Lz
    TSIDTrajectory["Swing_xdot"]=Phase2_xdot
    TSIDTrajectory["Swing_ydot"]=Phase2_ydot
    TSIDTrajectory["Swing_zdot"]=Phase2_zdot
    TSIDTrajectory["Swing_Ldotx"]=Phase2_Ldotx
    TSIDTrajectory["Swing_Ldoty"]=Phase2_Ldoty
    TSIDTrajectory["Swing_Ldotz"]=Phase2_Ldotz

    #DoubleSupport Phase
    TSIDTrajectory["DoubleSupport_TimeSeries"]=Phase3_TimeSeries
    TSIDTrajectory["DoubleSupport_x"]=Phase3_x
    TSIDTrajectory["DoubleSupport_y"]=Phase3_y
    TSIDTrajectory["DoubleSupport_z"]=Phase3_z
    TSIDTrajectory["DoubleSupport_Lx"]=Phase3_Lx
    TSIDTrajectory["DoubleSupport_Ly"]=Phase3_Ly
    TSIDTrajectory["DoubleSupport_Lz"]=Phase3_Lz
    TSIDTrajectory["DoubleSupport_xdot"]=Phase3_xdot
    TSIDTrajectory["DoubleSupport_ydot"]=Phase3_ydot
    TSIDTrajectory["DoubleSupport_zdot"]=Phase3_zdot
    TSIDTrajectory["DoubleSupport_Ldotx"]=Phase3_Ldotx
    TSIDTrajectory["DoubleSupport_Ldoty"]=Phase3_Ldoty
    TSIDTrajectory["DoubleSupport_Ldotz"]=Phase3_Ldotz

    #Contact config
    TSIDTrajectory["Init_PL"]=[PLx_init,PLy_init,PLz_init]
    TSIDTrajectory["Init_PR"]=[PRx_init,PRy_init,PRz_init]
    TSIDTrajectory["Landing_P"] = list(np.concatenate((px_res,py_res,pz_res),axis=None))
    TSIDTrajectory["LeftSwingFlag"]=LeftSwingFlag
    TSIDTrajectory["RightSwingFlag"]=RightSwingFlag

    TISD_Trajectories.append(TSIDTrajectory)

    if roundIdx == 0:
        x_result.append(x_traj)
        y_result.append(y_traj)
        z_result.append(z_traj)
        xdot_result.append(xdot_traj)
        ydot_result.append(ydot_traj)
        zdot_result.append(zdot_traj)
        Lx_result.append(Lx_traj)
        Ly_result.append(Ly_traj)
        Lz_result.append(Lz_traj)
        Ldotx_result.append(Ldotx_traj)
        Ldoty_result.append(Ldoty_traj)
        Ldotz_result.append(Ldotz_traj)

        timeseries.append(timeline)
    else:
        x_result.append(x_traj[1:])
        y_result.append(y_traj[1:])
        z_result.append(z_traj[1:])
        xdot_result.append(xdot_traj[1:])
        ydot_result.append(ydot_traj[1:])
        zdot_result.append(zdot_traj[1:])
        Lx_result.append(Lx_traj[1:])
        Ly_result.append(Ly_traj[1:])
        Lz_result.append(Lz_traj[1:])
        Ldotx_result.append(Ldotx_traj[1:])
        Ldoty_result.append(Ldoty_traj[1:])
        Ldotz_result.append(Ldotz_traj[1:])
        timeseries.append(timeline[1:])

    #print("z_traj",z_traj)

#x_result = np.concatenate(x_result[startStepNum:EndStepNum+1])   
#y_result = np.concatenate(y_result[startStepNum:EndStepNum+1]) 
#z_result = np.concatenate(z_result[startStepNum:EndStepNum+1]) 
#xdot_result = np.concatenate(xdot_result[startStepNum:EndStepNum+1])         
#ydot_result = np.concatenate(ydot_result[startStepNum:EndStepNum+1])    
#zdot_result = np.concatenate(zdot_result[startStepNum:EndStepNum+1])    
#Lx_result = np.concatenate(Lx_result[startStepNum:EndStepNum+1])    
#Ly_result = np.concatenate(Ly_result[startStepNum:EndStepNum+1])         
#Lz_result = np.concatenate(Lz_result[startStepNum:EndStepNum+1])   
#Ldotx_result = np.concatenate(Ldotx_result[startStepNum:EndStepNum+1])    
#Ldoty_result = np.concatenate(Ldoty_result[startStepNum:EndStepNum+1])         
#Ldotz_result = np.concatenate(Ldotz_result[startStepNum:EndStepNum+1])         
#timeseries = np.concatenate(timeseries[startStepNum:EndStepNum+1])

#CoM result
CoM_Traj = {"timeseries":timeseries,
            "x_result":x_result,
            "y_result":y_result,
            "z_result":z_result,
            "xdot_result":xdot_result,
            "ydot_result":ydot_result,
            "zdot_result":zdot_result,
            "Lx_result":Lx_result,
            "Ly_result":Ly_result,
            "Lz_result":Lz_result,
            "Ldotx_result":Ldotx_result,
            "Ldoty_result":Ldoty_result,
            "Ldotz_result":Ldotz_result,
}

#print(CoM_Traj["x_result"][8])

for traj_idx in range(len(NLP_Traj["x_result"])):
    if traj_idx%2 == 0:
        if traj_idx == 0:
            if traj_idx <= len(NLP_Traj["x_result"])-1:
                plt.plot(NLP_Traj["timeseries"][traj_idx],NLP_Traj[query_traj][traj_idx],color = 'r', label='NLP',linestyle='dashed')
            if traj_idx <= len(CoM_Traj["x_result"])-1:
                plt.plot(CoM_Traj["timeseries"][traj_idx],CoM_Traj[query_traj][traj_idx],color = 'b', label='CoM',linestyle='dashed')
        else:
            if traj_idx <= len(NLP_Traj["x_result"])-1:
                plt.plot(NLP_Traj["timeseries"][traj_idx],NLP_Traj[query_traj][traj_idx],color = 'r',linestyle='dashed')
            if traj_idx <= len(CoM_Traj["x_result"])-1:
                plt.plot(CoM_Traj["timeseries"][traj_idx],CoM_Traj[query_traj][traj_idx],color = 'b', linestyle='dashed')
    else:
        if traj_idx <= len(CoM_Traj["x_result"])-1:
            plt.plot(CoM_Traj["timeseries"][traj_idx],CoM_Traj[query_traj][traj_idx],color = 'b')
        if traj_idx <= len(NLP_Traj["x_result"])-1:
            plt.plot(NLP_Traj["timeseries"][traj_idx],NLP_Traj[query_traj][traj_idx],color = 'r')

plt.xlabel('Time')
plt.ylabel(query_traj)

plt.title(filename + ' - ' + query_traj)

#plt.xlim(0,12.8)

print(len(NLP_Traj[query_traj]))

plt.legend(loc="upper left")
plt.show()






