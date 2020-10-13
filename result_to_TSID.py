import pickle
import numpy as np
from Tools import *

import matplotlib.pyplot as plt #Matplotlib
from mpl_toolkits.mplot3d import Axes3D

filename = '/home/ggory15/git/icra2021/data/Terrain/4LookAhead_Trial0.p'

with open(filename, 'rb') as f:
    data = pickle.load(f)

#Make a result container
TISD_Trajectories = []

Level1_VarIndex = data["VarIdx_of_All_Levels"]["Level1_Var_Index"]
Level2_VarIndex = data["VarIdx_of_All_Levels"]["Level2_Var_Index"]

Trajectories = data["Trajectory_of_All_Rounds"]
CasadiParameters = data["CasadiParameters"]

all_res = []

timeseries = []

time_offset = 0

Terrain_flag = False

for roundIdx in range(len(Trajectories)):

    #print(roundIdx)

    traj = Trajectories[roundIdx]
    casadiParams = CasadiParameters[roundIdx]
    casadiPrevParams = []
    if roundIdx > 0:
        casadiPrevParams = CasadiParameters[roundIdx - 1]
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

    #clearflat = False
    #Clear L Ldot
    #if clearflat == True:
    #    Lx_traj = np.full((len(Lx_traj),),0)
    #    Ly_traj = np.full((len(Ly_traj),),0)
    #    Lz_traj = np.full((len(Lz_traj),),0)

    #    Ldotx_traj = np.full((len(Ldotx_traj),),0)
    #    Ldoty_traj = np.full((len(Ldoty_traj),),0)
    #    Ldotz_traj = np.full((len(Ldotz_traj),),0)

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

    #Get FootStep/Terrain Quaternions
    if "TerrainModel" in data:
        Allpatches = data["TerrainModel"]
        Allquat = []

        for patch in Allpatches:
            quat = getQuaternion(patch)
            Allquat.append(quat)
        Terrain_flag = True

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
    TSIDTrajectory["FootStep_Quaternions"] = Allquat
    TSIDTrajectory["LeftSwingFlag"]=LeftSwingFlag
    TSIDTrajectory["RightSwingFlag"]=RightSwingFlag

    if Terrain_flag is False:
        TSIDTrajectory["Init_L_quat"]=np.array([0,0,0,1])
        TSIDTrajectory["Init_R_quat"]=np.array([0,0,0,1])
        TSIDTrajectory["Landing_quat"]=np.array([0,0,0,1])
    else:   
        if (roundIdx == 0) or (roundIdx > len(Trajectories)-2):
            TSIDTrajectory["Init_L_quat"]=np.array([0,0,0,1])
            TSIDTrajectory["Init_R_quat"]=np.array([0,0,0,1])
        elif (roundIdx == 1):
            if casadiPrevParams[0] == 1.0:
                TSIDTrajectory["Init_L_quat"]= Allquat[roundIdx-1]
                TSIDTrajectory["Init_R_quat"]=np.array([0,0,0,1])   
            else:
                TSIDTrajectory["Init_R_quat"]= Allquat[roundIdx-1]
                TSIDTrajectory["Init_L_quat"]= np.array([0,0,0,1])  
        else:
            if casadiPrevParams[0] == 1.0:
                TSIDTrajectory["Init_L_quat"]= Allquat[roundIdx-1]
                TSIDTrajectory["Init_R_quat"]= Allquat[roundIdx-2]
            else:
                TSIDTrajectory["Init_R_quat"]= Allquat[roundIdx-1]
                TSIDTrajectory["Init_L_quat"]= Allquat[roundIdx-2]    

        if (roundIdx < len(Trajectories) - 3):
            TSIDTrajectory["Landing_quat"] = Allquat[roundIdx]
        else:
            TSIDTrajectory["Landing_quat"] = Allquat[len(Trajectories) -3]    

    print ("Phase #", roundIdx)
    print ("Init_R", TSIDTrajectory["Init_R_quat"])
    print ("Init_L", TSIDTrajectory["Init_L_quat"])
    print ("Landing_quat", TSIDTrajectory["Landing_quat"])
    print ("")
    TISD_Trajectories.append(TSIDTrajectory)

    #print(Ts_level2_res)
    #print("y motion:", np.max(y_traj) - np.min(y_traj))
    collected_traj = x_traj

    if roundIdx == 0:
        all_res.append(collected_traj)
        timeseries.append(timeline)
    else:
        all_res.append(collected_traj[1:])
        timeseries.append(timeline[1:])

    #print("z_traj",z_traj)

all_res = np.concatenate(all_res)
timeseries = np.concatenate(timeseries)

#plt.plot(timeseries,all_res)

#plt.show()



#Dump data into pickled file
DumpedResult = {"TSID_Trajectories": TISD_Trajectories,
}
pickle.dump(DumpedResult, open("/home/ggory15/git/icra2021/TSID_Trajectory"'.p', "wb"))  # save it into a file named save.p





