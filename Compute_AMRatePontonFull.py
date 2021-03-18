import pickle
import numpy as np

import matplotlib.pyplot as plt #Matplotlib
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(precision=4)

fullpath = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/FixedDuration/PontonFull_06/darpa_like_left_first_Ponton_previous/"+"2LookAhead_Trial0.p"

RoundNum = 0

#Load file
with open(fullpath, 'rb') as f:
    data = pickle.load(f)

Level1_VarIndex = data["VarIdx_of_All_Levels"]["Level1_Var_Index"]
Level2_VarIndex = data["VarIdx_of_All_Levels"]["Level2_Var_Index"]

Trajectories = data["Trajectory_of_All_Rounds"]
casadiParameters = data["CasadiParameters"]

PLx_init = casadiParams[14]
PLy_init = casadiParams[15]
PLz_init = casadiParams[16]

PRx_init = casadiParams[17]
PRy_init = casadiParams[18]
PRz_init = casadiParams[19]

traj = Trajectories[RoundNum]

px_res = traj[Level1_VarIndex["px"][0]:Level1_VarIndex["px"][1]+1]
py_res = traj[Level1_VarIndex["py"][0]:Level1_VarIndex["py"][1]+1]
pz_res = traj[Level1_VarIndex["pz"][0]:Level1_VarIndex["pz"][1]+1]

#Get Initial Double Support Contact Locations for the second level
#Assume always start from the left

if StepNum%2==0:
    PL_INIT = np.concatenate(((px_res,py_res,pz_res)),axis=None)
    PR_INIT = np.concatenate(((PRx_init,PRy_init,PRz_init)),axis=None)
elif StepNum%2==1:
    PL_INIT = np.concatenate(((PLx_init,PLy_init,PLz_init)),axis=None)
    PR_INIT = np.concatenate(((px_res,py_res,pz_res)),axis=None)

#Get Swing Phase Contact Locations
if StepNum%2==0:
    PL_Swing = np.concatenate(((px_res,py_res,pz_res)),axis=None)
    PR_Swing = np.concatenate(((PRx_init,PRy_init,PRz_init)),axis=None)
elif StepNum%2==1:
    PL_Swing = np.concatenate(((PLx_init,PLy_init,PLz_init)),axis=None)
    PR_Swing = np.concatenate(((px_res,py_res,pz_res)),axis=None)





    
    
    
    
    




