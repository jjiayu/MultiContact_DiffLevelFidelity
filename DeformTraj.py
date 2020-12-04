import numpy as np
import pickle

import matplotlib.pyplot as plt #Matplotlib
from mpl_toolkits.mplot3d import Axes3D

newPy = -0.26

#filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotion/Knitro/antfarm_ref/10LookAhead_Trial0.p"
filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotion/Knitro/flat_patches/" + '10LookAhead_Trial0.p'
#filename = "/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotion/Knitro/flat_ref/" + '10LookAhead_Trial0.p'

with open(filename, 'rb') as f:
    data = pickle.load(f)

#Get Result Data
Level1_VarIndex = data["VarIdx_of_All_Levels"]["Level1_Var_Index"]
Level2_VarIndex = data["VarIdx_of_All_Levels"]["Level2_Var_Index"]
Trajectories = data["Trajectory_of_All_Rounds"]
oldTrajectories = data["Trajectory_of_All_Rounds"]
Px_result = data["Px_fullres"]
Py_result = data["Py_fullres"]
Pz_result = data["Pz_fullres"]

PL_init_Full_res = data["PL_init_fullres"]
PR_init_Full_res = data["PR_init_fullres"]

#For the round 1
traj = Trajectories[1]
old_traj = Trajectories[1]
x_traj = traj[Level1_VarIndex["x"][0]:Level1_VarIndex["x"][1]+1]
y_traj = traj[Level1_VarIndex["y"][0]:Level1_VarIndex["y"][1]+1]
old_y_traj = y_traj
z_traj = traj[Level1_VarIndex["z"][0]:Level1_VarIndex["z"][1]+1]

#x_traj_fcontact_phase1 = x_traj[0:7] - Px_result[0] #first phase follow the first 
#x_traj_fcontact_phase2 = x_traj[7:14] - Px_result[1]
#x_traj_fcontact_phase3 = x_traj[14:] - Px_result[1]

y_traj_fcontact_phase1 = y_traj[0:7] - Py_result[0]
y_traj_fcontact_phase2 = y_traj[7:14] - Py_result[1]
y_traj_fcontact_phase3 = y_traj[14:] - Py_result[1]

#z_traj_fcontact_phase1 = z_traj[0:7] - Pz_result[0]
#z_traj_fcontact_phase2 = z_traj[7:14] - Pz_result[1]
#z_traj_fcontact_phase3 = z_traj[14:] - Pz_result[1]

y_traj_adapted_phase1 = y_traj_fcontact_phase1 + Py_result[0]
y_traj_adapted_phase2 = y_traj_fcontact_phase2 + newPy
y_traj_adapted_phase3 = y_traj_fcontact_phase3 + newPy

y_traj_adapted = np.concatenate((y_traj_adapted_phase1,y_traj_adapted_phase2,y_traj_adapted_phase3),axis=None)
ydot_traj_adapted = np.diff(y_traj_adapted)
ydot_traj_adapted = np.concatenate((ydot_traj_adapted,ydot_traj_adapted[-1]),axis=None)

# plt.plot(y_traj)
# plt.plot(y_traj_adapted)
# plt.show()

#update traj
#print(traj[Level1_VarIndex["y"][0]:Level1_VarIndex["y"][1]+1])
traj[Level1_VarIndex["y"][0]:Level1_VarIndex["y"][1]+1] = y_traj_adapted
#print(traj[Level1_VarIndex["y"][0]:Level1_VarIndex["y"][1]+1])
traj[Level1_VarIndex["ydot"][0]:Level1_VarIndex["ydot"][1]+1] = ydot_traj_adapted
Trajectories[1] = traj
#print(Trajectories[1][Level1_VarIndex["y"][0]:Level1_VarIndex["y"][1]+1])

#print(np.sum(old_y_traj-np.array(traj[Level1_VarIndex["y"][0]:Level1_VarIndex["y"][1]+1])))



#For the round 2
traj = Trajectories[2]
x_traj = traj[Level1_VarIndex["x"][0]:Level1_VarIndex["x"][1]+1]
y_traj = traj[Level1_VarIndex["y"][0]:Level1_VarIndex["y"][1]+1]
z_traj = traj[Level1_VarIndex["z"][0]:Level1_VarIndex["z"][1]+1]

# plt.plot(y_traj)
# plt.show()

y_trdaj_fcontact_phase1 = y_traj[0:7] - Py_result[1]
y_traj_fcontact_phase2 = y_traj[7:14] - Py_result[2]
y_traj_fcontact_phase3 = y_traj[14:] - Py_result[2]

y_traj_adapted_phase1 = y_traj_fcontact_phase1 + newPy
y_traj_adapted_phase2 = y_traj_fcontact_phase2 + Py_result[2]
y_traj_adapted_phase3 = y_traj_fcontact_phase3 + Py_result[2]

y_traj_adapted = np.concatenate((y_traj_adapted_phase1,y_traj_adapted_phase2,y_traj_adapted_phase3),axis=None)

ydot_traj_adapted = np.diff(y_traj_adapted)
ydot_traj_adapted = np.concatenate((ydot_traj_adapted,ydot_traj_adapted[-1]),axis=None)

# # plt.plot(y_traj)
# # plt.plot(y_traj_adapted)
# # plt.show()

#update traj
traj[Level1_VarIndex["y"][0]:Level1_VarIndex["y"][1]+1] = y_traj_adapted
traj[Level1_VarIndex["ydot"][0]:Level1_VarIndex["ydot"][1]+1] = ydot_traj_adapted
Trajectories[2] = traj

data["Trajectory_of_All_Rounds"] = Trajectories

print(np.sum(oldTrajectories-np.array(data["Trajectory_of_All_Rounds"])))
#x_traj_fcontact = np.concatenate((x_traj_fcontact_phase1,x_traj_fcontact_phase2,x_traj_fcontact_phase3),axis=None)
#y_traj_fcontact = np.concatenate((y_traj_fcontact_phase1,y_traj_fcontact_phase2,y_traj_fcontact_phase3),axis=None)
#z_traj_fcontact = np.concatenate((z_traj_fcontact_phase1,z_traj_fcontact_phase2,z_traj_fcontact_phase3),axis=None)

pickle.dump(data, open('/home/jiayu/Desktop/MultiContact_DiffLevelFidelity/RefMotion/Knitro/DeformedTraj'+'/'+'10LookAhead_Trial0'+'.p', "wb"))  # save it into a file named save.p
