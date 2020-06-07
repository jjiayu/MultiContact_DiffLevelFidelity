import numpy as np #Numpy
import matplotlib.pyplot as plt #Matplotlib
from mpl_toolkits.mplot3d import Axes3D

def PlotNLPStep(x_opt = None, fig=None, var_index=None, PL_init = None, PR_init = None, LeftSwing = None, RightSwing = None):
    #-----------------------------------------------------------------------------------------------------------------------
    #Plot Result
    if fig==None:
        fig=plt.figure()
    
    ax = Axes3D(fig)

    var_index = var_index["Level1_Var_Index"]
    #ax = fig.add_subplot(111, projection="3d")

    #ax.plot3D(x_res,y_res,z_res,color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)

    x_res = x_opt[var_index["x"][0]:var_index["x"][1]+1]
    x_res = np.array(x_res)
    print('x_res: ',x_res)
    y_res = x_opt[var_index["y"][0]:var_index["y"][1]+1]
    y_res = np.array(y_res)
    print('y_res: ',y_res)
    z_res = x_opt[var_index["z"][0]:var_index["z"][1]+1]
    z_res = np.array(z_res)
    print('z_res: ',z_res)
    Lx_res = x_opt[var_index["Lx"][0]:var_index["Lx"][1]+1]
    Lx_res = np.array(Lx_res)
    print('Lx_res: ',Lx_res)
    Ly_res = x_opt[var_index["Ly"][0]:var_index["Ly"][1]+1]
    Ly_res = np.array(Ly_res)
    print('Ly_res: ',Ly_res)
    Lz_res = x_opt[var_index["Lz"][0]:var_index["Lz"][1]+1]
    Lz_res = np.array(Lz_res)
    print('Lz_res: ',Lz_res)
    Ldotx_res = x_opt[var_index["Ldotx"][0]:var_index["Ldotx"][1]+1]
    Ldotx_res = np.array(Ldotx_res)
    print('Ldotx_res: ',Ldotx_res)
    Ldoty_res = x_opt[var_index["Ldoty"][0]:var_index["Ldoty"][1]+1]
    Ldoty_res = np.array(Ldoty_res)
    print('Ldoty_res: ',Ldoty_res)
    Ldotz_res = x_opt[var_index["Ldotz"][0]:var_index["Ldotz"][1]+1]
    Ldotz_res = np.array(Ldotz_res)
    print('Ldotz_res: ',Ldotz_res)
    px_res = x_opt[var_index["px"][0]:var_index["px"][1]+1]
    px_res = np.array(px_res)
    px_res = x_opt[var_index["px"][0]:var_index["px"][1]+1]
    px_res = np.array(px_res)
    print('px_res: ',px_res)
    py_res = x_opt[var_index["py"][0]:var_index["py"][1]+1]
    py_res = np.array(py_res)
    print('py_res: ',py_res)
    pz_res = x_opt[var_index["pz"][0]:var_index["pz"][1]+1]
    pz_res = np.array(pz_res)
    print('pz_res: ',pz_res)

    #CoM Trajectory
    ax.plot3D(x_res,y_res,z_res,color='green', linestyle='dashed', linewidth=2, markersize=12)
    #Initial Footstep Locations
    ax.scatter(PL_init[0], PL_init[1], PL_init[2], c='r', marker='o', linewidth = 10) 
    ax.scatter(PR_init[0], PR_init[1], PR_init[2], c='b', marker='o', linewidth = 10) 
    #Swing Foot
    if LeftSwing == 1:
        StepColor = 'r'
    if RightSwing == 1:
        StepColor = 'b'
    ax.scatter(px_res, py_res, pz_res, c=StepColor, marker='o', linewidth = 10) 

    ax.set_xlim3d(x_res[0]-0.2, px_res[-1]+0.35)
    ax.set_ylim3d(-0.5,0.5)
    ax.set_zlim3d(0,0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

    return ax

def Plot_Pure_Kinematics_Plan(x_opt = None, fig=None, var_index=None, PL_init = None, PR_init = None, LeftSwing = None, RightSwing = None):
    #-----------------------------------------------------------------------------------------------------------------------
    #Plot Result
    if fig==None:
        fig=plt.figure()
    
    ax = Axes3D(fig)
    #ax = fig.add_subplot(111, projection="3d")
    print(var_index["Level1_Var_Index"])

    var_index = var_index["Level1_Var_Index"]

    #ax.plot3D(x_res,y_res,z_res,color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)

    x_res = x_opt[var_index["x"][0]:var_index["x"][1]+1]
    x_res = np.array(x_res)
    print('x_res: ',x_res)
    y_res = x_opt[var_index["y"][0]:var_index["y"][1]+1]
    y_res = np.array(y_res)
    print('y_res: ',y_res)
    z_res = x_opt[var_index["z"][0]:var_index["z"][1]+1]
    z_res = np.array(z_res)
    print('z_res: ',z_res)
    px_res = x_opt[var_index["px"][0]:var_index["px"][1]+1]
    px_res = np.array(px_res)
    px_res = x_opt[var_index["px"][0]:var_index["px"][1]+1]
    px_res = np.array(px_res)
    print('px_res: ',px_res)
    py_res = x_opt[var_index["py"][0]:var_index["py"][1]+1]
    py_res = np.array(py_res)
    print('py_res: ',py_res)
    pz_res = x_opt[var_index["pz"][0]:var_index["pz"][1]+1]
    pz_res = np.array(pz_res)
    print('pz_res: ',pz_res)

    #CoM Trajectory
    ax.plot3D(x_res,y_res,z_res,color='green', linestyle='dashed', linewidth=2, markersize=12)
    #Initial Footstep Locations
    ax.scatter(PL_init[0], PL_init[1], PL_init[2], c='r', marker='o', linewidth = 10) 
    ax.scatter(PR_init[0], PR_init[1], PR_init[2], c='b', marker='o', linewidth = 10) 
    #Swing Foot
    if LeftSwing == 1:
        StepColor = 'r'
    if RightSwing == 1:
        StepColor = 'b'
    ax.scatter(px_res, py_res, pz_res, c=StepColor, marker='o', linewidth = 10) 

    ax.set_xlim3d(x_res[0]-0.2, px_res[-1]+0.35)
    ax.set_ylim3d(-0.5,0.5)
    ax.set_zlim3d(0,0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

    return ax

def Plot_Both_Levels(x_opt = None, fig=None, var_index=None, PL_init = None, PR_init = None, LeftSwing = None, RightSwing = None):

    #-----------------------------------------------------------------------------------------------------------------------
    #Plot First Level Result
    if fig==None:
        fig=plt.figure()
    
    ax = Axes3D(fig)

    #save the complete var_index
    var_index_temp =var_index

    #get the first level var_index
    var_index = var_index_temp["Level1_Var_Index"]
    #ax = fig.add_subplot(111, projection="3d")

    x_res = x_opt[var_index["x"][0]:var_index["x"][1]+1]
    x_res = np.array(x_res)
    print('x_res: ',x_res)
    y_res = x_opt[var_index["y"][0]:var_index["y"][1]+1]
    y_res = np.array(y_res)
    print('y_res: ',y_res)
    z_res = x_opt[var_index["z"][0]:var_index["z"][1]+1]
    z_res = np.array(z_res)
    print('z_res: ',z_res)
    Lx_res = x_opt[var_index["Lx"][0]:var_index["Lx"][1]+1]
    Lx_res = np.array(Lx_res)
    print('Lx_res: ',Lx_res)
    Ly_res = x_opt[var_index["Ly"][0]:var_index["Ly"][1]+1]
    Ly_res = np.array(Ly_res)
    print('Ly_res: ',Ly_res)
    Lz_res = x_opt[var_index["Lz"][0]:var_index["Lz"][1]+1]
    Lz_res = np.array(Lz_res)
    print('Lz_res: ',Lz_res)
    Ldotx_res = x_opt[var_index["Ldotx"][0]:var_index["Ldotx"][1]+1]
    Ldotx_res = np.array(Ldotx_res)
    print('Ldotx_res: ',Ldotx_res)
    Ldoty_res = x_opt[var_index["Ldoty"][0]:var_index["Ldoty"][1]+1]
    Ldoty_res = np.array(Ldoty_res)
    print('Ldoty_res: ',Ldoty_res)
    Ldotz_res = x_opt[var_index["Ldotz"][0]:var_index["Ldotz"][1]+1]
    Ldotz_res = np.array(Ldotz_res)
    print('Ldotz_res: ',Ldotz_res)
    px_res = x_opt[var_index["px"][0]:var_index["px"][1]+1]
    px_res = np.array(px_res)
    print('px_res: ',px_res)
    py_res = x_opt[var_index["py"][0]:var_index["py"][1]+1]
    py_res = np.array(py_res)
    print('py_res: ',py_res)
    pz_res = x_opt[var_index["pz"][0]:var_index["pz"][1]+1]
    pz_res = np.array(pz_res)
    print('pz_res: ',pz_res)

    #CoM Trajectory
    ax.plot3D(x_res,y_res,z_res,color='green', linestyle='dashed', linewidth=2, markersize=12)
    #Initial Footstep Locations
    ax.scatter(PL_init[0], PL_init[1], PL_init[2], c='r', marker='o', linewidth = 10) 
    ax.scatter(PR_init[0], PR_init[1], PR_init[2], c='b', marker='o', linewidth = 10) 
    #Swing Foot
    if LeftSwing == 1:
        StepColor = 'r'
    if RightSwing == 1:
        StepColor = 'b'
    ax.scatter(px_res, py_res, pz_res, c=StepColor, marker='o', linewidth = 10) 
    
    #-----------------------------------------------------------------------------------------------------------------------
    #Plot Second Level Result

    #Get Second Level result
    x_opt = x_opt[var_index["Ts"][1]+1:]

    #Get Second Level Result Index
    var_index = var_index_temp["Level2_Var_Index"]

    x_res = x_opt[var_index["x"][0]:var_index["x"][1]+1]
    x_res = np.array(x_res)
    print('x_res: ',x_res)
    y_res = x_opt[var_index["y"][0]:var_index["y"][1]+1]
    y_res = np.array(y_res)
    print('y_res: ',y_res)
    z_res = x_opt[var_index["z"][0]:var_index["z"][1]+1]
    z_res = np.array(z_res)
    print('z_res: ',z_res)
    px_res = x_opt[var_index["px"][0]:var_index["px"][1]+1]
    px_res = np.array(px_res)
    px_res = x_opt[var_index["px"][0]:var_index["px"][1]+1]
    px_res = np.array(px_res)
    print('px_res: ',px_res)
    py_res = x_opt[var_index["py"][0]:var_index["py"][1]+1]
    py_res = np.array(py_res)
    print('py_res: ',py_res)
    pz_res = x_opt[var_index["pz"][0]:var_index["pz"][1]+1]
    pz_res = np.array(pz_res)
    print('pz_res: ',pz_res)

    #CoM Trajectory
    ax.plot3D(x_res,y_res,z_res,color='green', linestyle='dashed', linewidth=2, markersize=12)
    
    #Swing Foot
    if LeftSwing == 1:
        StepColor = 'b'
        ax.scatter(px_res[0::2], py_res[0::2], pz_res[0::2], c=StepColor, marker='o', linewidth = 10) 
        StepColor = 'r'
        ax.scatter(px_res[1::2], py_res[1::2], pz_res[1::2], c=StepColor, marker='o', linewidth = 10) 
    if RightSwing == 1:
        StepColor = 'r'
        ax.scatter(px_res[0::2], py_res[0::2], pz_res[0::2], c=StepColor, marker='o', linewidth = 10) 
        StepColor = 'b'
        ax.scatter(px_res[1::2], py_res[1::2], pz_res[1::2], c=StepColor, marker='o', linewidth = 10) 


    ax.set_xlim3d(x_res[0]-0.2, px_res[-1]+0.35)
    ax.set_ylim3d(-0.5,0.5)
    ax.set_zlim3d(0,0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()