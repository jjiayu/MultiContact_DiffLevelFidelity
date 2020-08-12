import numpy as np #Numpy
import matplotlib.pyplot as plt #Matplotlib
from mpl_toolkits.mplot3d import Axes3D

def PlotSurface(Surface = None, ax = None):
    #Make the Rectangle, start from the top right coner, and move counter clockwise
    SurfRect = np.append(Surface,[Surface[0]],axis = 0)

    cx = [c[0] for c in SurfRect]
    cy = [c[1] for c in SurfRect]
    cz = [c[2] for c in SurfRect]
    ax.plot(cx, cy, cz)


def PlotNLPStep(x_opt = None, fig=None, var_index=None, PL_init = None, PR_init = None, LeftSwing = None, RightSwing = None, AllSurfaces = None):
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
    #print('x_res: ',x_res)
    y_res = x_opt[var_index["y"][0]:var_index["y"][1]+1]
    y_res = np.array(y_res)
    #print('y_res: ',y_res)
    z_res = x_opt[var_index["z"][0]:var_index["z"][1]+1]
    z_res = np.array(z_res)
    #print('z_res: ',z_res)
    xdot_res = x_opt[var_index["xdot"][0]:var_index["xdot"][1]+1]
    xdot_res = np.array(xdot_res)
    #print('xdot_res: ',xdot_res)
    ydot_res = x_opt[var_index["ydot"][0]:var_index["ydot"][1]+1]
    ydot_res = np.array(ydot_res)
    #print('ydot_res: ',ydot_res)
    zdot_res = x_opt[var_index["zdot"][0]:var_index["zdot"][1]+1]
    zdot_res = np.array(zdot_res)
    #print('zdot_res: ',zdot_res)
    Lx_res = x_opt[var_index["Lx"][0]:var_index["Lx"][1]+1]
    Lx_res = np.array(Lx_res)
    #print('Lx_res: ',Lx_res)
    Ly_res = x_opt[var_index["Ly"][0]:var_index["Ly"][1]+1]
    Ly_res = np.array(Ly_res)
    #print('Ly_res: ',Ly_res)
    Lz_res = x_opt[var_index["Lz"][0]:var_index["Lz"][1]+1]
    Lz_res = np.array(Lz_res)
    #print('Lz_res: ',Lz_res)
    Ldotx_res = x_opt[var_index["Ldotx"][0]:var_index["Ldotx"][1]+1]
    Ldotx_res = np.array(Ldotx_res)
    #print('Ldotx_res: ',Ldotx_res)
    Ldoty_res = x_opt[var_index["Ldoty"][0]:var_index["Ldoty"][1]+1]
    Ldoty_res = np.array(Ldoty_res)
    #print('Ldoty_res: ',Ldoty_res)
    Ldotz_res = x_opt[var_index["Ldotz"][0]:var_index["Ldotz"][1]+1]
    Ldotz_res = np.array(Ldotz_res)
    #print('Ldotz_res: ',Ldotz_res)
    px_res = x_opt[var_index["px"][0]:var_index["px"][1]+1]
    px_res = np.array(px_res)
    px_res = x_opt[var_index["px"][0]:var_index["px"][1]+1]
    px_res = np.array(px_res)
    #print('px_res: ',px_res)
    py_res = x_opt[var_index["py"][0]:var_index["py"][1]+1]
    py_res = np.array(py_res)
    #print('py_res: ',py_res)
    pz_res = x_opt[var_index["pz"][0]:var_index["pz"][1]+1]
    pz_res = np.array(pz_res)
    #print('pz_res: ',pz_res)
    Ts_res = x_opt[var_index["Ts"][0]:var_index["Ts"][1]+1]
    Ts_res = np.array(Ts_res)
    #print('Ts_res: ',Ts_res)

    #CoM Trajectory
    ax.plot3D(x_res,y_res,z_res,color='blue', linestyle='dashed', linewidth=2, markersize=12)
    #Initial Footstep Locations
    ax.scatter(PL_init[0], PL_init[1], PL_init[2], c='r', marker='o', linewidth = 10) 
    ax.scatter(PR_init[0], PR_init[1], PR_init[2], c='b', marker='o', linewidth = 10) 
    #Swing Foot
    if LeftSwing == 1:
        StepColor = 'r'
    if RightSwing == 1:
        StepColor = 'b'
    ax.scatter(px_res, py_res, pz_res, c=StepColor, marker='o', linewidth = 10) 

    #Pring surfaces
    for surf in AllSurfaces:
        print(surf)
        PlotSurface(Surface = surf, ax = ax)

    ax.set_xlim3d(x_res[0]-0.2, px_res[-1]+0.35)
    ax.set_ylim3d(-0.5,0.5)
    ax.set_zlim3d(0,0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

    return ax

def Plot_Pure_Kinematics_Plan(x_opt = None, fig=None, var_index=None, PL_init = None, PR_init = None, LeftSwing = None, RightSwing = None, AllSurfaces = None):
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
        StepColor = 'b'
        ax.scatter(px_res[0::2], py_res[0::2], pz_res[0::2], c=StepColor, marker='o', linewidth = 10) 
        StepColor = 'r'
        ax.scatter(px_res[1::2], py_res[1::2], pz_res[1::2], c=StepColor, marker='o', linewidth = 10) 
    if RightSwing == 1:
        StepColor = 'r'
        ax.scatter(px_res[0::2], py_res[0::2], pz_res[0::2], c=StepColor, marker='o', linewidth = 10) 
        StepColor = 'b'
        ax.scatter(px_res[1::2], py_res[1::2], pz_res[1::2], c=StepColor, marker='o', linewidth = 10) 

    #Print Surfaces
    for surf in AllSurfaces:
        print(surf)
        PlotSurface(Surface = surf, ax = ax)

    ax.set_xlim3d(x_res[0]-0.2, px_res[-1]+0.35)
    ax.set_ylim3d(-0.5,0.5)
    ax.set_zlim3d(0,0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

    return ax

def Plot_Both_Levels(x_opt = None, fig=None, var_index=None, PL_init = None, PR_init = None, LeftSwing = None, RightSwing = None, AllSurfaces = None):

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
    #print('x_res: ',x_res)
    y_res = x_opt[var_index["y"][0]:var_index["y"][1]+1]
    y_res = np.array(y_res)
    #print('y_res: ',y_res)
    z_res = x_opt[var_index["z"][0]:var_index["z"][1]+1]
    z_res = np.array(z_res)
    #print('z_res: ',z_res)
    xdot_res = x_opt[var_index["xdot"][0]:var_index["xdot"][1]+1]
    xdot_res = np.array(xdot_res)
    #print('xdot_res: ',xdot_res)
    ydot_res = x_opt[var_index["ydot"][0]:var_index["ydot"][1]+1]
    ydot_res = np.array(ydot_res)
    #print('ydot_res: ',ydot_res)
    zdot_res = x_opt[var_index["zdot"][0]:var_index["zdot"][1]+1]
    zdot_res = np.array(zdot_res)
    #print('zdot_res: ',zdot_res)
    Lx_res = x_opt[var_index["Lx"][0]:var_index["Lx"][1]+1]
    Lx_res = np.array(Lx_res)
    #print('Lx_res: ',Lx_res)
    Ly_res = x_opt[var_index["Ly"][0]:var_index["Ly"][1]+1]
    Ly_res = np.array(Ly_res)
    #print('Ly_res: ',Ly_res)
    Lz_res = x_opt[var_index["Lz"][0]:var_index["Lz"][1]+1]
    Lz_res = np.array(Lz_res)
    #print('Lz_res: ',Lz_res)
    Ldotx_res = x_opt[var_index["Ldotx"][0]:var_index["Ldotx"][1]+1]
    Ldotx_res = np.array(Ldotx_res)
    #print('Ldotx_res: ',Ldotx_res)
    Ldoty_res = x_opt[var_index["Ldoty"][0]:var_index["Ldoty"][1]+1]
    Ldoty_res = np.array(Ldoty_res)
    #print('Ldoty_res: ',Ldoty_res)
    Ldotz_res = x_opt[var_index["Ldotz"][0]:var_index["Ldotz"][1]+1]
    Ldotz_res = np.array(Ldotz_res)
    #print('Ldotz_res: ',Ldotz_res)
    px_res = x_opt[var_index["px"][0]:var_index["px"][1]+1]
    px_res = np.array(px_res)
    #print('px_res: ',px_res)
    py_res = x_opt[var_index["py"][0]:var_index["py"][1]+1]
    py_res = np.array(py_res)
    #print('py_res: ',py_res)
    pz_res = x_opt[var_index["pz"][0]:var_index["pz"][1]+1]
    pz_res = np.array(pz_res)
    #print('pz_res: ',pz_res)
    Ts_res = x_opt[var_index["Ts"][0]:var_index["Ts"][1]+1]
    Ts_res = np.array(Ts_res)
    #print('Ts_res: ',Ts_res)

    x_start = x_res[0]

    #CoM Trajectory
    ax.plot3D(x_res,y_res,z_res,color='blue', linestyle='dashed', linewidth=2, markersize=12)
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

    if var_index: #is any

        x_res = x_opt[var_index["x"][0]:var_index["x"][1]+1]
        x_res = np.array(x_res)
        #print('x_res: ',x_res)
        y_res = x_opt[var_index["y"][0]:var_index["y"][1]+1]
        y_res = np.array(y_res)
        #print('y_res: ',y_res)
        z_res = x_opt[var_index["z"][0]:var_index["z"][1]+1]
        z_res = np.array(z_res)
        #print('z_res: ',z_res)
        px_res = x_opt[var_index["px"][0]:var_index["px"][1]+1]
        px_res = np.array(px_res)
        #print('px_res: ',px_res)
        py_res = x_opt[var_index["py"][0]:var_index["py"][1]+1]
        py_res = np.array(py_res)
        #print('py_res: ',py_res)
        pz_res = x_opt[var_index["pz"][0]:var_index["pz"][1]+1]
        pz_res = np.array(pz_res)
        #print('pz_res: ',pz_res)

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


    #Print Surfaces
    for surf in AllSurfaces:
        print(surf)
        PlotSurface(Surface = surf, ax = ax)


    ax.set_xlim3d(x_start-0.2, px_res[-1]+0.35)
    ax.set_ylim3d(-0.5,0.5)
    ax.set_zlim3d(0,0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def PlotSingleOptimiation_and_PrintResult(x_opt = None, var_index=None, PL_init = None, PR_init = None, LeftSwing = None, RightSwing = None, PrintSecondLevel = False, PlotNLP = None, PlotBothLevel = None, AllSurfaces = None):
    #Print Result
    #First Level
    
    print("First Level NLP Results:")

    #save the complete var_index
    var_index_complete =var_index
    #get the first level var_index
    var_index = var_index_complete["Level1_Var_Index"]

    x_res = x_opt[var_index["x"][0]:var_index["x"][1]+1]
    x_res = np.array(x_res)
    print('x_res: ',x_res)

    y_res = x_opt[var_index["y"][0]:var_index["y"][1]+1]
    y_res = np.array(y_res)
    print('y_res: ',y_res)

    z_res = x_opt[var_index["z"][0]:var_index["z"][1]+1]
    z_res = np.array(z_res)
    print('z_res: ',z_res)

    xdot_res = x_opt[var_index["xdot"][0]:var_index["xdot"][1]+1]
    xdot_res = np.array(xdot_res)
    print('xdot_res: ',xdot_res)

    ydot_res = x_opt[var_index["ydot"][0]:var_index["ydot"][1]+1]
    ydot_res = np.array(ydot_res)
    print('ydot_res: ',ydot_res)

    zdot_res = x_opt[var_index["zdot"][0]:var_index["zdot"][1]+1]
    zdot_res = np.array(zdot_res)
    print('zdot_res: ',zdot_res)

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

    Ts_res = x_opt[var_index["Ts"][0]:var_index["Ts"][1]+1]
    Ts_res = np.array(Ts_res)
    print('Ts_res: ',Ts_res)

    FR1x_res = x_opt[var_index["FR1x"][0]:var_index["FR1x"][1]+1]
    FR1x_res = np.array(FR1x_res)
    print('FR1x_res: ',FR1x_res)

    FR1y_res = x_opt[var_index["FR1y"][0]:var_index["FR1y"][1]+1]
    FR1y_res = np.array(FR1y_res)
    print('FR1y_res: ',FR1y_res)

    FR1z_res = x_opt[var_index["FR1z"][0]:var_index["FR1z"][1]+1]
    FR1z_res = np.array(FR1z_res)
    print('FR1z_res: ',FR1z_res)

    #Backup the complete optimization result
    x_opt_complete = x_opt
    
    
    if PrintSecondLevel == True and var_index_complete["Level2_Var_Index"]:

        
        #Get Second Level result
        x_opt = x_opt[var_index["Ts"][1]+1:]
        
        #Get Second Level Result Index
        var_index = var_index_complete["Level2_Var_Index"]

        x_res = x_opt[var_index["x"][0]:var_index["x"][1]+1]
        x_res = np.array(x_res)

        xdot_res = x_opt[var_index["xdot"][0]:var_index["xdot"][1]+1]
        xdot_res = np.array(xdot_res)

        y_res = x_opt[var_index["y"][0]:var_index["y"][1]+1]
        y_res = np.array(y_res)

        ydot_res = x_opt[var_index["ydot"][0]:var_index["ydot"][1]+1]
        ydot_res = np.array(ydot_res)

        z_res = x_opt[var_index["z"][0]:var_index["z"][1]+1]
        z_res = np.array(z_res)

        zdot_res = x_opt[var_index["zdot"][0]:var_index["zdot"][1]+1]
        zdot_res = np.array(zdot_res)

        px_res = x_opt[var_index["px"][0]:var_index["px"][1]+1]
        px_res = np.array(px_res)

        py_res = x_opt[var_index["py"][0]:var_index["py"][1]+1]
        py_res = np.array(py_res)

        pz_res = x_opt[var_index["pz"][0]:var_index["pz"][1]+1]
        pz_res = np.array(pz_res)

        Ts_res = x_opt[var_index["Ts"][0]:var_index["Ts"][1]+1]
        Ts_res = np.array(Ts_res)

        if PrintSecondLevel == True:
            print("Second Level Results:")
            print('x_res: ',x_res)
            print('y_res: ',y_res)
            print('z_res: ',z_res)
            print('xdot_res: ',xdot_res)
            print('ydot_res: ',ydot_res)
            print('zdot_res: ',zdot_res)
            print('px_res: ',px_res)
            print('py_res: ',py_res)
            print('pz_res: ',pz_res)
            print('Ts_res: ',Ts_res)


    if PlotBothLevel == True:
        Plot_Both_Levels(x_opt=x_opt_complete,fig=None,var_index=var_index_complete,PL_init=PL_init,PR_init = PR_init, LeftSwing = LeftSwing, RightSwing = RightSwing, AllSurfaces = AllSurfaces)

    if PlotNLP == True:
        PlotNLPStep(x_opt=x_opt_complete,fig=None,var_index=var_index_complete,PL_init=PL_init,PR_init = PR_init, LeftSwing = LeftSwing, RightSwing = RightSwing, AllSurfaces = AllSurfaces)


def Plot_RHP_result(NumRounds = None, SwingLeftFirst = None, SwingRightFirst = None, x_fullres = None, y_fullres = None, z_fullres = None, PL_init_fullres = None, PR_init_fullres = None, Px_fullres = None, Py_fullres = None, Pz_fullres = None, AllSurfaces = None):
    #Plot Final Result
    fig=plt.figure()
    ax = Axes3D(fig)

    #Initial Footstep Locations
    ax.scatter(PL_init_fullres[0], PL_init_fullres[1], PL_init_fullres[2], c='r', marker='o', linewidth = 10) 
    ax.scatter(PR_init_fullres[0], PR_init_fullres[1], PR_init_fullres[2], c='b', marker='o', linewidth = 10) 

    for roundNum in range(NumRounds):
        #Decide Colors
        if roundNum%2 == 0:
            CoM_Color = 'green'
            
            if SwingLeftFirst == 1: #Even round, swing left
                P_Color = "red"
            elif SwingRightFirst == 1: #Even round, swing right
                P_Color = "blue"

        elif roundNum%2 == 1:
            CoM_Color = 'blue'

            if SwingLeftFirst == 1: #odd round, swing right
                P_Color = "blue"
            elif SwingRightFirst == 1: #odd round, swing left
                P_Color = "red"
        #Plot CoM
        ax.plot3D(x_fullres[roundNum],y_fullres[roundNum],z_fullres[roundNum],color=CoM_Color, linestyle='dashed', linewidth=2, markersize=12)
        #Plot Step
        ax.scatter(Px_fullres[roundNum], Py_fullres[roundNum], Pz_fullres[roundNum], c=P_Color, marker='o', linewidth = 10) 

    #Draw Terrain

    for surf in AllSurfaces:
        print(surf)
        PlotSurface(Surface = surf, ax = ax)

    ax.set_xlim3d(x_fullres[0][0]-0.2, x_fullres[-1][-1]+0.35)
    ax.set_ylim3d(-0.5,0.5)
    ax.set_zlim3d(0,0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def PlotFirstLevelAcceleration(Nk_Local=5, x_opt = None, fig=None, var_index=None, G = 9.80665, m = 95, plotAxis = "x"):
    #-----------------------------------------------------------------------------------------------------------------------
    #Plot Result
    #if fig==None:
    #    fig=plt.figure()
    #
    #ax = Axes3D(fig)

    #Number of NLP Phases
    NLPphase=3

    #parameter setup
    N_K = Nk_Local*NLPphase + 1
    #Time Span Setup
    tau_upper_limit = 1
    tauStepLength = tau_upper_limit/(N_K-1) #Get the interval length, total number of knots - 1

    var_index = var_index["Level1_Var_Index"]

    #Get Forces
    FL1x_res = x_opt[var_index["FL1x"][0]:var_index["FL1x"][1]+1]
    FL1x_res = np.array(FL1x_res)
    FL1y_res = x_opt[var_index["FL1y"][0]:var_index["FL1y"][1]+1]
    FL1y_res = np.array(FL1y_res)
    FL1z_res = x_opt[var_index["FL1z"][0]:var_index["FL1z"][1]+1]
    FL1z_res = np.array(FL1z_res)

    FL2x_res = x_opt[var_index["FL2x"][0]:var_index["FL2x"][1]+1]
    FL2x_res = np.array(FL2x_res)
    FL2y_res = x_opt[var_index["FL2y"][0]:var_index["FL2y"][1]+1]
    FL2y_res = np.array(FL2y_res)
    FL2z_res = x_opt[var_index["FL2z"][0]:var_index["FL2z"][1]+1]
    FL2z_res = np.array(FL2z_res)

    FL3x_res = x_opt[var_index["FL3x"][0]:var_index["FL3x"][1]+1]
    FL3x_res = np.array(FL3x_res)
    FL3y_res = x_opt[var_index["FL3y"][0]:var_index["FL3y"][1]+1]
    FL3y_res = np.array(FL3y_res)
    FL3z_res = x_opt[var_index["FL3z"][0]:var_index["FL3z"][1]+1]
    FL3z_res = np.array(FL3z_res)

    FL4x_res = x_opt[var_index["FL4x"][0]:var_index["FL4x"][1]+1]
    FL4x_res = np.array(FL4x_res)
    FL4y_res = x_opt[var_index["FL4y"][0]:var_index["FL4y"][1]+1]
    FL4y_res = np.array(FL4y_res)
    FL4z_res = x_opt[var_index["FL4z"][0]:var_index["FL4z"][1]+1]
    FL4z_res = np.array(FL4z_res)

    FR1x_res = x_opt[var_index["FR1x"][0]:var_index["FR1x"][1]+1]
    FR1x_res = np.array(FR1x_res)
    FR1y_res = x_opt[var_index["FR1y"][0]:var_index["FR1y"][1]+1]
    FR1y_res = np.array(FR1y_res)
    FR1z_res = x_opt[var_index["FR1z"][0]:var_index["FR1z"][1]+1]
    FR1z_res = np.array(FR1z_res)

    FR2x_res = x_opt[var_index["FR2x"][0]:var_index["FR2x"][1]+1]
    FR2x_res = np.array(FR2x_res)
    FR2y_res = x_opt[var_index["FR2y"][0]:var_index["FR2y"][1]+1]
    FR2y_res = np.array(FR2y_res)
    FR2z_res = x_opt[var_index["FR2z"][0]:var_index["FR2z"][1]+1]
    FR2z_res = np.array(FR2z_res)

    FR3x_res = x_opt[var_index["FR3x"][0]:var_index["FR3x"][1]+1]
    FR3x_res = np.array(FR3x_res)
    FR3y_res = x_opt[var_index["FR3y"][0]:var_index["FR3y"][1]+1]
    FR3y_res = np.array(FR3y_res)
    FR3z_res = x_opt[var_index["FR3z"][0]:var_index["FR3z"][1]+1]
    FR3z_res = np.array(FR3z_res)

    FR4x_res = x_opt[var_index["FR4x"][0]:var_index["FR4x"][1]+1]
    FR4x_res = np.array(FR4x_res)
    FR4y_res = x_opt[var_index["FR4y"][0]:var_index["FR4y"][1]+1]
    FR4y_res = np.array(FR4y_res)
    FR4z_res = x_opt[var_index["FR4z"][0]:var_index["FR4z"][1]+1]
    FR4z_res = np.array(FR4z_res)

    Ts_res = x_opt[var_index["Ts"][0]:var_index["Ts"][1]+1]
    Ts_res = np.array(Ts_res)


    Accx = []
    Accy = []
    Accz = []

    TimeSeries = []

    t = 0
    #Loop over all Phases (Knots)
    for Nph in range(NLPphase):
        
        #Decide Number of Knots
        if Nph == NLPphase-1:  #The last Knot belongs to the Last Phase
            Nk_ThisPhase = Nk_Local+1
        else:
            Nk_ThisPhase = Nk_Local

        #Decide Time Vector
        if Nph == 0: #first phase
            h = tauStepLength*NLPphase*(Ts_res[Nph]-0)
        else: #other phases
            h = tauStepLength*NLPphase*(Ts_res[Nph]-Ts_res[Nph-1])

        for Local_k_Count in range(Nk_ThisPhase):
            #Get knot number across the entire time line
            k = Nph*Nk_Local + Local_k_Count
            #print(k)

            #Add Cost Terms
            if k < N_K - 1:
                
                #make time vector
                TimeSeries.append(t)
                t = t + h

                Accx.append((FL1x_res[k]+FL2x_res[k]+FL3x_res[k]+FL4x_res[k]+FR1x_res[k]+FR2x_res[k]+FR3x_res[k]+FR4x_res[k])/m)
                Accy.append((FL1y_res[k]+FL2y_res[k]+FL3y_res[k]+FL4y_res[k]+FR1y_res[k]+FR2y_res[k]+FR3y_res[k]+FR4y_res[k])/m)
                Accz.append((FL1z_res[k]+FL2z_res[k]+FL3z_res[k]+FL4z_res[k]+FR1z_res[k]+FR2z_res[k]+FR3z_res[k]+FR4z_res[k])/m - G)

    if plotAxis == "x":
        plt.plot(TimeSeries,Accx)
        plt.ylabel('Acc X')
    elif plotAxis == "y":
        plt.plot(TimeSeries,Accy)
        plt.ylabel('Acc Y')
    elif plotAxis == "z":
        plt.plot(TimeSeries,Accz)
        plt.ylabel('Acc Z')
    
    plt.xlabel('Time')


    plt.show()