#Receding Horizon Planning Framework
import numpy as np #Numpy
import casadi as ca #Casadi
import matplotlib.pyplot as plt #Matplotlib
from mpl_toolkits.mplot3d import Axes3D
# Import SL1M modules
from sl1m.constants_and_tools import *
from sl1m.planner import *
from constraints import *
from Humanoid_ProblemDescription import *
from PlotResult import *
from Tools import *

#Initialization and Porblem Setup

#   Set Decimal Printing Precision
np.set_printoptions(precision=4)

#Terrain Definition/Surface Sequence
#Define Patches
#NOTE: The rectangle always stat from Top Right Corner, and the vertex move counterclockwise, it should be a list of numpy arrays
Patch1 = np.array([[0.5, 0.5, 0.], [-0.1, 0.5, 0.], [-0.1, -0.5, 0.], [0.5, -0.5, 0.]])
Patch2 = np.array([[5, 0.5, 0.], [-5, 0.5, 0.], [-5, -0.5, 0.], [5, -0.5, 0.]])
#Collect all patches for the final printing of the terrain
AllPatches = [Patch1,Patch2]
#Collect patche sequences, number of rows equals number of rounds
#ContactSeqs = [[Patch1,Patch2,Patch2,Patch2,Patch2],
#               [Patch2,Patch2,Patch2,Patch2,Patch2],
#               [Patch2,Patch2,Patch2,Patch2,Patch2],
#               [Patch2,Patch2,Patch2,Patch2,Patch2],
#               [Patch2,Patch2,Patch2,Patch2,Patch2],
#               [Patch2,Patch2,Patch2,Patch2,Patch2],
#               [Patch2,Patch2,Patch2,Patch2,Patch2],
#               [Patch2,Patch2,Patch2,Patch2,Patch2],
#               [Patch2,Patch2,Patch2,Patch2,Patch2],
#               [Patch2,Patch2,Patch2,Patch2,Patch2]]

ContactSeqs = [[Patch2,Patch2,Patch2,Patch2],
               [Patch2,Patch2,Patch2,Patch2],
               [Patch2,Patch2,Patch2,Patch2],
               [Patch2,Patch2,Patch2,Patch2],
               [Patch2,Patch2,Patch2,Patch2],
               [Patch2,Patch2,Patch2,Patch2],
               [Patch2,Patch2,Patch2,Patch2],
               [Patch2,Patch2,Patch2,Patch2],
               [Patch2,Patch2,Patch2,Patch2],
               [Patch2,Patch2,Patch2,Patch2]]

#   Define the Swing foot of the First Step
SwingLeftFirst = 1
SwingRightFirst = 0

#   Number of Rounds
#Nrounds = 15
Nrounds = len(ContactSeqs)

#   Initial Condition of the Robot
x_init = 0.0
y_init = 0.0
z_init = 0.6

xdot_init = 0.0
ydot_init = 0
zdot_init = 0

PLx_init = 0
PLy_init = 0.1
PLz_init = 0

PRx_init = 0
PRy_init = -0.1
PRz_init = 0

x_end = 3
y_end = 0
z_end = 0.6

xdot_end = 0
ydot_end = 0
zdot_end = 0

#Make Complete Result Container
PL_init_fullres = [PLx_init,PLy_init,PLz_init]
PR_init_fullres = [PRx_init,PRy_init,PRz_init]

x_fullres = []
y_fullres = []
z_fullres = []

xdot_fullres = []

Px_fullres = []
Py_fullres = []
Pz_fullres = []
costs = []

#StopRound initialisation
StopRound = Nrounds

#Number of Lookahead Steps per round
Nstep_lookahead = len(ContactSeqs[0])

#   Build Solver fo normal rounds (Start from Swing Phase)
#solver_init, DecisionVars_lb_init, DecisionVars_ub_init, glb_init, gub_init, var_index_init = BuildSolver(FirstLevel = "NLP_SingleStep", SecondLevel = "CoM_Dynamics", ConservativeFirstStep = False, m = 95,NumSurfaces = Nstep_lookahead,FirstRoundFlag=True)
#solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index = BuildSolver(FirstLevel = "NLP_SingleStep", SecondLevel = "CoM_Dynamics", ConservativeFirstStep = True, m = 95, NumSurfaces = Nstep_lookahead,FirstRoundFlag=True)
#solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index = BuildSolver(FirstLevel = "NLP_SingleStep", SecondLevel = "Pure_Kinematics_Check", ConservativeFirstStep = True, m = 95, NumSurfaces = Nstep_lookahead,FirstRoundFlag=True)
#solver_init, DecisionVars_lb_init, DecisionVars_ub_init, glb_init, gub_init, var_index_init = BuildSolver(FirstLevel = "NLP_SingleStep", SecondLevel = None, ConservativeFirstStep = False, m = 95,NumSurfaces = Nstep_lookahead,FirstRoundFlag=True)
solver_init, DecisionVars_lb_init, DecisionVars_ub_init, glb_init, gub_init, var_index_init = BuildSolver(FirstLevel = "NLP_SingleStep", SecondLevel = "NLP_SecondLevel", ConservativeFirstStep = False, m = 95, NumSurfaces = Nstep_lookahead,FirstRoundFlag=True)

#   Build Solver for the First Round (Start from Initial Double Support)
#solver_normal, DecisionVars_lb_normal, DecisionVars_ub_normal, glb_normal, gub_normal, var_index_normal = BuildSolver(FirstLevel = "NLP_SingleStep", SecondLevel = "CoM_Dynamics", ConservativeFirstStep = False, m = 95,NumSurfaces = Nstep_lookahead,FirstRoundFlag=False)
#solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index = BuildSolver(FirstLevel = "NLP_SingleStep", SecondLevel = "CoM_Dynamics", ConservativeFirstStep = True, m = 95, NumSurfaces = Nstep_lookahead,FirstRoundFlag=False)
#solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index = BuildSolver(FirstLevel = "NLP_SingleStep", SecondLevel = "Pure_Kinematics_Check", ConservativeFirstStep = True, m = 95, NumSurfaces = Nstep_lookahead,FirstRoundFlag=False)
#solver_normal, DecisionVars_lb_normal, DecisionVars_ub_normal, glb_normal, gub_normal, var_index_normal = BuildSolver(FirstLevel = "NLP_SingleStep", SecondLevel = None, ConservativeFirstStep = False, m = 95,NumSurfaces = Nstep_lookahead,FirstRoundFlag=False)
solver_normal, DecisionVars_lb_normal, DecisionVars_ub_normal, glb_normal, gub_normal, var_index_normal = BuildSolver(FirstLevel = "NLP_SingleStep", SecondLevel = "NLP_SecondLevel", ConservativeFirstStep = False, m = 95, NumSurfaces = Nstep_lookahead,FirstRoundFlag=False)


#Main For Loop
for roundNum in range(Nrounds):
    
    #Moving Targets
    #x_end = x_end + 0.2 #fixed time #0.5 - variable time

    print("The ", roundNum, "Round:")

    if roundNum == 0:

        #Switch Solver
        solver = solver_init

        #   Generate Initial Guess
        #   Random Initial Guess
        #       Shuffle the Random Seed Generator
        np.random.seed()
        DecisionVarsShape = DecisionVars_lb_init.shape
        DecisionVarsStart_init = DecisionVars_lb_init + np.multiply(np.random.rand(DecisionVarsShape[0],).flatten(),(DecisionVars_ub_init-DecisionVars_lb_init))#   Fixed Value Initial Guess

        #   backup var_index
        var_index_Level1 = var_index_init["Level1_Var_Index"]

        #switch var index
        var_index = var_index_init

        if SwingLeftFirst == 1:
            #Swing the Left
            LeftSwingFlag = 1
            RightSwingFlag = 0
        elif SwingRightFirst == 1:
            #Swing the Right
            LeftSwingFlag = 0
            RightSwingFlag = 1
        
        ##FirstRoundFlag
        #FirstRoundFlag = 1
        
        #Get Parameters for Half-space representation of Patches
        #Convert to Half Space Representation
        Patches = ContactSeqs[roundNum]
        HalfSpaceSeq = []
        for patch in Patches:
            #NOTE: Rotate the Raw Surfaces Sequences
            HalfSpacePatch = convert_surface_to_inequality(patch.T)
            HalfSpacePatch = np.concatenate(HalfSpacePatch,axis=None)
            HalfSpaceSeq.append(HalfSpacePatch)
        #Lamp all variables together
        HalfSpaceSeq = np.concatenate(HalfSpaceSeq,axis=None)
        #print(HalfSpaceSeq)
        
        ParaList = np.concatenate((LeftSwingFlag,RightSwingFlag,
            x_init,y_init,z_init,
            xdot_init,ydot_init,zdot_init,
            PLx_init,PLy_init,PLz_init,
            PRx_init,PRy_init,PRz_init,
            x_end,y_end,z_end,
            xdot_end,ydot_end,zdot_end,HalfSpaceSeq),axis=None)

        res = solver(x0=DecisionVarsStart_init, p = ParaList, lbx = DecisionVars_lb_init, ubx = DecisionVars_ub_init, lbg = glb_init, ubg = gub_init)

        x_opt = res["x"]
        x_opt = x_opt.full().flatten()

        x_opt_left = x_opt
        x_opt_right = x_opt

        #PlotSingleOptimiation_and_PrintResult(x_opt = x_opt, var_index=var_index, PL_init = np.array([PLx_init,PLy_init,PLz_init]), PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag, PrintSecondLevel = False, PlotNLP = True, PlotBothLevel = True)

    elif roundNum > 0:

        #Switch Solver
        solver = solver_normal

        #   Generate Initial Guess
        #   Random Initial Guess
        #       Shuffle the Random Seed Generator
        np.random.seed()
        DecisionVarsShape = DecisionVars_lb_normal.shape
        DecisionVarsStart_normal = DecisionVars_lb_normal + np.multiply(np.random.rand(DecisionVarsShape[0],).flatten(),(DecisionVars_ub_normal-DecisionVars_lb_normal))#   Fixed Value Initial Guess

        #   backup var_index
        if roundNum == 1:
            var_index_Level1 = var_index_init["Level1_Var_Index"]
        else:
            var_index_Level1 = var_index_normal["Level1_Var_Index"]

        #switch var index
        var_index = var_index_normal

        #Update Initial Condition
        x_res = x_opt[var_index_Level1["x"][0]:var_index_Level1["x"][1]+1]
        x_init = x_res[-1]
        y_res = x_opt[var_index_Level1["y"][0]:var_index_Level1["y"][1]+1]
        y_init = y_res[-1]
        z_res = x_opt[var_index_Level1["z"][0]:var_index_Level1["z"][1]+1]
        z_init = z_res[-1]

        xdot_res  = x_opt[var_index_Level1["xdot"][0]:var_index_Level1["xdot"][1]+1]
        xdot_init = xdot_res[-1]
        ydot_res  = x_opt[var_index_Level1["ydot"][0]:var_index_Level1["ydot"][1]+1]
        ydot_init = ydot_res[-1]
        zdot_res  = x_opt[var_index_Level1["zdot"][0]:var_index_Level1["zdot"][1]+1]
        zdot_init = zdot_res[-1]

        if SwingLeftFirst == 1:
            if roundNum%2 == 0:#Even (The First phase)
                #Swing the Left
                LeftSwingFlag = 1
                RightSwingFlag = 0
                
                #Update Initial Foot Location 
                px_res = x_opt[var_index_Level1["px"][0]:var_index_Level1["px"][1]+1]
                PRx_init = px_res[-1]
                py_res = x_opt[var_index_Level1["py"][0]:var_index_Level1["py"][1]+1]
                PRy_init = py_res[-1]
                pz_res = x_opt[var_index_Level1["pz"][0]:var_index_Level1["pz"][1]+1]
                PRz_init = pz_res[-1]

            elif roundNum%2 == 1:#odd (The Second phase)
                #Swing the Right
                LeftSwingFlag = 0
                RightSwingFlag = 1         

                #Update Initial Foot Location
                px_res = x_opt[var_index_Level1["px"][0]:var_index_Level1["px"][1]+1]
                PLx_init = px_res[-1]
                py_res = x_opt[var_index_Level1["py"][0]:var_index_Level1["py"][1]+1]
                PLy_init = py_res[-1]
                pz_res = x_opt[var_index_Level1["pz"][0]:var_index_Level1["pz"][1]+1]
                PLz_init = pz_res[-1]      

        elif SwingRightFirst == 1:
            if roundNum%2 == 0:#Even (The First phase)
                #Swing the Right
                LeftSwingFlag = 0
                RightSwingFlag = 1
                
                #Update Initial Foot Location 
                px_res = x_opt[var_index_Level1["px"][0]:var_index_Level1["px"][1]+1]
                PLx_init = px_res[-1]
                py_res = x_opt[var_index_Level1["py"][0]:var_index_Level1["py"][1]+1]
                PLy_init = py_res[-1]
                pz_res = x_opt[var_index_Level1["pz"][0]:var_index_Level1["pz"][1]+1]
                PLz_init = pz_res[-1]

            elif roundNum%2 == 1:#odd (The Second phase)
                #Swing the Left
                LeftSwingFlag = 1
                RightSwingFlag = 0         

                #Update Initial Foot Location
                px_res = x_opt[var_index_Level1["px"][0]:var_index_Level1["px"][1]+1]
                PRx_init = px_res[-1]
                py_res = x_opt[var_index_Level1["py"][0]:var_index_Level1["py"][1]+1]
                PRy_init = py_res[-1]
                pz_res = x_opt[var_index_Level1["pz"][0]:var_index_Level1["pz"][1]+1]
                PRz_init = pz_res[-1]            

        #Get Parameters for Half-space representation of Patches
        #Convert to Half Space Representation
        Patches = ContactSeqs[roundNum]
        HalfSpaceSeq = []
        for patch in Patches:
            #NOTE: Rotate the Raw Surfaces Sequences
            HalfSpacePatch = convert_surface_to_inequality(patch.T)
            HalfSpacePatch = np.concatenate(HalfSpacePatch,axis=None)
            HalfSpaceSeq.append(HalfSpacePatch)
        #Lamp all variables together
        HalfSpaceSeq = np.concatenate(HalfSpaceSeq,axis=None)
        #print(HalfSpaceSeq)

        ##FirstRoundFlag
        #FirstRoundFlag = 0

        #Build Parameter Vector
        ParaList = np.concatenate((LeftSwingFlag,RightSwingFlag,
            x_init,y_init,z_init,
            xdot_init,ydot_init,zdot_init,
            PLx_init,PLy_init,PLz_init,
            PRx_init,PRy_init,PRz_init,
            x_end,y_end,z_end,
            xdot_end,ydot_end,zdot_end,HalfSpaceSeq),axis=None)

        if SwingLeftFirst == 1:
            if roundNum%2 == 0:#Even (The First phase)
                if roundNum > 1:
                    #Swing the Left
                    res = solver(x0=x_opt_left, p = ParaList, lbx = DecisionVars_lb_normal, ubx = DecisionVars_ub_normal, lbg = glb_normal, ubg = gub_normal)
                else:
                    res = solver(x0=DecisionVarsStart_normal, p = ParaList, lbx = DecisionVars_lb_normal, ubx = DecisionVars_ub_normal, lbg = glb_normal, ubg = gub_normal)
                #res = solver(x0=DecisionVars_init, p = ParaList, lbx = DecisionVars_lb, ubx = DecisionVars_ub, lbg = glb, ubg = gub)
                x_opt = res["x"]
                x_opt = x_opt.full().flatten()  
                x_opt_left = x_opt

            elif roundNum%2 == 1:#odd (The Second phase)
                if roundNum > 1:
                    #Swing the Right
                    res = solver(x0=x_opt_right, p = ParaList, lbx = DecisionVars_lb_normal, ubx = DecisionVars_ub_normal, lbg = glb_normal, ubg = gub_normal)
                else:
                    res = solver(x0=DecisionVarsStart_normal, p = ParaList, lbx = DecisionVars_lb_normal, ubx = DecisionVars_ub_normal, lbg = glb_normal, ubg = gub_normal)
                #res = solver(x0=DecisionVars_init, p = ParaList, lbx = DecisionVars_lb, ubx = DecisionVars_ub, lbg = glb, ubg = gub)
                x_opt = res["x"]
                x_opt = x_opt.full().flatten()  
                x_opt_right = x_opt     

        elif SwingRightFirst == 1:
            if roundNum%2 == 0:#Even (The First phase)
                if roundNum > 1:
                    #Swing the Right
                    res = solver(x0=x_opt_right, p = ParaList, lbx = DecisionVars_lb_normal, ubx = DecisionVars_ub_normal, lbg = glb_normal, ubg = gub_normal)
                else:
                    res = solver(x0=DecisionVarsStart_normal, p = ParaList, lbx = DecisionVars_lb_normal, ubx = DecisionVars_ub_normal, lbg = glb_normal, ubg = gub_normal)
                #res = solver(x0=DecisionVars_init, p = ParaList, lbx = DecisionVars_lb, ubx = DecisionVars_ub, lbg = glb, ubg = gub)
                x_opt = res["x"]
                x_opt = x_opt.full().flatten()  
                x_opt_right = x_opt     

            elif roundNum%2 == 1:#odd (The Second phase)
                if roundNum > 1:
                    #Swing the Left
                    res = solver(x0=x_opt_left, p = ParaList, lbx = DecisionVars_lb_normal, ubx = DecisionVars_ub_normal, lbg = glb_normal, ubg = gub_normal)
                else:
                    res = solver(x0=DecisionVarsStart_normal, p = ParaList, lbx = DecisionVars_lb_normal, ubx = DecisionVars_ub_normal, lbg = glb_normal, ubg = gub_normal)
                #res = solver(x0=DecisionVars_init, p = ParaList, lbx = DecisionVars_lb, ubx = DecisionVars_ub, lbg = glb, ubg = gub)
                x_opt = res["x"]
                x_opt = x_opt.full().flatten()  
                x_opt_left = x_opt
     
    if solver.stats()["success"] == True:
        print("Round ", roundNum, solver.stats()["success"])
        #save result
        x_fullres.append(x_opt[var_index_Level1["x"][0]:var_index_Level1["x"][1]+1])
        #print(x_fullres)
        y_fullres.append(x_opt[var_index_Level1["y"][0]:var_index_Level1["y"][1]+1])
        z_fullres.append(x_opt[var_index_Level1["z"][0]:var_index_Level1["z"][1]+1])
        xdot_res_temp = x_opt[var_index_Level1["xdot"][0]:var_index_Level1["xdot"][1]+1]
        xdot_fullres.append(x_opt[var_index_Level1["xdot"][0]:var_index_Level1["xdot"][1]+1])
        Px_fullres.append(x_opt[var_index_Level1["px"][0]:var_index_Level1["px"][1]+1])
        Py_fullres.append(x_opt[var_index_Level1["py"][0]:var_index_Level1["py"][1]+1])
        Pz_fullres.append(x_opt[var_index_Level1["pz"][0]:var_index_Level1["pz"][1]+1])
        #Get Cost
        firstLevelCost = FirstLevelCost(x_opt=x_opt,var_index=var_index,G = 9.80665,m=95)
        costs.append(firstLevelCost)

        #Plot xdot of the first Level


    elif solver.stats()["success"] == False:
        StopRound = roundNum
        print("Fail at Round ", roundNum)
        break

    #for two levels
    PlotSingleOptimiation_and_PrintResult(x_opt = x_opt, var_index=var_index, PL_init = np.array([PLx_init,PLy_init,PLz_init]), PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag, PrintSecondLevel = True, PlotNLP = True, PlotBothLevel = True, AllSurfaces = AllPatches)
    
    #plot acceleration curves
    PlotFirstLevelAcceleration(x_opt = x_opt, var_index=var_index, plotAxis = "x")
    PlotFirstLevelAcceleration(x_opt = x_opt, var_index=var_index, plotAxis = "y")
    PlotFirstLevelAcceleration(x_opt = x_opt, var_index=var_index, plotAxis = "z")
    #for NLP only
    #PlotSingleOptimiation_and_PrintResult(x_opt = x_opt, var_index=var_index, PL_init = np.array([PLx_init,PLy_init,PLz_init]), PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag, PrintSecondLevel = False, PlotNLP = True, PlotBothLevel = False, AllSurfaces = AllPatches)


#plot full result
Plot_RHP_result(NumRounds = StopRound, SwingLeftFirst = SwingLeftFirst, SwingRightFirst = SwingRightFirst, x_fullres = x_fullres, y_fullres = y_fullres, z_fullres = z_fullres, PL_init_fullres = PL_init_fullres, PR_init_fullres = PR_init_fullres, Px_fullres = Px_fullres, Py_fullres = Py_fullres, Pz_fullres = Pz_fullres, AllSurfaces = AllPatches)
#Calculate Accumulated Cost
AccumCost = np.sum(costs)
print("Accumulated Cost is: ",AccumCost)