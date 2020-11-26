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
from TerrainGeneration import *
import os as os
import sys
import pickle
from NLP_Ref_Traj_Constructor import *
from getFirstLevelTerminal_State import *

#Initialization and Porblem Setup

#Get Parameters from Command line
TerrainName = sys.argv[1]

InitSeedType = sys.argv[2] #"random" or "previous"

ChosenSolver = sys.argv[3] #"NLP" or "CoM"

NumofLookAhead = int(sys.argv[4])

ShowFigure = sys.argv[5] #"True" or "False"

NumofRounds = int(sys.argv[6])

TrialNum = int(sys.argv[7])

ResultSavingFolder = sys.argv[8]

#Clear the logging file
#open("test.txt", "w").close()

ProvideInitialSeed = False

print("==================================================================")
print("A new Round")
print(" ")

print("Initial Seed Type: ", InitSeedType)
print("Chosen Solver: ", ChosenSolver)
print("Number of LookAhead: ",NumofLookAhead)


#   Set Decimal Printing Precision
np.set_printoptions(precision=4)

#get terrain set up
#AllPatches, ContactSeqs, TerrainTangentsX, TerrainTangentsY, TerrainNorms = TerrainSelection(name = "single_obstacle")
AllPatches, ContactSeqs, TerrainTangentsX, TerrainTangentsY, TerrainNorms = TerrainSelection(name = TerrainName, NumRounds = NumofRounds, NumContactSequence = NumofLookAhead)

#Show terrains
if ShowFigure == "True":
    PlotTerrain(AllSurfaces = AllPatches)

#Initial Contact Tagents and Norms
PL_init_TangentX = np.array([1,0,0])
PL_init_TangentY = np.array([0,1,0])
PL_init_Norm = np.array([0,0,1])
PR_init_TangentX = np.array([1,0,0])
PR_init_TangentY = np.array([0,1,0])
PR_init_Norm = np.array([0,0,1])

#ContactSeqs = [[Patch2,Patch2],
#               [Patch2,Patch2],
#               [Patch2,Patch2],
#               [Patch2,Patch2],
#               [Patch2,Patch2],
#               [Patch2,Patch2],
#               [Patch2,Patch2],
#               [Patch2,Patch2],
#               [Patch2,Patch2],
#               [Patch2,Patch2]]

#   Define the Swing foot of the First Step
SwingLeftFirst = 1
SwingRightFirst = 0

#   Number of Rounds
#Nrounds = 15
Nrounds = len(ContactSeqs)

#   Initial Condition of the Robot
x_init = 0.6
y_init = 0
z_init = 0.75

xdot_init = 0.0
ydot_init = 0
zdot_init = 0

Lx_init = 0
Ly_init = 0
Lz_init = 0

Ldotx_init = 0
Ldoty_init = 0
Ldotz_init = 0

PLx_init = 0.6
PLy_init = 0.1
PLz_init = 0

PRx_init = 0.6
PRy_init = -0.1
PRz_init = 0

x_end = 20
y_end = 0
z_end = 0.75

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
Fullcosts = []
Acc_cost = []
Momentum_Cost = []
MomentumRate_Cost = []
TerminalCost = []
CasadiParameters = []

#Full result container
AllRoundTrajectory = []

#StopRound initialisation
StopRound = Nrounds

#Number of Lookahead Steps per round
Nstep_lookahead = len(ContactSeqs[0])

#   Build Solver fo normal rounds (Start from Swing Phase)
#solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index = BuildSolver(FirstLevel = "NLP_SingleStep", SecondLevel = "CoM_Dynamics", ConservativeFirstStep = False, m = 95,NumSurfaces = Nstep_lookahead)
#solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index = BuildSolver(FirstLevel = "NLP_SingleStep", SecondLevel = "CoM_Dynamics", ConservativeFirstStep = True, m = 95, NumSurfaces = Nstep_lookahead)
#solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index = BuildSolver(FirstLevel = "NLP_SingleStep", SecondLevel = "Pure_Kinematics_Check", ConservativeFirstStep = True, m = 95, NumSurfaces = Nstep_lookahead)
#solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index = BuildSolver(FirstLevel = "NLP_SingleStep", SecondLevel = None, ConservativeFirstStep = False, m = 95,NumSurfaces = Nstep_lookahead)
#solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index = BuildSolver(FirstLevel = "NLP_SingleStep", SecondLevel = "NLP_SecondLevel", ConservativeFirstStep = False, m = 95, NumSurfaces = Nstep_lookahead)

if ChosenSolver == "NLP":
    if NumofLookAhead == 1:
        solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index = BuildSolver(FirstLevel = "NLP_SingleStep", SecondLevel = None, ConservativeFirstStep = False, m = 95,NumSurfaces = Nstep_lookahead)
    else:
        solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index = BuildSolver(FirstLevel = "NLP_SingleStep", SecondLevel = "NLP_SecondLevel", ConservativeFirstStep = False, m = 95, NumSurfaces = Nstep_lookahead)
elif ChosenSolver == "CoM":
    if NumofLookAhead == 1:
        solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index = BuildSolver(FirstLevel = "NLP_SingleStep", SecondLevel = None, ConservativeFirstStep = False, m = 95,NumSurfaces = Nstep_lookahead)
    else:
        solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index = BuildSolver(FirstLevel = "NLP_SingleStep", SecondLevel = "CoM_Dynamics", ConservativeFirstStep = False, m = 95,NumSurfaces = Nstep_lookahead)
elif ChosenSolver == "Mix":
    if NumofLookAhead == 1:
        solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index = BuildSolver(FirstLevel = "NLP_SingleStep", SecondLevel = None, ConservativeFirstStep = False, m = 95,NumSurfaces = Nstep_lookahead)
    else:
        solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index = BuildSolver(FirstLevel = "NLP_SingleStep", SecondLevel = "Mixure", ConservativeFirstStep = False, m = 95, NumSurfaces = Nstep_lookahead)

#   Generate Initial Guess
#   Random Initial Guess
#       Shuffle the Random Seed Generator
np.random.seed()
DecisionVarsShape = DecisionVars_lb.shape
DecisionVars_init = DecisionVars_lb + np.multiply(np.random.rand(DecisionVarsShape[0],).flatten(),(DecisionVars_ub - DecisionVars_lb))#   Fixed Value Initial Guess

#   backup var_index
var_index_Level1 = var_index["Level1_Var_Index"]

#Main For Loop
for roundNum in range(Nrounds):
    
    #Moving Targets
    #x_end = x_end + 0.2 #fixed time #0.5 - variable time

    print("The ", roundNum, "Round:")

    #Get Terminal State of the first level
    x_end_level1,y_end_level1,z_end_level1,xdot_end_level1,ydot_end_level1,zdot_end_level1,px_level1,py_level1,pz_level1,Lx_end_level1,Ly_end_level1,Lz_end_level1 = GetFirstLevelTerminalState(RoundNum = roundNum)
    Level1_ref_traj = GetFirstLevelTrajectory(RoundNum = roundNum)

    if roundNum == 0:

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

        #Get Reference Trajectory
        if ChosenSolver == "CoM":
            x_traj_ref, y_traj_ref, z_traj_ref, xdot_traj_ref, ydot_traj_ref, zdot_traj_ref, FLx_traj_ref, FLy_traj_ref, FLz_traj_ref, FRx_traj_ref, FRy_traj_ref, FRz_traj_ref, Px_seq_ref, Py_seq_ref, Pz_seq_ref, SwitchingTimeVec_ref, px_init_ref, py_init_ref, pz_init_ref = NLP_ref_trajectory_construction(StartStepNum = roundNum, LookAheadSteps = NumofLookAhead)
            #x_traj_ref, y_traj_ref, z_traj_ref, xdot_traj_ref, ydot_traj_ref, zdot_traj_ref, FLx_traj_ref, FLy_traj_ref, FLz_traj_ref, FRx_traj_ref, FRy_traj_ref, FRz_traj_ref, Px_seq_ref, Py_seq_ref, Pz_seq_ref, SwitchingTimeVec_ref = NLP_ref_trajectory_from_SecondLevel(StartStepNum = roundNum, LookAheadSteps = NumofLookAhead)

            ParaList = np.concatenate((LeftSwingFlag,RightSwingFlag,
                x_init,y_init,z_init,
                xdot_init,ydot_init,zdot_init,
                Lx_init,Ly_init,Lz_init,
                Ldotx_init,Ldoty_init,Ldotz_init,
                PLx_init,PLy_init,PLz_init,
                PRx_init,PRy_init,PRz_init,
                x_end,y_end,z_end,
                xdot_end,ydot_end,zdot_end,
                HalfSpaceSeq,
                TerrainTangentsX[roundNum],TerrainTangentsY[roundNum],TerrainNorms[roundNum],
                PL_init_TangentX,PL_init_TangentY,PL_init_Norm,
                PR_init_TangentX,PR_init_TangentY,PR_init_Norm,
                x_traj_ref,y_traj_ref,z_traj_ref,
                xdot_traj_ref,ydot_traj_ref,zdot_traj_ref,
                FLx_traj_ref,FLy_traj_ref,FLz_traj_ref,
                FRx_traj_ref,FRy_traj_ref,FRz_traj_ref,
                SwitchingTimeVec_ref,
                Px_seq_ref,Py_seq_ref,Pz_seq_ref,
                x_end_level1,y_end_level1,z_end_level1,
                xdot_end_level1,ydot_end_level1,zdot_end_level1,
                px_level1,py_level1,pz_level1,
                Lx_end_level1,Ly_end_level1,Lz_end_level1,
                Level1_ref_traj,
                px_init_ref, py_init_ref, pz_init_ref),axis=None)

        else:
            x_nlplevel2_ref, y_nlplevel2_ref, z_nlplevel2_ref, xdot_nlplevel2_ref, ydot_nlplevel2_ref, zdot_nlplevel2_ref, Lx_nlplevel2_ref, Ly_nlplevel2_ref, Lz_nlplevel2_ref, Ldotx_nlplevel2_ref, Ldoty_nlplevel2_ref, Ldotz_nlplevel2_ref, FL1x_nlplevel2_ref, FL1y_nlplevel2_ref, FL1z_nlplevel2_ref, FL2x_nlplevel2_ref, FL2y_nlplevel2_ref, FL2z_nlplevel2_ref, FL3x_nlplevel2_ref, FL3y_nlplevel2_ref, FL3z_nlplevel2_ref, FL4x_nlplevel2_ref, FL4y_nlplevel2_ref, FL4z_nlplevel2_ref, FR1x_nlplevel2_ref, FR1y_nlplevel2_ref, FR1z_nlplevel2_ref, FR2x_nlplevel2_ref, FR2y_nlplevel2_ref, FR2z_nlplevel2_ref, FR3x_nlplevel2_ref, FR3y_nlplevel2_ref, FR3z_nlplevel2_ref, FR4x_nlplevel2_ref, FR4y_nlplevel2_ref, FR4z_nlplevel2_ref, Px_seq_nlplevel2_ref, Py_seq_nlplevel2_ref, Pz_seq_nlplevel2_ref, SwitchingTimeVec_nlplevel2_ref, px_init_seq_nlplevel2_ref, py_init_seq_nlplevel2_ref, pz_init_seq_nlplevel2_ref = ref_trajectory_construction_for_NLPSecondLevel(StartStepNum = roundNum, LookAheadSteps = NumofLookAhead)
            
            print("SwitchingTimeDuration: ",SwitchingTimeVec_nlplevel2_ref)

            ParaList = np.concatenate((LeftSwingFlag,RightSwingFlag,
                x_init,y_init,z_init,
                xdot_init,ydot_init,zdot_init,
                Lx_init,Ly_init,Lz_init,
                Ldotx_init,Ldoty_init,Ldotz_init,
                PLx_init,PLy_init,PLz_init,
                PRx_init,PRy_init,PRz_init,
                x_end,y_end,z_end,
                xdot_end,ydot_end,zdot_end,
                HalfSpaceSeq,
                TerrainTangentsX[roundNum],TerrainTangentsY[roundNum],TerrainNorms[roundNum],
                PL_init_TangentX,PL_init_TangentY,PL_init_Norm,
                PR_init_TangentX,PR_init_TangentY,PR_init_Norm,
                x_end_level1,y_end_level1,z_end_level1,
                xdot_end_level1,ydot_end_level1,zdot_end_level1,
                px_level1,py_level1,pz_level1,
                Lx_end_level1,Ly_end_level1,Lz_end_level1,
                Level1_ref_traj,
                x_nlplevel2_ref, y_nlplevel2_ref, z_nlplevel2_ref, 
                xdot_nlplevel2_ref, ydot_nlplevel2_ref, zdot_nlplevel2_ref, 
                Lx_nlplevel2_ref, Ly_nlplevel2_ref, Lz_nlplevel2_ref, 
                Ldotx_nlplevel2_ref, Ldoty_nlplevel2_ref, Ldotz_nlplevel2_ref, 
                FL1x_nlplevel2_ref, FL1y_nlplevel2_ref, FL1z_nlplevel2_ref, 
                FL2x_nlplevel2_ref, FL2y_nlplevel2_ref, FL2z_nlplevel2_ref, 
                FL3x_nlplevel2_ref, FL3y_nlplevel2_ref, FL3z_nlplevel2_ref, 
                FL4x_nlplevel2_ref, FL4y_nlplevel2_ref, FL4z_nlplevel2_ref, 
                FR1x_nlplevel2_ref, FR1y_nlplevel2_ref, FR1z_nlplevel2_ref, 
                FR2x_nlplevel2_ref, FR2y_nlplevel2_ref, FR2z_nlplevel2_ref, 
                FR3x_nlplevel2_ref, FR3y_nlplevel2_ref, FR3z_nlplevel2_ref, 
                FR4x_nlplevel2_ref, FR4y_nlplevel2_ref, FR4z_nlplevel2_ref, 
                SwitchingTimeVec_nlplevel2_ref,
                Px_seq_nlplevel2_ref, Py_seq_nlplevel2_ref, Pz_seq_nlplevel2_ref, 
                px_init_seq_nlplevel2_ref, py_init_seq_nlplevel2_ref, pz_init_seq_nlplevel2_ref),axis=None)

        
        #For 1 step lookahead checking local minima only
        if NumofLookAhead  == 1 and InitSeedType == "random":
            if ProvideInitialSeed == True:
                DecisionVars_init = GetFirstLevelTrajectory(RoundNum = roundNum)


        res = solver(x0=DecisionVars_init, p = ParaList, lbx = DecisionVars_lb, ubx = DecisionVars_ub, lbg = glb, ubg = gub)

        x_opt = res["x"]
        x_opt = x_opt.full().flatten()

        x_opt_left = x_opt
        x_opt_right = x_opt

        #PlotSingleOptimiation_and_PrintResult(x_opt = x_opt, var_index=var_index, PL_init = np.array([PLx_init,PLy_init,PLz_init]), PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag, PrintSecondLevel = False, PlotNLP = True, PlotBothLevel = True)

    elif roundNum > 0:

        #Update Initial Condition --- from previous rounds
        x_res = x_opt[var_index_Level1["x"][0]:var_index_Level1["x"][1]+1]
        x_init = x_res[-1]
        print("x_init:",x_init)
        y_res = x_opt[var_index_Level1["y"][0]:var_index_Level1["y"][1]+1]
        y_init = y_res[-1]
        print("y_init:",y_init)
        z_res = x_opt[var_index_Level1["z"][0]:var_index_Level1["z"][1]+1]
        z_init = z_res[-1]
        print("z_init:",z_init)

        Lx_res = x_opt[var_index_Level1["Lx"][0]:var_index_Level1["Lx"][1]+1]
        Lx_init = Lx_res[-1]
        print("Lx_init:",Lx_init)
        Ly_res = x_opt[var_index_Level1["Ly"][0]:var_index_Level1["Ly"][1]+1]
        Ly_init = Ly_res[-1]
        print("Ly_init:",Ly_init)
        Lz_res = x_opt[var_index_Level1["Lz"][0]:var_index_Level1["Lz"][1]+1]
        Lz_init = Lz_res[-1]
        print("Lz_init:",Lz_init)

        Ldotx_res = x_opt[var_index_Level1["Ldotx"][0]:var_index_Level1["Ldotx"][1]+1]
        Ldotx_init = Ldotx_res[-1]
        Ldoty_res = x_opt[var_index_Level1["Ldoty"][0]:var_index_Level1["Ldoty"][1]+1]
        Ldoty_init = Ldoty_res[-1]
        Ldotz_res = x_opt[var_index_Level1["Ldotz"][0]:var_index_Level1["Ldotz"][1]+1]
        Ldotz_init = Ldotz_res[-1]

        xdot_res  = x_opt[var_index_Level1["xdot"][0]:var_index_Level1["xdot"][1]+1]
        xdot_init = xdot_res[-1]
        print("xdot_init:",xdot_init)
        ydot_res  = x_opt[var_index_Level1["ydot"][0]:var_index_Level1["ydot"][1]+1]
        ydot_init = ydot_res[-1]
        print("xdot_init:",ydot_init)
        zdot_res  = x_opt[var_index_Level1["zdot"][0]:var_index_Level1["zdot"][1]+1]
        zdot_init = zdot_res[-1]
        print("zdot_init:",zdot_init)

        # #----------------------------------
        # x_res = Level1_ref_traj[var_index_Level1["x"][0]:var_index_Level1["x"][1]+1]
        # x_init = x_res[0]
        # print("x_init:",x_init)
        # y_res = Level1_ref_traj[var_index_Level1["y"][0]:var_index_Level1["y"][1]+1]
        # y_init = y_res[0]
        # print("y_init:",y_init)
        # z_res = Level1_ref_traj[var_index_Level1["z"][0]:var_index_Level1["z"][1]+1]
        # z_init = z_res[0]
        # print("z_init:",z_init)
        # Lx_res = Level1_ref_traj[var_index_Level1["Lx"][0]:var_index_Level1["Lx"][1]+1]
        # Lx_init = Lx_res[0]
        # print("Lx_init:",Lx_init)
        # Ly_res = Level1_ref_traj[var_index_Level1["Ly"][0]:var_index_Level1["Ly"][1]+1]
        # Ly_init = Ly_res[0]
        # print("Ly_init:",Ly_init)
        # Lz_res = Level1_ref_traj[var_index_Level1["Lz"][0]:var_index_Level1["Lz"][1]+1]
        # Lz_init = Lz_res[0]
        # print("Lz_init:",Lz_init)

        # Ldotx_res = Level1_ref_traj[var_index_Level1["Ldotx"][0]:var_index_Level1["Ldotx"][1]+1]
        # Ldotx_init = Ldotx_res[0]
        # Ldoty_res = Level1_ref_traj[var_index_Level1["Ldoty"][0]:var_index_Level1["Ldoty"][1]+1]
        # Ldoty_init = Ldoty_res[0]
        # Ldotz_res = Level1_ref_traj[var_index_Level1["Ldotz"][0]:var_index_Level1["Ldotz"][1]+1]
        # Ldotz_init = Ldotz_res[0]

        # xdot_res  = Level1_ref_traj[var_index_Level1["xdot"][0]:var_index_Level1["xdot"][1]+1]
        # xdot_init = xdot_res[0]
        # ydot_res  = Level1_ref_traj[var_index_Level1["ydot"][0]:var_index_Level1["ydot"][1]+1]
        # ydot_init = ydot_res[0]
        # zdot_res  = Level1_ref_traj[var_index_Level1["zdot"][0]:var_index_Level1["zdot"][1]+1]
        # zdot_init = zdot_res[0]

        print("Terrain Tangent X:", TerrainTangentsX[roundNum-1][0])
        print("Terrain Tangent Y:", TerrainTangentsY[roundNum-1][0])
        print("Terrain Norm:", TerrainNorms[roundNum-1][0])

        if SwingLeftFirst == 1:
            if roundNum%2 == 0:#Even (The First phase)
                #Swing the Left
                LeftSwingFlag = 1
                RightSwingFlag = 0
                
                #Update Initial Foot Location and Terrain Tangents Norms
                px_res = x_opt[var_index_Level1["px"][0]:var_index_Level1["px"][1]+1]
                PRx_init = px_res[-1]
                py_res = x_opt[var_index_Level1["py"][0]:var_index_Level1["py"][1]+1]
                PRy_init = py_res[-1]
                pz_res = x_opt[var_index_Level1["pz"][0]:var_index_Level1["pz"][1]+1]
                PRz_init = pz_res[-1]
                #Init Terrain Tangent and Norm
                PR_init_TangentX = TerrainTangentsX[roundNum-1][0]
                PR_init_TangentY = TerrainTangentsY[roundNum-1][0]
                PR_init_Norm = TerrainNorms[roundNum-1][0]

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
                #Init Terrain Tangent and Norm
                PL_init_TangentX = TerrainTangentsX[roundNum-1][0]
                PL_init_TangentY = TerrainTangentsY[roundNum-1][0]
                PL_init_Norm = TerrainNorms[roundNum-1][0]

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
                #Init Terrain Tangent and Norm
                PL_init_TangentX = TerrainTangentsX[roundNum-1][0]
                PL_init_TangentY = TerrainTangentsY[roundNum-1][0]
                PL_init_Norm = TerrainNorms[roundNum-1][0]

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
                #Init Terrain Tangent and Norm
                PR_init_TangentX = TerrainTangentsX[roundNum-1][0]
                PR_init_TangentY = TerrainTangentsY[roundNum-1][0]
                PR_init_Norm = TerrainNorms[roundNum-1][0]

        print("PL init: ", PLx_init, PLy_init, PLz_init)
        print("PR init: ", PRx_init, PRy_init, PRz_init)

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

        #Get Reference Trajectory
        if ChosenSolver == "CoM":
            x_traj_ref, y_traj_ref, z_traj_ref, xdot_traj_ref, ydot_traj_ref, zdot_traj_ref, FLx_traj_ref, FLy_traj_ref, FLz_traj_ref, FRx_traj_ref, FRy_traj_ref, FRz_traj_ref, Px_seq_ref, Py_seq_ref, Pz_seq_ref, SwitchingTimeVec_ref, px_init_ref, py_init_ref, pz_init_ref = NLP_ref_trajectory_construction(StartStepNum = roundNum, LookAheadSteps = NumofLookAhead)
            #x_traj_ref, y_traj_ref, z_traj_ref, xdot_traj_ref, ydot_traj_ref, zdot_traj_ref, FLx_traj_ref, FLy_traj_ref, FLz_traj_ref, FRx_traj_ref, FRy_traj_ref, FRz_traj_ref, Px_seq_ref, Py_seq_ref, Pz_seq_ref, SwitchingTimeVec_ref = NLP_ref_trajectory_from_SecondLevel(StartStepNum = roundNum, LookAheadSteps = NumofLookAhead)

            ParaList = np.concatenate((LeftSwingFlag,RightSwingFlag,
                x_init,y_init,z_init,
                xdot_init,ydot_init,zdot_init,
                Lx_init,Ly_init,Lz_init,
                Ldotx_init,Ldoty_init,Ldotz_init,
                PLx_init,PLy_init,PLz_init,
                PRx_init,PRy_init,PRz_init,
                x_end,y_end,z_end,
                xdot_end,ydot_end,zdot_end,
                HalfSpaceSeq,
                TerrainTangentsX[roundNum],TerrainTangentsY[roundNum],TerrainNorms[roundNum],
                PL_init_TangentX,PL_init_TangentY,PL_init_Norm,
                PR_init_TangentX,PR_init_TangentY,PR_init_Norm,
                x_traj_ref,y_traj_ref,z_traj_ref,
                xdot_traj_ref,ydot_traj_ref,zdot_traj_ref,
                FLx_traj_ref,FLy_traj_ref,FLz_traj_ref,
                FRx_traj_ref,FRy_traj_ref,FRz_traj_ref,
                SwitchingTimeVec_ref,
                Px_seq_ref,Py_seq_ref,Pz_seq_ref,
                x_end_level1,y_end_level1,z_end_level1,
                xdot_end_level1,ydot_end_level1,zdot_end_level1,
                px_level1,py_level1,pz_level1,
                Lx_end_level1,Ly_end_level1,Lz_end_level1,
                Level1_ref_traj,
                px_init_ref, py_init_ref, pz_init_ref),axis=None)
            
        else:
            x_nlplevel2_ref, y_nlplevel2_ref, z_nlplevel2_ref, xdot_nlplevel2_ref, ydot_nlplevel2_ref, zdot_nlplevel2_ref, Lx_nlplevel2_ref, Ly_nlplevel2_ref, Lz_nlplevel2_ref, Ldotx_nlplevel2_ref, Ldoty_nlplevel2_ref, Ldotz_nlplevel2_ref, FL1x_nlplevel2_ref, FL1y_nlplevel2_ref, FL1z_nlplevel2_ref, FL2x_nlplevel2_ref, FL2y_nlplevel2_ref, FL2z_nlplevel2_ref, FL3x_nlplevel2_ref, FL3y_nlplevel2_ref, FL3z_nlplevel2_ref, FL4x_nlplevel2_ref, FL4y_nlplevel2_ref, FL4z_nlplevel2_ref, FR1x_nlplevel2_ref, FR1y_nlplevel2_ref, FR1z_nlplevel2_ref, FR2x_nlplevel2_ref, FR2y_nlplevel2_ref, FR2z_nlplevel2_ref, FR3x_nlplevel2_ref, FR3y_nlplevel2_ref, FR3z_nlplevel2_ref, FR4x_nlplevel2_ref, FR4y_nlplevel2_ref, FR4z_nlplevel2_ref, Px_seq_nlplevel2_ref, Py_seq_nlplevel2_ref, Pz_seq_nlplevel2_ref, SwitchingTimeVec_nlplevel2_ref, px_init_seq_nlplevel2_ref, py_init_seq_nlplevel2_ref, pz_init_seq_nlplevel2_ref = ref_trajectory_construction_for_NLPSecondLevel(StartStepNum = roundNum, LookAheadSteps = NumofLookAhead)
            ParaList = np.concatenate((LeftSwingFlag,RightSwingFlag,
                x_init,y_init,z_init,
                xdot_init,ydot_init,zdot_init,
                Lx_init,Ly_init,Lz_init,
                Ldotx_init,Ldoty_init,Ldotz_init,
                PLx_init,PLy_init,PLz_init,
                PRx_init,PRy_init,PRz_init,
                x_end,y_end,z_end,
                xdot_end,ydot_end,zdot_end,
                HalfSpaceSeq,
                TerrainTangentsX[roundNum],TerrainTangentsY[roundNum],TerrainNorms[roundNum],
                PL_init_TangentX,PL_init_TangentY,PL_init_Norm,
                PR_init_TangentX,PR_init_TangentY,PR_init_Norm,
                x_end_level1,y_end_level1,z_end_level1,
                xdot_end_level1,ydot_end_level1,zdot_end_level1,
                px_level1,py_level1,pz_level1,
                Lx_end_level1,Ly_end_level1,Lz_end_level1,
                Level1_ref_traj,
                x_nlplevel2_ref, y_nlplevel2_ref, z_nlplevel2_ref, 
                xdot_nlplevel2_ref, ydot_nlplevel2_ref, zdot_nlplevel2_ref, 
                Lx_nlplevel2_ref, Ly_nlplevel2_ref, Lz_nlplevel2_ref, 
                Ldotx_nlplevel2_ref, Ldoty_nlplevel2_ref, Ldotz_nlplevel2_ref, 
                FL1x_nlplevel2_ref, FL1y_nlplevel2_ref, FL1z_nlplevel2_ref, 
                FL2x_nlplevel2_ref, FL2y_nlplevel2_ref, FL2z_nlplevel2_ref, 
                FL3x_nlplevel2_ref, FL3y_nlplevel2_ref, FL3z_nlplevel2_ref, 
                FL4x_nlplevel2_ref, FL4y_nlplevel2_ref, FL4z_nlplevel2_ref, 
                FR1x_nlplevel2_ref, FR1y_nlplevel2_ref, FR1z_nlplevel2_ref, 
                FR2x_nlplevel2_ref, FR2y_nlplevel2_ref, FR2z_nlplevel2_ref, 
                FR3x_nlplevel2_ref, FR3y_nlplevel2_ref, FR3z_nlplevel2_ref, 
                FR4x_nlplevel2_ref, FR4y_nlplevel2_ref, FR4z_nlplevel2_ref, 
                SwitchingTimeVec_nlplevel2_ref,
                Px_seq_nlplevel2_ref, Py_seq_nlplevel2_ref, Pz_seq_nlplevel2_ref, 
                px_init_seq_nlplevel2_ref, py_init_seq_nlplevel2_ref, pz_init_seq_nlplevel2_ref),axis=None)

        if InitSeedType == "random":
            #Shuffle the Random Seed Generator
            np.random.seed()
            DecisionVarsShape = DecisionVars_lb.shape
            DecisionVars_init = DecisionVars_lb + np.multiply(np.random.rand(DecisionVarsShape[0],).flatten(),(DecisionVars_ub - DecisionVars_lb))#   Fixed Value Initial Guess
            
            #For 1 step lookahead checking local minima only
            if NumofLookAhead  == 1:
                if ProvideInitialSeed == True:
                    DecisionVars_init = GetFirstLevelTrajectory(RoundNum = roundNum)

            res = solver(x0=DecisionVars_init, p = ParaList, lbx = DecisionVars_lb, ubx = DecisionVars_ub, lbg = glb, ubg = gub)
            x_opt = res["x"]
            x_opt = x_opt.full().flatten()  

        elif InitSeedType == "previous":
            if SwingLeftFirst == 1:
                if roundNum%2 == 0:#Even (The First phase)
                    #Swing the Left
                    res = solver(x0=x_opt_left, p = ParaList, lbx = DecisionVars_lb, ubx = DecisionVars_ub, lbg = glb, ubg = gub)
                    #res = solver(x0=DecisionVars_init, p = ParaList, lbx = DecisionVars_lb, ubx = DecisionVars_ub, lbg = glb, ubg = gub)
                    x_opt = res["x"]
                    x_opt = x_opt.full().flatten()  
                    x_opt_left = x_opt

                elif roundNum%2 == 1:#odd (The Second phase)
                    #Swing the Right
                    res = solver(x0=x_opt_right, p = ParaList, lbx = DecisionVars_lb, ubx = DecisionVars_ub, lbg = glb, ubg = gub)
                    #res = solver(x0=DecisionVars_init, p = ParaList, lbx = DecisionVars_lb, ubx = DecisionVars_ub, lbg = glb, ubg = gub)
                    x_opt = res["x"]
                    x_opt = x_opt.full().flatten()  
                    x_opt_right = x_opt     

            elif SwingRightFirst == 1:
                if roundNum%2 == 0:#Even (The First phase)
                    #Swing the Right
                    res = solver(x0=x_opt_right, p = ParaList, lbx = DecisionVars_lb, ubx = DecisionVars_ub, lbg = glb, ubg = gub)
                    #res = solver(x0=DecisionVars_init, p = ParaList, lbx = DecisionVars_lb, ubx = DecisionVars_ub, lbg = glb, ubg = gub)
                    x_opt = res["x"]
                    x_opt = x_opt.full().flatten()  
                    x_opt_right = x_opt     

                elif roundNum%2 == 1:#odd (The Second phase)
                    #Swing the Left
                    res = solver(x0=x_opt_left, p = ParaList, lbx = DecisionVars_lb, ubx = DecisionVars_ub, lbg = glb, ubg = gub)
                    #res = solver(x0=DecisionVars_init, p = ParaList, lbx = DecisionVars_lb, ubx = DecisionVars_ub, lbg = glb, ubg = gub)
                    x_opt = res["x"]
                    x_opt = x_opt.full().flatten()  
                    x_opt_left = x_opt

    #print(solver.stats())
     
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
        firstLevelAccCost,firstLevelMomentumCost,firstLevelMomentumRateCost = FirstLevelCost(x_opt=x_opt,var_index=var_index,G = 9.80665,m=95)
        #Fullcosts.append(firstLevelFullCost)
        Acc_cost.append(firstLevelAccCost)
        Momentum_Cost.append(firstLevelMomentumCost)
        MomentumRate_Cost.append(firstLevelMomentumRateCost)
        TerminalCost.append(10*(x_fullres[-1][-1]-x_end)**2 + 10*(y_fullres[-1][-1]-y_end)**2 + 1000*(z_fullres[-1][-1]-z_end)**2)
        #Save the trajectory of current round
        AllRoundTrajectory.append(x_opt)
        CasadiParameters.append(ParaList)

        #Compute timings
        #ProgramTime = TotalRunTime - solver.stats()["t_proc_nlp_hess_l"] - solver.stats()["t_proc_nlp_grad"] - solver.stats()["t_proc_nlp_gf_jg"] - solver.stats()["t_proc_nlp_fg"]

        TotalRunTime = round(solver.stats()["t_proc_total"],4)
        print("Total Program Time: ", TotalRunTime)

    elif solver.stats()["success"] == False:
        StopRound = roundNum
        print("Fail at Round ", roundNum)
        TotalRunTime = round(solver.stats()["t_proc_total"],4)
        print("Total Program Time: ", TotalRunTime)
        break

    #===========================
    #NOTE: Remove comment to enbale Plot function
    #for two levels
    if ShowFigure == "True":
        PlotSingleOptimiation_and_PrintResult(x_opt = x_opt, var_index=var_index, PL_init = np.array([PLx_init,PLy_init,PLz_init]), PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag, PrintFirstLevel=True, PrintSecondLevel = True, PlotNLP = True, PlotBothLevel = True, AllSurfaces = AllPatches, RoundNum = roundNum)
    elif ShowFigure == "False":
        PlotSingleOptimiation_and_PrintResult(x_opt = x_opt, var_index=var_index, PL_init = np.array([PLx_init,PLy_init,PLz_init]), PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag, PrintFirstLevel=True, PrintSecondLevel = True, PlotNLP = False, PlotBothLevel = False, AllSurfaces = AllPatches, RoundNum = roundNum)
    #===========================



#===========================
#NOTE: Remove comment to enbale Plot function
#plot full result
if ShowFigure == "True":
    Plot_RHP_result(NumRounds = StopRound, SwingLeftFirst = SwingLeftFirst, SwingRightFirst = SwingRightFirst, x_fullres = x_fullres, y_fullres = y_fullres, z_fullres = z_fullres, PL_init_fullres = PL_init_fullres, PR_init_fullres = PR_init_fullres, Px_fullres = Px_fullres, Py_fullres = Py_fullres, Pz_fullres = Pz_fullres, AllSurfaces = AllPatches)

#===========================
#Compute Total Cost
TotalCost = TerminalCost[-1]+np.sum(Fullcosts)
#Calculate Accumulated Cost
AccumAccCost = round(np.sum(Acc_cost),4)
AccumMomentumCost = round(np.sum(Momentum_Cost),4)
AccumMomentumRateCost = round(np.sum(MomentumRate_Cost),4)
Terminal_X_pos = round(x_fullres[-1][-1],4)
Terminal_Y_pos = round(y_fullres[-1][-1],4)
Terminal_Z_pos = round(z_fullres[-1][-1],4)
print("Accumulated Acc Cost is: ", AccumAccCost)
print("Accumulated Momentum Cost is: ", AccumMomentumCost)
print("Accumulated Momentum Rate Cost is: ",AccumMomentumRateCost)
print("Total Cost is: ",round(TotalCost,4))
print("Terminal Cost is: ", round(TerminalCost[-1],4))
print("Terminal X position is: ", Terminal_X_pos)
print("Terminal Y position is: ", Terminal_Y_pos)
print("Terminal Z position is: ", Terminal_Z_pos)

#Dump data into pickled file
DumpedResult = {"TerrainModel": AllPatches,
                "VarIdx_of_All_Levels": var_index,
                "Trajectory_of_All_Rounds":AllRoundTrajectory,
                "TotalCost":TotalCost,
                "TerminalCost":TerminalCost,
                "Accumulated_Acc_Cost":AccumAccCost,
                "Accumulated_Momentum_Cost":AccumMomentumCost,
                "Accumulated_MomentumRate_Cost":AccumMomentumRateCost,
                "StopRound":StopRound,
                "SwingLeftFirst":SwingLeftFirst,
                "SwingRightFirst":SwingRightFirst,
                "x_fullres":x_fullres,
                "y_fullres":y_fullres,
                "z_fullres":z_fullres,
                "PL_init_fullres":PL_init_fullres,
                "PR_init_fullres":PR_init_fullres,
                "Px_fullres":Px_fullres,
                "Py_fullres":Py_fullres,
                "Pz_fullres":Pz_fullres,
                "Terminal_X_pos":Terminal_X_pos,
                "Terminal_Y_pos":Terminal_Y_pos,
                "Terminal_Z_pos":Terminal_Z_pos,
                "CasadiParameters":CasadiParameters
}
pickle.dump(DumpedResult, open(ResultSavingFolder+'/'+str(NumofLookAhead)+'LookAhead_Trial'+str(TrialNum)+'.p', "wb"))  # save it into a file named save.p
