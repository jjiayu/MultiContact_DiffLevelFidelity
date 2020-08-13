#Receding Horizon Planning Framework
import numpy as np #Numpy
import casadi as ca #Casadi
from Humanoid_ProblemDescription_3Order_Bezier import *

#Initialization and Porblem Setup

#   Set Decimal Printing Precision
np.set_printoptions(precision=4)

#   Define the Swing foot of the First Step
LeftSwingFlag = 0
RightSwingFlag = 1

#   Initial Condition of the Robot
#   Starting CoM State
C_start = [0.,0.,0.55]
#   Starting CoM Velocity
Cdot_start = [0.0,0.,0.]
#   Starting CoM Acceleration
Cddot_start = [0.0,0.0,0.0]

#   Expected Terminal Condition of the Robot
C_end = [0,0,0.55]

#   Timing Configuration NOTE: May become a search variable
TimeVec = [0.2,0.4,0.2]
T = 0.8

#   Initial Contact Locations
PL_init = [0, 0.1,0]
PR_init = [0,-0.1,0]

#Build Solver
solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index = BuildSolver(FirstLevel = "Bezier_SingleStep_Discrete_Order3")

#Build initial Seed (for QP it is not that important)
np.random.seed()
DecisionVarsShape = DecisionVars_lb.shape
DecisionVars_init = DecisionVars_lb + np.multiply(np.random.rand(DecisionVarsShape[0],).flatten(),(DecisionVars_ub-DecisionVars_lb))#   Fixed Value Initial Guess

#NOTE:Here we should have a for-loop to solve multiple solutions

P_next = [0,-0.1,0]

ParaList = np.concatenate((LeftSwingFlag,RightSwingFlag,C_start,Cdot_start,Cddot_start,C_end,TimeVec,T,PL_init,PR_init,P_next),axis=None)

res = solver(x0=DecisionVars_init, p = ParaList,lbx = DecisionVars_lb, ubx = DecisionVars_ub,lbg = glb, ubg = gub)

x_opt = res['x']
print(solver.stats()["success"])
print('x_opt: ', x_opt)
#print(res)