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
C_start = [0.,0.1,0.55]
#   Starting CoM Velocity
Cdot_start = [0.0,0.,0.]
#   Starting CoM Acceleration
Cddot_start = [0.0,0.0,0.0]

#   Expected Terminal Condition of the Robot
C_end = [5,0,0.55]

#   Timing Configuration NOTE: May become a search variable
TimeVec = [0.2,0.4,0.01]
T = TimeVec[0]+TimeVec[1]+TimeVec[2]
print(T)

#   Initial Contact Locations
PL_init = [0, 0.1,0]
PR_init = [0.1,-0.1,0]

#Build Solver
#solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index = BuildSolver(FirstLevel = "Bezier_SingleStep_Discrete_Order3")

solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index = BuildSolver(FirstLevel = "Bezier_SingleStep_Discrete_Order3", SecondLevel = "CoM_Dynamics_Fixed_Time", m = 95, NumPhase = 3)

#Build initial Seed (for QP it is not that important)
np.random.seed()
DecisionVarsShape = DecisionVars_lb.shape
DecisionVars_init = DecisionVars_lb + np.multiply(np.random.rand(DecisionVarsShape[0],).flatten(),(DecisionVars_ub-DecisionVars_lb))#   Fixed Value Initial Guess

#NOTE:Here we should have a for-loop to solve multiple solutions

P_next = [0.2,-0.2,0]

ParaList = np.concatenate((LeftSwingFlag,RightSwingFlag,C_start,Cdot_start,Cddot_start,C_end,TimeVec,T,PL_init,PR_init,P_next),axis=None)

res = solver(x0=DecisionVars_init, p = ParaList,lbx = DecisionVars_lb, ubx = DecisionVars_ub,lbg = glb, ubg = gub)

x_opt = res['x']
print(solver.stats()["success"])
#print('x_opt: ', x_opt)
print(res)

#print results
var_index_L1 = var_index["Level1_Var_Index"]

Cy_res = x_opt[var_index_L1["Cy"][0]:var_index_L1["Cy"][1]+1]
print("Cy_res")
print(Cy_res)

L0_res = x_opt[var_index_L1["L0"][0]:var_index_L1["L0"][1]+1]
print("L0_res")
print(L0_res)

L1_res = x_opt[var_index_L1["L1"][0]:var_index_L1["L1"][1]+1]
print("L1_res")
print(L1_res)

L2_res = x_opt[var_index_L1["L2"][0]:var_index_L1["L2"][1]+1]
print("L2_res")
print(L2_res)

L3_res = x_opt[var_index_L1["L3"][0]:var_index_L1["L3"][1]+1]
print("L3_res")
print(L3_res)

L4_res = x_opt[var_index_L1["L4"][0]:var_index_L1["L4"][1]+1]
print("L4_res")
print(L4_res)

FL1_init_p0_res = x_opt[var_index_L1["FL1_initdouble_p0"][0]:var_index_L1["FL1_initdouble_p0"][1]+1]
print("FL1_init_p0_res")
print(FL1_init_p0_res)

FL1_init_p1_res = x_opt[var_index_L1["FL1_initdouble_p1"][0]:var_index_L1["FL1_initdouble_p1"][1]+1]
print("FL1_init_p1_res")
print(FL1_init_p1_res)

FL1_swing_p0_res = x_opt[var_index_L1["FL1_swing_p0"][0]:var_index_L1["FL1_swing_p0"][1]+1]
print("FL1_swing_p0_res")
print(FL1_swing_p0_res)

FL1_swing_p1_res = x_opt[var_index_L1["FL1_swing_p1"][0]:var_index_L1["FL1_swing_p1"][1]+1]
print("FL1_swing_p1_res")
print(FL1_swing_p1_res)

FL1_double_p0_res = x_opt[var_index_L1["FL1_double_p0"][0]:var_index_L1["FL1_double_p0"][1]+1]
print("FL1_double_p0_res")
print(FL1_double_p0_res)

FL1_double_p1_res = x_opt[var_index_L1["FL1_double_p1"][0]:var_index_L1["FL1_double_p1"][1]+1]
print("FL1_double_p1_res")
print(FL1_double_p1_res)

FL2_init_p0_res = x_opt[var_index_L1["FL2_initdouble_p0"][0]:var_index_L1["FL2_initdouble_p0"][1]+1]
print("FL2_init_p0_res")
print(FL2_init_p0_res)

FL2_init_p1_res = x_opt[var_index_L1["FL2_initdouble_p1"][0]:var_index_L1["FL2_initdouble_p1"][1]+1]
print("FL2_init_p1_res")
print(FL2_init_p1_res)

FL2_swing_p0_res = x_opt[var_index_L1["FL2_swing_p0"][0]:var_index_L1["FL2_swing_p0"][1]+1]
print("FL2_swing_p0_res")
print(FL2_swing_p0_res)

FL2_swing_p1_res = x_opt[var_index_L1["FL2_swing_p1"][0]:var_index_L1["FL2_swing_p1"][1]+1]
print("FL2_swing_p1_res")
print(FL2_swing_p1_res)

FL2_double_p0_res = x_opt[var_index_L1["FL2_double_p0"][0]:var_index_L1["FL2_double_p0"][1]+1]
print("FL2_double_p0_res")
print(FL2_double_p0_res)

FL2_double_p1_res = x_opt[var_index_L1["FL2_double_p1"][0]:var_index_L1["FL2_double_p1"][1]+1]
print("FL2_double_p1_res")
print(FL2_double_p1_res)

FL3_init_p0_res = x_opt[var_index_L1["FL3_initdouble_p0"][0]:var_index_L1["FL3_initdouble_p0"][1]+1]
print("FL3_init_p0_res")
print(FL3_init_p0_res)

FL3_init_p1_res = x_opt[var_index_L1["FL3_initdouble_p1"][0]:var_index_L1["FL3_initdouble_p1"][1]+1]
print("FL3_init_p1_res")
print(FL3_init_p1_res)

FL3_swing_p0_res = x_opt[var_index_L1["FL3_swing_p0"][0]:var_index_L1["FL3_swing_p0"][1]+1]
print("FL3_swing_p0_res")
print(FL3_swing_p0_res)

FL3_swing_p1_res = x_opt[var_index_L1["FL3_swing_p1"][0]:var_index_L1["FL3_swing_p1"][1]+1]
print("FL3_swing_p1_res")
print(FL3_swing_p1_res)

FL3_double_p0_res = x_opt[var_index_L1["FL3_double_p0"][0]:var_index_L1["FL3_double_p0"][1]+1]
print("FL3_double_p0_res")
print(FL3_double_p0_res)

FL3_double_p1_res = x_opt[var_index_L1["FL3_double_p1"][0]:var_index_L1["FL3_double_p1"][1]+1]
print("FL3_double_p1_res")
print(FL3_double_p1_res)

FL4_init_p0_res = x_opt[var_index_L1["FL4_initdouble_p0"][0]:var_index_L1["FL4_initdouble_p0"][1]+1]
print("FL4_init_p0_res")
print(FL4_init_p0_res)

FL4_init_p1_res = x_opt[var_index_L1["FL4_initdouble_p1"][0]:var_index_L1["FL4_initdouble_p1"][1]+1]
print("FL4_init_p1_res")
print(FL4_init_p1_res)

FL4_swing_p0_res = x_opt[var_index_L1["FL4_swing_p0"][0]:var_index_L1["FL4_swing_p0"][1]+1]
print("FL4_swing_p0_res")
print(FL4_swing_p0_res)

FL4_swing_p1_res = x_opt[var_index_L1["FL4_swing_p1"][0]:var_index_L1["FL4_swing_p1"][1]+1]
print("FL4_swing_p1_res")
print(FL4_swing_p1_res)

FL4_double_p0_res = x_opt[var_index_L1["FL4_double_p0"][0]:var_index_L1["FL4_double_p0"][1]+1]
print("FL4_double_p0_res")
print(FL4_double_p0_res)

FL4_double_p1_res = x_opt[var_index_L1["FL4_double_p1"][0]:var_index_L1["FL4_double_p1"][1]+1]
print("FL4_double_p1_res")
print(FL4_double_p1_res)

FR1_init_p0_res = x_opt[var_index_L1["FR1_initdouble_p0"][0]:var_index_L1["FR1_initdouble_p0"][1]+1]
print("FR1_init_p0_res")
print(FR1_init_p0_res)

FR1_init_p1_res = x_opt[var_index_L1["FR1_initdouble_p1"][0]:var_index_L1["FR1_initdouble_p1"][1]+1]
print("FR1_init_p1_res")
print(FR1_init_p1_res)

FR1_swing_p0_res = x_opt[var_index_L1["FR1_swing_p0"][0]:var_index_L1["FR1_swing_p0"][1]+1]
print("FR1_swing_p0_res")
print(FR1_swing_p0_res)

FR1_swing_p1_res = x_opt[var_index_L1["FR1_swing_p1"][0]:var_index_L1["FR1_swing_p1"][1]+1]
print("FR1_swing_p1_res")
print(FR1_swing_p1_res)

FR1_double_p0_res = x_opt[var_index_L1["FR1_double_p0"][0]:var_index_L1["FR1_double_p0"][1]+1]
print("FR1_double_p0_res")
print(FR1_double_p0_res)

FR1_double_p1_res = x_opt[var_index_L1["FR1_double_p1"][0]:var_index_L1["FR1_double_p1"][1]+1]
print("FR1_double_p1_res")
print(FR1_double_p1_res)

FR2_init_p0_res = x_opt[var_index_L1["FR2_initdouble_p0"][0]:var_index_L1["FR2_initdouble_p0"][1]+1]
print("FR2_init_p0_res")
print(FR2_init_p0_res)

FR2_init_p1_res = x_opt[var_index_L1["FR2_initdouble_p1"][0]:var_index_L1["FR2_initdouble_p1"][1]+1]
print("FR2_init_p1_res")
print(FR2_init_p1_res)

FR2_swing_p0_res = x_opt[var_index_L1["FR2_swing_p0"][0]:var_index_L1["FR2_swing_p0"][1]+1]
print("FR2_swing_p0_res")
print(FR2_swing_p0_res)

FR2_swing_p1_res = x_opt[var_index_L1["FR2_swing_p1"][0]:var_index_L1["FR2_swing_p1"][1]+1]
print("FR2_swing_p1_res")
print(FR2_swing_p1_res)

FR2_double_p0_res = x_opt[var_index_L1["FR2_double_p0"][0]:var_index_L1["FR2_double_p0"][1]+1]
print("FR2_double_p0_res")
print(FR2_double_p0_res)

FR2_double_p1_res = x_opt[var_index_L1["FR2_double_p1"][0]:var_index_L1["FR2_double_p1"][1]+1]
print("FR2_double_p1_res")
print(FR2_double_p1_res)

FR3_init_p0_res = x_opt[var_index_L1["FR3_initdouble_p0"][0]:var_index_L1["FR3_initdouble_p0"][1]+1]
print("FR3_init_p0_res")
print(FR3_init_p0_res)

FR3_init_p1_res = x_opt[var_index_L1["FR3_initdouble_p1"][0]:var_index_L1["FR3_initdouble_p1"][1]+1]
print("FR3_init_p1_res")
print(FL3_init_p1_res)

FR3_swing_p0_res = x_opt[var_index_L1["FR3_swing_p0"][0]:var_index_L1["FR3_swing_p0"][1]+1]
print("FR3_swing_p0_res")
print(FR3_swing_p0_res)

FR3_swing_p1_res = x_opt[var_index_L1["FR3_swing_p1"][0]:var_index_L1["FR3_swing_p1"][1]+1]
print("FR3_swing_p1_res")
print(FR3_swing_p1_res)

FR3_double_p0_res = x_opt[var_index_L1["FR3_double_p0"][0]:var_index_L1["FR3_double_p0"][1]+1]
print("FR3_double_p0_res")
print(FR3_double_p0_res)

FR3_double_p1_res = x_opt[var_index_L1["FR3_double_p1"][0]:var_index_L1["FR3_double_p1"][1]+1]
print("FR3_double_p1_res")
print(FR3_double_p1_res)

FR4_init_p0_res = x_opt[var_index_L1["FR4_initdouble_p0"][0]:var_index_L1["FR4_initdouble_p0"][1]+1]
print("FR4_init_p0_res")
print(FR4_init_p0_res)

FR4_init_p1_res = x_opt[var_index_L1["FR4_initdouble_p1"][0]:var_index_L1["FR4_initdouble_p1"][1]+1]
print("FR4_init_p1_res")
print(FR4_init_p1_res)

FR4_swing_p0_res = x_opt[var_index_L1["FR4_swing_p0"][0]:var_index_L1["FR4_swing_p0"][1]+1]
print("FR4_swing_p0_res")
print(FR4_swing_p0_res)

FR4_swing_p1_res = x_opt[var_index_L1["FR4_swing_p1"][0]:var_index_L1["FR4_swing_p1"][1]+1]
print("FR4_swing_p1_res")
print(FR4_swing_p1_res)

FR4_double_p0_res = x_opt[var_index_L1["FR4_double_p0"][0]:var_index_L1["FR4_double_p0"][1]+1]
print("FR4_double_p0_res")
print(FR4_double_p0_res)

FR4_double_p1_res = x_opt[var_index_L1["FR4_double_p1"][0]:var_index_L1["FR4_double_p1"][1]+1]
print("FR4_double_p1_res")
print(FR4_double_p1_res)

#Plot trajectories

C_p0 = np.array(C_start)
C_p1 = T/3*np.array(Cdot_start) + C_p0
C_p2 = T**2/6*np.array(Cddot_start) + 2*C_p1 - C_p0

#init time tick
t = 0
#Loop over all Phases (Knots)
Nphase = 3
Nk_Local = 5
for Nph in range(Nphase):
    #Decide Number of Knots
    if Nph == Nphase-1:  #The last Knot belongs to the Last Phase
        Nk_ThisPhase = Nk_Local+1
    else:
        Nk_ThisPhase = Nk_Local

    #Compute time increment
    delta_t = TimeVec[Nph]/Nk_Local

    for Local_k_Count in range(Nk_ThisPhase):

        C_t = 1.0*C_p0*(1 - t/T)**3 + 3.0*C_p1*(t/T)*(1 - t/T)**2 + 3.0*C_p2*(t/T)**2*(1 - (t/T)) + 1.0*(t/T)**3*Cy_res

        L_t = 1.0*L0_res*(1 - t/T)**4 + 4.0*L1_res*(t/T)*(1 - t/T)**3 + 6.0*L2_res*(t/T)**2*(1 - t/T)**2 + 4.0*L3_res*(t/T)**3*(1 - t/T) + 1.0*L4_res*(t/T)**4

        #print(C_t[1])

        print("momentum")
        print(L_t)

        t = t + delta_t


#Check Newton Euler Equation
t = 0.
phase_t = t
#phase_t = t - TimeVec[0] - TimeVec[1]

G = [0,0,-9.80665]
m = 95

FL1_t = FL1_init_p0_res*(1.0 - 1.0*t/TimeVec[0]) + 1.0*FL1_init_p1_res*t/TimeVec[0]
FL2_t = FL2_init_p0_res*(1.0 - 1.0*t/TimeVec[0]) + 1.0*FL2_init_p1_res*t/TimeVec[0]
FL3_t = FL3_init_p0_res*(1.0 - 1.0*t/TimeVec[0]) + 1.0*FL3_init_p1_res*t/TimeVec[0]
FL4_t = FL4_init_p0_res*(1.0 - 1.0*t/TimeVec[0]) + 1.0*FL4_init_p1_res*t/TimeVec[0]

FR1_t = FR1_init_p0_res*(1.0 - 1.0*t/TimeVec[0]) + 1.0*FR1_init_p1_res*t/TimeVec[0]
FR2_t = FR2_init_p0_res*(1.0 - 1.0*t/TimeVec[0]) + 1.0*FR2_init_p1_res*t/TimeVec[0]
FR3_t = FR3_init_p0_res*(1.0 - 1.0*t/TimeVec[0]) + 1.0*FR3_init_p1_res*t/TimeVec[0]
FR4_t = FR4_init_p0_res*(1.0 - 1.0*t/TimeVec[0]) + 1.0*FR4_init_p1_res*t/TimeVec[0]

Ldot_t = (-L3_res + L4_res)*4/T*(t/T)**3 + (-L2_res + L3_res)*12.0/T*(t/T)**2*(1 - t/T)+ (-L1_res + L2_res)*12.0/T*(t/T)*(1 - t/T)**2+ (-L0_res + L1_res)*4.0/T*(1 - t/T)**3

Cddot_force = (FL1_t + FL2_t + FL3_t + FL4_t + FR1_t + FR2_t + FR3_t + FR4_t)/m - G

C_p0 = np.array(C_start)
C_p1 = T/3*np.array(Cdot_start) + C_p0
C_p2 = T**2/6*np.array(Cddot_start) + 2*C_p1 - C_p0
C_t = 1.0*C_p0*(1 - t/T)**3 + 3.0*C_p1*(t/T)*(1 - t/T)**2 + 3.0*C_p2*(t/T)**2*(1 - (t/T)) + 1.0*(t/T)**3*Cy_res
Cddot_t = 6.0*(t/T)*(C_p1 - 2*C_p2 + Cy_res)/T**2 + 6*(1.0 - 1.0*t/T)*(C_p0 - 2*C_p1 + C_p2)/T**2 - G
PL_init = np.array(PL_init)
PR_init = np.array(PR_init)

Ldot_force_t = np.cross(np.array(PL_init+np.array([0.11,0.06,0])-C_t).reshape((1,3)),np.array(FL1_t).reshape((1,3))) + np.cross(np.array(PL_init+np.array([0.11,-0.06,0])-C_t).reshape((1,3)),np.array(FL2_t).reshape((1,3))) + np.cross(np.array(PL_init+np.array([-0.11,0.06,0])-C_t).reshape((1,3)),np.array(FL3_t).reshape((1,3))) + np.cross(np.array(PL_init+np.array([-0.11,-0.06,0])-C_t).reshape((1,3)),np.array(FL4_t).reshape((1,3))) + np.cross(np.array(PR_init+np.array([0.11,0.06,0])-C_t).reshape((1,3)),np.array(FR1_t).reshape((1,3))) + np.cross(np.array(PR_init+np.array([0.11,-0.06,0])-C_t).reshape((1,3)),np.array(FR2_t).reshape((1,3))) + np.cross(np.array(PR_init+np.array([-0.11,0.06,0])-C_t).reshape((1,3)),np.array(FR3_t).reshape((1,3))) + np.cross(np.array(PR_init+np.array([-0.11,-0.06,0])-C_t).reshape((1,3)),np.array(FR4_t).reshape((1,3)))

print("Acc-force")
print(Cddot_force)
print("Acc")
print(Cddot_t)
print("Ldot")
print(Ldot_t)
print("Ldot_force")
print(Ldot_force_t)