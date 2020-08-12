#Description:
#   Functions for Building Problem Descriptions for Humanoid with Multi-fiedelity Planning Framework, using Bezier Curves to represent the trajectories
#   All coefficients computed from Steve's derive.py
#   wd repreent the coneficient for the cross produce
#   wu for the linear part
#   The first element of wd,wu is the coeficient of the unknown variable x
#   The second element of wd,wu is the constant
#   C_something means the multiplication is a cross product

#Coeficients
#Enforce Initial CoM Position, Velocity and Acceleration
#Results in 3 order CoM trajectory and 1 order CoM acceleration
#C(t) = 1.0*p0*(1 - t/T)**3 + 3.0*p1*(t/T)*(1 - t/T)**2 + 3.0*p2*(t/T)**2*(1 - (t/T)) + 1.0*(t/T)**3*x
#Cdot(t) = 3.0*(t/T)**2*(-p2 + x)/T + 6.0*(t/T)*(1 - t/T)*(-p1 + p2)/T + 3.0*(1 - t/T)**2*(-p0 + p1)/T
#Cddt(t) = 6.0*(t/T)*(p1 - 2*p2 + x)/T**2 + 6*(1.0 - 1.0*t/T)*(p0 - 2*p1 + p2)/T**2
#wd = [[0, (1.0*Cg*T**2*p0 - 12.0*Cp0*p1 + 6.0*Cp0*p2)/T**2], [2*Cp0/T**2, (Cg*T**2*p1 - 6*Cp0*p2 + 6*Cp1*p2)/T**2], [(-2*Cp0 + 6*Cp1)/T**2, (Cg*T**2*p2 - 6*Cp1*p2)/T**2], [(Cg*T**2 - 6*Cp1 + 12*Cp2)/T**2, 0]]
#wu = [[0, 6*(p0 - 2*p1 + p2)/T**2], [2/T**2, (4*p0 - 6*p1)/T**2], [4/T**2, (2*p0 - 6*p2)/T**2], [6/T**2, (6*p1 - 12*p2)/T**2]]

# Import Important Modules
import numpy as np #Numpy
import casadi as ca #Casadi
import matplotlib.pyplot as plt #Matplotlib
from mpl_toolkits.mplot3d import Axes3D
# Import SL1M modules
from sl1m.constants_and_tools import *
from sl1m.planner import *
from constraints import *

#   Set Decimal Printing Precision
np.set_printoptions(precision=4)

#FUNCTION: Build a single step Bezier Curve Problem, Discretized Formulation
#Parameters:
#   m: robot mass, default value set as the one of Talos
def Bezier_SingleStep_Discrete_Order3(m = 95, StandAlong = True, ParameterList = None):
    #print problem setup
    print("Bezier Problem Setup:")
    print("Constrain Initial Position and Initial Velocity, CoM curve in the order of 2 (3 control points)")
    print("No angular Momentum")

    #Define Parameters
    #   Gait Pattern, Each action is followed up by a double support phase
    GaitPattern = ['InitialDouble','Swing','DoubleSupport'] #,'RightSupport','DoubleSupport','LeftSupport','DoubleSupport'

    #   Number of Phases
    Nphase = len(GaitPattern)
    #   Number of Steps
    Nstep = 1
    #   Number of Knots per Phase - how many intervals do we have for a single phase; 10 intervals need 11 knots/ticks
    Nk_Local= 5
    #   Compute Number of Total knots/ticks, but the enumeration start from 0 to N_K-1
    N_K = Nk_Local*Nphase + 1 #+1 the last knots to finalize the plan
    #   Robot mass
    #m = 95 #kg
    G = [0,0,-9.80665] #kg/m^2
    #   Terrain Model
    #       Flat Terrain
    TerrainNorm = [0,0,1] 
    TerrainTangentX = [1,0,0]
    TerrainTangentY = [0,1,0]
    miu = 0.3
    #   Force Limits
    Fxlb = -300
    Fxub = 300
    Fylb = -300
    Fyub = 300
    Fzlb = -300
    Fzub = 300
    #-----------------------------------------------------------------------------------------------------------------------
    #Kinematics Constraint for Talos
    kinematicConstraints = genKinematicConstraints(left_foot_constraints, right_foot_constraints)
    K_CoM_Left = kinematicConstraints[0][0]
    k_CoM_Left = kinematicConstraints[0][1]
    K_CoM_Right = kinematicConstraints[1][0]
    k_CoM_Right = kinematicConstraints[1][1]
    #Relative Foot Constraint matrices
    relativeConstraints = genFootRelativeConstraints(right_foot_in_lf_frame_constraints, left_foot_in_rf_frame_constraints)
    Q_lf_in_rf = relativeConstraints[0][0] #named rf in lf, but representing lf in rf
    q_lf_in_rf = relativeConstraints[0][1] #named rf in lf, but representing lf in rf
    Q_rf_in_lf = relativeConstraints[1][0] #named lf in rf, but representing rf in lf
    q_rf_in_lf = relativeConstraints[1][1] #named lf in rf, but representing rf in lf
    #-----------------------------------------------------------------------------------------------------------------------

    #-----------------------------------------------------------------------------------------------------------------------
    #Get Casadi Parameters

    #Flags the first round
    ParaFirstRoundFlag = ParameterList["ParaFirstRoundFlag"]

    #Flags for Swing Legs (Defined as Parameters)
    ParaLeftSwingFlag = ParameterList["LeftSwingFlag"]
    ParaRightSwingFlag = ParameterList["RightSwingFlag"]

    #Initial CoM Position
    C_0 = ParameterList["CoM_0"]

    #Initial CoM Velocity
    Cdot_0 = ParameterList["CoMdot_0"]

    #Initial CoM Acceleration
    Cddot_0 = ParameterList["CoMddot_0"]

    #Terminal CoM Position
    C_end = ParameterList["CoM_end"]

    #Phase Duration Vector
    TimeVector = ParameterList["TimeVector"]

    #Total Time Duration
    T = ParameterList["TotalDuration"]

    #Initial Contact Locations
    PL_init = ParameterList["PL_Init"]
    PR_init = ParameterList["PR_Init"]

    #Next Contact Location
    P_next = ParameterList["P_next"]

    #-----------------------------------------------------------------------------------------------------------------------
    #Define Variable, and Bounds

    #Control Point for CoM trajectory (Only one free variable is allowed no matter the order of the curve, otherwise we have non-convex formulation)
    Cy = ca.SX.sym('Cy',3)
    Cy_lb = np.array([[-30]*(Cy.shape[0]*Cy.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Cy_ub = np.array([[30]*(Cy.shape[0]*Cy.shape[1])])

    #Control Points for Angular Momentum (order of 3)
    Ldot0 = ca.SX.sym('Ldot0',3)
    Ldot0_lb = np.array([[-500]*(Ldot0.shape[0]*Ldot0.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Ldot0_ub = np.array([[500]*(Ldot0.shape[0]*Ldot0.shape[1])])

    Ldot1 = ca.SX.sym('Ldot1',3)
    Ldot1_lb = np.array([[-500]*(Ldot1.shape[0]*Ldot1.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Ldot1_ub = np.array([[500]*(Ldot1.shape[0]*Ldot1.shape[1])])

    Ldot2 = ca.SX.sym('Ldot2',3)
    Ldot2_lb = np.array([[-500]*(Ldot2.shape[0]*Ldot2.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Ldot2_ub = np.array([[500]*(Ldot2.shape[0]*Ldot2.shape[1])])

    Ldot3 = ca.SX.sym('Ldot3',3)
    Ldot3_lb = np.array([[-500]*(Ldot3.shape[0]*Ldot3.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Ldot3_ub = np.array([[500]*(Ldot3.shape[0]*Ldot3.shape[1])])

    #Control Points for Force Trejctory (Temparrily set 2 free variables, in the order of 1), for each phase
    #   Can be implemented by a for loop, later consider about it
    #   Left Foot
    #   Contact Point 1
    #       InitDouble Support
    #           Control point 0
    FL1_initdouble_p0 = ca.SX.sym('FL1_initdouble_p0',3)
    FL1_initdouble_p0_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FL1_initdouble_p0_ub = np.array([Fxub,Fyub,Fzub])
    #           Control point 1
    FL1_initdouble_p1 = ca.SX.sym('FL1_initdouble_p1',3)
    FL1_initdouble_p1_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FL1_initdouble_p1_ub = np.array([Fxub,Fyub,Fzub])
    #       Swing
    #           Control point 0
    FL1_swing_p0 = ca.SX.sym('FL1_swing_p0',3)
    FL1_swing_p0_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FL1_swing_p0_ub = np.array([Fxub,Fyub,Fzub])
    #           Control point 1
    FL1_swing_p1 = ca.SX.sym('FL1_swing_p1',3)
    FL1_swing_p1_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FL1_swing_p1_ub = np.array([Fxub,Fyub,Fzub])
    #       DoubleSupport
    #           Control point 0
    FL1_double_p0 = ca.SX.sym('FL1_double_p0',3)
    FL1_double_p0_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FL1_double_p0_ub = np.array([Fxub,Fyub,Fzub])
    #           Control point 1
    FL1_double_p1 = ca.SX.sym('FL1_double_p1',3)
    FL1_double_p1_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FL1_double_p1_ub = np.array([Fxub,Fyub,Fzub])
    #   Contact Point 2
    #       InitDouble Support
    #           Control point 0
    FL2_initdouble_p0 = ca.SX.sym('FL2_initdouble_p0',3)
    FL2_initdouble_p0_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FL2_initdouble_p0_ub = np.array([Fxub,Fyub,Fzub])
    #           Control point 1
    FL2_initdouble_p1 = ca.SX.sym('FL2_initdouble_p1',3)
    FL2_initdouble_p1_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FL2_initdouble_p1_ub = np.array([Fxub,Fyub,Fzub])
    #       Swing
    #           Control point 0
    FL2_swing_p0 = ca.SX.sym('FL2_swing_p0',3)
    FL2_swing_p0_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FL2_swing_p0_ub = np.array([Fxub,Fyub,Fzub])
    #           Control point 1
    FL2_swing_p1 = ca.SX.sym('FL2_swing_p1',3)
    FL2_swing_p1_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FL2_swing_p1_ub = np.array([Fxub,Fyub,Fzub])
    #       DoubleSupport
    #           Control point 0
    FL2_double_p0 = ca.SX.sym('FL2_double_p0',3)
    FL2_double_p0_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FL2_double_p0_ub = np.array([Fxub,Fyub,Fzub])
    #           Control point 1
    FL2_double_p1 = ca.SX.sym('FL2_double_p1',3)
    FL2_double_p1_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FL2_double_p1_ub = np.array([Fxub,Fyub,Fzub])
    #   Contact Point 3
    #       InitDouble Support
    #           Control point 0
    FL3_initdouble_p0 = ca.SX.sym('FL3_initdouble_p0',3)
    FL3_initdouble_p0_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FL3_initdouble_p0_ub = np.array([Fxub,Fyub,Fzub])
    #           Control point 1
    FL3_initdouble_p1 = ca.SX.sym('FL3_initdouble_p1',3)
    FL3_initdouble_p1_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FL3_initdouble_p1_ub = np.array([Fxub,Fyub,Fzub])
    #       Swing
    #           Control point 0
    FL3_swing_p0 = ca.SX.sym('FL3_swing_p0',3)
    FL3_swing_p0_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FL3_swing_p0_ub = np.array([Fxub,Fyub,Fzub])
    #           Control point 1
    FL3_swing_p1 = ca.SX.sym('FL3_swing_p1',3)
    FL3_swing_p1_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FL3_swing_p1_ub = np.array([Fxub,Fyub,Fzub])
    #       DoubleSupport
    #           Control point 0
    FL3_double_p0 = ca.SX.sym('FL3_double_p0',3)
    FL3_double_p0_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FL3_double_p0_ub = np.array([Fxub,Fyub,Fzub])
    #           Control point 1
    FL3_double_p1 = ca.SX.sym('FL3_double_p1',3)
    FL3_double_p1_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FL3_double_p1_ub = np.array([Fxub,Fyub,Fzub])
    #   Contact Point 4
    #       InitDouble Support
    #           Control point 0
    FL4_initdouble_p0 = ca.SX.sym('FL4_initdouble_p0',3)
    FL4_initdouble_p0_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FL4_initdouble_p0_ub = np.array([Fxub,Fyub,Fzub])
    #           Control point 1
    FL4_initdouble_p1 = ca.SX.sym('FL4_initdouble_p1',3)
    FL4_initdouble_p1_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FL4_initdouble_p1_ub = np.array([Fxub,Fyub,Fzub])
    #       Swing
    #           Control point 0
    FL4_swing_p0 = ca.SX.sym('FL4_swing_p0',3)
    FL4_swing_p0_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FL4_swing_p0_ub = np.array([Fxub,Fyub,Fzub])
    #           Control point 1
    FL4_swing_p1 = ca.SX.sym('FL4_swing_p1',3)
    FL4_swing_p1_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FL4_swing_p1_ub = np.array([Fxub,Fyub,Fzub])
    #       DoubleSupport
    #           Control point 0
    FL4_double_p0 = ca.SX.sym('FL4_double_p0',3)
    FL4_double_p0_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FL4_double_p0_ub = np.array([Fxub,Fyub,Fzub])
    #           Control point 1
    FL4_double_p1 = ca.SX.sym('FL4_double_p1',3)
    FL4_double_p1_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FL4_double_p1_ub = np.array([Fxub,Fyub,Fzub])
    
    #   Right Foot
    #   Contact Point 1
    #       InitDouble Support
    #           Control point 0
    FR1_initdouble_p0 = ca.SX.sym('FR1_initdouble_p0',3)
    FR1_initdouble_p0_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FR1_initdouble_p0_ub = np.array([Fxub,Fyub,Fzub])
    #           Control point 1
    FR1_initdouble_p1 = ca.SX.sym('FR1_initdouble_p1',3)
    FR1_initdouble_p1_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FR1_initdouble_p1_ub = np.array([Fxub,Fyub,Fzub])
    #       Swing
    #           Control point 0
    FR1_swing_p0 = ca.SX.sym('FR1_swing_p0',3)
    FR1_swing_p0_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FR1_swing_p0_ub = np.array([Fxub,Fyub,Fzub])
    #           Control point 1
    FR1_swing_p1 = ca.SX.sym('FR1_swing_p1',3)
    FR1_swing_p1_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FR1_swing_p1_ub = np.array([Fxub,Fyub,Fzub])
    #       DoubleSupport
    #           Control point 0
    FR1_double_p0 = ca.SX.sym('FR1_double_p0',3)
    FR1_double_p0_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FR1_double_p0_ub = np.array([Fxub,Fyub,Fzub])
    #           Control point 1
    FR1_double_p1 = ca.SX.sym('FR1_double_p1',3)
    FR1_double_p1_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FR1_double_p1_ub = np.array([Fxub,Fyub,Fzub])
    #   Contact Point 2
    #       InitDouble Support
    #           Control point 0
    FR2_initdouble_p0 = ca.SX.sym('FR2_initdouble_p0',3)
    FR2_initdouble_p0_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FR2_initdouble_p0_ub = np.array([Fxub,Fyub,Fzub])
    #           Control point 1
    FR2_initdouble_p1 = ca.SX.sym('FR2_initdouble_p1',3)
    FR2_initdouble_p1_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FR2_initdouble_p1_ub = np.array([Fxub,Fyub,Fzub])
    #       Swing
    #           Control point 0
    FR2_swing_p0 = ca.SX.sym('FR2_swing_p0',3)
    FR2_swing_p0_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FR2_swing_p0_ub = np.array([Fxub,Fyub,Fzub])
    #           Control point 1
    FR2_swing_p1 = ca.SX.sym('FR2_swing_p1',3)
    FR2_swing_p1_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FR2_swing_p1_ub = np.array([Fxub,Fyub,Fzub])
    #       DoubleSupport
    #           Control point 0
    FR2_double_p0 = ca.SX.sym('FR2_double_p0',3)
    FR2_double_p0_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FR2_double_p0_ub = np.array([Fxub,Fyub,Fzub])
    #           Control point 1
    FR2_double_p1 = ca.SX.sym('FR2_double_p1',3)
    FR2_double_p1_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FR2_double_p1_ub = np.array([Fxub,Fyub,Fzub])
    #   Contact Point 3
    #       InitDouble Support
    #           Control point 0
    FR3_initdouble_p0 = ca.SX.sym('FR3_initdouble_p0',3)
    FR3_initdouble_p0_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FR3_initdouble_p0_ub = np.array([Fxub,Fyub,Fzub])
    #           Control point 1
    FR3_initdouble_p1 = ca.SX.sym('FR3_initdouble_p1',3)
    FR3_initdouble_p1_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FR3_initdouble_p1_ub = np.array([Fxub,Fyub,Fzub])
    #       Swing
    #           Control point 0
    FR3_swing_p0 = ca.SX.sym('FR3_swing_p0',3)
    FR3_swing_p0_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FR3_swing_p0_ub = np.array([Fxub,Fyub,Fzub])
    #           Control point 1
    FR3_swing_p1 = ca.SX.sym('FR3_swing_p1',3)
    FR3_swing_p1_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FR3_swing_p1_ub = np.array([Fxub,Fyub,Fzub])
    #       DoubleSupport
    #           Control point 0
    FR3_double_p0 = ca.SX.sym('FR3_double_p0',3)
    FR3_double_p0_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FR3_double_p0_ub = np.array([Fxub,Fyub,Fzub])
    #           Control point 1
    FR3_double_p1 = ca.SX.sym('FR3_double_p1',3)
    FR3_double_p1_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FR3_double_p1_ub = np.array([Fxub,Fyub,Fzub])
    #   Contact Point 4
    #       InitDouble Support
    #           Control point 0
    FR4_initdouble_p0 = ca.SX.sym('FR4_initdouble_p0',3)
    FR4_initdouble_p0_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FR4_initdouble_p0_ub = np.array([Fxub,Fyub,Fzub])
    #           Control point 1
    FR4_initdouble_p1 = ca.SX.sym('FR4_initdouble_p1',3)
    FR4_initdouble_p1_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FR4_initdouble_p1_ub = np.array([Fxub,Fyub,Fzub])
    #       Swing
    #           Control point 0
    FR4_swing_p0 = ca.SX.sym('FR4_swing_p0',3)
    FR4_swing_p0_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FR4_swing_p0_ub = np.array([Fxub,Fyub,Fzub])
    #           Control point 1
    FR4_swing_p1 = ca.SX.sym('FR4_swing_p1',3)
    FR4_swing_p1_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FR4_swing_p1_ub = np.array([Fxub,Fyub,Fzub])
    #       DoubleSupport
    #           Control point 0
    FR4_double_p0 = ca.SX.sym('FR4_double_p0',3)
    FR4_double_p0_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FR4_double_p0_ub = np.array([Fxub,Fyub,Fzub])
    #           Control point 1
    FR4_double_p1 = ca.SX.sym('FR4_double_p1',3)
    FR4_double_p1_lb = np.array([Fxlb,Fylb,Fzlb]) #particular way of generating lists in python, [value]*number of elements
    FR4_double_p1_ub = np.array([Fxub,Fyub,Fzub])

    #   Collect all Decision Variables
    DecisionVars = ca.vertcat(Cy, Ldot0, Ldot1, Ldot2, Ldot3, FL1_initdouble_p0, FL1_initdouble_p1, FL1_swing_p0, FL1_swing_p1, FL1_double_p0, FL1_double_p1, FL2_initdouble_p0, FL2_initdouble_p1, FL2_swing_p0, FL2_swing_p1, FL2_double_p0, FL2_double_p1, FL3_initdouble_p0, FL3_initdouble_p1, FL3_swing_p0, FL3_swing_p1, FL3_double_p0, FL3_double_p1, FL4_initdouble_p0, FL4_initdouble_p1, FL4_swing_p0, FL4_swing_p1, FL4_double_p0, FL4_double_p1, FR1_initdouble_p0, FR1_initdouble_p1, FR1_swing_p0, FR1_swing_p1, FR1_double_p0, FR1_double_p1, FR2_initdouble_p0, FR2_initdouble_p1, FR2_swing_p0, FR2_swing_p1, FR2_double_p0, FR2_double_p1, FR3_initdouble_p0, FR3_initdouble_p1, FR3_swing_p0, FR3_swing_p1, FR3_double_p0, FR3_double_p1, FR4_initdouble_p0, FR4_initdouble_p1, FR4_swing_p0, FR4_swing_p1, FR4_double_p0, FR4_double_p1)
    #print(DecisionVars)
    DecisionVarsShape = DecisionVars.shape

    #   Collect all lower bound and upper bound
    DecisionVars_lb = np.concatenate(((Cy_lb, Ldot0_lb, Ldot1_lb, Ldot2_lb, Ldot3_lb, FL1_initdouble_p0_lb, FL1_initdouble_p1_lb, FL1_swing_p0_lb, FL1_swing_p1_lb, FL1_double_p0_lb, FL1_double_p1_lb, FL2_initdouble_p0_lb, FL2_initdouble_p1_lb, FL2_swing_p0_lb, FL2_swing_p1_lb, FL2_double_p0_lb, FL2_double_p1_lb, FL3_initdouble_p0_lb, FL3_initdouble_p1_lb, FL3_swing_p0_lb, FL3_swing_p1_lb, FL3_double_p0_lb, FL3_double_p1_lb, FL4_initdouble_p0_lb, FL4_initdouble_p1_lb, FL4_swing_p0_lb, FL4_swing_p1_lb, FL4_double_p0_lb, FL4_double_p1_lb, FR1_initdouble_p0_lb, FR1_initdouble_p1_lb, FR1_swing_p0_lb, FR1_swing_p1_lb, FR1_double_p0_lb, FR1_double_p1_lb, FR2_initdouble_p0_lb, FR2_initdouble_p1_lb, FR2_swing_p0_lb, FR2_swing_p1_lb, FR2_double_p0_lb, FR2_double_p1_lb, FR3_initdouble_p0_lb, FR3_initdouble_p1_lb, FR3_swing_p0_lb, FR3_swing_p1_lb, FR3_double_p0_lb, FR3_double_p1_lb, FR4_initdouble_p0_lb, FR4_initdouble_p1_lb, FR4_swing_p0_lb, FR4_swing_p1_lb, FR4_double_p0_lb, FR4_double_p1_lb)),axis=None)
    DecisionVars_ub = np.concatenate(((Cy_ub, Ldot0_ub, Ldot1_ub, Ldot2_ub, Ldot3_ub, FL1_initdouble_p0_ub, FL1_initdouble_p1_ub, FL1_swing_p0_ub, FL1_swing_p1_ub, FL1_double_p0_ub, FL1_double_p1_ub, FL2_initdouble_p0_ub, FL2_initdouble_p1_ub, FL2_swing_p0_ub, FL2_swing_p1_ub, FL2_double_p0_ub, FL2_double_p1_ub, FL3_initdouble_p0_ub, FL3_initdouble_p1_ub, FL3_swing_p0_ub, FL3_swing_p1_ub, FL3_double_p0_ub, FL3_double_p1_ub, FL4_initdouble_p0_ub, FL4_initdouble_p1_ub, FL4_swing_p0_ub, FL4_swing_p1_ub, FL4_double_p0_ub, FL4_double_p1_ub, FR1_initdouble_p0_ub, FR1_initdouble_p1_ub, FR1_swing_p0_ub, FR1_swing_p1_ub, FR1_double_p0_ub, FR1_double_p1_ub,  FR2_initdouble_p0_ub, FR2_initdouble_p1_ub, FR2_swing_p0_ub, FR2_swing_p1_ub, FR2_double_p0_ub, FR2_double_p1_ub, FR3_initdouble_p0_ub, FR3_initdouble_p1_ub, FR3_swing_p0_ub, FR3_swing_p1_ub, FR3_double_p0_ub, FR3_double_p1_ub, FR4_initdouble_p0_ub, FR4_initdouble_p1_ub, FR4_swing_p0_ub, FR4_swing_p1_ub, FR4_double_p0_ub, FR4_double_p1_ub)),axis=None)
    
    #   Compute Control points for CoM
    C_p0 = C_0
    C_p1 = T/3*Cdot_0 + C_p0
    C_p2 = T**2/6*Cddot_0 + 2*C_p1 - C_p0

    #-----------------------------------------------------------------------------------------------------------------------
    #Define Constrains and Running Cost
    g = []
    glb = []
    gub = []
    J = 0

    #Initial Position is Guaranteed by defined control points
    #Open Terminal Constraints

    #init time tick
    t = 0
    #Loop over all Phases (Knots)
    for Nph in range(Nphase):
        #Decide Number of Knots
        if Nph == Nphase-1:  #The last Knot belongs to the Last Phase
            Nk_ThisPhase = Nk_Local+1
        else:
            Nk_ThisPhase = Nk_Local

        #Compute time increment
        delta_t = TimeVector[Nph]/Nk_Local

        for Local_k_Count in range(Nk_ThisPhase):
            k = Nph*Nk_Local + Local_k_Count

            #Get CoM Knots
            C_t = 1.0*C_p0*(1 - t/T)**3 + 3.0*C_p1*(t/T)*(1 - t/T)**2 + 3.0*C_p2*(t/T)**2*(1 - (t/T)) + 1.0*(t/T)**3*Cy
            Cddot_t = 6.0*(t/T)*(C_p1 - 2*C_p2 + Cy)/T**2 + 6*(1.0 - 1.0*t/T)*(C_p0 - 2*C_p1 + C_p2)/T**2
            
            #Get Angular Momentum Rate
            Ldot_t = Ldot0*(1-t/T)**3 + Ldot1*3*(t/T)*(1-t/T)**2 + Ldot2*3*(t/T)**2*(1-t/T)+Ldot3*(t/T)**3

            #Get w
            wd_t = (1-t/T)**3*(ca.cross(ca.DM(G),C_p0) - 12/T**2*ca.cross(C_p0,C_p1) + 6/T**2*ca.cross(C_p0,C_p2)) + 3*(1-t/T)**2*(t/T)*(1/T**2*ca.cross(C_p0,Cy) + ca.cross(ca.DM(G),C_p1) - 6/T**2*ca.cross(C_p0,C_p2) + 6/T**2*ca.cross(C_p1,C_p2)) + 3*(1-t/T)*(t/T)**2*(-2/T**2*ca.cross(C_p0,Cy) + 6/T**2*ca.cross(C_p1,Cy) + ca.cross(ca.DM(G),C_p2) - 6/T**2*ca.cross(C_p1,C_p2)) + (t/T)**3*(ca.cross(ca.DM(G),Cy) - 6/T**2*ca.cross(C_p1,Cy) + 12/T**2*ca.cross(C_p2,Cy))
            wu_t = (1-t/T)**3*(6*(C_p0 - 2*C_p1 + C_p2)/T**2) + 3*(1-t/T)**2*(t/T)*(2/T**2*Cy + (4*C_p0 - 6*C_p1)/T**2) + 3*(1-t/T)*(t/T)**2*(4/T**2*Cy + (2*C_p0 - 6*C_p2)/T**2) + (t/T)**3*(6/T**2*Cy + (6*C_p1 - 12*C_p2)/T**2)

            if GaitPattern[Nph]=='InitialDouble':
                #Get Forces Knots Depend on Phases
                #Contact Point 1
                FL1_t = FL1_initdouble_p0*(1.0 - 1.0*t/T) + 1.0*FL1_initdouble_p1*t/T
                FR1_t = FR1_initdouble_p0*(1.0 - 1.0*t/T) + 1.0*FR1_initdouble_p1*t/T
                #Contact Point 2
                FL2_t = FL2_initdouble_p0*(1.0 - 1.0*t/T) + 1.0*FL2_initdouble_p1*t/T
                FR2_t = FR2_initdouble_p0*(1.0 - 1.0*t/T) + 1.0*FR2_initdouble_p1*t/T
                #Contact Point 3
                FL3_t = FL3_initdouble_p0*(1.0 - 1.0*t/T) + 1.0*FL3_initdouble_p1*t/T
                FR3_t = FR3_initdouble_p0*(1.0 - 1.0*t/T) + 1.0*FR3_initdouble_p1*t/T
                #Contact Point 4
                FL4_t = FL4_initdouble_p0*(1.0 - 1.0*t/T) + 1.0*FL4_initdouble_p1*t/T
                FR4_t = FR4_initdouble_p0*(1.0 - 1.0*t/T) + 1.0*FR4_initdouble_p1*t/T

                #Kinematics Constraint
                #   CoM in the Left foot
                g.append(ca.if_else(ParaFirstRoundFlag,K_CoM_Left@(C_t-PL_init)-ca.DM(k_CoM_Left),np.full((len(k_CoM_Left),),-1)))
                glb.append(np.full((len(k_CoM_Left),),-np.inf))
                gub.append(np.full((len(k_CoM_Left),),0))

                #   CoM in the Right foot
                g.append(ca.if_else(ParaFirstRoundFlag,K_CoM_Right@(C_t-PR_init)-ca.DM(k_CoM_Right),np.full((len(k_CoM_Right),),-1)))
                glb.append(np.full((len(k_CoM_Right),),-np.inf))
                gub.append(np.full((len(k_CoM_Right),),0))

                #Angular Dynamics
                if k<N_K-1: #double check the knot number is valid
                    g.append(m*wd_t + Ldot_t -ca.cross(PL_init+np.array([0.11,0.06,0]),FL1_t)+ca.cross(PL_init+np.array([0.11,-0.06,0]),FL2_t)+ca.cross(PL_init+np.array([-0.11,0.06,0]),FL3_t)+ca.cross(PL_init+np.array([-0.11,-0.06,0]),FL4_t)+ca.cross(PR_init+np.array([0.11,0.06,0]),FR1_t)+ca.cross(PR_init+np.array([0.11,-0.06,0]),FR2_t)+ca.cross(PR_init+np.array([-0.11,0.06,0]),FR3_t)+ca.cross(PR_init+np.array([-0.11,-0.06,0]),FR4_t))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))
            
            elif GaitPattern[Nph]=='Swing':
                #Get Forces Knots Depend on Phases
                #Contact Point 1
                FL1_t = FL1_swing_p0*(1.0 - 1.0*t/T) + 1.0*FL1_swing_p1*t/T
                FR1_t = FR1_swing_p0*(1.0 - 1.0*t/T) + 1.0*FR1_swing_p1*t/T
                #Contact Point 2
                FL2_t = FL2_swing_p0*(1.0 - 1.0*t/T) + 1.0*FL2_swing_p1*t/T
                FR2_t = FR2_swing_p0*(1.0 - 1.0*t/T) + 1.0*FR2_swing_p1*t/T
                #Contact Point 3
                FL3_t = FL3_swing_p0*(1.0 - 1.0*t/T) + 1.0*FL3_swing_p1*t/T
                FR3_t = FR3_swing_p0*(1.0 - 1.0*t/T) + 1.0*FR3_swing_p1*t/T
                #Contact Point 4
                FL4_t = FL4_swing_p0*(1.0 - 1.0*t/T) + 1.0*FL4_swing_p1*t/T
                FR4_t = FR4_swing_p0*(1.0 - 1.0*t/T) + 1.0*FR4_swing_p1*t/T

                #   Complementarity Condition
                #   If LEFT Foot is SWING (RIGHT is STATONARY), then Zero Forces for the LEFT Foot
                g.append(ca.if_else(ParaLeftSwingFlag,FL1_t,np.array([0,0,0])))
                glb.append(np.array([0,0,0]))
                gub.append(np.array([0,0,0]))

                g.append(ca.if_else(ParaLeftSwingFlag,FL2_t,np.array([0,0,0])))
                glb.append(np.array([0,0,0]))
                gub.append(np.array([0,0,0]))

                g.append(ca.if_else(ParaLeftSwingFlag,FL3_t,np.array([0,0,0])))
                glb.append(np.array([0,0,0]))
                gub.append(np.array([0,0,0]))

                g.append(ca.if_else(ParaLeftSwingFlag,FL4_t,np.array([0,0,0])))
                glb.append(np.array([0,0,0]))
                gub.append(np.array([0,0,0]))

                #   If RIGHT Foot is SWING (LEFT is STATIONARY), then Zero Forces for the RIGHT Foot
                g.append(ca.if_else(ParaRightSwingFlag,FR1_t,np.array([0,0,0])))
                glb.append(np.array([0,0,0]))
                gub.append(np.array([0,0,0]))

                g.append(ca.if_else(ParaRightSwingFlag,FR2_t,np.array([0,0,0])))
                glb.append(np.array([0,0,0]))
                gub.append(np.array([0,0,0]))

                g.append(ca.if_else(ParaRightSwingFlag,FR3_t,np.array([0,0,0])))
                glb.append(np.array([0,0,0]))
                gub.append(np.array([0,0,0]))

                g.append(ca.if_else(ParaRightSwingFlag,FR4_t,np.array([0,0,0])))
                glb.append(np.array([0,0,0]))
                gub.append(np.array([0,0,0]))

                #Kinematics Constraint and Angular Dynamics Constraint

                #IF LEFT Foot is SWING (RIGHT FOOT is STATIONARY)
                #Kinematics Constraint
                #   CoM in the RIGHT Foot
                g.append(ca.if_else(ParaLeftSwingFlag, K_CoM_Right@(C_t-PR_init)-ca.DM(k_CoM_Right), np.full((len(k_CoM_Right),),-1)))
                glb.append(np.full((len(k_CoM_Right),),-np.inf))
                gub.append(np.full((len(k_CoM_Right),),0))

                #   Angular Dynamics(Right Support)
                if k<N_K-1:
                    g.append(ca.if_else(ParaLeftSwingFlag, m*wd_t + Ldot_t -ca.cross(PR_init+np.array([0.11,0.06,0]),FR1_t)+ca.cross(PR_init+np.array([0.11,-0.06,0]),FR2_t)+ca.cross(PR_init+np.array([-0.11,0.06,0]),FR3_t)+ca.cross(PR_init+np.array([-0.11,-0.06,0]),FR4_t), np.array([0,0,0])))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))

                #If RIGHT foot is SWING (LEFT is STATIONARY), Then LEFT Foot is the Support FOOT
                #Kinematics Constraint
                #   CoM in the Left foot
                g.append(ca.if_else(ParaRightSwingFlag, K_CoM_Left@(C_t-PL_init)-ca.DM(k_CoM_Left), np.full((len(k_CoM_Left),),-1)))
                glb.append(np.full((len(k_CoM_Left),),-np.inf))
                gub.append(np.full((len(k_CoM_Left),),0))

                #   Angular Dynamics(Left Support)
                if k<N_K-1:
                    g.append(ca.if_else(ParaRightSwingFlag, m*wd_t + Ldot_t -ca.cross(PL_init+np.array([0.11,0.06,0]),FL1_t)+ca.cross(PL_init+np.array([0.11,-0.06,0]),FL2_t)+ca.cross(PL_init+np.array([-0.11,0.06,0]),FL3_t)+ca.cross(PL_init+np.array([-0.11,-0.06,0]),FL4_t), np.array([0,0,0])))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))

            elif GaitPattern[Nph]=='DoubleSupport':
                #Get Forces Knots Depend on Phases
                #Contact Point 1
                FL1_t = FL1_double_p0*(1.0 - 1.0*t/T) + 1.0*FL1_double_p1*t/T
                FR1_t = FR1_double_p0*(1.0 - 1.0*t/T) + 1.0*FR1_double_p1*t/T
                #Contact Point 2
                FL2_t = FL2_double_p0*(1.0 - 1.0*t/T) + 1.0*FL2_double_p1*t/T
                FR2_t = FR2_double_p0*(1.0 - 1.0*t/T) + 1.0*FR2_double_p1*t/T
                #Contact Point 3
                FL3_t = FL3_double_p0*(1.0 - 1.0*t/T) + 1.0*FL3_double_p1*t/T
                FR3_t = FR3_double_p0*(1.0 - 1.0*t/T) + 1.0*FR3_double_p1*t/T
                #Contact Point 4
                FL4_t = FL4_double_p0*(1.0 - 1.0*t/T) + 1.0*FL4_double_p1*t/T
                FR4_t = FR4_double_p0*(1.0 - 1.0*t/T) + 1.0*FR4_double_p1*t/T
                
                #Kinematic Constraint
                
                #IF LEFT Foot is SWING (RIGHT FOOT is STATIONARY)
                #Kinematics Constraint
                #   CoM in the RIGHT Foot (Init Foot)
                g.append(ca.if_else(ParaLeftSwingFlag, K_CoM_Right@(C_t-PR_init)-ca.DM(k_CoM_Right), np.full((len(k_CoM_Right),),-1)))
                glb.append(np.full((len(k_CoM_Right),),-np.inf))
                gub.append(np.full((len(k_CoM_Right),),0))
                #   CoM in the LEFT foot (Moved/Swing - PL_k)
                g.append(ca.if_else(ParaLeftSwingFlag, K_CoM_Left@(C_t-P_next)-ca.DM(k_CoM_Left), np.full((len(k_CoM_Left),),-1)))
                glb.append(np.full((len(k_CoM_Left),),-np.inf))
                gub.append(np.full((len(k_CoM_Left),),0))

                #Angular Dynamics
                #Swing Left
                g.append(ca.if_else(ParaLeftSwingFlag, m*wd_t + Ldot_t - ca.cross(P_next+np.array([0.11,0.06,0]),FL1_t)+ca.cross(P_next+np.array([0.11,-0.06,0]),FL2_t)+ca.cross(P_next+np.array([-0.11,0.06,0]),FL3_t)+ca.cross(P_next+np.array([-0.11,-0.06,0]),FL4_t)+ca.cross(PR_init+np.array([0.11,0.06,0]),FR1_t)+ca.cross(PR_init+np.array([0.11,-0.06,0]),FR2_t)+ca.cross(PR_init+np.array([-0.11,0.06,0]),FR3_t)+ca.cross(PR_init+np.array([-0.11,-0.06,0]),FR4_t),np.array([0,0,0])))
                glb.append(np.array([0,0,0]))
                gub.append(np.array([0,0,0]))
                
                #if RIGHT Foot is SWING (LEFT FOOT is STATIONARY)
                #Kinematics Constraint
                #   CoM in the Left foot (Init Foot)
                g.append(ca.if_else(ParaRightSwingFlag, K_CoM_Left@(C_t-PL_init)-ca.DM(k_CoM_Left),np.full((len(k_CoM_Left),),-1)))
                glb.append(np.full((len(k_CoM_Left),),-np.inf))
                gub.append(np.full((len(k_CoM_Left),),0))
                #   CoM in the Right foot (Moved/Swing - PR_k) 
                g.append(ca.if_else(ParaRightSwingFlag,K_CoM_Right@(C_t-P_next)-ca.DM(k_CoM_Right),np.full((len(k_CoM_Right),),-1)))
                glb.append(np.full((len(k_CoM_Right),),-np.inf))
                gub.append(np.full((len(k_CoM_Right),),0))

                #Angular Dynamics
                #Swing Right
                g.append(ca.if_else(ParaRightSwingFlag, m*wd_t + Ldot_t -ca.cross(PL_init+np.array([0.11,0.06,0]),FL1_t)+ca.cross(PL_init+np.array([0.11,-0.06,0]),FL2_t)+ca.cross(PL_init+np.array([-0.11,0.06,0]),FL3_t)+ca.cross(PL_init+np.array([-0.11,-0.06,0]),FL4_t)+ca.cross(P_next+np.array([0.11,0.06,0]),FR1_t)+ca.cross(P_next+np.array([0.11,-0.06,0]),FR2_t)+ca.cross(P_next+np.array([-0.11,0.06,0]),FR3_t)+ca.cross(P_next+np.array([-0.11,-0.06,0]),FR4_t),np.array([0,0,0])))
                glb.append(np.array([0,0,0]))
                gub.append(np.array([0,0,0]))

            #-------------------------------------
            #Unilateral Forces and Friction Cone
            
            #Unilateral Forces
            #Left Foot 1
            g.append(FL1_t.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #Left Foot 2
            g.append(FL2_t.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #Left Foot 3
            g.append(FL3_t.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #Left Foot 4
            g.append(FL4_t.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])

            #Right Foot 1
            g.append(FR1_t.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #Right Foot 2
            g.append(FR2_t.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #Right Foot 3
            g.append(FR3_t.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #Right Foot 4
            g.append(FR4_t.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])

            #Friction Cone
            #   Left Foot 1 x-axis Set 1
            g.append(FL1_t.T@TerrainTangentX - miu*FL1_t.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Left Foot 1 x-axis Set 2
            g.append(FL1_t.T@TerrainTangentX + miu*FL1_t.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #   Left Foot 1 y-axis Set 1
            g.append(FL1_t.T@TerrainTangentY - miu*FL1_t.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Left Foot 1 Y-axis Set 2
            g.append(FL1_t.T@TerrainTangentY + miu*FL1_t.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])

            #   Left Foot 2 x-axis Set 1
            g.append(FL2_t.T@TerrainTangentX - miu*FL2_t.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Left Foot 2 x-axis Set 2
            g.append(FL2_t.T@TerrainTangentX + miu*FL2_t.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #   Left Foot 2 y-axis Set 1
            g.append(FL2_t.T@TerrainTangentY - miu*FL2_t.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Left Foot 2 Y-axis Set 2
            g.append(FL2_t.T@TerrainTangentY + miu*FL2_t.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])

            #   Left Foot 3 x-axis Set 1
            g.append(FL3_t.T@TerrainTangentX - miu*FL3_t.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Left Foot 3 x-axis Set 2
            g.append(FL3_t.T@TerrainTangentX + miu*FL3_t.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #   Left Foot 3 y-axis Set 1
            g.append(FL3_t.T@TerrainTangentY - miu*FL3_t.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Left Foot 3 Y-axis Set 2
            g.append(FL3_t.T@TerrainTangentY + miu*FL3_t.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])

            #   Left Foot 4 x-axis Set 1
            g.append(FL4_t.T@TerrainTangentX - miu*FL4_t.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Left Foot 4 x-axis Set 2
            g.append(FL4_t.T@TerrainTangentX + miu*FL4_t.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #   Left Foot 4 y-axis Set 1
            g.append(FL4_t.T@TerrainTangentY - miu*FL4_t.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Left Foot 4 Y-axis Set 2
            g.append(FL4_t.T@TerrainTangentY + miu*FL4_t.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])

            #   Right Foot 1 x-axis Set 1
            g.append(FR1_t.T@TerrainTangentX - miu*FR1_t.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Right Foot 1 x-axis Set 2
            g.append(FR1_t.T@TerrainTangentX + miu*FR1_t.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #   Right Foot 1 Y-axis Set 1
            g.append(FR1_t.T@TerrainTangentY - miu*FR1_t.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Right Foot 1 Y-axis Set 2
            g.append(FR1_t.T@TerrainTangentY + miu*FR1_t.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])

            #   Right Foot 2 x-axis Set 1
            g.append(FR2_t.T@TerrainTangentX - miu*FR2_t.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Right Foot 2 x-axis Set 2
            g.append(FR2_t.T@TerrainTangentX + miu*FR2_t.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #   Right Foot 2 Y-axis Set 1
            g.append(FR2_t.T@TerrainTangentY - miu*FR2_t.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Right Foot 2 Y-axis Set 2
            g.append(FR2_t.T@TerrainTangentY + miu*FR2_t.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])

            #   Right Foot 3 x-axis Set 1
            g.append(FR3_t.T@TerrainTangentX - miu*FR3_t.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Right Foot 3 x-axis Set 2
            g.append(FR3_t.T@TerrainTangentX + miu*FR3_t.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #   Right Foot 3 Y-axis Set 1
            g.append(FR3_t.T@TerrainTangentY - miu*FR3_t.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Right Foot 3 Y-axis Set 2
            g.append(FR3_t.T@TerrainTangentY + miu*FR3_t.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])

            #   Right Foot 4 x-axis Set 1
            g.append(FR4_t.T@TerrainTangentX - miu*FR4_t.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Right Foot 4 x-axis Set 2
            g.append(FR4_t.T@TerrainTangentX + miu*FR4_t.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #   Right Foot 4 Y-axis Set 1
            g.append(FR4_t.T@TerrainTangentY - miu*FR4_t.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Right Foot 4 Y-axis Set 2
            g.append(FR4_t.T@TerrainTangentY + miu*FR4_t.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])

            #Linear Dynamics
            g.append(m*wu_t - m*ca.DM(G) - FL1_t - FL2_t - FL3_t - FL4_t - FR1_t - FR2_t - FR3_t - FR4_t)
            glb.append(np.array([0,0,0]))
            gub.append(np.array([0,0,0]))

            J = J + delta_t*Cddot_t[0]**2 + delta_t*Cddot_t[1]**2+ delta_t*Cddot_t[2]**2 + delta_t*Ldot_t[0]**2 + delta_t*Ldot_t[1]**2 +delta_t*Ldot_t[2]**2

            #Update time 
            t = t + delta_t

    #Add Terminal Cost
    #J = J + 10000*(C_terminal[0]-C_end[0])**2 + (C_terminal[1]-C_end[1])**2 + 10000*(C_terminal[2]-C_end[2])**2

    #Compute Terminal CoM Position
    C_terminal = 1.0*C_p0*(1 - T/T)**3 + 3.0*C_p1*(T/T)*(1 - T/T)**2 + 3.0*C_p2*(T/T)**2*(1 - (T/T)) + 1.0*(T/T)**3*Cy

    #Relative Foot Constraints
    #   For init phase
    g.append(Q_rf_in_lf@(PR_init-PL_init))
    glb.append(np.full((len(q_rf_in_lf),),-np.inf))
    gub.append(q_rf_in_lf)

    
    #If LEFT foot is SWING (RIGHT is STATIONARY), Then LEFT Foot should Stay in the polytpe of the RIGHT FOOT
    g.append(ca.if_else(ParaLeftSwingFlag,Q_lf_in_rf@(P_next-PR_init)-ca.DM(q_lf_in_rf),np.full((len(q_lf_in_rf),),-1)))
    glb.append(np.full((len(q_lf_in_rf),),-np.inf))
    gub.append(np.full((len(q_lf_in_rf),),0))

    #If RIGHT foot is SWING (LEFT is STATIONARY), Then RIGHT Foot should stay in the polytope of the LEFT Foot
    g.append(ca.if_else(ParaRightSwingFlag,Q_rf_in_lf@(P_next-PL_init)-ca.DM(q_rf_in_lf),np.full((len(q_rf_in_lf),),-1)))
    glb.append(np.full((len(q_rf_in_lf),),-np.inf))
    gub.append(np.full((len(q_rf_in_lf),),0))

    #-----------------------------------------------------------------------------------------------------------------------
    #Get Variable Index - !!!This is the pure Index, when try to get the array using other routines, we need to add "+1" at the last index due to Python indexing conventions
    Cy_index = (0,2)
    Ldot0_index = (Cy_index[0]+3,Cy_index[1]+3)
    Ldot1_index = (Ldot0_index[0]+3,Ldot0_index[1]+3)
    Ldot2_index = (Ldot1_index[0]+3,Ldot1_index[1]+3)
    Ldot3_index = (Ldot2_index[0]+3,Ldot2_index[1]+3)
    FL1_initdouble_p0_index = (Ldot3_index[0]+3,Ldot3_index[1]+3)
    FL1_initdouble_p1_index = (FL1_initdouble_p0_index[0]+3,FL1_initdouble_p0_index[1]+3)
    FL1_swing_p0_index = (FL1_initdouble_p1_index[0]+3,FL1_initdouble_p1_index[1]+3)
    FL1_swing_p1_index = (FL1_swing_p0_index[0]+3,FL1_swing_p0_index[1]+3)
    FL1_double_p0_index = (FL1_swing_p1_index[0]+3,FL1_swing_p1_index[1]+3)
    FL1_double_p1_index = (FL1_double_p0_index[0]+3,FL1_double_p0_index[1]+3)

    FL2_initdouble_p0_index = (FL1_double_p1_index[0]+3,FL1_double_p1_index[1]+3)
    FL2_initdouble_p1_index = (FL2_initdouble_p0_index[0]+3,FL2_initdouble_p0_index[1]+3)
    FL2_swing_p0_index = (FL2_initdouble_p1_index[0]+3,FL2_initdouble_p1_index[1]+3)
    FL2_swing_p1_index = (FL2_swing_p0_index[0]+3,FL2_swing_p0_index[1]+3)
    FL2_double_p0_index = (FL2_swing_p1_index[0]+3,FL2_swing_p1_index[1]+3)
    FL2_double_p1_index = (FL2_double_p0_index[0]+3,FL2_double_p0_index[1]+3)

    FL3_initdouble_p0_index = (FL2_double_p1_index[0]+3,FL2_double_p1_index[1]+3)
    FL3_initdouble_p1_index = (FL3_initdouble_p0_index[0]+3,FL3_initdouble_p0_index[1]+3)
    FL3_swing_p0_index = (FL3_initdouble_p1_index[0]+3,FL3_initdouble_p1_index[1]+3)
    FL3_swing_p1_index = (FL3_swing_p0_index[0]+3,FL3_swing_p0_index[1]+3)
    FL3_double_p0_index = (FL3_swing_p1_index[0]+3,FL3_swing_p1_index[1]+3)
    FL3_double_p1_index = (FL3_double_p0_index[0]+3,FL3_double_p0_index[1]+3)

    FL4_initdouble_p0_index = (FL3_double_p1_index[0]+3,FL3_double_p1_index[1]+3)
    FL4_initdouble_p1_index = (FL4_initdouble_p0_index[0]+3,FL4_initdouble_p0_index[1]+3)
    FL4_swing_p0_index = (FL4_initdouble_p1_index[0]+3,FL4_initdouble_p1_index[1]+3)
    FL4_swing_p1_index = (FL4_swing_p0_index[0]+3,FL4_swing_p0_index[1]+3)
    FL4_double_p0_index = (FL4_swing_p1_index[0]+3,FL4_swing_p1_index[1]+3)
    FL4_double_p1_index = (FL4_double_p0_index[0]+3,FL4_double_p0_index[1]+3)

    FR1_initdouble_p0_index = (FL4_double_p1_index[0]+3,FL4_double_p1_index[1]+3)
    FR1_initdouble_p1_index = (FR1_initdouble_p0_index[0]+3,FR1_initdouble_p0_index[1]+3)
    FR1_swing_p0_index = (FR1_initdouble_p1_index[0]+3,FR1_initdouble_p1_index[1]+3)
    FR1_swing_p1_index = (FR1_swing_p0_index[0]+3,FR1_swing_p0_index[1]+3)
    FR1_double_p0_index = (FR1_swing_p1_index[0]+3,FR1_swing_p1_index[1]+3)
    FR1_double_p1_index = (FR1_double_p0_index[0]+3,FR1_double_p0_index[1]+3)

    FR2_initdouble_p0_index = (FR1_double_p1_index[0]+3,FR1_double_p1_index[1]+3)
    FR2_initdouble_p1_index = (FR2_initdouble_p0_index[0]+3,FR2_initdouble_p0_index[1]+3)
    FR2_swing_p0_index = (FR2_initdouble_p1_index[0]+3,FR2_initdouble_p1_index[1]+3)
    FR2_swing_p1_index = (FR2_swing_p0_index[0]+3,FR2_swing_p0_index[1]+3)
    FR2_double_p0_index = (FR2_swing_p1_index[0]+3,FR2_swing_p1_index[1]+3)
    FR2_double_p1_index = (FR2_double_p0_index[0]+3,FR2_double_p0_index[1]+3)

    FR3_initdouble_p0_index = (FR2_double_p1_index[0]+3,FR2_double_p1_index[1]+3)
    FR3_initdouble_p1_index = (FR3_initdouble_p0_index[0]+3,FL3_initdouble_p0_index[1]+3)
    FR3_swing_p0_index = (FR3_initdouble_p1_index[0]+3,FR3_initdouble_p1_index[1]+3)
    FR3_swing_p1_index = (FR3_swing_p0_index[0]+3,FR3_swing_p0_index[1]+3)
    FR3_double_p0_index = (FR3_swing_p1_index[0]+3,FR3_swing_p1_index[1]+3)
    FR3_double_p1_index = (FR3_double_p0_index[0]+3,FR3_double_p0_index[1]+3)

    FR4_initdouble_p0_index = (FR3_double_p1_index[0]+3,FR3_double_p1_index[1]+3)
    FR4_initdouble_p1_index = (FR4_initdouble_p0_index[0]+3,FR4_initdouble_p0_index[1]+3)
    FR4_swing_p0_index = (FR4_initdouble_p1_index[0]+3,FR4_initdouble_p1_index[1]+3)
    FR4_swing_p1_index = (FR4_swing_p0_index[0]+3,FR4_swing_p0_index[1]+3)
    FR4_double_p0_index = (FR4_swing_p1_index[0]+3,FR4_swing_p1_index[1]+3)
    FR4_double_p1_index = (FR4_double_p0_index[0]+3,FR4_double_p0_index[1]+3)
    
    var_index = {"Cy":Cy_index,
                 "Ldot0": Ldot0_index,
                 "Ldot1": Ldot1_index,
                 "Ldot2": Ldot2_index,
                 "Ldot3": Ldot3_index,
                 "FL1_initdouble_p0": FL1_initdouble_p0_index,
                 "FL1_initdouble_p1": FL1_initdouble_p1_index,
                 "FL1_swing_p0": FL1_swing_p0_index,
                 "FL1_swing_p1": FL1_swing_p1_index,
                 "FL1_double_p0":FL1_double_p0_index,
                 "FL1_double_p1":FL1_double_p1_index,
                 "FL2_initdouble_p0": FL2_initdouble_p0_index,
                 "FL2_initdouble_p1": FL2_initdouble_p1_index,
                 "FL2_swing_p0": FL2_swing_p0_index,
                 "FL2_swing_p1": FL2_swing_p1_index,
                 "FL2_double_p0":FL2_double_p0_index,
                 "FL2_double_p1":FL2_double_p1_index,
                 "FL3_initdouble_p0": FL3_initdouble_p0_index,
                 "FL3_initdouble_p1": FL3_initdouble_p1_index,
                 "FL3_swing_p0": FL3_swing_p0_index,
                 "FL3_swing_p1": FL3_swing_p1_index,
                 "FL3_double_p0":FL3_double_p0_index,
                 "FL3_double_p1":FL3_double_p1_index,
                 "FL4_initdouble_p0": FL4_initdouble_p0_index,
                 "FL4_initdouble_p1": FL4_initdouble_p1_index,
                 "FL4_swing_p0": FL4_swing_p0_index,
                 "FL4_swing_p1": FL4_swing_p1_index,
                 "FL4_double_p0":FL4_double_p0_index,
                 "FL4_double_p1":FL4_double_p1_index,
                 "FR1_initdouble_p0": FR1_initdouble_p0_index,
                 "FR1_initdouble_p1": FR1_initdouble_p1_index,
                 "FR1_swing_p0": FR1_swing_p0_index,
                 "FR1_swing_p1": FR1_swing_p1_index,
                 "FR1_double_p0":FR1_double_p0_index,
                 "FR1_double_p1":FR1_double_p1_index,
                 "FR2_initdouble_p0": FR2_initdouble_p0_index,
                 "FR2_initdouble_p1": FR2_initdouble_p1_index,
                 "FR2_swing_p0": FR2_swing_p0_index,
                 "FR2_swing_p1": FR2_swing_p1_index,
                 "FR2_double_p0":FR2_double_p0_index,
                 "FR2_double_p1":FR2_double_p1_index,
                 "FR3_initdouble_p0": FR3_initdouble_p0_index,
                 "FR3_initdouble_p1": FR3_initdouble_p1_index,
                 "FR3_swing_p0": FR3_swing_p0_index,
                 "FR3_swing_p1": FR3_swing_p1_index,
                 "FR3_double_p0":FR3_double_p0_index,
                 "FR3_double_p1":FR3_double_p1_index,
                 "FR4_initdouble_p0": FR4_initdouble_p0_index,
                 "FR4_initdouble_p1": FR4_initdouble_p1_index,
                 "FR4_swing_p0": FR4_swing_p0_index,
                 "FR4_swing_p1": FR4_swing_p1_index,
                 "FR4_double_p0":FR4_double_p0_index,
                 "FR4_double_p1":FR4_double_p1_index,
    }

    return DecisionVars, DecisionVars_lb, DecisionVars_ub, J, g, glb, gub, C_terminal, var_index
    #return solver, DecisionVars, DecisionVars_lb, DecisionVars_ub, J, g, glb, gub, var_index

def BuildSolver(FirstLevel = None, SecondLevel = None, m = 95, NumPhase = 3):
    #Check if the First Level is selected properly
    assert FirstLevel != None, "First Level is Not Selected."

    #Define Parameter Vector
    #-------------------------------------------
    #Define Solver Parameter Vector
    #Flag for defining the First Round
    ParaFirstRoundFlag = ca.SX.sym("ParaFirstRoundFlag")

    #Flags for Swing Legs (Defined as Parameters)
    ParaLeftSwingFlag = ca.SX.sym("LeftSwingFlag")
    ParaRightSwingFlag = ca.SX.sym("RightSwingFlag")

    #Initial CoM Position
    C_0 = ca.SX.sym("CoM_init",3)

    #Initial CoM Velocity
    Cdot_0 = ca.SX.sym("CoMdot_init",3)

    #Initial CoM acceleration
    Cddot_0 = ca.SX.sym("CoMddot_init",3)

    #Terminal CoM Postion in the future 
    C_end = ca.SX.sym("CoM_end",3)

    #Time Vector
    TimeVector = ca.SX.sym("TimeVector",NumPhase)

    #Total Time Duration
    T = ca.SX.sym("TimeDuration")

    #Initial Contact Locations
    PL_init = ca.SX.sym("PL_Init",3)
    PR_init = ca.SX.sym("PR_Init",3)

    #Next Contact Location
    P_next = ca.SX.sym("P_next",3)

    #   Collect all Parameters
    ParaList = {"ParaFirstRoundFlag":ParaFirstRoundFlag,
                "LeftSwingFlag":ParaLeftSwingFlag,
                "RightSwingFlag":ParaRightSwingFlag,
                "CoM_0":C_0,
                "CoMdot_0":Cdot_0,
                "CoMddot_0":Cddot_0,
                "CoM_end":C_end,
                "TimeVector":TimeVector,
                "TotalDuration":T,
                "PL_Init":PL_init,
                "PR_Init":PR_init,
                "P_next":P_next,
    }

    paras = ca.vertcat(ParaFirstRoundFlag,ParaLeftSwingFlag,ParaRightSwingFlag,C_0,Cdot_0,Cddot_0,C_end,TimeVector,T,PL_init,PR_init,P_next)

    #Identify the Fidelity Type of the whole framework, Used to tell the First Level to set Constraints Accordingly
    if SecondLevel == None:
        SingleFidelity = True
    else:
        SingleFidelity = False

    #Bulding the First Level
    if FirstLevel == "Bezier_SingleStep_Discrete_Order3":
        var_L1, var_lb_L1, var_ub_L1, J_L1, g_L1, glb_L1, gub_L1, CoM_end_L1, var_index_L1 = Bezier_SingleStep_Discrete_Order3(m = m, ParameterList = ParaList)
    else:
        print("Print Not implemented or Wrong Solver Build Enumeration")        

    #!!!!!Building the Second Level
    if SecondLevel == None:
        print("No Second Level")
        var_index_L2 = []
    else:
        print("Not Implemented")

    #Set-up Terminal Cost Here and Sum over all costs
    if SecondLevel == None: #No second Level
        #   Collect the variables, terminal cost set as the end of the single first level
        J = J_L1
        J = J + 100*(CoM_end_L1[0] - C_end[0])**2 + 100*(CoM_end_L1[1] - C_end[1])**2 + 100*(CoM_end_L1[2] - C_end[2])**2
    else:
        print("Second Level not Implemented")

    #Lamp all Levels
    #   No Second Level
    if SecondLevel == None:
        DecisionVars = var_L1
        DecisionVars_lb = var_lb_L1
        DecisionVars_ub = var_ub_L1
        #need to reshape constraints
        g = ca.vertcat(*g_L1)
        glb = np.concatenate(glb_L1)
        gub = np.concatenate(gub_L1)
        #var_index = {"Level1_Var_Index": var_index_Level1}
    #   With Second Level
    else:
        print("Different Fidelity not implemented")
        
    #Collect all Variable Index
    var_index = {"Level1_Var_Index": var_index_L1,
                 "Level2_Var_Index": var_index_L2,
    }
    
    #   Build Solvers
    #Build Solver
    prob = {'x': DecisionVars, 'f': J, 'g': g, 'p': paras}
    solver = ca.qpsol('solver', 'gurobi', prob)

    return solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index



solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index = BuildSolver(FirstLevel = "Bezier_SingleStep_Discrete_Order3")

np.random.seed()
DecisionVarsShape = DecisionVars_lb.shape
DecisionVars_init = DecisionVars_lb + np.multiply(np.random.rand(DecisionVarsShape[0],).flatten(),(DecisionVars_ub-DecisionVars_lb))#   Fixed Value Initial Guess

FirstRoundFlag = 1
LeftSwingFlag = 0
RightSwingFlag = 1
C_start = [0.,0.,0.45]
Cdot_start = [0.0,0.,0.]
Cddot_start = [0.0,0.0,0.0]
C_end = [10,0,0.5]
TimeVec = [0.2,0.4,0.2]
T = 0.8
PL_init = [0, 0.1,0]
PR_init = [0,-0.1,0]
P_next = [0.1,-0.1,0]

ParaList = np.concatenate((FirstRoundFlag,LeftSwingFlag,RightSwingFlag,C_start,Cdot_start,Cddot_start,C_end,TimeVec,T,PL_init,PR_init,P_next),axis=None)

#res = solver(x0=DecisionVars_init, p = ParaList, lbx = DecisionVars_lb, ubx = DecisionVars_ub, lbg = glb, ubg = gub)

res = solver(x0=DecisionVars_init, p = ParaList,lbx = DecisionVars_lb, ubx = DecisionVars_ub,lbg = glb, ubg = gub)

x_opt = res['x']
print(solver.stats()["success"])
print('x_opt: ', x_opt)