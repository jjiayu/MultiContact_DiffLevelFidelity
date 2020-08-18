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
    L0 = ca.SX.sym('L0',3)
    L0_lb = np.array([[-5]*(L0.shape[0]*L0.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    L0_ub = np.array([[5]*(L0.shape[0]*L0.shape[1])])

    L1 = ca.SX.sym('L1',3)
    L1_lb = np.array([[-5]*(L1.shape[0]*L1.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    L1_ub = np.array([[5]*(L1.shape[0]*L1.shape[1])])

    L2 = ca.SX.sym('L2',3)
    L2_lb = np.array([[-5]*(L2.shape[0]*L2.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    L2_ub = np.array([[5]*(L2.shape[0]*L2.shape[1])])

    L3 = ca.SX.sym('L3',3)
    L3_lb = np.array([[-5]*(L3.shape[0]*L3.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    L3_ub = np.array([[5]*(L3.shape[0]*L3.shape[1])])

    L4 = ca.SX.sym('L4',3)
    L4_lb = np.array([[-5]*(L4.shape[0]*L4.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    L4_ub = np.array([[5]*(L4.shape[0]*L4.shape[1])])

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
    DecisionVars = ca.vertcat(Cy, L0, L1, L2, L3, L4, FL1_initdouble_p0, FL1_initdouble_p1, FL1_swing_p0, FL1_swing_p1, FL1_double_p0, FL1_double_p1, FL2_initdouble_p0, FL2_initdouble_p1, FL2_swing_p0, FL2_swing_p1, FL2_double_p0, FL2_double_p1, FL3_initdouble_p0, FL3_initdouble_p1, FL3_swing_p0, FL3_swing_p1, FL3_double_p0, FL3_double_p1, FL4_initdouble_p0, FL4_initdouble_p1, FL4_swing_p0, FL4_swing_p1, FL4_double_p0, FL4_double_p1, FR1_initdouble_p0, FR1_initdouble_p1, FR1_swing_p0, FR1_swing_p1, FR1_double_p0, FR1_double_p1, FR2_initdouble_p0, FR2_initdouble_p1, FR2_swing_p0, FR2_swing_p1, FR2_double_p0, FR2_double_p1, FR3_initdouble_p0, FR3_initdouble_p1, FR3_swing_p0, FR3_swing_p1, FR3_double_p0, FR3_double_p1, FR4_initdouble_p0, FR4_initdouble_p1, FR4_swing_p0, FR4_swing_p1, FR4_double_p0, FR4_double_p1)
    #print(DecisionVars)
    DecisionVarsShape = DecisionVars.shape

    #   Collect all lower bound and upper bound
    DecisionVars_lb = np.concatenate(((Cy_lb, L0_lb, L1_lb, L2_lb, L3_lb, L4_lb, FL1_initdouble_p0_lb, FL1_initdouble_p1_lb, FL1_swing_p0_lb, FL1_swing_p1_lb, FL1_double_p0_lb, FL1_double_p1_lb, FL2_initdouble_p0_lb, FL2_initdouble_p1_lb, FL2_swing_p0_lb, FL2_swing_p1_lb, FL2_double_p0_lb, FL2_double_p1_lb, FL3_initdouble_p0_lb, FL3_initdouble_p1_lb, FL3_swing_p0_lb, FL3_swing_p1_lb, FL3_double_p0_lb, FL3_double_p1_lb, FL4_initdouble_p0_lb, FL4_initdouble_p1_lb, FL4_swing_p0_lb, FL4_swing_p1_lb, FL4_double_p0_lb, FL4_double_p1_lb, FR1_initdouble_p0_lb, FR1_initdouble_p1_lb, FR1_swing_p0_lb, FR1_swing_p1_lb, FR1_double_p0_lb, FR1_double_p1_lb, FR2_initdouble_p0_lb, FR2_initdouble_p1_lb, FR2_swing_p0_lb, FR2_swing_p1_lb, FR2_double_p0_lb, FR2_double_p1_lb, FR3_initdouble_p0_lb, FR3_initdouble_p1_lb, FR3_swing_p0_lb, FR3_swing_p1_lb, FR3_double_p0_lb, FR3_double_p1_lb, FR4_initdouble_p0_lb, FR4_initdouble_p1_lb, FR4_swing_p0_lb, FR4_swing_p1_lb, FR4_double_p0_lb, FR4_double_p1_lb)),axis=None)
    DecisionVars_ub = np.concatenate(((Cy_ub, L0_ub, L1_ub, L2_ub, L3_ub, L4_lb, FL1_initdouble_p0_ub, FL1_initdouble_p1_ub, FL1_swing_p0_ub, FL1_swing_p1_ub, FL1_double_p0_ub, FL1_double_p1_ub, FL2_initdouble_p0_ub, FL2_initdouble_p1_ub, FL2_swing_p0_ub, FL2_swing_p1_ub, FL2_double_p0_ub, FL2_double_p1_ub, FL3_initdouble_p0_ub, FL3_initdouble_p1_ub, FL3_swing_p0_ub, FL3_swing_p1_ub, FL3_double_p0_ub, FL3_double_p1_ub, FL4_initdouble_p0_ub, FL4_initdouble_p1_ub, FL4_swing_p0_ub, FL4_swing_p1_ub, FL4_double_p0_ub, FL4_double_p1_ub, FR1_initdouble_p0_ub, FR1_initdouble_p1_ub, FR1_swing_p0_ub, FR1_swing_p1_ub, FR1_double_p0_ub, FR1_double_p1_ub,  FR2_initdouble_p0_ub, FR2_initdouble_p1_ub, FR2_swing_p0_ub, FR2_swing_p1_ub, FR2_double_p0_ub, FR2_double_p1_ub, FR3_initdouble_p0_ub, FR3_initdouble_p1_ub, FR3_swing_p0_ub, FR3_swing_p1_ub, FR3_double_p0_ub, FR3_double_p1_ub, FR4_initdouble_p0_ub, FR4_initdouble_p1_ub, FR4_swing_p0_ub, FR4_swing_p1_ub, FR4_double_p0_ub, FR4_double_p1_ub)),axis=None)
    
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
            
            #Get Angular Momentum (for cost computation)
            L_t = 1.0*L0*(1 - t/T)**4 + 4.0*L1*(t/T)*(1 - t/T)**3 + 6.0*L2*(t/T)**2*(1 - t/T)**2 + 4.0*L3*(t/T)**3*(1 - t/T) + 1.0*L4*(t/T)**4

            #Get Angular Momentum Rate, derivative of Angular Momentum rate
            #Ldot_t = Ldot0*(1-t/T)**3 + Ldot1*3*(t/T)*(1-t/T)**2 + Ldot2*3*(t/T)**2*(1-t/T)+Ldot3*(t/T)**3
            Ldot_t = (-L3 + L4)*4/T*(t/T)**3 + (-L2 + L3)*12.0/T*(t/T)**2*(1 - t/T)+ (-L1 + L2)*12.0/T*(t/T)*(1 - t/T)**2+ (-L0 + L1)*4.0/T*(1 - t/T)**3

            #wu --- for linear part
            #Waypoints for wu
            wu_0 = 6*(C_p0 - 2*C_p1 + C_p2)/T**2
            wu_1 = 2/T**2*Cy + (4*C_p0 - 6*C_p1)/T**2
            wu_2 = 4/T**2*Cy + (2*C_p0 - 6*C_p2)/T**2
            wu_3 = 6/T**2*Cy + (6*C_p1 - 12*C_p2)/T**2

            wu_t = (1-t/T)**3*wu_0 + 3*(1-t/T)**2*(t/T)*wu_1 + 3*(1-t/T)*(t/T)**2*wu_2 + (t/T)**3*wu_3

            #wd ---- for cross product
            #Waypoints for wd
            wd_0 = (1.0*T**2*ca.cross(ca.DM(G),C_p0) - 12.0*ca.cross(C_p0,C_p1) + 6.0*ca.cross(C_p0,C_p2))/T**2
            wd_1 = 2*ca.cross(C_p0,Cy)/T**2 + (T**2*ca.cross(ca.DM(G),C_p1) - 6*ca.cross(C_p0,C_p2) + 6*ca.cross(C_p1,C_p2))/T**2
            wd_2 = (-2*ca.cross(C_p0,Cy) + 6*ca.cross(C_p1,Cy))/T**2 + (T**2*ca.cross(ca.DM(G),C_p2) - 6*ca.cross(C_p1,C_p2))/T**2
            wd_3 = (T**2*ca.cross(ca.DM(G),Cy) - 6*ca.cross(C_p1,Cy) + 12*ca.cross(C_p2,Cy))/T**2

            wd_t = (1-t/T)**3*wd_0 + 3*(1-t/T)**2*(t/T)*wd_1 + 3*(1-t/T)*(t/T)**2*wd_2 + (t/T)**3*wd_3

            if GaitPattern[Nph]=='InitialDouble':
                #Get Forces Knots Depend on Phases
                #Contact Point 1
                FL1_t = FL1_initdouble_p0*(1.0 - 1.0*t/TimeVector[0]) + 1.0*FL1_initdouble_p1*t/TimeVector[0]
                FR1_t = FR1_initdouble_p0*(1.0 - 1.0*t/TimeVector[0]) + 1.0*FR1_initdouble_p1*t/TimeVector[0]
                #Contact Point 2
                FL2_t = FL2_initdouble_p0*(1.0 - 1.0*t/TimeVector[0]) + 1.0*FL2_initdouble_p1*t/TimeVector[0]
                FR2_t = FR2_initdouble_p0*(1.0 - 1.0*t/TimeVector[0]) + 1.0*FR2_initdouble_p1*t/TimeVector[0]
                #Contact Point 3
                FL3_t = FL3_initdouble_p0*(1.0 - 1.0*t/TimeVector[0]) + 1.0*FL3_initdouble_p1*t/TimeVector[0]
                FR3_t = FR3_initdouble_p0*(1.0 - 1.0*t/TimeVector[0]) + 1.0*FR3_initdouble_p1*t/TimeVector[0]
                #Contact Point 4
                FL4_t = FL4_initdouble_p0*(1.0 - 1.0*t/TimeVector[0]) + 1.0*FL4_initdouble_p1*t/TimeVector[0]
                FR4_t = FR4_initdouble_p0*(1.0 - 1.0*t/TimeVector[0]) + 1.0*FR4_initdouble_p1*t/TimeVector[0]

                #Kinematics Constraint
                #   CoM in the Left foot
                g.append(K_CoM_Left@(C_t-PL_init)-ca.DM(k_CoM_Left))
                glb.append(np.full((len(k_CoM_Left),),-np.inf))
                gub.append(np.full((len(k_CoM_Left),),0))

                #   CoM in the Right foot
                g.append(K_CoM_Right@(C_t-PR_init)-ca.DM(k_CoM_Right))
                glb.append(np.full((len(k_CoM_Right),),-np.inf))
                gub.append(np.full((len(k_CoM_Right),),0))

                #Angular Dynamics
                if k<N_K-1: #double check the knot number is valid
                    g.append(m*wd_t + Ldot_t - ca.cross(PL_init+np.array([0.11,0.06,0]),FL1_t) - ca.cross(PL_init+np.array([0.11,-0.06,0]),FL2_t) - ca.cross(PL_init+np.array([-0.11,0.06,0]),FL3_t) - ca.cross(PL_init+np.array([-0.11,-0.06,0]),FL4_t) - ca.cross(PR_init+np.array([0.11,0.06,0]),FR1_t) - ca.cross(PR_init+np.array([0.11,-0.06,0]),FR2_t) - ca.cross(PR_init+np.array([-0.11,0.06,0]),FR3_t) - ca.cross(PR_init+np.array([-0.11,-0.06,0]),FR4_t))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))
            
            elif GaitPattern[Nph]=='Swing':
                #Get Forces Knots Depend on Phases
                #Contact Point 1
                FL1_t = FL1_swing_p0*(1.0 - 1.0*(t-TimeVector[0])/TimeVector[1]) + 1.0*FL1_swing_p1*(t-TimeVector[0])/TimeVector[1]
                FR1_t = FR1_swing_p0*(1.0 - 1.0*(t-TimeVector[0])/TimeVector[1]) + 1.0*FR1_swing_p1*(t-TimeVector[0])/TimeVector[1]
                #Contact Point 2
                FL2_t = FL2_swing_p0*(1.0 - 1.0*(t-TimeVector[0])/TimeVector[1]) + 1.0*FL2_swing_p1*(t-TimeVector[0])/TimeVector[1]
                FR2_t = FR2_swing_p0*(1.0 - 1.0*(t-TimeVector[0])/TimeVector[1]) + 1.0*FR2_swing_p1*(t-TimeVector[0])/TimeVector[1]
                #Contact Point 3
                FL3_t = FL3_swing_p0*(1.0 - 1.0*(t-TimeVector[0])/TimeVector[1]) + 1.0*FL3_swing_p1*(t-TimeVector[0])/TimeVector[1]
                FR3_t = FR3_swing_p0*(1.0 - 1.0*(t-TimeVector[0])/TimeVector[1]) + 1.0*FR3_swing_p1*(t-TimeVector[0])/TimeVector[1]
                #Contact Point 4
                FL4_t = FL4_swing_p0*(1.0 - 1.0*(t-TimeVector[0])/TimeVector[1]) + 1.0*FL4_swing_p1*(t-TimeVector[0])/TimeVector[1]
                FR4_t = FR4_swing_p0*(1.0 - 1.0*(t-TimeVector[0])/TimeVector[1]) + 1.0*FR4_swing_p1*(t-TimeVector[0])/TimeVector[1]

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
                    g.append(ca.if_else(ParaLeftSwingFlag, m*wd_t + Ldot_t - ca.cross(PR_init+np.array([0.11,0.06,0]),FR1_t) - ca.cross(PR_init+np.array([0.11,-0.06,0]),FR2_t) - ca.cross(PR_init+np.array([-0.11,0.06,0]),FR3_t) - ca.cross(PR_init+np.array([-0.11,-0.06,0]),FR4_t), np.array([0,0,0])))
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
                    g.append(ca.if_else(ParaRightSwingFlag, m*wd_t + Ldot_t - ca.cross(PL_init+np.array([0.11,0.06,0]),FL1_t) - ca.cross(PL_init+np.array([0.11,-0.06,0]),FL2_t) - ca.cross(PL_init+np.array([-0.11,0.06,0]),FL3_t) - ca.cross(PL_init+np.array([-0.11,-0.06,0]),FL4_t), np.array([0,0,0])))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))

            elif GaitPattern[Nph]=='DoubleSupport':
                #Get Forces Knots Depend on Phases
                #Contact Point 1
                FL1_t = FL1_double_p0*(1.0 - 1.0*(t-TimeVector[1]-TimeVector[0])/TimeVector[2]) + 1.0*FL1_double_p1*(t-TimeVector[1]-TimeVector[0])/TimeVector[2]
                FR1_t = FR1_double_p0*(1.0 - 1.0*(t-TimeVector[1]-TimeVector[0])/TimeVector[2]) + 1.0*FR1_double_p1*(t-TimeVector[1]-TimeVector[0])/TimeVector[2]
                #Contact Point 2
                FL2_t = FL2_double_p0*(1.0 - 1.0*(t-TimeVector[1]-TimeVector[0])/TimeVector[2]) + 1.0*FL2_double_p1*(t-TimeVector[1]-TimeVector[0])/TimeVector[2]
                FR2_t = FR2_double_p0*(1.0 - 1.0*(t-TimeVector[1]-TimeVector[0])/TimeVector[2]) + 1.0*FR2_double_p1*(t-TimeVector[1]-TimeVector[0])/TimeVector[2]
                #Contact Point 3
                FL3_t = FL3_double_p0*(1.0 - 1.0*(t-TimeVector[1]-TimeVector[0])/TimeVector[2]) + 1.0*FL3_double_p1*(t-TimeVector[1]-TimeVector[0])/TimeVector[2]
                FR3_t = FR3_double_p0*(1.0 - 1.0*(t-TimeVector[1]-TimeVector[0])/TimeVector[2]) + 1.0*FR3_double_p1*(t-TimeVector[1]-TimeVector[0])/TimeVector[2]
                #Contact Point 4
                FL4_t = FL4_double_p0*(1.0 - 1.0*(t-TimeVector[1]-TimeVector[0])/TimeVector[2]) + 1.0*FL4_double_p1*(t-TimeVector[1]-TimeVector[0])/TimeVector[2]
                FR4_t = FR4_double_p0*(1.0 - 1.0*(t-TimeVector[1]-TimeVector[0])/TimeVector[2]) + 1.0*FR4_double_p1*(t-TimeVector[1]-TimeVector[0])/TimeVector[2]
                
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
                g.append(ca.if_else(ParaLeftSwingFlag, m*wd_t + Ldot_t - ca.cross(P_next+np.array([0.11,0.06,0]),FL1_t) - ca.cross(P_next+np.array([0.11,-0.06,0]),FL2_t) - ca.cross(P_next+np.array([-0.11,0.06,0]),FL3_t) - ca.cross(P_next+np.array([-0.11,-0.06,0]),FL4_t) - ca.cross(PR_init+np.array([0.11,0.06,0]),FR1_t) - ca.cross(PR_init+np.array([0.11,-0.06,0]),FR2_t) - ca.cross(PR_init+np.array([-0.11,0.06,0]),FR3_t) - ca.cross(PR_init+np.array([-0.11,-0.06,0]),FR4_t),np.array([0,0,0])))
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
                g.append(ca.if_else(ParaRightSwingFlag, m*wd_t + Ldot_t - ca.cross(PL_init+np.array([0.11,0.06,0]),FL1_t) - ca.cross(PL_init+np.array([0.11,-0.06,0]),FL2_t) - ca.cross(PL_init+np.array([-0.11,0.06,0]),FL3_t) - ca.cross(PL_init+np.array([-0.11,-0.06,0]),FL4_t) - ca.cross(P_next+np.array([0.11,0.06,0]),FR1_t) - ca.cross(P_next+np.array([0.11,-0.06,0]),FR2_t) - ca.cross(P_next+np.array([-0.11,0.06,0]),FR3_t) - ca.cross(P_next+np.array([-0.11,-0.06,0]),FR4_t),np.array([0,0,0])))
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

            J = J + delta_t*Cddot_t[0]**2 + delta_t*Cddot_t[1]**2+ delta_t*Cddot_t[2]**2 + delta_t*L_t[0]**2 + delta_t*L_t[1]**2 +delta_t*L_t[2]**2

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
    L0_index = (Cy_index[0]+3,Cy_index[1]+3)
    L1_index = (L0_index[0]+3,L0_index[1]+3)
    L2_index = (L1_index[0]+3,L1_index[1]+3)
    L3_index = (L2_index[0]+3,L2_index[1]+3)
    L4_index = (L3_index[0]+3,L3_index[1]+3)
    FL1_initdouble_p0_index = (L4_index[0]+3,L4_index[1]+3)
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
    FR3_initdouble_p1_index = (FR3_initdouble_p0_index[0]+3,FR3_initdouble_p0_index[1]+3)
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
                 "L0": L0_index,
                 "L1": L1_index,
                 "L2": L2_index,
                 "L3": L3_index,
                 "L4": L4_index,
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

def CoM_Dynamics_Fixed_Time(m = 95, Nsteps = 4, StandAlong = False, StaticStop = False, ParameterList = None):
    #------------------------------------------------
    #Parameter Setup
    #   Set up Gait pattern
    GaitPattern = ["InitialDouble","Swing","DoubleSupport"] + ["Swing","DoubleSupport"]*(Nsteps-1)
    #   Number of Phases: Nsteps*2 + 1 (Initial Double Support)
    Nphase = len(GaitPattern)
    #   Number of Knots per Phase - how many intervals do we have for a single phase; 10 intervals need 11 knots/ticks
    Nk_Local= 5
    #   Compute Number of Total knots/ticks, but the enumeration start from 0 to N_K-1
    N_K = Nk_Local*Nphase + 1 #+1 the last knots to finalize the plan
    #   Robot mass
    #m = 95 #kg
    G = 9.80665 #kg/m^2
    #   Terrain Model
    #       Flat Terrain
    TerrainNorm = [0,0,1] 
    TerrainTangentX = [1,0,0]
    TerrainTangentY = [0,1,0]
    miu = 0.3
    #   Force Limits
    Fxlb = -300*4
    Fxub = 300*4
    Fylb = -300*4
    Fyub = 300*4
    Fzlb = -300*4
    Fzub = 300*4
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
    
    #Define Initial and Terminal Conditions, Get from casadi Parameters
    x_init = C_0[0]
    y_init = C_0[1]
    z_init = C_0[2]

    xdot_init = Cdot_0[0]
    ydot_init = Cdot_0[1]
    zdot_init = Cdot_0[2]

    PLx_init = PL_init[0]
    PLy_init = PL_init[1]
    PLz_init = PL_init[2]
    PL_init = ca.vertcat(PLx_init,PLy_init,PLz_init)

    PRx_init = PR_init[0]
    PRy_init = PR_init[1]
    PRz_init = PR_init[2]
    PR_init = ca.vertcat(PRx_init,PRy_init,PRz_init)

    x_end = C_end[0]
    y_end = C_end[1]
    z_end = C_end[2]

    #Surface Patches
    #SurfParas = ParameterList["SurfParas"]

    #-----------------------------------------------------------------------------------------------------------------------
    #Define Variables and Bounds, Parameters
    #   CoM Position x-axis
    x = ca.SX.sym('x',N_K)
    x_lb = np.array([[0]*(x.shape[0]*x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    x_ub = np.array([[10]*(x.shape[0]*x.shape[1])])
    #   CoM Position y-axis
    y = ca.SX.sym('y',N_K)
    y_lb = np.array([[-0.75]*(y.shape[0]*y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    y_ub = np.array([[0.75]*(y.shape[0]*y.shape[1])])
    #   CoM Position z-axis
    z = ca.SX.sym('z',N_K)
    z_lb = np.array([[0.55]*(z.shape[0]*z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    z_ub = np.array([[1]*(z.shape[0]*z.shape[1])])
    #   CoM Velocity x-axis
    xdot = ca.SX.sym('xdot',N_K)#0.75 old
    xdot_lb = np.array([[-0.75]*(xdot.shape[0]*xdot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    xdot_ub = np.array([[0.75]*(xdot.shape[0]*xdot.shape[1])])
    #   CoM Velocity y-axis
    ydot = ca.SX.sym('ydot',N_K)
    ydot_lb = np.array([[-0.75]*(ydot.shape[0]*ydot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    ydot_ub = np.array([[0.75]*(ydot.shape[0]*ydot.shape[1])])
    #   CoM Velocity z-axis
    zdot = ca.SX.sym('zdot',N_K)
    zdot_lb = np.array([[-0.75]*(zdot.shape[0]*zdot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    zdot_ub = np.array([[0.75]*(zdot.shape[0]*zdot.shape[1])])

    #Left Foot Force
    #Left Foot x-axis
    FLx = ca.SX.sym('FLx',N_K)
    FLx_lb = np.array([[Fxlb]*(FLx.shape[0]*FLx.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FLx_ub = np.array([[Fxub]*(FLx.shape[0]*FLx.shape[1])])
    #Left Foot y-axis
    FLy = ca.SX.sym('FLy',N_K)
    FLy_lb = np.array([[Fylb]*(FLy.shape[0]*FLy.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FLy_ub = np.array([[Fyub]*(FLy.shape[0]*FLy.shape[1])])
    #Left Foot z-axis
    FLz = ca.SX.sym('FLz',N_K)
    FLz_lb = np.array([[Fzlb]*(FLz.shape[0]*FLz.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FLz_ub = np.array([[Fzub]*(FLz.shape[0]*FLz.shape[1])])

    #Right Contact Force
    #Right Foot x-axis
    FRx = ca.SX.sym('FRx',N_K)
    FRx_lb = np.array([[Fxlb]*(FRx.shape[0]*FRx.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FRx_ub = np.array([[Fxub]*(FRx.shape[0]*FRx.shape[1])])
    #Right Foot y-axis
    FRy = ca.SX.sym('FRy',N_K)
    FRy_lb = np.array([[Fylb]*(FRy.shape[0]*FRy.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FRy_ub = np.array([[Fyub]*(FRy.shape[0]*FRy.shape[1])])
    #Right Foot z-axis
    FRz = ca.SX.sym('FRz',N_K)
    FRz_lb = np.array([[Fzlb]*(FRz.shape[0]*FRz.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FRz_ub = np.array([[Fzub]*(FRz.shape[0]*FRz.shape[1])])

    #   Init Contact Location
    #   px
    px_init = ca.SX.sym('px_init')
    px_init_lb = np.array([-1])
    px_init_ub = np.array([10])

    #   py
    py_init = ca.SX.sym('py_init')
    py_init_lb = np.array([-0.75])
    py_init_ub = np.array([0.75])

    #   pz
    pz_init = ca.SX.sym('pz_init')
    pz_init_lb = np.array([-1])
    pz_init_ub = np.array([1])

    #   Contact Location Sequence
    px = [] 
    py = []
    pz = []
    px_lb = []
    py_lb = []
    pz_lb = []
    px_ub = []
    py_ub = []
    pz_ub = []
    for stepIdx in range(Nsteps):
        pxtemp = ca.SX.sym('px'+str(stepIdx+1)) #0 + 1
        px.append(pxtemp)
        px_lb.append(np.array([-0.5]))
        px_ub.append(np.array([10]))

        pytemp = ca.SX.sym('py'+str(stepIdx+1))
        py.append(pytemp)
        py_lb.append(np.array([-1]))
        py_ub.append(np.array([1]))

        #   Foot steps are all staying on the ground
        pztemp = ca.SX.sym('pz'+str(stepIdx+1))
        pz.append(pztemp)
        pz_lb.append(np.array([0]))
        pz_ub.append(np.array([0]))

    #Switching Time Vector
    #Ts = []
    #Ts_lb = []
    #Ts_ub = []
    #for n_phase in range(Nphase):
    #    Tstemp = ca.SX.sym('Ts'+str(n_phase+1)) #0 + 1 + ....
    #    Ts.append(Tstemp)
    #    Ts_lb.append(np.array([0.05]))
    #    Ts_ub.append(np.array([2.0*(Nphase+1)])) #Consider the First Level

    #   Collect all Decision Variables
    DecisionVars = ca.vertcat(x,y,z,
                              xdot,ydot,zdot,
                              FLx,FLy,FLz,
                              FRx,FRy,FRz,
                              px_init,py_init,pz_init,
                              *px,*py,*pz)#,*Ts)
    #print(DecisionVars)
    DecisionVarsShape = DecisionVars.shape

    #   Collect all lower bound and upper bound
    DecisionVars_lb = np.concatenate(((x_lb,y_lb,z_lb,xdot_lb,ydot_lb,zdot_lb,FLx_lb,FLy_lb,FLz_lb,FRx_lb,FRy_lb,FRz_lb,px_init_lb,py_init_lb,pz_init_lb,px_lb,py_lb,pz_lb)),axis=None)
    DecisionVars_ub = np.concatenate(((x_ub,y_ub,z_ub,xdot_ub,ydot_ub,zdot_ub,FLx_ub,FLy_ub,FLz_ub,FRx_ub,FRy_ub,FRz_ub,px_init_ub,py_init_ub,pz_init_ub,px_ub,py_ub,pz_ub)),axis=None)
    
    #Time Span Setup
    #tau_upper_limit = 1
    #tauStepLength = tau_upper_limit/(N_K-1) #Get the interval length, total number of knots - 1

    #-----------------------------------------------------------------------------------------------------------------------
    #Define Constrains and Running Cost
    g = []
    glb = []
    gub = []
    J = 0

    ##Initial and Termianl Conditions
    ##   Terminal CoM y-axis
    #g.append(y[-1]-y_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    ##   Terminal CoM z-axis
    #g.append(z[-1]-z_end)
    #glb.append(np.array([0]))    
    #gub.append(np.array([0]))

    #if StaticStop == True:
        #   Terminal CoM Velocity x-axis
        #g.append(xdot[-1])
        #glb.append(np.array([0]))    
        #gub.append(np.array([0]))

        #   Terminal CoM Velocity y-axis
        #g.append(ydot[-1])
        #glb.append(np.array([0]))
        #gub.append(np.array([0]))

        #   Terminal CoM Velocity z-axis
        #g.append(zdot[-1])
        #glb.append(np.array([0]))    
        #gub.append(np.array([0]))

        #   Terminal CoM Acceleration x-axis
        #g.append(FLx[-2]/m+FRx[-2]/m)
        #glb.append(np.array([0]))    
        #gub.append(np.array([0]))

        #   Terminal CoM Acceleration y-axis
        #g.append(FLy[-2]/m+FRy[-2]/m)
        #glb.append(np.array([0]))    
        #gub.append(np.array([0]))

        #   Terminal CoM Acceleration z-axis
        #g.append(FLz[-2]/m+FRz[-2]/m - G)
        #glb.append(np.array([0]))    
        #gub.append(np.array([0]))

    h_doublesupport = 0.2
    h_swing = 0.4

    #Loop over all Phases (Knots)
    for Nph in range(Nphase):
        #Decide Number of Knots
        if Nph == Nphase-1:  #The last Knot belongs to the Last Phase
            Nk_ThisPhase = Nk_Local+1
        else:
            Nk_ThisPhase = Nk_Local       

        ##Decide Time Vector
        #if Nph == 0: #first phase
        #    h = tauStepLength*Nphase*(Ts[Nph]-0)
        #else: #other phases
        #    h = tauStepLength*Nphase*(Ts[Nph]-Ts[Nph-1]) 

        if GaitPattern[Nph]=='Swing':
            h = h_swing/Nk_Local
        elif GaitPattern[Nph]=='DoubleSupport' or 'InitialDouble':
            h = h_doublesupport/Nk_Local


        for Local_k_Count in range(Nk_ThisPhase):
            #Get knot number across the entire time line
            k = Nph*Nk_Local + Local_k_Count
            #print(k)
            #------------------------------------------
            #Build useful vectors
            #   Forces
            FL_k = ca.vertcat(FLx[k],FLy[k],FLz[k])
            FR_k = ca.vertcat(FRx[k],FRy[k],FRz[k])
            #   CoM
            CoM_k = ca.vertcat(x[k],y[k],z[k])

            #-------------------------------------------
            #Phase dependent Constraints (CoM Kinematics and Angular Dynamics)
            if GaitPattern[Nph]=='InitialDouble':
                #initial support foot (the landing foot from the first phase)
                p_init = ca.vertcat(px_init,py_init,pz_init)
                
                #If First Level Swing the Left, then the second level initial double support phase has p_init as the left support, PR_init as the right support
                #Kinematics Constraint
                #   CoM in the Left foot
                g.append(ca.if_else(ParaLeftSwingFlag,K_CoM_Left@(CoM_k-p_init)-ca.DM(k_CoM_Left),np.full((len(k_CoM_Left),),-1)))
                glb.append(np.full((len(k_CoM_Left),),-np.inf))
                gub.append(np.full((len(k_CoM_Left),),0))

                #   CoM in the Right foot
                g.append(ca.if_else(ParaLeftSwingFlag,K_CoM_Right@(CoM_k-PR_init)-ca.DM(k_CoM_Right),np.full((len(k_CoM_Right),),-1)))
                glb.append(np.full((len(k_CoM_Right),),-np.inf))
                gub.append(np.full((len(k_CoM_Right),),0))

                #If First Level Swing the Right, then the second level initial double support phase has p_init as the Right support, PL_init as the Left support
                #Kinematics Constraint
                #   CoM in the Left foot
                g.append(ca.if_else(ParaRightSwingFlag,K_CoM_Left@(CoM_k-PL_init)-ca.DM(k_CoM_Left),np.full((len(k_CoM_Left),),-1)))
                glb.append(np.full((len(k_CoM_Left),),-np.inf))
                gub.append(np.full((len(k_CoM_Left),),0))

                #   CoM in the Right foot
                g.append(ca.if_else(ParaRightSwingFlag,K_CoM_Right@(CoM_k-p_init)-ca.DM(k_CoM_Right),np.full((len(k_CoM_Right),),-1)))
                glb.append(np.full((len(k_CoM_Right),),-np.inf))
                gub.append(np.full((len(k_CoM_Right),),0))

            elif GaitPattern[Nph]=='Swing':
                
                if (Nph-1)//2 == 0:
                    #!!!!!!Pass from the first Level!!!!!!
                    P_k_current = ca.vertcat(px_init,py_init,pz_init) #ca.vertcat(px[-1],py[-1],pz[-1])
                    #!!!!!!
                    P_k_next = ca.vertcat(px[Nph//2],py[Nph//2],pz[Nph//2])
                else:
                    P_k_current = ca.vertcat(px[Nph//2-1],py[Nph//2-1],pz[Nph//2-1])
                    P_k_next = ca.vertcat(px[Nph//2],py[Nph//2],pz[Nph//2])

                if ((Nph-1)//2)%2 == 0: #even number steps
                    
                    #------------------------------------
                    #Zero Forces

                    #If First Level Swing the Left, then the second level Even Number Phases (the first Phase) Swing the Right -> Left -> Right
                    #RIGHT FOOT has zero forces
                    g.append(ca.if_else(ParaLeftSwingFlag,FR_k,np.array([0,0,0])))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))
                    
                    #If First Levvel Swing the Right, then the second level Even Number Phases (the first phase) Swing the Left -> Right -> Left
                    #LEFT FOOT has zero Forces
                    g.append(ca.if_else(ParaRightSwingFlag,FL_k,np.array([0,0,0])))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))

                    #------------------------------------
                    #CoM Kinematics

                    #If First Level Swing the Left, then the second level Even Number Phases (the first Phase) Swing the Right -> Left -> Right
                    #Left foot in contact for p_current, right foot is going to land as p_next
                    g.append(ca.if_else(ParaLeftSwingFlag, K_CoM_Left@(CoM_k-P_k_current)-ca.DM(k_CoM_Left), np.full((len(k_CoM_Left),),-1)))
                    glb.append(np.full((len(k_CoM_Left),),-np.inf))
                    gub.append(np.full((len(k_CoM_Left),),0))

                    #If First Levvel Swing the Right, then the second level Even Number Phases (the first phase) Swing the Left -> Right -> Left
                    #Right foot in contact for p_current, left foot is going to land at p_next
                    g.append(ca.if_else(ParaRightSwingFlag, K_CoM_Right@(CoM_k-P_k_current)-ca.DM(k_CoM_Right), np.full((len(k_CoM_Right),),-1)))
                    glb.append(np.full((len(k_CoM_Right),),-np.inf))
                    gub.append(np.full((len(k_CoM_Right),),0))

                elif ((Nph-1)//2)%2 == 1: #odd number steps
                    #------------------------------------
                    #Zero Forces

                    #If the first level swings the Left, then the second level for Odd Number Phases (the second phase) swings left -> Right -> Left
                    #LEFT FOOT has zero forces
                    g.append(ca.if_else(ParaLeftSwingFlag,FL_k,np.array([0,0,0])))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))

                    #If the first level swings the Right, then the second level for Odd Number Phases (the second phase) swings Right -> Left -> Right
                    #RIGHT FOOT has zero forces
                    g.append(ca.if_else(ParaRightSwingFlag,FR_k,np.array([0,0,0])))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))
                    
                    #------------------------------------
                    #CoM Kinematics
                    #If the first level swings the Left, then the second level for Odd Number Phases (the second phase) swings left -> Right -> Left
                    #Right foot in contact for p_current, left foot is going to land at p_next
                    g.append(ca.if_else(ParaLeftSwingFlag, K_CoM_Right@(CoM_k-P_k_current)-ca.DM(k_CoM_Right), np.full((len(k_CoM_Right),),-1)))
                    glb.append(np.full((len(k_CoM_Right),),-np.inf))
                    gub.append(np.full((len(k_CoM_Right),),0))

                    #If the first level swings the Right, then the second level for Odd Number Phases (the second phase) swings Right -> Left -> Right
                    #Left foot in contact for p_current, right foot is going to land as p_next
                    g.append(ca.if_else(ParaRightSwingFlag, K_CoM_Left@(CoM_k-P_k_current)-ca.DM(k_CoM_Left), np.full((len(k_CoM_Left),),-1)))
                    glb.append(np.full((len(k_CoM_Left),),-np.inf))
                    gub.append(np.full((len(k_CoM_Left),),0))

            elif GaitPattern[Nph]=='DoubleSupport':
                
                #CoM Kinematic Constraint
                if (Nph-1-1)//2 == 0:
                    #!!!!!!Pass from the first Level!!!!!!
                    P_k_current = ca.vertcat(px_init,py_init,pz_init) #ca.vertcat(px[-1],py[-1],pz[-1])
                    #!!!!!!
                    P_k_next = ca.vertcat(px[(Nph-1)//2],py[(Nph-1)//2],pz[(Nph-1)//2])
                else:
                    P_k_current = ca.vertcat(px[(Nph-1)//2-1],py[(Nph-1)//2-1],pz[(Nph-1)//2-1])
                    P_k_next = ca.vertcat(px[(Nph-1)//2],py[(Nph-1)//2],pz[(Nph-1)//2])

                if ((Nph-1-1)//2)%2 == 0: #even number steps
                    #If First Level Swing the Left, then the second level Even Number Phases (the first Phase) Swing the Right -> Left -> Right
                    #Left foot in contact for p_current, right foot is going to land as p_next
                    g.append(ca.if_else(ParaLeftSwingFlag, K_CoM_Left@(CoM_k-P_k_current)-ca.DM(k_CoM_Left), np.full((len(k_CoM_Left),),-1)))
                    glb.append(np.full((len(k_CoM_Left),),-np.inf))
                    gub.append(np.full((len(k_CoM_Left),),0))

                    g.append(ca.if_else(ParaLeftSwingFlag, K_CoM_Right@(CoM_k-P_k_next)-ca.DM(k_CoM_Right), np.full((len(k_CoM_Right),),-1)))
                    glb.append(np.full((len(k_CoM_Right),),-np.inf))
                    gub.append(np.full((len(k_CoM_Right),),0))

                    #If First Levvel Swing the Right, then the second level Even Number Phases (the first phase) Swing the Left -> Right -> Left
                    #Right foot in contact for p_current, left foot is going to land at p_next
                    g.append(ca.if_else(ParaRightSwingFlag, K_CoM_Right@(CoM_k-P_k_current)-ca.DM(k_CoM_Right), np.full((len(k_CoM_Right),),-1)))
                    glb.append(np.full((len(k_CoM_Right),),-np.inf))
                    gub.append(np.full((len(k_CoM_Right),),0))

                    g.append(ca.if_else(ParaRightSwingFlag, K_CoM_Left@(CoM_k-P_k_next)-ca.DM(k_CoM_Left), np.full((len(k_CoM_Left),),-1)))
                    glb.append(np.full((len(k_CoM_Left),),-np.inf))
                    gub.append(np.full((len(k_CoM_Left),),0))

                elif ((Nph-1-1)//2)%2 == 1: #odd number steps
                    #If the first level swings the Left, then the second level for Odd Number Phases (the second phase) swings left -> Right -> Left
                    #Right foot in contact for p_current, left foot is going to land at p_next
                    g.append(ca.if_else(ParaLeftSwingFlag, K_CoM_Right@(CoM_k-P_k_current)-ca.DM(k_CoM_Right), np.full((len(k_CoM_Right),),-1)))
                    glb.append(np.full((len(k_CoM_Right),),-np.inf))
                    gub.append(np.full((len(k_CoM_Right),),0))

                    g.append(ca.if_else(ParaLeftSwingFlag, K_CoM_Left@(CoM_k-P_k_next)-ca.DM(k_CoM_Left), np.full((len(k_CoM_Left),),-1)))
                    glb.append(np.full((len(k_CoM_Left),),-np.inf))
                    gub.append(np.full((len(k_CoM_Left),),0))

                    #If the first level swings the Right, then the second level for Odd Number Phases (the second phase) swings Right -> Left -> Right
                    #Left foot in contact for p_current, right foot is going to land as p_next
                    g.append(ca.if_else(ParaRightSwingFlag, K_CoM_Left@(CoM_k-P_k_current)-ca.DM(k_CoM_Left), np.full((len(k_CoM_Left),),-1)))
                    glb.append(np.full((len(k_CoM_Left),),-np.inf))
                    gub.append(np.full((len(k_CoM_Left),),0))

                    g.append(ca.if_else(ParaRightSwingFlag, K_CoM_Right@(CoM_k-P_k_current)-ca.DM(k_CoM_Right), np.full((len(k_CoM_Right),),-1)))
                    glb.append(np.full((len(k_CoM_Right),),-np.inf))
                    gub.append(np.full((len(k_CoM_Right),),0))

            #-------------------------------------
            #Unilateral Forces and Friction Cone
            
            #Unilateral Forces
            #Left Foot
            g.append(FL_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #Right Foot
            g.append(FR_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])

            #Friction Cone
            #   Left Foot x-axis Set 1
            g.append(FL_k.T@TerrainTangentX - miu*FL_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Left Foot x-axis Set 2
            g.append(FL_k.T@TerrainTangentX + miu*FL_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #   Left Foot y-axis Set 1
            g.append(FL_k.T@TerrainTangentY - miu*FL_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Left Foot Y-axis Set 2
            g.append(FL_k.T@TerrainTangentY + miu*FL_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])

            #   Right Foot x-axis Set 1
            g.append(FR_k.T@TerrainTangentX - miu*FR_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Right Foot x-axis Set 2
            g.append(FR_k.T@TerrainTangentX + miu*FR_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #   Right Foot Y-axis Set 1
            g.append(FR_k.T@TerrainTangentY - miu*FR_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Right Foot Y-axis Set 2
            g.append(FR_k.T@TerrainTangentY + miu*FR_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])

            #-------------------------------------
            #Dynamics Constraint
            if k < N_K - 1: #N_k - 1 the enumeration of the last knot, -1 the knot before the last knot
                #First-order Dynamics x-axis
                g.append(x[k+1] - x[k] - h*xdot[k])
                glb.append(np.array([0]))
                gub.append(np.array([0]))

                #First-order Dynamics y-axis
                g.append(y[k+1] - y[k] - h*ydot[k])
                glb.append(np.array([0]))
                gub.append(np.array([0]))

                #First-order Dynamics z-axis
                g.append(z[k+1] - z[k] - h*zdot[k])
                glb.append(np.array([0]))
                gub.append(np.array([0]))

                #Second-order Dynamics x-axis
                g.append(xdot[k+1] - xdot[k] - h*(FLx[k]/m+FRx[k]/m))
                glb.append(np.array([0]))
                gub.append(np.array([0]))

                #Second-order Dynamics y-axis
                g.append(ydot[k+1] - ydot[k] - h*(FLy[k]/m+FRy[k]/m))
                glb.append(np.array([0]))
                gub.append(np.array([0]))

                #Second-order Dynamics z-axis
                g.append(zdot[k+1] - zdot[k] - h*(FLz[k]/m+FRz[k]/m - G))
                glb.append(np.array([0]))
                gub.append(np.array([0]))
            
            #---------------------------------------------------------
            # #Add Cost Terms
            if k < N_K - 1:
                J = J + h*(FLx[k]/m+FRx[k]/m)**2 + h*(FLy[k]/m+FRy[k]/m)**2 + h*(FLz[k]/m+FRz[k]/m - G)**2

    #-------------------------------------
    #Relative Footstep Constraint
    for step_cnt in range(Nsteps):
        if step_cnt == 0:
            #!!!!!!Pass from the first Level!!!!!!
            P_k_current = ca.vertcat(px_init,py_init,pz_init) #ca.vertcat(px[-1],py[-1],pz[-1])
            #!!!!!!
            P_k_next = ca.vertcat(px[step_cnt],py[step_cnt],pz[step_cnt])
        else:
            P_k_current = ca.vertcat(px[step_cnt-1],py[step_cnt-1],pz[step_cnt-1])
            P_k_next = ca.vertcat(px[step_cnt],py[step_cnt],pz[step_cnt])

        if step_cnt%2 == 0: #even number steps
            #If First Level Swing the Left, then the second level Even Number Phases (the first Phase) Swing the Right -> Left -> Right
            #Left foot in contact for p_current, right foot is going to land as p_next
            #Relative Swing Foot Location (rf in lf)
            g.append(ca.if_else(ParaLeftSwingFlag,Q_rf_in_lf@(P_k_next-P_k_current)-ca.DM(q_rf_in_lf),np.full((len(q_rf_in_lf),),-1)))
            glb.append(np.full((len(q_rf_in_lf),),-np.inf))
            gub.append(np.full((len(q_rf_in_lf),),0))

            #If First Levvel Swing the Right, then the second level Even Number Phases (the first phase) Swing the Left -> Right -> Left
            #Right foot in contact for p_current, left foot is going to land at p_next
            #Relative Swing Foot Location (lf in rf)
            g.append(ca.if_else(ParaRightSwingFlag,Q_lf_in_rf@(P_k_next-P_k_current)-ca.DM(q_lf_in_rf),np.full((len(q_lf_in_rf),),-1)))
            glb.append(np.full((len(q_lf_in_rf),),-np.inf))
            gub.append(np.full((len(q_lf_in_rf),),0))

        elif step_cnt%2 == 1: #odd number steps
            #If the first level swings the Left, then the second level for Odd Number Phases (the second phase) swings left -> Right -> Left
            #Right foot in contact for p_current, left foot is going to land at p_next
            #Relative Swing Foot Location (lf in rf)
            g.append(ca.if_else(ParaLeftSwingFlag,Q_lf_in_rf@(P_k_next-P_k_current)-ca.DM(q_lf_in_rf),np.full((len(q_lf_in_rf),),-1)))
            glb.append(np.full((len(q_lf_in_rf),),-np.inf))
            gub.append(np.full((len(q_lf_in_rf),),0))

            #If the first level swings the Right, then the second level for Odd Number Phases (the second phase) swings Right -> Left -> Right
            #Left foot in contact for p_current, right foot is going to land as p_next
            #Relative Swing Foot Location (rf in lf)
            g.append(ca.if_else(ParaRightSwingFlag,Q_rf_in_lf@(P_k_next-P_k_current)-ca.DM(q_rf_in_lf),np.full((len(q_rf_in_lf),),-1)))
            glb.append(np.full((len(q_rf_in_lf),),-np.inf))
            gub.append(np.full((len(q_rf_in_lf),),0))

    #Foot Step Constraint
    #for PatchNum in range(Nsteps):
    #    #Get Footstep Vector
    #    P_vector = ca.vertcat(px[PatchNum],py[PatchNum],pz[PatchNum])

        #Get Half Space Representation
    #    SurfParaTemp = SurfParas[20+PatchNum*20:19+(PatchNum+1)*20+1]
    #    SurfK = SurfParaTemp[0:11+1]
    #    SurfK = ca.reshape(SurfK,3,4)
    #    SurfK = SurfK.T #NOTE: This process is due to casadi naming convention to have first row to be x1,x2,x3
    #    SurfE = SurfParaTemp[11+1:11+3+1]
    #    Surfk = SurfParaTemp[14+1:14+4+1]
    #    Surfe = SurfParaTemp[-1]

        #FootStep Constraint
        #Inequality
    #    g.append(SurfK@P_vector - Surfk)
    #    glb.append(np.full((4,),-np.inf))
    #    gub.append(np.full((4,),0))
        #print(FirstSurfK@p_next - FirstSurfk)

        #Equality
    #    g.append(SurfE.T@P_vector - Surfe)
    #    glb.append(np.array([0]))
    #    gub.append(np.array([0]))

    #Switching Time Constraint
    #for phase_cnt in range(Nphase):
    #    if GaitPattern[phase_cnt] == 'InitialDouble':
    #        g.append(Ts[phase_cnt])
    #        glb.append(np.array([0.05]))
    #        gub.append(np.array([0.3]))
    #    elif GaitPattern[phase_cnt] == 'Swing':
    #        if phase_cnt == 0:
    #            g.append(Ts[phase_cnt]-0)
    #            glb.append(np.array([0.05]))
    #            gub.append(np.array([0.9]))
    #        else:
    #            g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
    #            glb.append(np.array([0.05]))
    #            gub.append(np.array([0.9]))
    #    elif GaitPattern[phase_cnt] == 'DoubleSupport':
    #        g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
    #        glb.append(np.array([0.05]))
    #        gub.append(np.array([0.3])) #0.1 - 0.3

    #-----------------------------------------------------------------------------------------------------------------------
    #Get Variable Index - !!!This is the pure Index, when try to get the array using other routines, we need to add "+1" at the last index due to Python indexing conventions
    x_index = (0,N_K-1) #First set of variables start counting from 0, The enumeration of the last knot is N_K-1
    y_index = (x_index[1]+1,x_index[1]+N_K)
    z_index = (y_index[1]+1,y_index[1]+N_K)
    xdot_index = (z_index[1]+1,z_index[1]+N_K)
    ydot_index = (xdot_index[1]+1,xdot_index[1]+N_K)
    zdot_index = (ydot_index[1]+1,ydot_index[1]+N_K)
    FLx_index = (zdot_index[1]+1,zdot_index[1]+N_K)
    FLy_index = (FLx_index[1]+1,FLx_index[1]+N_K)
    FLz_index = (FLy_index[1]+1,FLy_index[1]+N_K)
    FRx_index = (FLz_index[1]+1,FLz_index[1]+N_K)
    FRy_index = (FRx_index[1]+1,FRx_index[1]+N_K)
    FRz_index = (FRy_index[1]+1,FRy_index[1]+N_K)
    px_init_index = (FRz_index[1]+1,FRz_index[1]+1)
    py_init_index = (px_init_index[1]+1,px_init_index[1]+1)
    pz_init_index = (py_init_index[1]+1,py_init_index[1]+1)
    px_index = (pz_init_index[1]+1,pz_init_index[1]+Nsteps)
    py_index = (px_index[1]+1,px_index[1]+Nsteps)
    pz_index = (py_index[1]+1,py_index[1]+Nsteps)
    #Ts_index = (pz_index[1]+1,pz_index[1]+Nphase)

    var_index = {"x":x_index,
                 "y":y_index,
                 "z":z_index,
                 "xdot":xdot_index,
                 "ydot":ydot_index,
                 "zdot":zdot_index,
                 "FLx":FLx_index,
                 "FLy":FLy_index,
                 "FLz":FLz_index,
                 "FRx":FRx_index,
                 "FRy":FRy_index,
                 "FRz":FRz_index,
                 "px_init":px_init_index,
                 "py_init":py_init_index,
                 "pz_init":pz_init_index,
                 "px":px_index,
                 "py":py_index,
                 "pz":pz_index,
    }
    #             "Ts":Ts_index,
    #}

    return DecisionVars, DecisionVars_lb, DecisionVars_ub, J, g, glb, gub, var_index

def BuildSolver(FirstLevel = None, SecondLevel = None, m = 95, NumPhase = 3):
    #Check if the First Level is selected properly
    assert FirstLevel != None, "First Level is Not Selected."

    #Define Parameter Vector
    #-------------------------------------------
    #Define Solver Parameter Vector

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
    ParaList = {"LeftSwingFlag":ParaLeftSwingFlag,
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

    paras = ca.vertcat(ParaLeftSwingFlag,ParaRightSwingFlag,C_0,Cdot_0,Cddot_0,C_end,TimeVector,T,PL_init,PR_init,P_next)

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
    elif SecondLevel == "CoM_Dynamics_Fixed_Time":
        var_L2, var_lb_L2, var_ub_L2, J_L2, g_L2, glb_L2, gub_L2, var_index_L2 = CoM_Dynamics_Fixed_Time(m = m, StandAlong = False,  ParameterList = ParaList, Nsteps = 4)#Here is the total number of steps
    else:
        print("Not Implemented")

    #Set-up Terminal Cost Here and Sum over all costs
    if SecondLevel == None: #No second Level
        #   Collect the variables, terminal cost set as the end of the single first level
        J = J_L1
        J = J + 100*(CoM_end_L1[0] - C_end[0])**2 + 100*(CoM_end_L1[1] - C_end[1])**2 + 100*(CoM_end_L1[2] - C_end[2])**2
    else:
        J = J_L1 + J_L2
        x_L2 = var_L2[var_index_L2["x"][0]:var_index_L2["x"][1]+1]
        y_L2 = var_L2[var_index_L2["y"][0]:var_index_L2["y"][1]+1]
        z_L2 = var_L2[var_index_L2["z"][0]:var_index_L2["z"][1]+1]
        xdot_L2 = var_L2[var_index_L2["xdot"][0]:var_index_L2["xdot"][1]+1]
        ydot_L2 = var_L2[var_index_L2["ydot"][0]:var_index_L2["ydot"][1]+1]
        zdot_L2 = var_L2[var_index_L2["zdot"][0]:var_index_L2["zdot"][1]+1]
        J = J + 100*(x_L2[-1]-C_end[0])**2 + 100*(y_L2[-1]-C_end[1])**2 + 100*(z_L2[-1]-C_end[2])**2
        print("Second Level not Implemented")

        #Get Terminal State of the First Level
        Cy = var_L1[var_index_L1["Cy"][0]:var_index_L1["Cy"][1]+1]
        C_p0 = C_0
        C_p1 = T/3*Cdot_0 + C_p0
        C_p2 = T**2/6*Cddot_0 + 2*C_p1 - C_p0
        C_L1_end = Cy
        Cdot_L1_end = 3.0*T**2*(-C_p2 + Cy)/T + 6.0*T*(1 - T)*(-C_p1 + C_p2)/T + 3.0*(1 - T)**2*(-C_p0 + C_p1)/T

        #Deal with Connections between the first level and the second level
        gConnect = []
        gConnect_lb = []
        gConnect_ub = []

        #   Initial Condition x-axis (Connect to the First Level)
        gConnect.append(var_L2[var_index_L2["x"][0]]-C_L1_end[0])
        gConnect_lb.append(np.array([0]))
        gConnect_ub.append(np.array([0]))

        #   Initial Condition y-axis (Connect to the First Level)
        gConnect.append(var_L2[var_index_L2["y"][0]]-C_L1_end[1])
        gConnect_lb.append(np.array([0]))
        gConnect_ub.append(np.array([0]))

        #   Initial Condition z-axis (Connect to the First Level)
        gConnect.append(var_L2[var_index_L2["z"][0]]-C_L1_end[2])
        gConnect_lb.append(np.array([0]))
        gConnect_ub.append(np.array([0]))

        #   xdot
        gConnect.append(var_L2[var_index_L2["xdot"][0]]-Cdot_L1_end[0])
        gConnect_lb.append(np.array([0]))
        gConnect_ub.append(np.array([0]))

        #   ydot
        gConnect.append(var_L2[var_index_L2["ydot"][0]]-Cdot_L1_end[1])
        gConnect_lb.append(np.array([0]))
        gConnect_ub.append(np.array([0]))

        #   zdot
        gConnect.append(var_L2[var_index_L2["zdot"][0]]-Cdot_L1_end[2])
        gConnect_lb.append(np.array([0]))
        gConnect_ub.append(np.array([0]))

        #   Initial Contact Location x-axis (Connect to the First Level)
        gConnect.append(var_L2[var_index_L2["px_init"][0]]-P_next[0])
        gConnect_lb.append(np.array([0]))
        gConnect_ub.append(np.array([0]))

        #   Initial Contact Location y-axis (Connect to the First Level)
        gConnect.append(var_L2[var_index_L2["py_init"][0]]-P_next[1])
        gConnect_lb.append(np.array([0]))
        gConnect_ub.append(np.array([0]))

        #   Initial Contact Location y-axis (Connect to the First Level)
        gConnect.append(var_L2[var_index_L2["pz_init"][0]]-P_next[2])
        gConnect_lb.append(np.array([0]))
        gConnect_ub.append(np.array([0]))

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
        DecisionVars = ca.vertcat(var_L1,var_L2)
        DecisionVars_lb  = np.concatenate((var_lb_L1,var_lb_L2),axis=None)
        DecisionVars_ub = np.concatenate((var_ub_L1,var_ub_L2),axis=None)
        g = ca.vertcat(*g_L1,*g_L2,*gConnect)
        #   Convert shape of glb and gub
        glb_L1 = np.concatenate((glb_L1),axis=None)
        gub_L1 = np.concatenate((gub_L1),axis=None)

        glb_L2 = np.concatenate((glb_L2),axis=None)
        gub_L2 = np.concatenate((gub_L2),axis=None)

        gConnect_lb = np.concatenate((gConnect_lb),axis=None)
        gConnect_ub = np.concatenate((gConnect_ub),axis=None)
        #Get all constraint bounds
        glb = np.concatenate((glb_L1,glb_L2,gConnect_lb),axis=None)
        gub = np.concatenate((gub_L1,gub_L2,gConnect_ub),axis=None)
        
        print(glb)

    #Collect all Variable Index
    var_index = {"Level1_Var_Index": var_index_L1,
                 "Level2_Var_Index": var_index_L2,
    }
    
    #   Build Solvers
    #Build Solver
    opts = {}
    opts["error_on_fail"] = False
    prob = {'x': DecisionVars, 'f': J, 'g': g, 'p': paras}
    solver = ca.qpsol('solver', 'gurobi', prob, opts)

    return solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index
