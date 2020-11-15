#Description:
#   Functions for Building Problem Descriptions for Humanoid with Multi-fiedelity Planning Framework

# Import Important Modules
import numpy as np #Numpy
import casadi as ca #Casadi
import matplotlib.pyplot as plt #Matplotlib
from mpl_toolkits.mplot3d import Axes3D
from Constraint_Builder import *
# Import SL1M modules
#from sl1m.constants_and_tools import *
#from sl1m.planner import *
#from constraints import * 
from sl1m.problem_definition import *

from sl1m.planner_scenarios.talos.constraints import *

#Import NLP motion reference
#from NLP_Reference_Traj import *


#FUNCTION: Build a single step NLP problem
#Parameters:
#   m: robot mass, default value set as the one of Talos
def NLP_SingleStep(m = 95, StandAlong = True, ConservativeEnd = False, ParameterList = None, CentralY = False):
    #-----------------------------------------------------------------------------------------------------------------------
    #Define Parameters
    #   Gait Pattern, Each action is followed up by a double support phase
    GaitPattern = ['InitialDouble','Swing','DoubleSupport'] #,'RightSupport','DoubleSupport','LeftSupport','DoubleSupport'
    #   Number of Phases
    Nphase = len(GaitPattern)
    #   Number of Steps
    Nstep = 1
    #   Number of Knots per Phase - how many intervals do we have for a single phase; 10 intervals need 11 knots/ticks
    Nk_Local= 7
    #   Compute Number of Total knots/ticks, but the enumeration start from 0 to N_K-1
    N_K = Nk_Local*Nphase + 1 #+1 the last knots to finalize the plan
    #   Robot mass
    #m = 95 #kg
    G = 9.80665 #kg/m^2
    ##   Terrain Model
    ##       Flat Terrain
    #TerrainNorm = [0,0,1] 
    #TerrainTangentX = [1,0,0]
    #TerrainTangentY = [0,1,0]
    #Friction Coefficient
    miu = 0.3
    #   Force Limits
    F_bound = 400
    Fxlb = -F_bound
    Fxub = F_bound
    Fylb = -F_bound
    Fyub = F_bound
    Fzlb = -F_bound
    Fzub = F_bound
    #   Angular Momentum Bounds
    L_bound = 2.5
    Ldot_bound = 3.5
    Lub = L_bound
    Llb = -L_bound
    Ldotub = Ldot_bound
    Ldotlb = -Ldot_bound

    #Minimum y-axis foot location
    py_lower_limit = 0.04
    #Lowest z
    z_lowest = 0.7
    z_highest = 0.8
    #-----------------------------------------------------------------------------------------------------------------------
    #Kinematics Constraint for Talos
    #kinematicConstraints = genKinematicConstraints(left_foot_constraints,right_foot_constraints)
    K_CoM_Right,k_CoM_Right = right_foot_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]))
    K_CoM_Left,k_CoM_Left = left_foot_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]))

    #K_CoM_Left = kinematicConstraints[0][0]
    #k_CoM_Left = kinematicConstraints[0][1]
    #K_CoM_Right = kinematicConstraints[1][0]
    #k_CoM_Right = kinematicConstraints[1][1]
    #Relative Foot Constraint matrices
    
    #relativeConstraints = genFootRelativeConstraints(right_foot_in_lf_frame_constraints,left_foot_in_rf_frame_constraints)
    Q_rf_in_lf,q_rf_in_lf = right_foot_in_lf_frame_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]))
    Q_lf_in_rf,q_lf_in_rf = left_foot_in_rf_frame_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]))

    #Q_rf_in_lf = relativeConstraints[0][0] #named lf in rf, but representing rf in lf
    #q_rf_in_lf = relativeConstraints[0][1] #named lf in rf, but representing rf in lf
    #Q_lf_in_rf = relativeConstraints[1][0] #named rf in lf, but representing lf in rf
    #q_lf_in_rf = relativeConstraints[1][1] #named rf in lf, but representing lf in rf


    #-----------------------------------------------------------------------------------------------------------------------

    #-----------------------------------------------------------------------------------------------------------------------
    #Define Initial and Terminal Conditions
    #x_init = ca.SX.sym('x_init')
    #y_init = ca.SX.sym('y_init')
    #z_init = ca.SX.sym('z_init')

    x_init = ParameterList["x_init"]
    y_init = ParameterList["y_init"]
    z_init = ParameterList["z_init"]

    xdot_init = ParameterList["xdot_init"]
    ydot_init = ParameterList["ydot_init"]
    zdot_init = ParameterList["zdot_init"]

    Lx_init = ParameterList["Lx_init"]
    Ly_init = ParameterList["Ly_init"]
    Lz_init = ParameterList["Lz_init"]

    Ldotx_init = ParameterList["Ldotx_init"]
    Ldoty_init = ParameterList["Ldoty_init"]
    Ldotz_init = ParameterList["Ldotz_init"]

    PLx_init = ParameterList["PLx_init"]
    PLy_init = ParameterList["PLy_init"]
    PLz_init = ParameterList["PLz_init"]
    PL_init = ca.vertcat(PLx_init,PLy_init,PLz_init)

    PRx_init = ParameterList["PRx_init"]
    PRy_init = ParameterList["PRy_init"]
    PRz_init = ParameterList["PRz_init"]
    PR_init = ca.vertcat(PRx_init,PRy_init,PRz_init)

    x_end = ParameterList["x_end"]
    y_end = ParameterList["y_end"]
    z_end = ParameterList["z_end"]

    xdot_end = ParameterList["xdot_end"]
    ydot_end = ParameterList["ydot_end"]
    zdot_end = ParameterList["zdot_end"]

    #Lx_end = 0
    #Ly_end = 0
    #Lz_end = 0

    #Ldotx_end = 0
    #Ldoty_end = 0
    #Ldotz_end = 0

    #Flags for Swing Legs (Defined as Parameters)
    ParaLeftSwingFlag = ParameterList["LeftSwingFlag"]
    ParaRightSwingFlag = ParameterList["RightSwingFlag"]

    #Surfaces (Only the First One)
    SurfParas = ParameterList["SurfParas"]
    FirstSurfPara = SurfParas[0:19+1]
    #print(FirstSurPara)
    #   Process the Parameters
    #   FirstSurfK, the matrix
    FirstSurfK = FirstSurfPara[0:11+1]
    FirstSurfK = ca.reshape(FirstSurfK,3,4)
    FirstSurfK = FirstSurfK.T #NOTE: This process is due to casadi naming convention to have first row to be x1,x2,x3
    #   FirstSurfE, the vector for equality constraint
    FirstSurfE = FirstSurfPara[11+1:11+3+1]
    #   FirstSurfk, the vector fo inquality constraint
    FirstSurfk = FirstSurfPara[14+1:14+4+1]
    #   FirstSurfe, the vector fo inquality constraint
    FirstSurfe = FirstSurfPara[-1]

    #Tangents and Norms
    #Initial Contact Norm and Tangents
    PL_init_Norm = ParameterList["PL_init_Norm"]
    PL_init_TangentX = ParameterList["PL_init_TangentX"]
    PL_init_TangentY = ParameterList["PL_init_TangentY"]
    PR_init_Norm = ParameterList["PR_init_Norm"]
    PR_init_TangentX = ParameterList["PR_init_TangentX"]
    PR_init_TangentY = ParameterList["PR_init_TangentY"]
    
    #Future Contact Norm and Tangents
    SurfNorms = ParameterList["SurfNorms"]                
    SurfTangentsX = ParameterList["SurfTangentsX"]
    SurfTangentsY = ParameterList["SurfTangentsY"]

    ##First Round Flag (If yes, we have an initial double support phase, if not, we don't have an initial double support phase)
    #ParaFirstRoundFlag = ParameterList["FirstRoundFlag"]

    #-----------------------------------------------------------------------------------------------------------------------
    #Define Variables and Bounds, Parameters
    #   CoM Position x-axis
    x = ca.SX.sym('x',N_K)
    x_lb = np.array([[0]*(x.shape[0]*x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    x_ub = np.array([[30]*(x.shape[0]*x.shape[1])])
    #   CoM Position y-axis
    y = ca.SX.sym('y',N_K)
    y_lb = np.array([[-1]*(y.shape[0]*y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    y_ub = np.array([[1]*(y.shape[0]*y.shape[1])])
    #   CoM Position z-axis
    z = ca.SX.sym('z',N_K)
    z_lb = np.array([[[z_lowest]]*(z.shape[0]*z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    z_ub = np.array([[z_highest]*(z.shape[0]*z.shape[1])])
    #z_lb = np.array([[0.55]*(z.shape[0]*z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #z_ub = np.array([[0.65]*(z.shape[0]*z.shape[1])])
    #   CoM Velocity x-axis
    xdot = ca.SX.sym('xdot',N_K)
    xdot_lb = np.array([[-1.5]*(xdot.shape[0]*xdot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    xdot_ub = np.array([[1.5]*(xdot.shape[0]*xdot.shape[1])])
    #   CoM Velocity y-axis
    ydot = ca.SX.sym('ydot',N_K)
    ydot_lb = np.array([[-1.5]*(ydot.shape[0]*ydot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    ydot_ub = np.array([[1.5]*(ydot.shape[0]*ydot.shape[1])])
    #   CoM Velocity z-axis
    zdot = ca.SX.sym('zdot',N_K)
    zdot_lb = np.array([[-1.5]*(zdot.shape[0]*zdot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    zdot_ub = np.array([[1.5]*(zdot.shape[0]*zdot.shape[1])])
    #   Angular Momentum x-axis
    Lx = ca.SX.sym('Lx',N_K)
    Lx_lb = np.array([[Llb]*(Lx.shape[0]*Lx.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Lx_ub = np.array([[Lub]*(Lx.shape[0]*Lx.shape[1])])
    #   Angular Momentum y-axis
    Ly = ca.SX.sym('Ly',N_K)
    Ly_lb = np.array([[Llb]*(Ly.shape[0]*Ly.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Ly_ub = np.array([[Lub]*(Ly.shape[0]*Ly.shape[1])])
    #   Angular Momntum y-axis
    Lz = ca.SX.sym('Lz',N_K)
    Lz_lb = np.array([[Llb]*(Lz.shape[0]*Lz.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Lz_ub = np.array([[Lub]*(Lz.shape[0]*Lz.shape[1])])
    #   Angular Momentum rate x-axis
    Ldotx = ca.SX.sym('Ldotx',N_K)
    Ldotx_lb = np.array([[Ldotlb]*(Ldotx.shape[0]*Ldotx.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Ldotx_ub = np.array([[Ldotub]*(Ldotx.shape[0]*Ldotx.shape[1])])
    #   Angular Momentum y-axis
    Ldoty = ca.SX.sym('Ldoty',N_K)
    Ldoty_lb = np.array([[Ldotlb]*(Ldoty.shape[0]*Ldoty.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Ldoty_ub = np.array([[Ldotub]*(Ldoty.shape[0]*Ldoty.shape[1])])
    #   Angular Momntum z-axis
    Ldotz = ca.SX.sym('Ldotz',N_K)
    Ldotz_lb = np.array([[Ldotlb]*(Ldotz.shape[0]*Ldotz.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Ldotz_ub = np.array([[Ldotub]*(Ldotz.shape[0]*Ldotz.shape[1])])
    #left Foot Forces
    #Left Foot Contact Point 1 x-axis
    FL1x = ca.SX.sym('FL1x',N_K)
    FL1x_lb = np.array([[Fxlb]*(FL1x.shape[0]*FL1x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL1x_ub = np.array([[Fxub]*(FL1x.shape[0]*FL1x.shape[1])])
    #Left Foot Contact Point 1 y-axis
    FL1y = ca.SX.sym('FL1y',N_K)
    FL1y_lb = np.array([[Fylb]*(FL1y.shape[0]*FL1y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL1y_ub = np.array([[Fyub]*(FL1y.shape[0]*FL1y.shape[1])])
    #Left Foot Contact Point 1 z-axis
    FL1z = ca.SX.sym('FL1z',N_K)
    FL1z_lb = np.array([[Fzlb]*(FL1z.shape[0]*FL1z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL1z_ub = np.array([[Fzub]*(FL1z.shape[0]*FL1z.shape[1])])
    #Left Foot Contact Point 2 x-axis
    FL2x = ca.SX.sym('FL2x',N_K)
    FL2x_lb = np.array([[Fxlb]*(FL2x.shape[0]*FL2x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL2x_ub = np.array([[Fxub]*(FL2x.shape[0]*FL2x.shape[1])])
    #Left Foot Contact Point 2 y-axis
    FL2y = ca.SX.sym('FL2y',N_K)
    FL2y_lb = np.array([[Fylb]*(FL2y.shape[0]*FL2y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL2y_ub = np.array([[Fyub]*(FL2y.shape[0]*FL2y.shape[1])])
    #Left Foot Contact Point 2 z-axis
    FL2z = ca.SX.sym('FL2z',N_K)
    FL2z_lb = np.array([[Fzlb]*(FL2z.shape[0]*FL2z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL2z_ub = np.array([[Fzub]*(FL2z.shape[0]*FL2z.shape[1])])
    #Left Foot Contact Point 3 x-axis
    FL3x = ca.SX.sym('FL3x',N_K)
    FL3x_lb = np.array([[Fxlb]*(FL3x.shape[0]*FL3x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL3x_ub = np.array([[Fxub]*(FL3x.shape[0]*FL3x.shape[1])])
    #Left Foot Contact Point 3 y-axis
    FL3y = ca.SX.sym('FL3y',N_K)
    FL3y_lb = np.array([[Fylb]*(FL3y.shape[0]*FL3y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL3y_ub = np.array([[Fyub]*(FL3y.shape[0]*FL3y.shape[1])])
    #Left Foot Contact Point 3 z-axis
    FL3z = ca.SX.sym('FL3z',N_K)
    FL3z_lb = np.array([[Fzlb]*(FL3z.shape[0]*FL3z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL3z_ub = np.array([[Fzub]*(FL3z.shape[0]*FL3z.shape[1])])
    #Left Foot Contact Point 4 x-axis
    FL4x = ca.SX.sym('FL4x',N_K)
    FL4x_lb = np.array([[Fxlb]*(FL4x.shape[0]*FL4x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL4x_ub = np.array([[Fxub]*(FL4x.shape[0]*FL4x.shape[1])])
    #Left Foot Contact Point 4 y-axis
    FL4y = ca.SX.sym('FL4y',N_K)
    FL4y_lb = np.array([[Fylb]*(FL4y.shape[0]*FL4y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL4y_ub = np.array([[Fyub]*(FL4y.shape[0]*FL4y.shape[1])])
    #Left Foot Contact Point 4 z-axis
    FL4z = ca.SX.sym('FL4z',N_K)
    FL4z_lb = np.array([[Fzlb]*(FL4z.shape[0]*FL4z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL4z_ub = np.array([[Fzub]*(FL4z.shape[0]*FL4z.shape[1])])

    #Right Contact Force x-axis
    #Right Foot Contact Point 1 x-axis
    FR1x = ca.SX.sym('FR1x',N_K)
    FR1x_lb = np.array([[Fxlb]*(FR1x.shape[0]*FR1x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR1x_ub = np.array([[Fxub]*(FR1x.shape[0]*FR1x.shape[1])])
    #Right Foot Contact Point 1 y-axis
    FR1y = ca.SX.sym('FR1y',N_K)
    FR1y_lb = np.array([[Fylb]*(FR1y.shape[0]*FR1y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR1y_ub = np.array([[Fyub]*(FR1y.shape[0]*FR1y.shape[1])])
    #Right Foot Contact Point 1 z-axis
    FR1z = ca.SX.sym('FR1z',N_K)
    FR1z_lb = np.array([[Fzlb]*(FR1z.shape[0]*FR1z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR1z_ub = np.array([[Fzub]*(FR1z.shape[0]*FR1z.shape[1])])
    #Right Foot Contact Point 2 x-axis
    FR2x = ca.SX.sym('FR2x',N_K)
    FR2x_lb = np.array([[Fxlb]*(FR2x.shape[0]*FR2x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR2x_ub = np.array([[Fxub]*(FR2x.shape[0]*FR2x.shape[1])])
    #Right Foot Contact Point 2 y-axis
    FR2y = ca.SX.sym('FR2y',N_K)
    FR2y_lb = np.array([[Fylb]*(FR2y.shape[0]*FR2y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR2y_ub = np.array([[Fyub]*(FR2y.shape[0]*FR2y.shape[1])])
    #Right Foot Contact Point 2 z-axis
    FR2z = ca.SX.sym('FR2z',N_K)
    FR2z_lb = np.array([[Fzlb]*(FR2z.shape[0]*FR2z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR2z_ub = np.array([[Fzub]*(FR2z.shape[0]*FR2z.shape[1])])
    #Right Foot Contact Point 3 x-axis
    FR3x = ca.SX.sym('FR3x',N_K)
    FR3x_lb = np.array([[Fxlb]*(FR3x.shape[0]*FR3x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR3x_ub = np.array([[Fxub]*(FR3x.shape[0]*FR3x.shape[1])])
    #Right Foot Contact Point 3 y-axis
    FR3y = ca.SX.sym('FR3y',N_K)
    FR3y_lb = np.array([[Fylb]*(FR3y.shape[0]*FR3y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR3y_ub = np.array([[Fyub]*(FR3y.shape[0]*FR3y.shape[1])])
    #Right Foot Contact Point 3 z-axis
    FR3z = ca.SX.sym('FR3z',N_K)
    FR3z_lb = np.array([[Fzlb]*(FR3z.shape[0]*FR3z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR3z_ub = np.array([[Fzub]*(FR3z.shape[0]*FR3z.shape[1])])
    #Right Foot Contact Point 4 x-axis
    FR4x = ca.SX.sym('FR4x',N_K)
    FR4x_lb = np.array([[Fxlb]*(FR4x.shape[0]*FR4x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR4x_ub = np.array([[Fxub]*(FR4x.shape[0]*FR4x.shape[1])])
    #Right Foot Contact Point 4 y-axis
    FR4y = ca.SX.sym('FR4y',N_K)
    FR4y_lb = np.array([[Fylb]*(FR4y.shape[0]*FR4y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR4y_ub = np.array([[Fyub]*(FR4y.shape[0]*FR4y.shape[1])])
    #Right Foot Contact Point 4 z-axis
    FR4z = ca.SX.sym('FR4z',N_K)
    FR4z_lb = np.array([[Fzlb]*(FR4z.shape[0]*FR4z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR4z_ub = np.array([[Fzub]*(FR4z.shape[0]*FR4z.shape[1])])

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
    for stepIdx in range(Nstep):
        pxtemp = ca.SX.sym('px'+str(stepIdx+1)) #0 + 1
        px.append(pxtemp)
        px_lb.append(np.array([-1]))
        px_ub.append(np.array([30]))

        pytemp = ca.SX.sym('py'+str(stepIdx+1))
        py.append(pytemp)
        py_lb.append(np.array([-1]))
        py_ub.append(np.array([1]))

        #   Foot steps are all staying on the ground
        pztemp = ca.SX.sym('pz'+str(stepIdx+1))
        pz.append(pztemp)
        pz_lb.append(np.array([-5]))
        pz_ub.append(np.array([5]))

    #Switching Time Vector
    Ts = []
    Ts_lb = []
    Ts_ub = []
    for n_phase in range(Nphase):
        Tstemp = ca.SX.sym('Ts'+str(n_phase+1)) #0 + 1 + ....
        Ts.append(Tstemp)
        Ts_lb.append(np.array([0.05]))
        Ts_ub.append(np.array([2.0]))

    #!!!!
    #Note: Parameters need to be change according the level Setup
    #!!!!
    #paras = ca.vertcat(ParaLeftSwingFlag,ParaRightSwingFlag,x_init,y_init,z_init,PLx_init,PLy_init,PLz_init,PRx_init,PRy_init,PRz_init,x_end,y_end,z_end)

    #   Collect all Decision Variables
    DecisionVars = ca.vertcat(x,y,z,xdot,ydot,zdot,Lx,Ly,Lz,Ldotx,Ldoty,Ldotz,FL1x,FL1y,FL1z,FL2x,FL2y,FL2z,FL3x,FL3y,FL3z,FL4x,FL4y,FL4z,FR1x,FR1y,FR1z,FR2x,FR2y,FR2z,FR3x,FR3y,FR3z,FR4x,FR4y,FR4z,*px,*py,*pz,*Ts)
    #print(DecisionVars)
    DecisionVarsShape = DecisionVars.shape

    #   Collect all lower bound and upper bound
    DecisionVars_lb = np.concatenate(((x_lb,y_lb,z_lb,xdot_lb,ydot_lb,zdot_lb,Lx_lb,Ly_lb,Lz_lb,Ldotx_lb,Ldoty_lb,Ldotz_lb,FL1x_lb,FL1y_lb,FL1z_lb,FL2x_lb,FL2y_lb,FL2z_lb,FL3x_lb,FL3y_lb,FL3z_lb,FL4x_lb,FL4y_lb,FL4z_lb,FR1x_lb,FR1y_lb,FR1z_lb,FR2x_lb,FR2y_lb,FR2z_lb,FR3x_lb,FR3y_lb,FR3z_lb,FR4x_lb,FR4y_lb,FR4z_lb,px_lb,py_lb,pz_lb,Ts_lb)),axis=None)
    DecisionVars_ub = np.concatenate(((x_ub,y_ub,z_ub,xdot_ub,ydot_ub,zdot_ub,Lx_ub,Ly_ub,Lz_ub,Ldotx_ub,Ldoty_ub,Ldotz_ub,FL1x_ub,FL1y_ub,FL1z_ub,FL2x_ub,FL2y_ub,FL2z_ub,FL3x_ub,FL3y_ub,FL3z_ub,FL4x_ub,FL4y_ub,FL4z_ub,FR1x_ub,FR1y_ub,FR1z_ub,FR2x_ub,FR2y_ub,FR2z_ub,FR3x_ub,FR3y_ub,FR3z_ub,FR4x_ub,FR4y_ub,FR4z_ub,px_ub,py_ub,pz_ub,Ts_ub)),axis=None)

    #Time Span Setup
    tau_upper_limit = 1
    tauStepLength = tau_upper_limit/(N_K-1) #Get the interval length, total number of knots - 1

    #-----------------------------------------------------------------------------------------------------------------------
    #Define Constrains and Running Cost
    g = []
    glb = []
    gub = []
    J = 0

    #Initial and Termianl Conditions
    #   Initial CoM x-axis
    g.append(x[0]-x_init)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

    #   Initial CoM y-axis
    g.append(y[0]-y_init)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

    #   Initial CoM z-axis
    g.append(z[0]-z_init)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

    #   Initial CoM Velocity x-axis
    g.append(xdot[0]-xdot_init)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

    #   Initial CoM Velocity y-axis
    g.append(ydot[0]-ydot_init)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

    #   Initial CoM Velocity z-axis
    g.append(zdot[0]-zdot_init)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

    #   Initial Angular Momentum x-axis
    g.append(Lx[0]-Lx_init)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

    #   Initial Angular Momentum y-axis
    g.append(Ly[0]-Ly_init)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

    #   Initial Angular Momentum z-axis
    g.append(Lz[0]-Lz_init)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

    # #   Initial Angular Momentum rate x-axis
    # g.append(Ldotx[0]-Ldotx_init)
    # glb.append(np.array([0]))
    # gub.append(np.array([0]))

    # #   Initial Angular Momentum rate y-axis
    # g.append(Ldoty[0]-Ldoty_init)
    # glb.append(np.array([0]))
    # gub.append(np.array([0]))

    # #   Initial Angular Momentum rate z-axis
    # g.append(Ldotz[0]-Ldotz_init)
    # glb.append(np.array([0]))
    # gub.append(np.array([0]))

    #if StandAlong == True:
    #    #   Terminal CoM y-axis
    #    g.append(y[-1]-y_end)
    #    glb.append(np.array([0]))
    #    gub.append(np.array([0]))

    #    #   Terminal CoM z-axis
    #    g.append(z[-1]-z_end)
    #    glb.append(np.array([0]))
    #    gub.append(np.array([0]))

    if ConservativeEnd == True:
        #   Terminal CoM Velocity x-axis
        g.append(xdot[-1]-xdot_end)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        #   Terminal CoM Velocity y-axis
        g.append(ydot[-1]-ydot_end)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        #   Terminal CoM Velocity z-axis
        g.append(zdot[-1]-zdot_end)
        glb.append(np.array([0]))
        gub.append(np.array([0]))


        #   Terminal CoM Velocity x-axis
        g.append(xdot[-1]-xdot_end)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        #   Terminal CoM Velocity y-axis
        g.append(ydot[-1]-ydot_end)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        #   Terminal CoM Velocity z-axis
        g.append(zdot[-1]-zdot_end)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        #  Terminal Angular Momentum x-axis
        g.append(Lx[-1]-Lx_end)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        #  Terminal Angular Momentum y-axis
        g.append(Ly[-1]-Ly_end)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        #  Terminal Angular Momentum z-axis
        g.append(Lz[-1]-Lz_end)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

    #   Terminal Angular Momentum Rate x-axis
    #g.append(Ldotx[-1]-Ldotx_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    #   Terminal Angular Momentum Rate y-axis
    #g.append(Ldoty[-1]-Ldoty_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    #   Terminal Angular Momentum Rate z-axis
    #g.append(Ldotz[-1]-Ldotz_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    # #---------------------
    # #Terminal Conditions
    # g.append(x[-1]-0.7578)
    # glb.append(np.array([0]))
    # gub.append(np.array([0]))

    # g.append(y[-1]-3.7915e-02)
    # glb.append(np.array([0]))
    # gub.append(np.array([0]))

    # g.append(z[-1]-0.7658)
    # glb.append(np.array([0]))
    # gub.append(np.array([0]))

    # g.append(xdot[-1]-0.3647)
    # glb.append(np.array([0]))
    # gub.append(np.array([0]))

    # g.append(ydot[-1]-0.1572)
    # glb.append(np.array([0]))
    # gub.append(np.array([0]))

    # g.append(zdot[-1]-0.0218)
    # glb.append(np.array([0]))
    # gub.append(np.array([0]))

    # g.append(Lx[-1]+1.1823e-03)
    # glb.append(np.array([0]))
    # gub.append(np.array([0]))

    # g.append(Ly[-1]+1.1943e-04)
    # glb.append(np.array([0]))
    # gub.append(np.array([0]))

    # g.append(Lz[-1]-1.6479e-10)
    # glb.append(np.array([0]))
    # gub.append(np.array([0]))

    # g.append(px[0]-0.9267)
    # glb.append(np.array([0]))
    # gub.append(np.array([0]))

    # g.append(py[0]-0.148)
    # glb.append(np.array([0]))
    # gub.append(np.array([0]))

    #Loop over all Phases (Knots)
    for Nph in range(Nphase):
        #Decide Number of Knots
        if Nph == Nphase-1:  #The last Knot belongs to the Last Phase
            Nk_ThisPhase = Nk_Local+1
        else:
            Nk_ThisPhase = Nk_Local       

        #Decide Time Vector
        if Nph == 0: #first phase
            h = tauStepLength*Nphase*(Ts[Nph]-0)
        else: #other phases
            h = tauStepLength*Nphase*(Ts[Nph]-Ts[Nph-1]) 

        for Local_k_Count in range(Nk_ThisPhase):
            #Get knot number across the entire time line
            k = Nph*Nk_Local + Local_k_Count
            #print(k)

            #------------------------------------------
            #Build useful vectors
            #   Forces
            FL1_k = ca.vertcat(FL1x[k],FL1y[k],FL1z[k])
            FL2_k = ca.vertcat(FL2x[k],FL2y[k],FL2z[k])
            FL3_k = ca.vertcat(FL3x[k],FL3y[k],FL3z[k])
            FL4_k = ca.vertcat(FL4x[k],FL4y[k],FL4z[k])

            FR1_k = ca.vertcat(FR1x[k],FR1y[k],FR1z[k])
            FR2_k = ca.vertcat(FR2x[k],FR2y[k],FR2z[k])
            FR3_k = ca.vertcat(FR3x[k],FR3y[k],FR3z[k])
            FR4_k = ca.vertcat(FR4x[k],FR4y[k],FR4z[k])
            #   CoM
            CoM_k = ca.vertcat(x[k],y[k],z[k])
            #   Angular Momentum
            if k<N_K-1: #N_K-1 the enumeration of the last knot, k<N_K-1 the one before the last knot
                Ldot_current = ca.vertcat(Ldotx[k],Ldoty[k],Ldotz[k])
                Ldot_next = ca.vertcat(Ldotx[k+1],Ldotx[k+1],Ldotx[k+1])
            #-------------------------------------------

            #-------------------------------------------
            #Phase dependent Constraints (CoM Kinematics and Angular Dynamics)
            if GaitPattern[Nph]=='InitialDouble':
                #Kinematics Constraint
                #   CoM in the Left foot
                g.append(K_CoM_Left@(CoM_k-PL_init)-ca.DM(k_CoM_Left))
                glb.append(np.full((len(k_CoM_Left),),-np.inf))
                gub.append(np.full((len(k_CoM_Left),),0))

                #   CoM in the Right foot
                g.append(K_CoM_Right@(CoM_k-PR_init)-ca.DM(k_CoM_Right))
                glb.append(np.full((len(k_CoM_Right),),-np.inf))
                gub.append(np.full((len(k_CoM_Right),),0))
                
                #Angular Dynamics
                #Definition of Contact Points of a foot
                #P3----------------P1
                #|                  |
                #|                  |
                #|                  |
                #P4----------------P2

                if k<N_K-1: #double check the knot number is valid
                    g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = None, Ldot_current = Ldot_current, h = h, PL = PL_init, PL_TangentX = PL_init_TangentX, PL_TangentY = PL_init_TangentY, PR = PR_init, PR_TangentX = PR_init_TangentX, PR_TangentY = PR_init_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                    # g.append(L_next - L_current - h*(ca.cross((PL_init+0.11*PL_init_TangentX+0.06*PL_init_TangentY-CoM_k),FL1_k) + 
                    #                                        ca.cross((PL_init+0.11*PL_init_TangentX-0.06*PL_init_TangentY-CoM_k),FL2_k) + 
                    #                                        ca.cross((PL_init-0.11*PL_init_TangentX+0.06*PL_init_TangentY-CoM_k),FL3_k) + 
                    #                                        ca.cross((PL_init-0.11*PL_init_TangentX-0.06*PL_init_TangentY-CoM_k),FL4_k) + 
                    #                                        ca.cross((PR_init+0.11*PR_init_TangentX+0.06*PR_init_TangentY-CoM_k),FR1_k) + 
                    #                                        ca.cross((PR_init+0.11*PR_init_TangentX-0.06*PR_init_TangentY-CoM_k),FR2_k) + 
                    #                                        ca.cross((PR_init-0.11*PR_init_TangentX+0.06*PR_init_TangentY-CoM_k),FR3_k) + 
                    #                                        ca.cross((PR_init-0.11*PR_init_TangentX-0.06*PR_init_TangentY-CoM_k),FR4_k)))
                    # #g.append(Ldot_next-Ldot_current-h*(ca.cross((PL_init+np.array([0.11,0.06,0])-CoM_k),FL1_k)+ca.cross((PL_init+np.array([0.11,-0.06,0])-CoM_k),FL2_k)+ca.cross((PL_init+np.array([-0.11,0.06,0])-CoM_k),FL3_k)+ca.cross((PL_init+np.array([-0.11,-0.06,0])-CoM_k),FL4_k)+ca.cross((PR_init+np.array([0.11,0.06,0])-CoM_k),FR1_k)+ca.cross((PR_init+np.array([0.11,-0.06,0])-CoM_k),FR2_k)+ca.cross((PR_init+np.array([-0.11,0.06,0])-CoM_k),FR3_k)+ca.cross((PR_init+np.array([-0.11,-0.06,0])-CoM_k),FR4_k)))
                    # glb.append(np.array([0,0,0]))
                    # gub.append(np.array([0,0,0]))
                else:
                    print("Initial Double Stage - Angular Dynamics Constraint - Knot number exceeds limit")
                
                #Unilateral Force Constraints for all Suppport foot
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, F_k = FL1_k, TerrainNorm = PL_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, F_k = FL2_k, TerrainNorm = PL_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, F_k = FL3_k, TerrainNorm = PL_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, F_k = FL4_k, TerrainNorm = PL_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, F_k = FR1_k, TerrainNorm = PR_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, F_k = FR2_k, TerrainNorm = PR_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, F_k = FR3_k, TerrainNorm = PR_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, F_k = FR4_k, TerrainNorm = PR_init_Norm)

                #Friction Cone
                #Initial phase, no Leg Swing First Inidcators
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, F_k = FL1_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, F_k = FL2_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, F_k = FL3_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, F_k = FL4_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu)

                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, F_k = FR1_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, F_k = FR2_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, F_k = FR3_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, F_k = FR4_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu)

            elif GaitPattern[Nph]=='Swing':
                #Kinematics Constraint and Angular Dynamics Constraint

                #IF LEFT Foot is SWING (RIGHT FOOT is STATIONARY)
                #Kinematics Constraint
                #   CoM in the RIGHT Foot
                g.append(ca.if_else(ParaLeftSwingFlag, K_CoM_Right@(CoM_k-PR_init)-ca.DM(k_CoM_Right), np.full((len(k_CoM_Right),),-1)))
                glb.append(np.full((len(k_CoM_Right),),-np.inf))
                gub.append(np.full((len(k_CoM_Right),),0))

                #Angular Dynamics (Right Support)
                if k<N_K-1:
                    g, glb, gub = Angular_Momentum_Rate_Swing(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, Ldot_current = Ldot_current, h = h, P = PR_init, P_TangentX = PR_init_TangentX, P_TangentY = PR_init_TangentY, CoM_k = CoM_k, F1_k = FR1_k, F2_k = FR2_k, F3_k = FR3_k, F4_k = FR4_k)

                    # g.append(ca.if_else(ParaLeftSwingFlag, L_next - L_current - h*(ca.cross((PR_init+0.11*PR_init_TangentX+0.06*PR_init_TangentY-CoM_k),FR1_k) + 
                    #                                                                      ca.cross((PR_init+0.11*PR_init_TangentX-0.06*PR_init_TangentY-CoM_k),FR2_k) + 
                    #                                                                      ca.cross((PR_init-0.11*PR_init_TangentX+0.06*PR_init_TangentY-CoM_k),FR3_k) + 
                    #                                                                      ca.cross((PR_init-0.11*PR_init_TangentX-0.06*PR_init_TangentY-CoM_k),FR4_k)), np.array([0,0,0])))
                    # #g.append(ca.if_else(ParaLeftSwingFlag, Ldot_next-Ldot_current-h*(ca.cross((PR_init+np.array([0.11,0.06,0])-CoM_k),FR1_k)+ca.cross((PR_init+np.array([0.11,-0.06,0])-CoM_k),FR2_k)+ca.cross((PR_init+np.array([-0.11,0.06,0])-CoM_k),FR3_k)+ca.cross((PR_init+np.array([-0.11,-0.06,0])-CoM_k),FR4_k)), np.array([0,0,0])))
                    # glb.append(np.array([0,0,0]))
                    # gub.append(np.array([0,0,0]))

                #If RIGHT foot is SWING (LEFT is STATIONARY), Then LEFT Foot is the Support FOOT
                #Kinematics Constraint
                #   CoM in the Left foot
                g.append(ca.if_else(ParaRightSwingFlag, K_CoM_Left@(CoM_k-PL_init)-ca.DM(k_CoM_Left), np.full((len(k_CoM_Left),),-1)))
                glb.append(np.full((len(k_CoM_Left),),-np.inf))
                gub.append(np.full((len(k_CoM_Left),),0))

                #Angular Dynamics (Left Support)
                if k<N_K-1:
                    g, glb, gub = Angular_Momentum_Rate_Swing(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, Ldot_current = Ldot_current, h = h, P = PL_init, P_TangentX = PL_init_TangentX, P_TangentY = PL_init_TangentY, CoM_k = CoM_k, F1_k = FL1_k, F2_k = FL2_k, F3_k = FL3_k, F4_k = FL4_k)

                    # g.append(ca.if_else(ParaRightSwingFlag, L_next - L_current - h*(ca.cross((PL_init+0.11*PL_init_TangentX+0.06*PL_init_TangentY-CoM_k),FL1_k) + 
                    #                                                                       ca.cross((PL_init+0.11*PL_init_TangentX-0.06*PL_init_TangentY-CoM_k),FL2_k) + 
                    #                                                                       ca.cross((PL_init-0.11*PL_init_TangentX+0.06*PL_init_TangentY-CoM_k),FL3_k) + 
                    #                                                                       ca.cross((PL_init-0.11*PL_init_TangentX-0.06*PL_init_TangentY-CoM_k),FL4_k)), np.array([0,0,0])))
                    # #g.append(ca.if_else(ParaRightSwingFlag,Ldot_next-Ldot_current-h*(ca.cross((PL_init+np.array([0.11,0.06,0])-CoM_k),FL1_k)+ca.cross((PL_init+np.array([0.11,-0.06,0])-CoM_k),FL2_k)+ca.cross((PL_init+np.array([-0.11,0.06,0])-CoM_k),FL3_k)+ca.cross((PL_init+np.array([-0.11,-0.06,0])-CoM_k),FL4_k)), np.array([0,0,0])))
                    # glb.append(np.array([0,0,0]))
                    # gub.append(np.array([0,0,0]))

                #Unilateral Constraint
                #
                # if the Left foot Swings, then the right foot should have unilateral constraints
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainNorm = PR_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainNorm = PR_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainNorm = PR_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainNorm = PR_init_Norm)
                # if the Right foot Swings, then the Left foot should have unilateral constraints
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainNorm = PL_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainNorm = PL_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainNorm = PL_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainNorm = PL_init_Norm)

                #Zero Force Constrain
                # if the Left Foot Swings, then the Left foot should have zero forces
                g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k)
                g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k)
                g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k)
                g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k)

                # if the Right Foot Swing, then the Right foot should have zero forces
                g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k)
                g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k)
                g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k)
                g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k)

                #Friction Cone Constraint
                #If swing the Left foot first, then friction cone enforced on the Right Foot
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu)
                #If swing the Right Foot First, then the friction cone enforced on the Left Foot
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu)

            elif GaitPattern[Nph]=='DoubleSupport':
                #Kinematic Constraint and Angular Dynamics
                
                #IF LEFT Foot is SWING (RIGHT FOOT is STATIONARY)
                #Kinematics Constraint
                #   CoM in the RIGHT Foot (Init Foot)
                g.append(ca.if_else(ParaLeftSwingFlag, K_CoM_Right@(CoM_k-PR_init)-ca.DM(k_CoM_Right), np.full((len(k_CoM_Right),),-1)))
                glb.append(np.full((len(k_CoM_Right),),-np.inf))
                gub.append(np.full((len(k_CoM_Right),),0))
                #   CoM in the LEFT foot (Moved/Swing - PL_k)
                PL_k = ca.vertcat(*px,*py,*pz) #Moved/Swing Foot landing position
                g.append(ca.if_else(ParaLeftSwingFlag, K_CoM_Left@(CoM_k-PL_k)-ca.DM(k_CoM_Left), np.full((len(k_CoM_Left),),-1)))
                glb.append(np.full((len(k_CoM_Left),),-np.inf))
                gub.append(np.full((len(k_CoM_Left),),0))
                #Angular Dynamics (Double Support)
                #Terrain Tangent and Norm
                PL_k_Norm = SurfNorms[0:3]
                PL_k_TangentX = SurfTangentsX[0:3]
                PL_k_TangentY = SurfTangentsY[0:3]
                
                if k<N_K-1:
                    g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, Ldot_current = Ldot_current, h = h, PL = PL_k, PL_TangentX = PL_k_TangentX, PL_TangentY = PL_k_TangentY, PR = PR_init, PR_TangentX = PR_init_TangentX, PR_TangentY = PR_init_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)

                    # g.append(ca.if_else(ParaLeftSwingFlag, L_next - L_current - h*(ca.cross((PL_k+0.11*PL_k_TangentX+0.06*PL_k_TangentY-CoM_k),FL1_k) + 
                    #                                                                      ca.cross((PL_k+0.11*PL_k_TangentX-0.06*PL_k_TangentY-CoM_k),FL2_k) + 
                    #                                                                      ca.cross((PL_k-0.11*PL_k_TangentX+0.06*PL_k_TangentY-CoM_k),FL3_k) + 
                    #                                                                      ca.cross((PL_k-0.11*PL_k_TangentX-0.06*PL_k_TangentY-CoM_k),FL4_k) +
                    #                                                                      ca.cross((PR_init+0.11*PR_init_TangentX+0.06*PR_init_TangentY-CoM_k),FR1_k) + 
                    #                                                                      ca.cross((PR_init+0.11*PR_init_TangentX-0.06*PR_init_TangentY-CoM_k),FR2_k) + 
                    #                                                                      ca.cross((PR_init-0.11*PR_init_TangentX+0.06*PR_init_TangentY-CoM_k),FR3_k) + 
                    #                                                                      ca.cross((PR_init-0.11*PR_init_TangentX-0.06*PR_init_TangentY-CoM_k),FR4_k)), np.array([0,0,0])))
                    # #g.append(ca.if_else(ParaLeftSwingFlag,Ldot_next-Ldot_current-h*(ca.cross((PL_k+np.array([0.11,0.06,0])-CoM_k),FL1_k)+ca.cross((PL_k+np.array([0.11,-0.06,0])-CoM_k),FL2_k)+ca.cross((PL_k+np.array([-0.11,0.06,0])-CoM_k),FL3_k)+ca.cross((PL_k+np.array([-0.11,-0.06,0])-CoM_k),FL4_k)+ca.cross((PR_init+np.array([0.11,0.06,0])-CoM_k),FR1_k)+ca.cross((PR_init+np.array([0.11,-0.06,0])-CoM_k),FR2_k)+ca.cross((PR_init+np.array([-0.11,0.06,0])-CoM_k),FR3_k)+ca.cross((PR_init+np.array([-0.11,-0.06,0])-CoM_k),FR4_k)),np.array([0,0,0])))
                    # glb.append(np.array([0,0,0]))
                    # gub.append(np.array([0,0,0]))
                
                #if RIGHT Foot is SWING (LEFT FOOT is STATIONARY)
                #Kinematics Constraint
                #   CoM in the Left foot (Init Foot)
                g.append(ca.if_else(ParaRightSwingFlag, K_CoM_Left@(CoM_k-PL_init)-ca.DM(k_CoM_Left),np.full((len(k_CoM_Left),),-1)))
                glb.append(np.full((len(k_CoM_Left),),-np.inf))
                gub.append(np.full((len(k_CoM_Left),),0))
                #   CoM in the Right foot (Moved/Swing - PR_k) 
                PR_k = ca.vertcat(*px,*py,*pz) #Moved/Swing Foot landing position
                g.append(ca.if_else(ParaRightSwingFlag,K_CoM_Right@(CoM_k-PR_k)-ca.DM(k_CoM_Right),np.full((len(k_CoM_Right),),-1)))
                glb.append(np.full((len(k_CoM_Right),),-np.inf))
                gub.append(np.full((len(k_CoM_Right),),0))

                #Angular Dynamics (Double Support)
                #Terrain Tangent and Norm
                PR_k_Norm = SurfNorms[0:3]
                PR_k_TangentX = SurfTangentsX[0:3]
                PR_k_TangentY = SurfTangentsY[0:3]
                
                if k<N_K-1:
                    g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, Ldot_current = Ldot_current, h = h, PL = PL_init, PL_TangentX = PL_init_TangentX, PL_TangentY = PL_init_TangentY, PR = PR_k, PR_TangentX = PR_k_TangentX, PR_TangentY = PR_k_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)

                    # g.append(ca.if_else(ParaRightSwingFlag, L_next - L_current - h*(ca.cross((PL_init+0.11*PL_init_TangentX+0.06*PL_init_TangentY-CoM_k),FL1_k) + 
                    #                                                                       ca.cross((PL_init+0.11*PL_init_TangentX-0.06*PL_init_TangentY-CoM_k),FL2_k) + 
                    #                                                                       ca.cross((PL_init-0.11*PL_init_TangentX+0.06*PL_init_TangentY-CoM_k),FL3_k) + 
                    #                                                                       ca.cross((PL_init-0.11*PL_init_TangentX-0.06*PL_init_TangentY-CoM_k),FL4_k) +
                    #                                                                       ca.cross((PR_k+0.11*PR_k_TangentX+0.06*PR_k_TangentY-CoM_k),FR1_k) + 
                    #                                                                       ca.cross((PR_k+0.11*PR_k_TangentX-0.06*PR_k_TangentY-CoM_k),FR2_k) + 
                    #                                                                       ca.cross((PR_k-0.11*PR_k_TangentX+0.06*PR_k_TangentY-CoM_k),FR3_k) + 
                    #                                                                       ca.cross((PR_k-0.11*PR_k_TangentX-0.06*PR_k_TangentY-CoM_k),FR4_k)), np.array([0,0,0])))                    
                    # #g.append(ca.if_else(ParaRightSwingFlag,Ldot_next-Ldot_current-h*(ca.cross((PL_init+np.array([0.11,0.06,0])-CoM_k),FL1_k)+ca.cross((PL_init+np.array([0.11,-0.06,0])-CoM_k),FL2_k)+ca.cross((PL_init+np.array([-0.11,0.06,0])-CoM_k),FL3_k)+ca.cross((PL_init+np.array([-0.11,-0.06,0])-CoM_k),FL4_k)+ca.cross((PR_k+np.array([0.11,0.06,0])-CoM_k),FR1_k)+ca.cross((PR_k+np.array([0.11,-0.06,0])-CoM_k),FR2_k)+ca.cross((PR_k+np.array([-0.11,0.06,0])-CoM_k),FR3_k)+ca.cross((PR_k+np.array([-0.11,-0.06,0])-CoM_k),FR4_k)),np.array([0,0,0])))
                    # glb.append(np.array([0,0,0]))
                    # gub.append(np.array([0,0,0]))

                #Unilater Constraints
                # Norm at the new landing surface
                Pnext_Norm = SurfNorms[0:3]
                Pnext_TangentX = SurfTangentsX[0:3]
                Pnext_TangentY = SurfTangentsY[0:3]

                #Case 1 
                # if swing the Left foot first, then the Left foot obey unilateral constraint on the New SurfaceNorm
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainNorm = Pnext_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainNorm = Pnext_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainNorm = Pnext_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainNorm = Pnext_Norm)
                # Then the Right foot oby the unilateral constraints on the Init Surface Norm
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainNorm = PR_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainNorm = PR_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainNorm = PR_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainNorm = PR_init_Norm)
                
                #Case 2
                # if swing the Right foot first, then the Right foot obey unilateral constraint on the New SurfaceNorm
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainNorm = Pnext_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainNorm = Pnext_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainNorm = Pnext_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainNorm = Pnext_Norm)
                # Then the Left foot oby the unilateral constraints on the Init Surface Norm
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainNorm = PL_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainNorm = PL_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainNorm = PL_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainNorm = PL_init_Norm)

                #Friction Cone
                
                #Case 1
                #If Swing the Left foot first, then the Left foot obey the friction cone constraint in the new landing place
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainTangentX = Pnext_TangentX, TerrainTangentY = Pnext_TangentY, TerrainNorm = Pnext_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainTangentX = Pnext_TangentX, TerrainTangentY = Pnext_TangentY, TerrainNorm = Pnext_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainTangentX = Pnext_TangentX, TerrainTangentY = Pnext_TangentY, TerrainNorm = Pnext_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainTangentX = Pnext_TangentX, TerrainTangentY = Pnext_TangentY, TerrainNorm = Pnext_Norm, miu = miu)
                #Then the Right foot obey the fricition cone constraint in the initial landing place
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu)

                #Case 2
                #If swing the Right foot first, then the Right foot obey the friction cone constraint in the new landing place
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainTangentX = Pnext_TangentX, TerrainTangentY = Pnext_TangentY, TerrainNorm = Pnext_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainTangentX = Pnext_TangentX, TerrainTangentY = Pnext_TangentY, TerrainNorm = Pnext_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainTangentX = Pnext_TangentX, TerrainTangentY = Pnext_TangentY, TerrainNorm = Pnext_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainTangentX = Pnext_TangentX, TerrainTangentY = Pnext_TangentY, TerrainNorm = Pnext_Norm, miu = miu)
                # then the Left foot obey the constaint on the initial landing place
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu)

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

                #First-order Angular Momentum Dynamics x-axis
                g.append(Lx[k+1] - Lx[k] - h*Ldotx[k])
                glb.append(np.array([0]))
                gub.append(np.array([0]))

                #First-order Angular Momentum Dynamics y-axis
                g.append(Ly[k+1] - Ly[k] - h*Ldoty[k])
                glb.append(np.array([0]))
                gub.append(np.array([0]))

                #First-order Angular Momentum Dynamics z-axis
                g.append(Lz[k+1] - Lz[k] - h*Ldotz[k])
                glb.append(np.array([0]))
                gub.append(np.array([0]))

                #Second-order Dynamics x-axis
                g.append(xdot[k+1] - xdot[k] - h*(FL1x[k]/m+FL2x[k]/m+FL3x[k]/m+FL4x[k]/m+FR1x[k]/m+FR2x[k]/m+FR3x[k]/m+FR4x[k]/m))
                glb.append(np.array([0]))
                gub.append(np.array([0]))

                #Second-order Dynamics y-axis
                g.append(ydot[k+1] - ydot[k] - h*(FL1y[k]/m+FL2y[k]/m+FL3y[k]/m+FL4y[k]/m+FR1y[k]/m+FR2y[k]/m+FR3y[k]/m+FR4y[k]/m))
                glb.append(np.array([0]))
                gub.append(np.array([0]))

                #Second-order Dynamics z-axis
                g.append(zdot[k+1] - zdot[k] - h*(FL1z[k]/m+FL2z[k]/m+FL3z[k]/m+FL4z[k]/m+FR1z[k]/m+FR2z[k]/m+FR3z[k]/m+FR4z[k]/m - G))
                glb.append(np.array([0]))
                gub.append(np.array([0]))
            
                ##Constant Acceleration
                #Accx_1  = FL1x[k]/m+FL2x[k]/m+FL3x[k]/m+FL4x[k]/m+FR1x[k]/m+FR2x[k]/m+FR3x[k]/m+FR4x[k]/m
                #Accx_2  = FL1x[k+1]/m+FL2x[k+1]/m+FL3x[k+1]/m+FL4x[k+1]/m+FR1x[k+1]/m+FR2x[k+1]/m+FR3x[k+1]/m+FR4x[k+1]/m
                
                #Accy_1  = FL1y[k]/m+FL2y[k]/m+FL3y[k]/m+FL4y[k]/m+FR1y[k]/m+FR2y[k]/m+FR3y[k]/m+FR4y[k]/m
                #Accy_2  = FL1y[k+1]/m+FL2y[k+1]/m+FL3y[k+1]/m+FL4y[k+1]/m+FR1y[k+1]/m+FR2y[k+1]/m+FR3y[k+1]/m+FR4y[k+1]/m

                #Accz_1  = FL1z[k]/m+FL2z[k]/m+FL3z[k]/m+FL4z[k]/m+FR1z[k]/m+FR2z[k]/m+FR3z[k]/m+FR4z[k]/m
                #Accz_2  = FL1z[k+1]/m+FL2z[k+1]/m+FL3z[k+1]/m+FL4z[k+1]/m+FR1z[k+1]/m+FR2z[k+1]/m+FR3z[k+1]/m+FR4z[k+1]/m

                #g.append(Accx_1-Accx_2)
                #glb.append(np.array([0]))
                #gub.append(np.array([0]))

                #g.append(Accy_1-Accy_2)
                #glb.append(np.array([0]))
                #gub.append(np.array([0]))

                #g.append(Accz_1-Accz_2)
                #glb.append(np.array([0]))
                #gub.append(np.array([0]))
            
            #Add Cost Terms
            if k < N_K - 1:
                #with angular momentum
                #J = J + h*Lx[k]**2 + h*Ly[k]**2 + h*Lz[k]**2 + h*(FL1x[k]/m+FL2x[k]/m+FL3x[k]/m+FL4x[k]/m+FR1x[k]/m+FR2x[k]/m+FR3x[k]/m+FR4x[k]/m)**2 + h*(FL1y[k]/m+FL2y[k]/m+FL3y[k]/m+FL4y[k]/m+FR1y[k]/m+FR2y[k]/m+FR3y[k]/m+FR4y[k]/m)**2 + h*(FL1z[k]/m+FL2z[k]/m+FL3z[k]/m+FL4z[k]/m+FR1z[k]/m+FR2z[k]/m+FR3z[k]/m+FR4z[k]/m - G)**2
                #No Angular momentum
                #J = J + h*(FL1x[k]/m+FL2x[k]/m+FL3x[k]/m+FL4x[k]/m+FR1x[k]/m+FR2x[k]/m+FR3x[k]/m+FR4x[k]/m)**2 + h*(FL1y[k]/m+FL2y[k]/m+FL3y[k]/m+FL4y[k]/m+FR1y[k]/m+FR2y[k]/m+FR3y[k]/m+FR4y[k]/m)**2 + h*(FL1z[k]/m+FL2z[k]/m+FL3z[k]/m+FL4z[k]/m+FR1z[k]/m+FR2z[k]/m+FR3z[k]/m+FR4z[k]/m - G)**2
                #With Angular momentum rate
                #J = J + h*Ldotx[k]**2 + h*Ldoty[k]**2 + h*Ldotz[k]**2 + h*(FL1x[k]/m+FL2x[k]/m+FL3x[k]/m+FL4x[k]/m+FR1x[k]/m+FR2x[k]/m+FR3x[k]/m+FR4x[k]/m)**2 + h*(FL1y[k]/m+FL2y[k]/m+FL3y[k]/m+FL4y[k]/m+FR1y[k]/m+FR2y[k]/m+FR3y[k]/m+FR4y[k]/m)**2 + h*(FL1z[k]/m+FL2z[k]/m+FL3z[k]/m+FL4z[k]/m+FR1z[k]/m+FR2z[k]/m+FR3z[k]/m+FR4z[k]/m - G)**2
                #With Angular momentum and angular momentum together
                J = J + h*Lx[k]**2 + h*Ly[k]**2 + h*Lz[k]**2 + h*Ldotx[k]**2 + h*Ldoty[k]**2 + h*Ldotz[k]**2 + h*(FL1x[k]/m+FL2x[k]/m+FL3x[k]/m+FL4x[k]/m+FR1x[k]/m+FR2x[k]/m+FR3x[k]/m+FR4x[k]/m)**2 + h*(FL1y[k]/m+FL2y[k]/m+FL3y[k]/m+FL4y[k]/m+FR1y[k]/m+FR2y[k]/m+FR3y[k]/m+FR4y[k]/m)**2 + h*(FL1z[k]/m+FL2z[k]/m+FL3z[k]/m+FL4z[k]/m+FR1z[k]/m+FR2z[k]/m+FR3z[k]/m+FR4z[k]/m - G)**2

    #Relative Foot Constraints
    #   For init phase
    g.append(Q_rf_in_lf@(PR_init-PL_init))
    glb.append(np.full((len(q_rf_in_lf),),-np.inf))
    gub.append(q_rf_in_lf)

    #   For the Double Support Phase
    p_next = ca.vertcat(*px,*py,*pz)
    
    #If LEFT foot is SWING (RIGHT is STATIONARY), Then LEFT Foot should Stay in the polytpe of the RIGHT FOOT
    g.append(ca.if_else(ParaLeftSwingFlag,Q_lf_in_rf@(p_next-PR_init)-ca.DM(q_lf_in_rf),np.full((len(q_lf_in_rf),),-1)))
    glb.append(np.full((len(q_lf_in_rf),),-np.inf))
    gub.append(np.full((len(q_lf_in_rf),),0))
    #Right - Stationary foot should also inthe polytope of the Swing foot
    g.append(ca.if_else(ParaLeftSwingFlag,Q_rf_in_lf@(PR_init-p_next)-ca.DM(q_rf_in_lf),np.full((len(q_rf_in_lf),),-1)))
    glb.append(np.full((len(q_rf_in_lf),),-np.inf))
    gub.append(np.full((len(q_rf_in_lf),),0))

    #If RIGHT foot is SWING (LEFT is STATIONARY), Then RIGHT Foot should stay in the polytope of the LEFT Foot
    g.append(ca.if_else(ParaRightSwingFlag,Q_rf_in_lf@(p_next-PL_init)-ca.DM(q_rf_in_lf),np.full((len(q_rf_in_lf),),-1)))
    glb.append(np.full((len(q_rf_in_lf),),-np.inf))
    gub.append(np.full((len(q_rf_in_lf),),0))
    #Left - Stationary foot should also in the polytope of the Swing foot
    g.append(ca.if_else(ParaRightSwingFlag,Q_lf_in_rf@(PL_init-p_next)-ca.DM(q_lf_in_rf),np.full((len(q_lf_in_rf),),-1)))
    glb.append(np.full((len(q_lf_in_rf),),-np.inf))
    gub.append(np.full((len(q_lf_in_rf),),0))

    #FootStep Constraint
    #P3----------------P1
    #|                  |
    #|                  |
    #|                  |
    #P4----------------P2
    
    # Norm at the new landing surface
    Pnext_Norm = SurfNorms[0:3]
    Pnext_TangentX = SurfTangentsX[0:3]
    Pnext_TangentY = SurfTangentsY[0:3]

    #Contact Point 1
    #Inequality
    g.append(FirstSurfK@(p_next + 0.11*Pnext_TangentX + 0.06*Pnext_TangentY) - FirstSurfk)
    glb.append(np.full((4,),-np.inf))
    gub.append(np.full((4,),0))
    #Equality
    g.append(FirstSurfE.T@(p_next + 0.11*Pnext_TangentX + 0.06*Pnext_TangentY) - FirstSurfe)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

    #Contact Point 2
    #Inequality
    g.append(FirstSurfK@(p_next + 0.11*Pnext_TangentX - 0.06*Pnext_TangentY) - FirstSurfk)
    glb.append(np.full((4,),-np.inf))
    gub.append(np.full((4,),0))
    #Equality
    g.append(FirstSurfE.T@(p_next + 0.11*Pnext_TangentX - 0.06*Pnext_TangentY) - FirstSurfe)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

    #Contact Point 3
    #Inequality
    g.append(FirstSurfK@(p_next - 0.11*Pnext_TangentX + 0.06*Pnext_TangentY) - FirstSurfk)
    glb.append(np.full((4,),-np.inf))
    gub.append(np.full((4,),0))
    #Equality
    g.append(FirstSurfE.T@(p_next - 0.11*Pnext_TangentX + 0.06*Pnext_TangentY) - FirstSurfe)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

    #Contact Point 4
    #Inequality
    g.append(FirstSurfK@(p_next + 0.11*Pnext_TangentX + 0.06*Pnext_TangentY) - FirstSurfk)
    glb.append(np.full((4,),-np.inf))
    gub.append(np.full((4,),0))
    #Equality
    g.append(FirstSurfE.T@(p_next + 0.11*Pnext_TangentX + 0.06*Pnext_TangentY) - FirstSurfe)
    glb.append(np.array([0]))
    gub.append(np.array([0]))


    #Approximate Kinematics Contraint to avoid footstep over-crossing from y=0
    
    if CentralY == True:
        #if Left swing the first, then the landing foot is the left
        g.append(ca.if_else(ParaLeftSwingFlag,p_next[1],np.array([1])))
        glb.append(np.array([py_lower_limit]))
        gub.append(np.array([np.inf]))

        #if Right swing the first, then the landing foot is the Right
        g.append(ca.if_else(ParaRightSwingFlag,p_next[1],np.array([-1])))
        glb.append(np.array([-np.inf]))
        gub.append(np.array([-py_lower_limit]))

    #Switching Time Constraint
    #NOTE: For Unconservative First Level, The range of double support is 0.1 to 0.3, The range of swing is 0.3 to 0.9
    #NOTE: For conservative First Level, The range of double support is 0.1 to 0.5, The range of swing is 0.3 to 0.9
    for phase_cnt in range(Nphase):
        if GaitPattern[phase_cnt] == 'InitialDouble':
            g.append(Ts[phase_cnt])
            glb.append(np.array([0.1])) #0.1 - 0.3
            gub.append(np.array([0.3]))
        elif GaitPattern[phase_cnt] == 'Swing':
            g.append(Ts[phase_cnt]-Ts[phase_cnt-1]) #0.6-1
            glb.append(np.array([0.5])) #0.5-0.7
            gub.append(np.array([0.7])) #0.4 - 0.9
        elif GaitPattern[phase_cnt] == 'DoubleSupport':
            g.append(Ts[phase_cnt]-Ts[phase_cnt-1])#0.05-0.3
            glb.append(np.array([0.1]))
            gub.append(np.array([0.3])) #0.1 - 0.3

    #-----------------------------------------------------------------------------------------------------------------------
    #Get Variable Index - !!!This is the pure Index, when try to get the array using other routines, we need to add "+1" at the last index due to Python indexing conventions
    x_index = (0,N_K-1) #First set of variables start counting from 0, The enumeration of the last knot is N_K-1
    y_index = (x_index[1]+1,x_index[1]+N_K)
    z_index = (y_index[1]+1,y_index[1]+N_K)
    xdot_index = (z_index[1]+1,z_index[1]+N_K)
    ydot_index = (xdot_index[1]+1,xdot_index[1]+N_K)
    zdot_index = (ydot_index[1]+1,ydot_index[1]+N_K)
    Lx_index = (zdot_index[1]+1,zdot_index[1]+N_K)
    Ly_index = (Lx_index[1]+1,Lx_index[1]+N_K)
    Lz_index = (Ly_index[1]+1,Ly_index[1]+N_K)
    Ldotx_index = (Lz_index[1]+1,Lz_index[1]+N_K)
    Ldoty_index = (Ldotx_index[1]+1,Ldotx_index[1]+N_K)
    Ldotz_index = (Ldoty_index[1]+1,Ldoty_index[1]+N_K)
    FL1x_index = (Ldotz_index[1]+1,Ldotz_index[1]+N_K)
    FL1y_index = (FL1x_index[1]+1,FL1x_index[1]+N_K)
    FL1z_index = (FL1y_index[1]+1,FL1y_index[1]+N_K)
    FL2x_index = (FL1z_index[1]+1,FL1z_index[1]+N_K)
    FL2y_index = (FL2x_index[1]+1,FL2x_index[1]+N_K)
    FL2z_index = (FL2y_index[1]+1,FL2y_index[1]+N_K)
    FL3x_index = (FL2z_index[1]+1,FL2z_index[1]+N_K)
    FL3y_index = (FL3x_index[1]+1,FL3x_index[1]+N_K)
    FL3z_index = (FL3y_index[1]+1,FL3y_index[1]+N_K)
    FL4x_index = (FL3z_index[1]+1,FL3z_index[1]+N_K)
    FL4y_index = (FL4x_index[1]+1,FL4x_index[1]+N_K)
    FL4z_index = (FL4y_index[1]+1,FL4y_index[1]+N_K)
    FR1x_index = (FL4z_index[1]+1,FL4z_index[1]+N_K)
    FR1y_index = (FR1x_index[1]+1,FR1x_index[1]+N_K)
    FR1z_index = (FR1y_index[1]+1,FR1y_index[1]+N_K)
    FR2x_index = (FR1z_index[1]+1,FR1z_index[1]+N_K)
    FR2y_index = (FR2x_index[1]+1,FR2x_index[1]+N_K)
    FR2z_index = (FR2y_index[1]+1,FR2y_index[1]+N_K)
    FR3x_index = (FR2z_index[1]+1,FR2z_index[1]+N_K)
    FR3y_index = (FR3x_index[1]+1,FR3x_index[1]+N_K)
    FR3z_index = (FR3y_index[1]+1,FR3y_index[1]+N_K)
    FR4x_index = (FR3z_index[1]+1,FR3z_index[1]+N_K)
    FR4y_index = (FR4x_index[1]+1,FR4x_index[1]+N_K)
    FR4z_index = (FR4y_index[1]+1,FR4y_index[1]+N_K)
    px_index = (FR4z_index[1]+1,FR4z_index[1]+Nstep)
    py_index = (px_index[1]+1,px_index[1]+Nstep)
    pz_index = (py_index[1]+1,py_index[1]+Nstep)
    Ts_index = (pz_index[1]+1,pz_index[1]+Nphase)

    var_index = {"x":x_index,
                 "y":y_index,
                 "z":z_index,
                 "xdot":xdot_index,
                 "ydot":ydot_index,
                 "zdot":zdot_index,
                 "Lx":Lx_index,
                 "Ly":Ly_index,
                 "Lz":Lz_index,
                 "Ldotx":Ldotx_index,
                 "Ldoty":Ldoty_index,
                 "Ldotz":Ldotz_index,
                 "FL1x":FL1x_index,
                 "FL1y":FL1y_index,
                 "FL1z":FL1z_index,
                 "FL2x":FL2x_index,
                 "FL2y":FL2y_index,
                 "FL2z":FL2z_index,
                 "FL3x":FL3x_index,
                 "FL3y":FL3y_index,
                 "FL3z":FL3z_index,
                 "FL4x":FL4x_index,
                 "FL4y":FL4y_index,
                 "FL4z":FL4z_index,
                 "FR1x":FR1x_index,
                 "FR1y":FR1y_index,
                 "FR1z":FR1z_index,
                 "FR2x":FR2x_index,
                 "FR2y":FR2y_index,
                 "FR2z":FR2z_index,
                 "FR3x":FR3x_index,
                 "FR3y":FR3y_index,
                 "FR3z":FR3z_index,
                 "FR4x":FR4x_index,
                 "FR4y":FR4y_index,
                 "FR4z":FR4z_index,
                 "px":px_index,
                 "py":py_index,
                 "pz":pz_index,
                 "Ts":Ts_index,
    }

    return DecisionVars, DecisionVars_lb, DecisionVars_ub, J, g, glb, gub, var_index

#NLP Second Level
def NLP_SecondLevel(m = 95, Nk_Local = 7, Nsteps = 1, ParameterList = None, StaticStop = False, NumPatches = None, CentralY = False):
    #-----------------------------------------------------------------------------------------------------------------------
    #Define Parameters
    #   Gait Pattern, Each action is followed up by a double support phase
    GaitPattern = ["InitialDouble","Swing","DoubleSupport"] + ["InitialDouble", "Swing","DoubleSupport"]*(Nsteps-1) #,'RightSupport','DoubleSupport','LeftSupport','DoubleSupport'

    #   Number of Phases
    Nphase = len(GaitPattern)
    #   Number of Steps
    #Nstep = 1
    #   Number of Knots per Phase - how many intervals do we have for a single phase; 10 intervals need 11 knots/ticks
    #Nk_Local= 5
    #   Compute Number of Total knots/ticks, but the enumeration start from 0 to N_K-1
    N_K = Nk_Local*Nphase + 1 #+1 the last knots to finalize the plan
    #print(N_K)
    #   Robot mass
    #m = 95 #kg
    G = 9.80665 #kg/m^2
    #   Terrain Model
    #       Flat Terrain
    #TerrainNorm = [0,0,1] 
    #TerrainTangentX = [1,0,0]
    #TerrainTangentY = [0,1,0]
    miu = 0.3
    #   Force Limits
    F_bound = 400
    Fxlb = -F_bound
    Fxub = F_bound
    Fylb = -F_bound
    Fyub = F_bound
    Fzlb = -F_bound
    Fzub = F_bound
    #   Angular Momentum Bounds
    L_bound = 2.5
    Ldot_bound = 3.5
    Lub = L_bound
    Llb = -L_bound
    Ldotub = Ldot_bound
    Ldotlb = -Ldot_bound
    #Minimum y-axis foot location
    py_lower_limit = 0.04
    #Lowest Z
    z_lowest = 0.7
    z_highest = 0.8
    #-----------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------
    #Kinematics Constraint for Talos
    #kinematicConstraints = genKinematicConstraints(left_foot_constraints,right_foot_constraints)
    K_CoM_Right,k_CoM_Right = right_foot_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]))
    K_CoM_Left,k_CoM_Left = left_foot_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]))

    #K_CoM_Left = kinematicConstraints[0][0]
    #k_CoM_Left = kinematicConstraints[0][1]
    #K_CoM_Right = kinematicConstraints[1][0]
    #k_CoM_Right = kinematicConstraints[1][1]
    #Relative Foot Constraint matrices
    
    #relativeConstraints = genFootRelativeConstraints(right_foot_in_lf_frame_constraints,left_foot_in_rf_frame_constraints)
    Q_rf_in_lf,q_rf_in_lf = right_foot_in_lf_frame_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]))
    Q_lf_in_rf,q_lf_in_rf = left_foot_in_rf_frame_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]))

    #Q_rf_in_lf = relativeConstraints[0][0] #named lf in rf, but representing rf in lf
    #q_rf_in_lf = relativeConstraints[0][1] #named lf in rf, but representing rf in lf
    #Q_lf_in_rf = relativeConstraints[1][0] #named rf in lf, but representing lf in rf
    #q_lf_in_rf = relativeConstraints[1][1] #named rf in lf, but representing lf in rf
    #-----------------------------------------------------------------------------------------------------------------------
    #Define Initial and Terminal Conditions
    #x_init = ca.SX.sym('x_init')
    #y_init = ca.SX.sym('y_init')
    #z_init = ca.SX.sym('z_init')

    x_init = ParameterList["x_init"]
    y_init = ParameterList["y_init"]
    z_init = ParameterList["z_init"]

    xdot_init = ParameterList["xdot_init"]
    ydot_init = ParameterList["ydot_init"]
    zdot_init = ParameterList["zdot_init"]

    #Lx_init = 0
    #Ly_init = 0
    #Lz_init = 0

    #Ldotx_init = 0
    #Ldoty_init = 0
    #Ldotz_init = 0

    PLx_init = ParameterList["PLx_init"]
    PLy_init = ParameterList["PLy_init"]
    PLz_init = ParameterList["PLz_init"]
    PL_init = ca.vertcat(PLx_init,PLy_init,PLz_init)

    PRx_init = ParameterList["PRx_init"]
    PRy_init = ParameterList["PRy_init"]
    PRz_init = ParameterList["PRz_init"]
    PR_init = ca.vertcat(PRx_init,PRy_init,PRz_init)

    x_end = ParameterList["x_end"]
    y_end = ParameterList["y_end"]
    z_end = ParameterList["z_end"]

    xdot_end = ParameterList["xdot_end"]
    ydot_end = ParameterList["ydot_end"]
    zdot_end = ParameterList["zdot_end"]

    #Lx_end = 0
    #Ly_end = 0
    #Lz_end = 0

    #Ldotx_end = 0
    #Ldoty_end = 0
    #Ldotz_end = 0

    #Flags for Swing Legs (Defined as Parameters)
    ParaLeftSwingFlag = ParameterList["LeftSwingFlag"]
    ParaRightSwingFlag = ParameterList["RightSwingFlag"]

    #Surfaces (Only the Second One)
    #Surface Patches
    SurfParas = ParameterList["SurfParas"]

    #Tangents and Norms
    #Initial Contact Norm and Tangents
    PL_init_Norm = ParameterList["PL_init_Norm"]
    PL_init_TangentX = ParameterList["PL_init_TangentX"]
    PL_init_TangentY = ParameterList["PL_init_TangentY"]
    PR_init_Norm = ParameterList["PR_init_Norm"]
    PR_init_TangentX = ParameterList["PR_init_TangentX"]
    PR_init_TangentY = ParameterList["PR_init_TangentY"]
    
    #Future Contact Norm and Tangents
    SurfNorms = ParameterList["SurfNorms"]                
    SurfTangentsX = ParameterList["SurfTangentsX"]
    SurfTangentsY = ParameterList["SurfTangentsY"]

    # #Refrence Trajectories
    # x_ref = ParameterList["x_ref"]
    # y_ref = ParameterList["y_ref"]
    # z_ref = ParameterList["z_ref"]
    # xdot_ref = ParameterList["xdot_ref"]
    # ydot_ref = ParameterList["ydot_ref"]
    # zdot_ref = ParameterList["zdot_ref"]
    # FLx_ref = ParameterList["FLx_ref"]
    # FLy_ref = ParameterList["FLy_ref"]
    # FLz_ref = ParameterList["FLz_ref"]
    # FRx_ref = ParameterList["FRx_ref"]
    # FRy_ref = ParameterList["FRy_ref"]
    # FRz_ref = ParameterList["FRz_ref"]
    # SwitchingTimeVec_ref = ParameterList["SwitchingTimeVec_ref"]
    # Px_seq_ref = ParameterList["Px_seq_ref"]
    # Py_seq_ref = ParameterList["Py_seq_ref"]
    # Pz_seq_ref = ParameterList["Pz_seq_ref"]

    #-----------------------------------------------------------------------------------------------------------------------
    #Define Variables and Bounds, Parameters
    #   CoM Position x-axis
    x = ca.SX.sym('x',N_K)
    x_lb = np.array([[0]*(x.shape[0]*x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    x_ub = np.array([[30]*(x.shape[0]*x.shape[1])])
    #   CoM Position y-axis
    y = ca.SX.sym('y',N_K)
    y_lb = np.array([[-1]*(y.shape[0]*y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    y_ub = np.array([[1]*(y.shape[0]*y.shape[1])])
    #   CoM Position z-axis
    z = ca.SX.sym('z',N_K)
    z_lb = np.array([[z_lowest]*(z.shape[0]*z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    z_ub = np.array([[z_highest]*(z.shape[0]*z.shape[1])])
    #   CoM Velocity x-axis
    xdot = ca.SX.sym('xdot',N_K)
    xdot_lb = np.array([[-1.5]*(xdot.shape[0]*xdot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    xdot_ub = np.array([[1.5]*(xdot.shape[0]*xdot.shape[1])])
    #   CoM Velocity y-axis
    ydot = ca.SX.sym('ydot',N_K)
    ydot_lb = np.array([[-1.5]*(ydot.shape[0]*ydot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    ydot_ub = np.array([[1.5]*(ydot.shape[0]*ydot.shape[1])])
    #   CoM Velocity z-axis
    zdot = ca.SX.sym('zdot',N_K)
    zdot_lb = np.array([[-1.5]*(zdot.shape[0]*zdot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    zdot_ub = np.array([[1.5]*(zdot.shape[0]*zdot.shape[1])])
    #   Angular Momentum x-axis
    Lx = ca.SX.sym('Lx',N_K)
    Lx_lb = np.array([[Llb]*(Lx.shape[0]*Lx.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Lx_ub = np.array([[Lub]*(Lx.shape[0]*Lx.shape[1])])
    #   Angular Momentum y-axis
    Ly = ca.SX.sym('Ly',N_K)
    Ly_lb = np.array([[Llb]*(Ly.shape[0]*Ly.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Ly_ub = np.array([[Lub]*(Ly.shape[0]*Ly.shape[1])])
    #   Angular Momntum y-axis
    Lz = ca.SX.sym('Lz',N_K)
    Lz_lb = np.array([[Llb]*(Lz.shape[0]*Lz.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Lz_ub = np.array([[Lub]*(Lz.shape[0]*Lz.shape[1])])
    #   Angular Momentum rate x-axis
    Ldotx = ca.SX.sym('Ldotx',N_K)
    Ldotx_lb = np.array([[Ldotlb]*(Ldotx.shape[0]*Ldotx.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Ldotx_ub = np.array([[Ldotub]*(Ldotx.shape[0]*Ldotx.shape[1])])
    #   Angular Momentum y-axis
    Ldoty = ca.SX.sym('Ldoty',N_K)
    Ldoty_lb = np.array([[Ldotlb]*(Ldoty.shape[0]*Ldoty.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Ldoty_ub = np.array([[Ldotub]*(Ldoty.shape[0]*Ldoty.shape[1])])
    #   Angular Momntum z-axis
    Ldotz = ca.SX.sym('Ldotz',N_K)
    Ldotz_lb = np.array([[Ldotlb]*(Ldotz.shape[0]*Ldotz.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Ldotz_ub = np.array([[Ldotub]*(Ldotz.shape[0]*Ldotz.shape[1])])
    #left Foot Forces
    #Left Foot Contact Point 1 x-axis
    FL1x = ca.SX.sym('FL1x',N_K)
    FL1x_lb = np.array([[Fxlb]*(FL1x.shape[0]*FL1x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL1x_ub = np.array([[Fxub]*(FL1x.shape[0]*FL1x.shape[1])])
    #Left Foot Contact Point 1 y-axis
    FL1y = ca.SX.sym('FL1y',N_K)
    FL1y_lb = np.array([[Fylb]*(FL1y.shape[0]*FL1y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL1y_ub = np.array([[Fyub]*(FL1y.shape[0]*FL1y.shape[1])])
    #Left Foot Contact Point 1 z-axis
    FL1z = ca.SX.sym('FL1z',N_K)
    FL1z_lb = np.array([[Fzlb]*(FL1z.shape[0]*FL1z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL1z_ub = np.array([[Fzub]*(FL1z.shape[0]*FL1z.shape[1])])
    #Left Foot Contact Point 2 x-axis
    FL2x = ca.SX.sym('FL2x',N_K)
    FL2x_lb = np.array([[Fxlb]*(FL2x.shape[0]*FL2x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL2x_ub = np.array([[Fxub]*(FL2x.shape[0]*FL2x.shape[1])])
    #Left Foot Contact Point 2 y-axis
    FL2y = ca.SX.sym('FL2y',N_K)
    FL2y_lb = np.array([[Fylb]*(FL2y.shape[0]*FL2y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL2y_ub = np.array([[Fyub]*(FL2y.shape[0]*FL2y.shape[1])])
    #Left Foot Contact Point 2 z-axis
    FL2z = ca.SX.sym('FL2z',N_K)
    FL2z_lb = np.array([[Fzlb]*(FL2z.shape[0]*FL2z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL2z_ub = np.array([[Fzub]*(FL2z.shape[0]*FL2z.shape[1])])
    #Left Foot Contact Point 3 x-axis
    FL3x = ca.SX.sym('FL3x',N_K)
    FL3x_lb = np.array([[Fxlb]*(FL3x.shape[0]*FL3x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL3x_ub = np.array([[Fxub]*(FL3x.shape[0]*FL3x.shape[1])])
    #Left Foot Contact Point 3 y-axis
    FL3y = ca.SX.sym('FL3y',N_K)
    FL3y_lb = np.array([[Fylb]*(FL3y.shape[0]*FL3y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL3y_ub = np.array([[Fyub]*(FL3y.shape[0]*FL3y.shape[1])])
    #Left Foot Contact Point 3 z-axis
    FL3z = ca.SX.sym('FL3z',N_K)
    FL3z_lb = np.array([[Fzlb]*(FL3z.shape[0]*FL3z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL3z_ub = np.array([[Fzub]*(FL3z.shape[0]*FL3z.shape[1])])
    #Left Foot Contact Point 4 x-axis
    FL4x = ca.SX.sym('FL4x',N_K)
    FL4x_lb = np.array([[Fxlb]*(FL4x.shape[0]*FL4x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL4x_ub = np.array([[Fxub]*(FL4x.shape[0]*FL4x.shape[1])])
    #Left Foot Contact Point 4 y-axis
    FL4y = ca.SX.sym('FL4y',N_K)
    FL4y_lb = np.array([[Fylb]*(FL4y.shape[0]*FL4y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL4y_ub = np.array([[Fyub]*(FL4y.shape[0]*FL4y.shape[1])])
    #Left Foot Contact Point 4 z-axis
    FL4z = ca.SX.sym('FL4z',N_K)
    FL4z_lb = np.array([[Fzlb]*(FL4z.shape[0]*FL4z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL4z_ub = np.array([[Fzub]*(FL4z.shape[0]*FL4z.shape[1])])

    #Right Contact Force x-axis
    #Right Foot Contact Point 1 x-axis
    FR1x = ca.SX.sym('FR1x',N_K)
    FR1x_lb = np.array([[Fxlb]*(FR1x.shape[0]*FR1x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR1x_ub = np.array([[Fxub]*(FR1x.shape[0]*FR1x.shape[1])])
    #Right Foot Contact Point 1 y-axis
    FR1y = ca.SX.sym('FR1y',N_K)
    FR1y_lb = np.array([[Fylb]*(FR1y.shape[0]*FR1y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR1y_ub = np.array([[Fyub]*(FR1y.shape[0]*FR1y.shape[1])])
    #Right Foot Contact Point 1 z-axis
    FR1z = ca.SX.sym('FR1z',N_K)
    FR1z_lb = np.array([[Fzlb]*(FR1z.shape[0]*FR1z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR1z_ub = np.array([[Fzub]*(FR1z.shape[0]*FR1z.shape[1])])
    #Right Foot Contact Point 2 x-axis
    FR2x = ca.SX.sym('FR2x',N_K)
    FR2x_lb = np.array([[Fxlb]*(FR2x.shape[0]*FR2x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR2x_ub = np.array([[Fxub]*(FR2x.shape[0]*FR2x.shape[1])])
    #Right Foot Contact Point 2 y-axis
    FR2y = ca.SX.sym('FR2y',N_K)
    FR2y_lb = np.array([[Fylb]*(FR2y.shape[0]*FR2y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR2y_ub = np.array([[Fyub]*(FR2y.shape[0]*FR2y.shape[1])])
    #Right Foot Contact Point 2 z-axis
    FR2z = ca.SX.sym('FR2z',N_K)
    FR2z_lb = np.array([[Fzlb]*(FR2z.shape[0]*FR2z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR2z_ub = np.array([[Fzub]*(FR2z.shape[0]*FR2z.shape[1])])
    #Right Foot Contact Point 3 x-axis
    FR3x = ca.SX.sym('FR3x',N_K)
    FR3x_lb = np.array([[Fxlb]*(FR3x.shape[0]*FR3x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR3x_ub = np.array([[Fxub]*(FR3x.shape[0]*FR3x.shape[1])])
    #Right Foot Contact Point 3 y-axis
    FR3y = ca.SX.sym('FR3y',N_K)
    FR3y_lb = np.array([[Fylb]*(FR3y.shape[0]*FR3y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR3y_ub = np.array([[Fyub]*(FR3y.shape[0]*FR3y.shape[1])])
    #Right Foot Contact Point 3 z-axis
    FR3z = ca.SX.sym('FR3z',N_K)
    FR3z_lb = np.array([[Fzlb]*(FR3z.shape[0]*FR3z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR3z_ub = np.array([[Fzub]*(FR3z.shape[0]*FR3z.shape[1])])
    #Right Foot Contact Point 4 x-axis
    FR4x = ca.SX.sym('FR4x',N_K)
    FR4x_lb = np.array([[Fxlb]*(FR4x.shape[0]*FR4x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR4x_ub = np.array([[Fxub]*(FR4x.shape[0]*FR4x.shape[1])])
    #Right Foot Contact Point 4 y-axis
    FR4y = ca.SX.sym('FR4y',N_K)
    FR4y_lb = np.array([[Fylb]*(FR4y.shape[0]*FR4y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR4y_ub = np.array([[Fyub]*(FR4y.shape[0]*FR4y.shape[1])])
    #Right Foot Contact Point 4 z-axis
    FR4z = ca.SX.sym('FR4z',N_K)
    FR4z_lb = np.array([[Fzlb]*(FR4z.shape[0]*FR4z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR4z_ub = np.array([[Fzub]*(FR4z.shape[0]*FR4z.shape[1])])

    #Initial Contact Location (need to connect to the first level)
    #   Plx
    px_init = ca.SX.sym('px_init')
    px_init_lb = np.array([-1])
    px_init_ub = np.array([30])

    #   py
    py_init = ca.SX.sym('py_init')
    py_init_lb = np.array([-1])
    py_init_ub = np.array([1])

    #   pz
    pz_init = ca.SX.sym('pz_init')
    pz_init_lb = np.array([-5])
    pz_init_ub = np.array([5])

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
        pxtemp = ca.SX.sym('px'+str(stepIdx)) #0 + 1
        px.append(pxtemp)
        px_lb.append(np.array([-1]))
        px_ub.append(np.array([30]))

        pytemp = ca.SX.sym('py'+str(stepIdx))
        py.append(pytemp)
        py_lb.append(np.array([-1]))
        py_ub.append(np.array([1]))

        #   Foot steps are all staying on the ground
        pztemp = ca.SX.sym('pz'+str(stepIdx))
        pz.append(pztemp)
        pz_lb.append(np.array([-5]))
        pz_ub.append(np.array([5]))

    #Switching Time Vector
    Ts = []
    Ts_lb = []
    Ts_ub = []
    for n_phase in range(Nphase):
        Tstemp = ca.SX.sym('Ts'+str(n_phase+1)) #0 + 1 + ....
        Ts.append(Tstemp)
        Ts_lb.append(np.array([0.05]))
        Ts_ub.append(np.array([2.0*(Nphase+1)]))

    #   Collect all Decision Variables
    DecisionVars = ca.vertcat(x,y,z,xdot,ydot,zdot,Lx,Ly,Lz,Ldotx,Ldoty,Ldotz,FL1x,FL1y,FL1z,FL2x,FL2y,FL2z,FL3x,FL3y,FL3z,FL4x,FL4y,FL4z,FR1x,FR1y,FR1z,FR2x,FR2y,FR2z,FR3x,FR3y,FR3z,FR4x,FR4y,FR4z,px_init,py_init,pz_init,*px,*py,*pz,*Ts)
    #print(DecisionVars)
    DecisionVarsShape = DecisionVars.shape

    #   Collect all lower bound and upper bound
    DecisionVars_lb = np.concatenate(((x_lb,y_lb,z_lb,xdot_lb,ydot_lb,zdot_lb,Lx_lb,Ly_lb,Lz_lb,Ldotx_lb,Ldoty_lb,Ldotz_lb,FL1x_lb,FL1y_lb,FL1z_lb,FL2x_lb,FL2y_lb,FL2z_lb,FL3x_lb,FL3y_lb,FL3z_lb,FL4x_lb,FL4y_lb,FL4z_lb,FR1x_lb,FR1y_lb,FR1z_lb,FR2x_lb,FR2y_lb,FR2z_lb,FR3x_lb,FR3y_lb,FR3z_lb,FR4x_lb,FR4y_lb,FR4z_lb,px_init_lb,py_init_lb,pz_init_lb,px_lb,py_lb,pz_lb,Ts_lb)),axis=None)
    DecisionVars_ub = np.concatenate(((x_ub,y_ub,z_ub,xdot_ub,ydot_ub,zdot_ub,Lx_ub,Ly_ub,Lz_ub,Ldotx_ub,Ldoty_ub,Ldotz_ub,FL1x_ub,FL1y_ub,FL1z_ub,FL2x_ub,FL2y_ub,FL2z_ub,FL3x_ub,FL3y_ub,FL3z_ub,FL4x_ub,FL4y_ub,FL4z_ub,FR1x_ub,FR1y_ub,FR1z_ub,FR2x_ub,FR2y_ub,FR2z_ub,FR3x_ub,FR3y_ub,FR3z_ub,FR4x_ub,FR4y_ub,FR4z_ub,px_init_ub,py_init_ub,pz_init_ub,px_ub,py_ub,pz_ub,Ts_ub)),axis=None)

    #Time Span Setup
    tau_upper_limit = 1
    tauStepLength = tau_upper_limit/(N_K-1) #Get the interval length, total number of knots - 1

    #-----------------------------------------------------------------------------------------------------------------------
    #Define Constrains and Running Cost
    g = []
    glb = []
    gub = []
    J = 0

    #Initial and Terminal Condition

    ##   Terminal CoM y-axis
    #g.append(y[-1]-y_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    ##   Terminal CoM z-axis
    #g.append(z[-1]-z_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    #if StaticStop == True:
    #    #   Terminal Zero CoM velocity x-axis
    #    g.append(xdot[-1])
    #    glb.append(np.array([0]))
    #    gub.append(np.array([0]))

    #    #   Terminal Zero CoM velocity y-axis
    #    g.append(ydot[-1])
    #    glb.append(np.array([0]))
    #    gub.append(np.array([0]))

    #    #   Terminal Zero CoM velocity z-axis
    #    g.append(zdot[-1])
    #    glb.append(np.array([0]))
    #    gub.append(np.array([0]))

    #   Terminal Angular Momentum x-axis
    #g.append(Lx[-1]-Lx_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    #   Terminal Angular Momentum y-axis
    #g.append(Ly[-1]-Ly_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    #   Terminal Angular Momentum z-axis
    #g.append(Lz[-1]-Lz_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    #   Terminal Angular Momentum Rate x-axis
    #g.append(Ldotx[-1]-Ldotx_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    #   Terminal Angular Momentum Rate y-axis
    #g.append(Ldoty[-1]-Ldoty_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    #   Terminal Angular Momentum Rate z-axis
    #g.append(Ldotz[-1]-Ldotz_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    #Loop over all Phases (Knots)
    for Nph in range(Nphase):
        #Decide Number of Knots
        if Nph == Nphase-1:  #The last Knot belongs to the Last Phase
            Nk_ThisPhase = Nk_Local+1
        else:
            Nk_ThisPhase = Nk_Local       

        #Decide Time Vector
        if Nph == 0: #first phase
            h = tauStepLength*Nphase*(Ts[Nph]-0)
        else: #other phases
            h = tauStepLength*Nphase*(Ts[Nph]-Ts[Nph-1]) 

        for Local_k_Count in range(Nk_ThisPhase):
            #Get knot number across the entire time line
            k = Nph*Nk_Local + Local_k_Count
            #print(k)

            #------------------------------------------
            #Build useful vectors
            #   Forces
            FL1_k = ca.vertcat(FL1x[k],FL1y[k],FL1z[k])
            FL2_k = ca.vertcat(FL2x[k],FL2y[k],FL2z[k])
            FL3_k = ca.vertcat(FL3x[k],FL3y[k],FL3z[k])
            FL4_k = ca.vertcat(FL4x[k],FL4y[k],FL4z[k])

            FR1_k = ca.vertcat(FR1x[k],FR1y[k],FR1z[k])
            FR2_k = ca.vertcat(FR2x[k],FR2y[k],FR2z[k])
            FR3_k = ca.vertcat(FR3x[k],FR3y[k],FR3z[k])
            FR4_k = ca.vertcat(FR4x[k],FR4y[k],FR4z[k])
            #   CoM
            CoM_k = ca.vertcat(x[k],y[k],z[k])
            #   Angular Momentum
            if k<N_K-1: #N_K-1 the enumeration of the last knot, k<N_K-1 the one before the last knot
                Ldot_current = ca.vertcat(Ldotx[k],Ldoty[k],Ldotz[k])
                Ldot_next = ca.vertcat(Ldotx[k+1],Ldoty[k+1],Ldotz[k+1])
            #-------------------------------------------

            #-------------------------------------------
            #Phase dependent Constraints (CoM Kinematics and Angular Dynamics)
            
            #Get Step Counter
            StepCnt = Nph//3

            #NOTE: The first phase (Initial Double) --- Needs special care
            if Nph == 0 and GaitPattern[Nph]=='InitialDouble':

                #initial support foot (the landing foot from the first phase)
                p_init = ca.vertcat(px_init,py_init,pz_init)
                p_init_TangentX = SurfTangentsX[0:3]
                p_init_TangentY = SurfTangentsY[0:3]
                p_init_Norm = SurfNorms[0:3]

                #Case 1
                #If First Level Swing the Left, the the 0 phase (InitDouble) has p_init as the left support, PR_init as the right support
                #Kinematics Constraint
                #CoM in Left (p_init)
                g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_init)
                #CoM in Right (PR_init)
                g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = PR_init)
                #Angular Dynamics
                if k<N_K-1:
                    g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, Ldot_current = Ldot_current, h = h, PL = p_init, PL_TangentX = p_init_TangentX, PL_TangentY = p_init_TangentY, PR = PR_init, PR_TangentX = PR_init_TangentX, PR_TangentY = PR_init_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                    #g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, Ldot_next = L_next, Ldot_current = L_current, h = h, PL = p_init, PL_TangentX = p_init_TangentX, PL_TangentY = p_init_TangentY, PR = PR_init, PR_TangentX = PR_init_TangentX, PR_TangentY = PR_init_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)

                #Unilateral Constraint
                #If the first level swing the Left foot first, then the Left foot is the landing foot (p_init), Left foot obey the unilateral constraint on p_init
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainNorm = p_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainNorm = p_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainNorm = p_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainNorm = p_init_Norm)
                #then the Right foot is obey the unilateral constraint on the PR_init
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainNorm = PR_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainNorm = PR_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainNorm = PR_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainNorm = PR_init_Norm)
                #Friction Cone Constraint
                #If the first level swing the Left foot first, then the Left foot is the landing foot (p_init), Left foot obey the friction cone constraint on p_init
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu)
                #then the right foot obeys the friction cone constraints on the PR_init
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu)
                
                #Case 2
                #If First Level Swing the Right, the the 0 phase (InitDouble) has p_init as the Right support, PL_init as the Left support
                #Kinematics Constraint
                #CoM in the Left foot
                g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = PL_init)
                #CoM in the Right foot
                g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_init)
                #Agnular Dynamics
                if k<N_K-1:
                    g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, Ldot_current = Ldot_current, h = h, PL = PL_init, PL_TangentX = PL_init_TangentX, PL_TangentY = PL_init_TangentY, PR = p_init, PR_TangentX = p_init_TangentX, PR_TangentY = p_init_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                    #g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, Ldot_next = L_next, Ldot_current = L_current, h = h, PL = PL_init, PL_TangentX = PL_init_TangentX, PL_TangentY = PL_init_TangentY, PR = p_init, PR_TangentX = p_init_TangentX, PR_TangentY = p_init_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                #Unilateral Constraint
                #If the first level swings the Right foot first, then the right foot is the landing foot (p_init), Right foot obeys the unilateral constraint on p_init
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainNorm = p_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainNorm = p_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainNorm = p_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainNorm = p_init_Norm)
                #then the Left foot obeys the unilateral constrint on the PL_init
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainNorm = PL_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainNorm = PL_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainNorm = PL_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainNorm = PL_init_Norm)                
                #Friction Cone Constraint
                #if the first level swing the right foot first, then the Right foot is the landing foot (p_init), Right foot obey the friction cone constraints on p_init
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu)
                #then the left foot obeys the friction cone constraint of PL_init
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu)
            
            #All other phases
            else:
                if GaitPattern[Nph]=='InitialDouble':
                    #Get contact location
                    if StepCnt == 1: #Step 1 needs special care (NOTE: Step Count Start from 0)
                        p_previous = ca.vertcat(px_init,py_init,pz_init)
                        p_previous_TangentX = SurfTangentsX[0:3]
                        p_previous_TangentY = SurfTangentsY[0:3]
                        p_previous_Norm = SurfNorms[0:3]

                        #In second level, Surfaces index is Step Vector Index (fpr px, py, pz, here is StepCnt-1) + 1
                        p_current = ca.vertcat(px[StepCnt-1],py[StepCnt-1],pz[StepCnt-1])
                        p_current_TangentX = SurfTangentsX[StepCnt*3:StepCnt*3+3]
                        p_current_TangentY = SurfTangentsY[StepCnt*3:StepCnt*3+3]
                        p_current_Norm = SurfNorms[StepCnt*3:StepCnt*3+3]

                    else: #Like Step 2, 3, 4 .....
                        p_previous = ca.vertcat(px[StepCnt-2],py[StepCnt-2],pz[StepCnt-2])
                        p_previous_TangentX = SurfTangentsX[(StepCnt-1)*3:(StepCnt-1)*3+3]
                        p_previous_TangentY = SurfTangentsY[(StepCnt-1)*3:(StepCnt-1)*3+3]
                        p_previous_Norm = SurfNorms[(StepCnt-1)*3:(StepCnt-1)*3+3]

                        p_current = ca.vertcat(px[StepCnt-1],py[StepCnt-1],pz[StepCnt-1])
                        p_current_TangentX = SurfTangentsX[StepCnt*3:StepCnt*3+3]
                        p_current_TangentY = SurfTangentsY[StepCnt*3:StepCnt*3+3]
                        p_current_Norm = SurfNorms[StepCnt*3:StepCnt*3+3]

                    #Give Constraint according to even and odd steps
                    if StepCnt%2 == 0: #Even Numbers of Footsteps
                        #Case 1
                        #If the first level swing the Left, then the Even Number of Steps in the Intial Double support phase have p_current as Left Support (Landed), p_previous as Right Support (Stationary)
                        #CoM in the Left (p_current)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_current)
                        #CoM in the Right (p_previous)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_previous)
                        #Angular Dynamics
                        if k<N_K-1:
                            g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, Ldot_current = Ldot_current, h = h, PL = p_current, PL_TangentX = p_current_TangentX, PL_TangentY = p_current_TangentY, PR = p_previous, PR_TangentX = p_previous_TangentX, PR_TangentY = p_previous_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                            #g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, Ldot_next = L_next, Ldot_current = L_current, h = h, PL = p_current, PL_TangentX = p_current_TangentX, PL_TangentY = p_current_TangentY, PR = p_previous, PR_TangentX = p_previous_TangentX, PR_TangentY = p_previous_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                        #Unilateral Constraint
                        #If the first level swing the Left foot first, then the Left foot is the landing foot (p_current), Left foot obey the unilateral constraint on p_current
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainNorm = p_current_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainNorm = p_current_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainNorm = p_current_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainNorm = p_current_Norm)
                        #then the Right foot is obey the unilateral constraint on the Stationary foot p_previous
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainNorm = p_previous_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainNorm = p_previous_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainNorm = p_previous_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainNorm = p_previous_Norm)
                        #Friction Cone Constraint
                        #If the first level swing the Left foot first, then the Left foot is the landing foot (p_current), Left foot obey the friction cone constraint on p_current
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        #then the right foot obeys the friction cone constraints on the on the Stationary foot p_previous
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        
                        #Case 2
                        #If the first level swing the Right, then the Even Number of Steps in the Intial Double support phase have p_current as Right Support (Landed), 
                        #p_previous as Left Support (Stationary)
                        #CoM in the Left (p_previous)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_previous)
                        #CoM in the Right (p_current)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_current)
                        #Angular Dynamics
                        if k<N_K-1:
                            g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, Ldot_current = Ldot_current, h = h, PL = p_previous, PL_TangentX = p_previous_TangentX, PL_TangentY = p_previous_TangentY, PR = p_current, PR_TangentX = p_current_TangentX, PR_TangentY = p_current_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                            #g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, Ldot_next = L_next, Ldot_current = L_current, h = h, PL = p_previous, PL_TangentX = p_previous_TangentX, PL_TangentY = p_previous_TangentY, PR = p_current, PR_TangentX = p_current_TangentX, PR_TangentY = p_current_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                        #Unilateral Constraint
                        #Left foot obey the unilateral constraint on p_previous
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainNorm = p_previous_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainNorm = p_previous_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainNorm = p_previous_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainNorm = p_previous_Norm)
                        #Right foot is obey the unilateral constraint on p_current
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainNorm = p_current_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainNorm = p_current_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainNorm = p_current_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainNorm = p_current_Norm)
                        #Friction Cone Constraint
                        #Left foot obey the friction cone constraint on p_previous
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        #then the right foot obeys the friction cone constraints on the on the Stationary foot p_previous
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)

                    elif StepCnt%2 == 1: #Odd Number of Steps
                        #Case 1
                        #If the first level swing the Left, then the Odd Number of Steps in the Intial Double support phase have p_current as Right Support (Landed), p_previous as Left Support (Stationary)
                        #CoM in the Left (p_previous)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_previous)
                        #CoM in the Right (p_current)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_current)
                        #Angular Dynamics
                        if k<N_K-1:
                            g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, Ldot_current = Ldot_current, h = h, PL = p_previous, PL_TangentX = p_previous_TangentX, PL_TangentY = p_previous_TangentY, PR = p_current, PR_TangentX = p_current_TangentX, PR_TangentY = p_current_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                            #g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, Ldot_next = L_next, Ldot_current = L_current, h = h, PL = p_previous, PL_TangentX = p_previous_TangentX, PL_TangentY = p_previous_TangentY, PR = p_current, PR_TangentX = p_current_TangentX, PR_TangentY = p_current_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                        #Unilateral Constraint
                        #Left foot obey the unilateral constraint on p_previous
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainNorm = p_previous_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainNorm = p_previous_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainNorm = p_previous_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainNorm = p_previous_Norm)
                        #Right foot is obey the unilateral constraint on p_current
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainNorm = p_current_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainNorm = p_current_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainNorm = p_current_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainNorm = p_current_Norm)
                        #Friction Cone Constraint
                        #Left foot obey the friction cone constraint on p_previous
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        #right foot obeys the friction cone constraints on p_current
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        
                        #Case 2
                        #If the first level swing the Right, then the Odd Number of Steps in the Intial Double support phase have p_current as Left Support (Landed), 
                        #p_previous as Right Support (Stationary)
                        #CoM in the Left (p_current)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_current)
                        #CoM in the Right (p_previous)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_previous)
                        #Angular Dynamics
                        if k<N_K-1:
                            g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, Ldot_current = Ldot_current, h = h, PL = p_current, PL_TangentX = p_current_TangentX, PL_TangentY = p_current_TangentY, PR = p_previous, PR_TangentX = p_previous_TangentX, PR_TangentY = p_previous_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                            #g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, Ldot_next = L_next, Ldot_current = L_current, h = h, PL = p_current, PL_TangentX = p_current_TangentX, PL_TangentY = p_current_TangentY, PR = p_previous, PR_TangentX = p_previous_TangentX, PR_TangentY = p_previous_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                        #Unilateral Constraint
                        #Left foot obey the unilateral constraint on p_current
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainNorm = p_current_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainNorm = p_current_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainNorm = p_current_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainNorm = p_current_Norm)
                        #Right foot is obey the unilateral constraint on p_previous
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainNorm = p_previous_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainNorm = p_previous_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainNorm = p_previous_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainNorm = p_previous_Norm)
                        #Friction Cone Constraint
                        #Left foot obey the friction cone constraint on p_current
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        #then the right foot obeys the friction cone constraints on the p_previous
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)

                elif GaitPattern[Nph]== 'Swing':
                    #Get contact location
                    if StepCnt == 0:#Special Case for the First Step (NOTE:Step 0)
                        p_stance = ca.vertcat(px_init,py_init,pz_init)
                        p_stance_TangentX = SurfTangentsX[0:3]
                        p_stance_TangentY = SurfTangentsY[0:3]
                        p_stance_Norm = SurfNorms[0:3]

                    else: #For other Steps, indexed as 1,2,3,4
                        p_stance = ca.vertcat(px[StepCnt-1],py[StepCnt-1],pz[StepCnt-1])
                        p_stance_TangentX = SurfTangentsX[StepCnt*3:StepCnt*3+3]
                        p_stance_TangentY = SurfTangentsY[StepCnt*3:StepCnt*3+3]
                        p_stance_Norm = SurfNorms[StepCnt*3:StepCnt*3+3]

                    #Give Constraint according to even and odd steps
                    if StepCnt%2 == 0: #Even Number of Steps
                        #Case 1
                        #If First Level Swing the Left, then the second level Even Number Phases (the first Phase) Swing the Right
                        #Left foot is the stance foot
                        #Right foot is floating
                        #Kinematics Constraint
                        #CoM in the Left (p_stance)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_stance)
                        #Angular Dynamics (Left Stance)
                        if k<N_K-1:
                            g, glb, gub = Angular_Momentum_Rate_Swing(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, Ldot_current = Ldot_current, h = h, P = p_stance, P_TangentX = p_stance_TangentX, P_TangentY = p_stance_TangentY, CoM_k = CoM_k, F1_k = FL1_k, F2_k = FL2_k, F3_k = FL3_k, F4_k = FL4_k)
                            #g, glb, gub = Angular_Momentum_Rate_Swing(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, Ldot_next = L_next, Ldot_current = L_current, h = h, P = p_stance, P_TangentX = p_stance_TangentX, P_TangentY = p_stance_TangentY, CoM_k = CoM_k, F1_k = FL1_k, F2_k = FL2_k, F3_k = FL3_k, F4_k = FL4_k)
                        #Zero Forces (Right Foot)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k)
                        #Unilateral Constraints on Left Foot p_stance
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainNorm = p_stance_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainNorm = p_stance_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainNorm = p_stance_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainNorm = p_stance_Norm)
                        #Friction Cone Constraint on Left Foot p_stance
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)

                        #Case 2
                        #If First Level Swing the Right, then the second level Even Number Phases (the first Phase) Swing the Left
                        #Right foot is the stance foot
                        #Left foot is floating
                        #Kinematics Constraint
                        #CoM in the Right (p_stance)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_stance)
                        #Angular Dynamics(Right Stance)
                        if k<N_K-1:
                            g, glb, gub = Angular_Momentum_Rate_Swing(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, Ldot_current = Ldot_current, h = h, P = p_stance, P_TangentX = p_stance_TangentX, P_TangentY = p_stance_TangentY, CoM_k = CoM_k, F1_k = FR1_k, F2_k = FR2_k, F3_k = FR3_k, F4_k = FR4_k)
                            #g, glb, gub = Angular_Momentum_Rate_Swing(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, Ldot_next = L_next, Ldot_current = L_current, h = h, P = p_stance, P_TangentX = p_stance_TangentX, P_TangentY = p_stance_TangentY, CoM_k = CoM_k, F1_k = FR1_k, F2_k = FR2_k, F3_k = FR3_k, F4_k = FR4_k)
                        #Zero Forces (Left Foot)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k)
                        #Unilateral Constraints on Right Foot p_stance
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainNorm = p_stance_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainNorm = p_stance_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainNorm = p_stance_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainNorm = p_stance_Norm)
                        #Friction Cone Constraint on Right Foot p_stance
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)

                    elif StepCnt%2 == 1: #Odd Number of Steps
                        #Case 1
                        #If First Level Swing the Left, then the second level Odd Number Steps Swing the Left
                        #Right foot is the stance foot
                        #Left foot is floating
                        #Kinematics Constraint
                        #CoM in the Right (p_stance)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_stance)
                        #Angular Dynamics (Right Stance)
                        if k<N_K-1:
                            g, glb, gub = Angular_Momentum_Rate_Swing(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, Ldot_current = Ldot_current, h = h, P = p_stance, P_TangentX = p_stance_TangentX, P_TangentY = p_stance_TangentY, CoM_k = CoM_k, F1_k = FR1_k, F2_k = FR2_k, F3_k = FR3_k, F4_k = FR4_k)
                            #g, glb, gub = Angular_Momentum_Rate_Swing(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, Ldot_next = L_next, Ldot_current = L_current, h = h, P = p_stance, P_TangentX = p_stance_TangentX, P_TangentY = p_stance_TangentY, CoM_k = CoM_k, F1_k = FR1_k, F2_k = FR2_k, F3_k = FR3_k, F4_k = FR4_k)
                        #Zero Forces (Left Foot)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k)
                        #Unilateral Constraints on Right Foot p_stance
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainNorm = p_stance_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainNorm = p_stance_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainNorm = p_stance_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainNorm = p_stance_Norm)
                        #Friction Cone Constraint on Right Foot p_stance
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)

                        #Case 2
                        #If First Level Swing the Right, then the second level Odd Number Steps Swing the Right
                        #Left foot is the stance foot
                        #Right foot is floating
                        #Kinematics Constraint
                        #CoM in the Left (p_stance)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_stance)
                        #Angular Dynamics (Left Stance)
                        if k<N_K-1:
                            g, glb, gub = Angular_Momentum_Rate_Swing(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, Ldot_current = Ldot_current, h = h, P = p_stance, P_TangentX = p_stance_TangentX, P_TangentY = p_stance_TangentY, CoM_k = CoM_k, F1_k = FL1_k, F2_k = FL2_k, F3_k = FL3_k, F4_k = FL4_k)
                            #g, glb, gub = Angular_Momentum_Rate_Swing(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, Ldot_next = L_next, Ldot_current = L_current, h = h, P = p_stance, P_TangentX = p_stance_TangentX, P_TangentY = p_stance_TangentY, CoM_k = CoM_k, F1_k = FL1_k, F2_k = FL2_k, F3_k = FL3_k, F4_k = FL4_k)
                        #Zero Forces (Right Foot)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k)
                        #Unilateral Constraints on Left Foot p_stance
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainNorm = p_stance_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainNorm = p_stance_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainNorm = p_stance_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainNorm = p_stance_Norm)
                        #Friction Cone Constraint on Left Foot p_stance
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)

                elif GaitPattern[Nph]=='DoubleSupport':
                    #Get contact location
                    if StepCnt == 0: #Special Case for the First Step (NOTE: Step 0)
                        p_stationary = ca.vertcat(px_init,py_init,pz_init)
                        p_stationary_TangentX = SurfTangentsX[0:3]
                        p_stationary_TangentY = SurfTangentsY[0:3]
                        p_stationary_Norm = SurfNorms[0:3]

                        p_land = ca.vertcat(px[StepCnt],py[StepCnt],pz[StepCnt])
                        p_land_TangentX = SurfTangentsX[(StepCnt+1)*3:(StepCnt+1)*3+3]
                        p_land_TangentY = SurfTangentsY[(StepCnt+1)*3:(StepCnt+1)*3+3]
                        p_land_Norm = SurfNorms[(StepCnt+1)*3:(StepCnt+1)*3+3]
                
                    else: #For other steps, indexed as 1,2,3,4
                        p_stationary = ca.vertcat(px[StepCnt-1],py[StepCnt-1],pz[StepCnt-1])
                        p_stationary_TangentX = SurfTangentsX[StepCnt*3:StepCnt*3+3]
                        p_stationary_TangentY = SurfTangentsY[StepCnt*3:StepCnt*3+3]
                        p_stationary_Norm = SurfNorms[StepCnt*3:StepCnt*3+3]

                        p_land = ca.vertcat(px[StepCnt],py[StepCnt],pz[StepCnt])
                        p_land_TangentX = SurfTangentsX[(StepCnt+1)*3:(StepCnt+1)*3+3]
                        p_land_TangentY = SurfTangentsY[(StepCnt+1)*3:(StepCnt+1)*3+3]
                        p_land_Norm = SurfNorms[(StepCnt+1)*3:(StepCnt+1)*3+3]

                    #Give Constraint according to even and odd steps
                    if StepCnt%2 == 0: #Even Number of Steps
                        #Case 1
                        #If First Level Swing the Left, then the second level Even Steps Swing the Right
                        #In Double Support Phase
                        #Left Foot is the Stationary
                        #Right Foot is the Land
                        #Kinemactics Constraint
                        #CoM in the Left (p_stationary)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_stationary)
                        #CoM in the Right (p_land)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_land)
                        #Angular Dynamics
                        if k<N_K-1:
                            g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, Ldot_current = Ldot_current, h = h, PL = p_stationary, PL_TangentX = p_stationary_TangentX, PL_TangentY = p_stationary_TangentY, PR = p_land, PR_TangentX = p_land_TangentX, PR_TangentY = p_land_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                            #g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, Ldot_next = L_next, Ldot_current = L_current, h = h, PL = p_stationary, PL_TangentX = p_stationary_TangentX, PL_TangentY = p_stationary_TangentY, PR = p_land, PR_TangentX = p_land_TangentX, PR_TangentY = p_land_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)

                        #Unilateral Constraint
                        #Left foot obey the unilateral constraint on p_stationary
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainNorm = p_stationary_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainNorm = p_stationary_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainNorm = p_stationary_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainNorm = p_stationary_Norm)
                        #then the Right foot is obey the unilateral constraint on the Stationary foot p_land
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainNorm = p_land_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainNorm = p_land_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainNorm = p_land_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainNorm = p_land_Norm)
                        #Friction Cone Constraint
                        #Left foot obey the friction cone constraint on p_stationary
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        #then the right foot obeys the friction cone constraints on the on p_land
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        
                        #Case 2
                        #If First Level Swing the Right, then the second level Even Steps Swing the Left
                        #In Double Support Phase
                        #Right Foot is the Stationary
                        #Left Foot is the Land
                        #Kinemactics Constraint
                        #CoM in the Left (p_land)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_land)
                        #CoM in the Right (p_stationary)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_stationary)
                        #Angular Dynamics
                        if k<N_K-1:
                            g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, Ldot_current = Ldot_current, h = h, PL = p_land, PL_TangentX = p_land_TangentX, PL_TangentY = p_land_TangentY, PR = p_stationary, PR_TangentX = p_stationary_TangentX, PR_TangentY = p_stationary_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                            #g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, Ldot_next = L_next, Ldot_current = L_current, h = h, PL = p_land, PL_TangentX = p_land_TangentX, PL_TangentY = p_land_TangentY, PR = p_stationary, PR_TangentX = p_stationary_TangentX, PR_TangentY = p_stationary_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                        #Unilateral Constraint
                        #Left foot obey the unilateral constraint on p_land
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainNorm = p_land_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainNorm = p_land_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainNorm = p_land_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainNorm = p_land_Norm)
                        #then the Right foot is obey the unilateral constraint on the Stationary foot p_stationary
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainNorm = p_stationary_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainNorm = p_stationary_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainNorm = p_stationary_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainNorm = p_stationary_Norm)
                        #Friction Cone Constraint
                        #Left foot obey the friction cone constraint on p_land
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        #then the right foot obeys the friction cone constraints on the on p_stationary
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        
                    elif StepCnt%2 == 1:#Odd Number of Steps
                        #Case 1
                        #If First Level Swing the Left, then the second level Odd Steps Swing the Left
                        #In Double Support Phase
                        #Right Foot is the Stationary
                        #Left Foot is the Land
                        #Kinemactics Constraint
                        #CoM in the Left (p_land)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_land)
                        #CoM in the Right (p_stationary)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_stationary)
                        #Angular Dynamics
                        if k<N_K-1:
                            g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, Ldot_current = Ldot_current, h = h, PL = p_land, PL_TangentX = p_land_TangentX, PL_TangentY = p_land_TangentY, PR = p_stationary, PR_TangentX = p_stationary_TangentX, PR_TangentY = p_stationary_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                            #g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, Ldot_next = L_next, Ldot_current = L_current, h = h, PL = p_land, PL_TangentX = p_land_TangentX, PL_TangentY = p_land_TangentY, PR = p_stationary, PR_TangentX = p_stationary_TangentX, PR_TangentY = p_stationary_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                        #Unilateral Constraint
                        #Left foot obey the unilateral constraint on p_land
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainNorm = p_land_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainNorm = p_land_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainNorm = p_land_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainNorm = p_land_Norm)
                        #then the Right foot is obey the unilateral constraint on the Stationary foot p_stationary
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainNorm = p_stationary_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainNorm = p_stationary_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainNorm = p_stationary_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainNorm = p_stationary_Norm)
                        #Friction Cone Constraint
                        #Left foot obey the friction cone constraint on p_land
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        #then the right foot obeys the friction cone constraints on the on p_stationary
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        
                        #Case 2
                        #If First Level Swing the Right, then the second level Odd Steps Swing the Right
                        #In Double Support Phase
                        #Left Foot is the Stationary
                        #Right Foot is the Land
                        #Kinematics Constraint
                        #CoM in the Left (p_stationary)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_stationary)
                        #CoM in the Right (p_land)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_land)
                        #Angular Dynamics
                        if k<N_K-1:
                            g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, Ldot_current = Ldot_current, h = h, PL = p_stationary, PL_TangentX = p_stationary_TangentX, PL_TangentY = p_stationary_TangentY, PR = p_land, PR_TangentX = p_land_TangentX, PR_TangentY = p_land_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                            #g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, Ldot_next = L_next, Ldot_current = L_current, h = h, PL = p_stationary, PL_TangentX = p_stationary_TangentX, PL_TangentY = p_stationary_TangentY, PR = p_land, PR_TangentX = p_land_TangentX, PR_TangentY = p_land_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                        #Unilateral Constraint
                        #Left foot obey the unilateral constraint on p_stationary
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainNorm = p_stationary_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainNorm = p_stationary_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainNorm = p_stationary_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainNorm = p_stationary_Norm)
                        #then the Right foot is obey the unilateral constraint on the Stationary foot p_land
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainNorm = p_land_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainNorm = p_land_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainNorm = p_land_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainNorm = p_land_Norm)
                        #Friction Cone Constraint
                        #Left foot obey the friction cone constraint on p_stationary
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        #then the right foot obeys the friction cone constraints on the on p_land
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        
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

                #First-order Angular Momentum Dynamics x-axis
                g.append(Lx[k+1] - Lx[k] - h*Ldotx[k])
                glb.append(np.array([0]))
                gub.append(np.array([0]))

                #First-order Angular Momentum Dynamics y-axis
                g.append(Ly[k+1] - Ly[k] - h*Ldoty[k])
                glb.append(np.array([0]))
                gub.append(np.array([0]))

                #First-order Angular Momentum Dynamics z-axis
                g.append(Lz[k+1] - Lz[k] - h*Ldotz[k])
                glb.append(np.array([0]))
                gub.append(np.array([0]))

                #Second-order Dynamics x-axis
                g.append(xdot[k+1] - xdot[k] - h*(FL1x[k]/m+FL2x[k]/m+FL3x[k]/m+FL4x[k]/m+FR1x[k]/m+FR2x[k]/m+FR3x[k]/m+FR4x[k]/m))
                glb.append(np.array([0]))
                gub.append(np.array([0]))

                #Second-order Dynamics y-axis
                g.append(ydot[k+1] - ydot[k] - h*(FL1y[k]/m+FL2y[k]/m+FL3y[k]/m+FL4y[k]/m+FR1y[k]/m+FR2y[k]/m+FR3y[k]/m+FR4y[k]/m))
                glb.append(np.array([0]))
                gub.append(np.array([0]))

                #Second-order Dynamics z-axis
                g.append(zdot[k+1] - zdot[k] - h*(FL1z[k]/m+FL2z[k]/m+FL3z[k]/m+FL4z[k]/m+FR1z[k]/m+FR2z[k]/m+FR3z[k]/m+FR4z[k]/m - G))
                glb.append(np.array([0]))
                gub.append(np.array([0]))
            
            #Add Cost Terms
            if k < N_K - 1:
                #with angular momentum
                #J = J + h*Lx[k]**2 + h*Ly[k]**2 + h*Lz[k]**2 + h*(FL1x[k]/m+FL2x[k]/m+FL3x[k]/m+FL4x[k]/m+FR1x[k]/m+FR2x[k]/m+FR3x[k]/m+FR4x[k]/m)**2 + h*(FL1y[k]/m+FL2y[k]/m+FL3y[k]/m+FL4y[k]/m+FR1y[k]/m+FR2y[k]/m+FR3y[k]/m+FR4y[k]/m)**2 + h*(FL1z[k]/m+FL2z[k]/m+FL3z[k]/m+FL4z[k]/m+FR1z[k]/m+FR2z[k]/m+FR3z[k]/m+FR4z[k]/m - G)**2
                #with angular momentum rate
                #J = J + h*Ldotx[k]**2 + h*Ldoty[k]**2 + h*Ldotz[k]**2 + h*(FL1x[k]/m+FL2x[k]/m+FL3x[k]/m+FL4x[k]/m+FR1x[k]/m+FR2x[k]/m+FR3x[k]/m+FR4x[k]/m)**2 + h*(FL1y[k]/m+FL2y[k]/m+FL3y[k]/m+FL4y[k]/m+FR1y[k]/m+FR2y[k]/m+FR3y[k]/m+FR4y[k]/m)**2 + h*(FL1z[k]/m+FL2z[k]/m+FL3z[k]/m+FL4z[k]/m+FR1z[k]/m+FR2z[k]/m+FR3z[k]/m+FR4z[k]/m - G)**2
                #No Angular momentum
                #J = J + h*(FL1x[k]/m+FL2x[k]/m+FL3x[k]/m+FL4x[k]/m+FR1x[k]/m+FR2x[k]/m+FR3x[k]/m+FR4x[k]/m)**2 + h*(FL1y[k]/m+FL2y[k]/m+FL3y[k]/m+FL4y[k]/m+FR1y[k]/m+FR2y[k]/m+FR3y[k]/m+FR4y[k]/m)**2 + h*(FL1z[k]/m+FL2z[k]/m+FL3z[k]/m+FL4z[k]/m+FR1z[k]/m+FR2z[k]/m+FR3z[k]/m+FR4z[k]/m - G)**2
                #With Angular momentum and angular momentum together
                J = J + h*Lx[k]**2 + h*Ly[k]**2 + h*Lz[k]**2 + h*Ldotx[k]**2 + h*Ldoty[k]**2 + h*Ldotz[k]**2 + h*(FL1x[k]/m+FL2x[k]/m+FL3x[k]/m+FL4x[k]/m+FR1x[k]/m+FR2x[k]/m+FR3x[k]/m+FR4x[k]/m)**2 + h*(FL1y[k]/m+FL2y[k]/m+FL3y[k]/m+FL4y[k]/m+FR1y[k]/m+FR2y[k]/m+FR3y[k]/m+FR4y[k]/m)**2 + h*(FL1z[k]/m+FL2z[k]/m+FL3z[k]/m+FL4z[k]/m+FR1z[k]/m+FR2z[k]/m+FR3z[k]/m+FR4z[k]/m - G)**2

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
            #Also, the stationary foot Left should stay in the polytope of the landed swing foot - RIGHT
            #NOTE: current - next now
            g.append(ca.if_else(ParaLeftSwingFlag,Q_lf_in_rf@(P_k_current-P_k_next)-ca.DM(q_lf_in_rf),np.full((len(q_lf_in_rf),),-1)))
            glb.append(np.full((len(q_lf_in_rf),),-np.inf))
            gub.append(np.full((len(q_lf_in_rf),),0))

            #If First Levvel Swing the Right, then the second level Even Number Phases (the first phase) Swing the Left -> Right -> Left
            #Right foot in contact for p_current, left foot is going to land at p_next
            #Relative Swing Foot Location (lf in rf)
            g.append(ca.if_else(ParaRightSwingFlag,Q_lf_in_rf@(P_k_next-P_k_current)-ca.DM(q_lf_in_rf),np.full((len(q_lf_in_rf),),-1)))
            glb.append(np.full((len(q_lf_in_rf),),-np.inf))
            gub.append(np.full((len(q_lf_in_rf),),0))
            #Also, the stationary foot Rigth should stay in the polytope of the landed swing foot - Left
            #NOTE: current - next now
            g.append(ca.if_else(ParaRightSwingFlag,Q_rf_in_lf@(P_k_current-P_k_next)-ca.DM(q_rf_in_lf),np.full((len(q_rf_in_lf),),-1)))
            glb.append(np.full((len(q_rf_in_lf),),-np.inf))
            gub.append(np.full((len(q_rf_in_lf),),0))

        elif step_cnt%2 == 1: #odd number steps
            #If the first level swings the Left, then the second level for Odd Number Phases (the second phase) swings left -> Right -> Left
            #Right foot in contact for p_current, left foot is going to land at p_next
            #Relative Swing Foot Location (lf in rf)
            g.append(ca.if_else(ParaLeftSwingFlag,Q_lf_in_rf@(P_k_next-P_k_current)-ca.DM(q_lf_in_rf),np.full((len(q_lf_in_rf),),-1)))
            glb.append(np.full((len(q_lf_in_rf),),-np.inf))
            gub.append(np.full((len(q_lf_in_rf),),0))
            #Also, the stationary foot Rigth should stay in the polytope of the landed swing foot - Left
            #NOTE: current - next now
            g.append(ca.if_else(ParaLeftSwingFlag,Q_rf_in_lf@(P_k_current-P_k_next)-ca.DM(q_rf_in_lf),np.full((len(q_rf_in_lf),),-1)))
            glb.append(np.full((len(q_rf_in_lf),),-np.inf))
            gub.append(np.full((len(q_rf_in_lf),),0))

            #If the first level swings the Right, then the second level for Odd Number Phases (the second phase) swings Right -> Left -> Right
            #Left foot in contact for p_current, right foot is going to land as p_next
            #Relative Swing Foot Location (rf in lf)
            g.append(ca.if_else(ParaRightSwingFlag,Q_rf_in_lf@(P_k_next-P_k_current)-ca.DM(q_rf_in_lf),np.full((len(q_rf_in_lf),),-1)))
            glb.append(np.full((len(q_rf_in_lf),),-np.inf))
            gub.append(np.full((len(q_rf_in_lf),),0))
            #Also, the stationary foot LEFT should stay in the polytope of the landed swing foot - RIGHT
            #NOTE: current - next now
            g.append(ca.if_else(ParaRightSwingFlag,Q_lf_in_rf@(P_k_current-P_k_next)-ca.DM(q_lf_in_rf),np.full((len(q_lf_in_rf),),-1)))
            glb.append(np.full((len(q_lf_in_rf),),-np.inf))
            gub.append(np.full((len(q_lf_in_rf),),0))

    #Foot Step Constraint
    #FootStep Constraint
    #P3----------------P1
    #|                  |
    #|                  |
    #|                  |
    #P4----------------P2
    for PatchNum in range(Nsteps):
        #Get Footstep Vector
        P_vector = ca.vertcat(px[PatchNum],py[PatchNum],pz[PatchNum])

        #Get Half Space Representation 
        #NOTE: In the second level, the terrain patch start from the second patch, indexed as 1
        SurfParaTemp = SurfParas[20+PatchNum*20:19+(PatchNum+1)*20+1]
        #print(SurfParaTemp)
        SurfK = SurfParaTemp[0:11+1]
        SurfK = ca.reshape(SurfK,3,4)
        SurfK = SurfK.T #NOTE: This process is due to casadi naming convention to have first row to be x1,x2,x3
        SurfE = SurfParaTemp[11+1:11+3+1]
        Surfk = SurfParaTemp[14+1:14+4+1]
        Surfe = SurfParaTemp[-1]

        #Terrain Tangent and Norms
        P_vector_TangentX = SurfTangentsX[(PatchNum+1)*3:(PatchNum+1)*3+3]
        P_vector_TangentY = SurfTangentsY[(PatchNum+1)*3:(PatchNum+1)*3+3]

        #Contact Point 1
        #Inequality
        g.append(SurfK@(P_vector + 0.11*P_vector_TangentX + 0.06*P_vector_TangentY) - Surfk)
        glb.append(np.full((4,),-np.inf))
        gub.append(np.full((4,),0))
        #Equality
        g.append(SurfE.T@(P_vector + 0.11*P_vector_TangentX + 0.06*P_vector_TangentY) - Surfe)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        #Contact Point 2
        #Inequality
        g.append(SurfK@(P_vector + 0.11*P_vector_TangentX - 0.06*P_vector_TangentY) - Surfk)
        glb.append(np.full((4,),-np.inf))
        gub.append(np.full((4,),0))
        #Equality
        g.append(SurfE.T@(P_vector + 0.11*P_vector_TangentX - 0.06*P_vector_TangentY) - Surfe)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        #Contact Point 3
        #Inequality
        g.append(SurfK@(P_vector - 0.11*P_vector_TangentX + 0.06*P_vector_TangentY) - Surfk)
        glb.append(np.full((4,),-np.inf))
        gub.append(np.full((4,),0))
        #Equality
        g.append(SurfE.T@(P_vector - 0.11*P_vector_TangentX + 0.06*P_vector_TangentY) - Surfe)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        #Contact Point 4
        #Inequality
        g.append(SurfK@(P_vector + 0.11*P_vector_TangentX + 0.06*P_vector_TangentY) - Surfk)
        glb.append(np.full((4,),-np.inf))
        gub.append(np.full((4,),0))
        #Equality
        g.append(SurfE.T@(P_vector + 0.11*P_vector_TangentX + 0.06*P_vector_TangentY) - Surfe)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        ##FootStep Constraint
        ##Inequality
        #g.append(SurfK@P_vector - Surfk)
        #glb.append(np.full((4,),-np.inf))
        #gub.append(np.full((4,),0))
        #print(FirstSurfK@p_next - FirstSurfk)

        ##Equality
        #g.append(SurfE.T@P_vector - Surfe)
        #glb.append(np.array([0]))
        #gub.append(np.array([0]))

    #Approximate Kinematics Constraint --- Disallow over-crossing of footsteps from y =0

    if CentralY == True:

        for step_cnt in range(Nsteps):
            P_k_next = ca.vertcat(px[step_cnt],py[step_cnt],pz[step_cnt])
    
            if step_cnt%2 == 0: #even number steps
                #If First Level Swing the Left, then the second level Even Number Phases (the first Phase) Swing the Right -> Left -> Right
                #Left foot in contact for p_current, right foot is going to land as p_next
                g.append(ca.if_else(ParaLeftSwingFlag,P_k_next[1],np.array([-1])))
                glb.append(np.array([-np.inf]))
                gub.append(np.array([-py_lower_limit]))
    
                #If First Levvel Swing the Right, then the second level Even Number Phases (the first phase) Swing the Left -> Right -> Left
                #Right foot in contact for p_current, left foot is going to land at p_next
                g.append(ca.if_else(ParaRightSwingFlag,P_k_next[1],np.array([1])))
                glb.append(np.array([py_lower_limit]))
                gub.append(np.array([np.inf]))
    
            elif step_cnt%2 == 1: #odd number steps
                #If the first level swings the Left, then the second level for Odd Number Phases (the second phase) swings left -> Right -> Left
                #Right foot in contact for p_current, left foot is going to land at p_next
                g.append(ca.if_else(ParaLeftSwingFlag,P_k_next[1],np.array([1])))
                glb.append(np.array([py_lower_limit]))
                gub.append(np.array([np.inf]))
    
                #If the first level swings the Right, then the second level for Odd Number Phases (the second phase) swings Right -> Left -> Right
                #Left foot in contact for p_current, right foot is going to land as p_next
                g.append(ca.if_else(ParaRightSwingFlag,P_k_next[1],np.array([-1])))
                glb.append(np.array([-np.inf]))
                gub.append(np.array([-py_lower_limit]))

    #Switching Time Constraint
    for phase_cnt in range(Nphase):
        if GaitPattern[phase_cnt] == 'InitialDouble':
            if phase_cnt == 0:
                g.append(Ts[phase_cnt] - 0)
                glb.append(np.array([0.1])) #0.1-0.3
                gub.append(np.array([0.3]))
            else:
                g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
                glb.append(np.array([0.1]))
                gub.append(np.array([0.3]))

        elif GaitPattern[phase_cnt] == 'Swing':
            g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
            glb.append(np.array([0.5]))
            gub.append(np.array([0.7]))

            #if phase_cnt == 0:
            #    g.append(Ts[phase_cnt]-0)#0.6-1
            #    glb.append(np.array([0.5]))#0.5 for NLP success
            #    gub.append(np.array([0.7]))
            #else:
            #    g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
            #    glb.append(np.array([0.5]))
            #    gub.append(np.array([0.7]))

        elif GaitPattern[phase_cnt] == 'DoubleSupport':
            g.append(Ts[phase_cnt]-Ts[phase_cnt-1]) #0.1-0.9
            glb.append(np.array([0.1]))
            gub.append(np.array([0.3])) #0.1-0.3

    #-----------------------------------------------------------------------------------------------------------------------
    #Get Variable Index - !!!This is the pure Index, when try to get the array using other routines, we need to add "+1" at the last index due to Python indexing conventions
    x_index = (0,N_K-1) #First set of variables start counting from 0, The enumeration of the last knot is N_K-1
    y_index = (x_index[1]+1,x_index[1]+N_K)
    z_index = (y_index[1]+1,y_index[1]+N_K)
    xdot_index = (z_index[1]+1,z_index[1]+N_K)
    ydot_index = (xdot_index[1]+1,xdot_index[1]+N_K)
    zdot_index = (ydot_index[1]+1,ydot_index[1]+N_K)
    Lx_index = (zdot_index[1]+1,zdot_index[1]+N_K)
    Ly_index = (Lx_index[1]+1,Lx_index[1]+N_K)
    Lz_index = (Ly_index[1]+1,Ly_index[1]+N_K)
    Ldotx_index = (Lz_index[1]+1,Lz_index[1]+N_K)
    Ldoty_index = (Ldotx_index[1]+1,Ldotx_index[1]+N_K)
    Ldotz_index = (Ldoty_index[1]+1,Ldoty_index[1]+N_K)
    FL1x_index = (Ldotz_index[1]+1,Ldotz_index[1]+N_K)
    FL1y_index = (FL1x_index[1]+1,FL1x_index[1]+N_K)
    FL1z_index = (FL1y_index[1]+1,FL1y_index[1]+N_K)
    FL2x_index = (FL1z_index[1]+1,FL1z_index[1]+N_K)
    FL2y_index = (FL2x_index[1]+1,FL2x_index[1]+N_K)
    FL2z_index = (FL2y_index[1]+1,FL2y_index[1]+N_K)
    FL3x_index = (FL2z_index[1]+1,FL2z_index[1]+N_K)
    FL3y_index = (FL3x_index[1]+1,FL3x_index[1]+N_K)
    FL3z_index = (FL3y_index[1]+1,FL3y_index[1]+N_K)
    FL4x_index = (FL3z_index[1]+1,FL3z_index[1]+N_K)
    FL4y_index = (FL4x_index[1]+1,FL4x_index[1]+N_K)
    FL4z_index = (FL4y_index[1]+1,FL4y_index[1]+N_K)
    FR1x_index = (FL4z_index[1]+1,FL4z_index[1]+N_K)
    FR1y_index = (FR1x_index[1]+1,FR1x_index[1]+N_K)
    FR1z_index = (FR1y_index[1]+1,FR1y_index[1]+N_K)
    FR2x_index = (FR1z_index[1]+1,FR1z_index[1]+N_K)
    FR2y_index = (FR2x_index[1]+1,FR2x_index[1]+N_K)
    FR2z_index = (FR2y_index[1]+1,FR2y_index[1]+N_K)
    FR3x_index = (FR2z_index[1]+1,FR2z_index[1]+N_K)
    FR3y_index = (FR3x_index[1]+1,FR3x_index[1]+N_K)
    FR3z_index = (FR3y_index[1]+1,FR3y_index[1]+N_K)
    FR4x_index = (FR3z_index[1]+1,FR3z_index[1]+N_K)
    FR4y_index = (FR4x_index[1]+1,FR4x_index[1]+N_K)
    FR4z_index = (FR4y_index[1]+1,FR4y_index[1]+N_K)
    px_init_index = (FR4z_index[1]+1,FR4z_index[1]+1)
    py_init_index = (px_init_index[1]+1,px_init_index[1]+1)
    pz_init_index = (py_init_index[1]+1,py_init_index[1]+1)
    px_index = (pz_init_index[1]+1,pz_init_index[1]+Nsteps)
    py_index = (px_index[1]+1,px_index[1]+Nsteps)
    pz_index = (py_index[1]+1,py_index[1]+Nsteps)
    Ts_index = (pz_index[1]+1,pz_index[1]+Nphase)

    var_index = {"x":x_index,
                 "y":y_index,
                 "z":z_index,
                 "xdot":xdot_index,
                 "ydot":ydot_index,
                 "zdot":zdot_index,
                 "Lx":Lx_index,
                 "Ly":Ly_index,
                 "Lz":Lz_index,
                 "Ldotx":Ldotx_index,
                 "Ldoty":Ldoty_index,
                 "Ldotz":Ldotz_index,
                 "FL1x":FL1x_index,
                 "FL1y":FL1y_index,
                 "FL1z":FL1z_index,
                 "FL2x":FL2x_index,
                 "FL2y":FL2y_index,
                 "FL2z":FL2z_index,
                 "FL3x":FL3x_index,
                 "FL3y":FL3y_index,
                 "FL3z":FL3z_index,
                 "FL4x":FL4x_index,
                 "FL4y":FL4y_index,
                 "FL4z":FL4z_index,
                 "FR1x":FR1x_index,
                 "FR1y":FR1y_index,
                 "FR1z":FR1z_index,
                 "FR2x":FR2x_index,
                 "FR2y":FR2y_index,
                 "FR2z":FR2z_index,
                 "FR3x":FR3x_index,
                 "FR3y":FR3y_index,
                 "FR3z":FR3z_index,
                 "FR4x":FR4x_index,
                 "FR4y":FR4y_index,
                 "FR4z":FR4z_index,
                 "px_init":px_init_index,
                 "py_init":py_init_index,
                 "pz_init":pz_init_index,
                 "px":px_index,
                 "py":py_index,
                 "pz":pz_index,
                 "Ts":Ts_index,
    }

    #print(DecisionVars[var_index["px_init"][0]:var_index["px_init"][1]+1])

    return DecisionVars, DecisionVars_lb, DecisionVars_ub, J, g, glb, gub, var_index

def CoM_Dynamics_Four_Points(m = 95, Nk_Local = 7, Nsteps = 1, ParameterList = None, StaticStop = False, NumPatches = None, CentralY = False):
    #-----------------------------------------------------------------------------------------------------------------------
    #Define Parameters
    #   Gait Pattern, Each action is followed up by a double support phase
    GaitPattern = ["InitialDouble","Swing","DoubleSupport"] + ["InitialDouble", "Swing","DoubleSupport"]*(Nsteps-1) #,'RightSupport','DoubleSupport','LeftSupport','DoubleSupport'

    PhaseDurationVec = [0.3, 0.5, 0.3]*(Nsteps)

    print(PhaseDurationVec)

    #   Number of Phases
    Nphase = len(GaitPattern)
    #   Number of Steps
    #Nstep = 1
    #   Number of Knots per Phase - how many intervals do we have for a single phase; 10 intervals need 11 knots/ticks
    #Nk_Local= 5
    #   Compute Number of Total knots/ticks, but the enumeration start from 0 to N_K-1
    N_K = Nk_Local*Nphase + 1 #+1 the last knots to finalize the plan
    #   Robot mass
    #m = 95 #kg
    G = 9.80665 #kg/m^2
    #   Terrain Model
    #       Flat Terrain
    #TerrainNorm = [0,0,1] 
    #TerrainTangentX = [1,0,0]
    #TerrainTangentY = [0,1,0]
    miu = 0.3
    #   Force Limits
    F_bound = 400
    Fxlb = -F_bound
    Fxub = F_bound
    Fylb = -F_bound
    Fyub = F_bound
    Fzlb = -F_bound
    Fzub = F_bound
    #   Angular Momentum Bounds
    L_bound = 2.5
    Ldot_bound = 3.5
    Lub = L_bound
    Llb = -L_bound
    Ldotub = Ldot_bound
    Ldotlb = -Ldot_bound
    #Minimum y-axis foot location
    py_lower_limit = 0.04
    #Lowest Z
    z_lowest = 0.7
    z_highest = 0.8
    #-----------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------
    #Kinematics Constraint for Talos
    #kinematicConstraints = genKinematicConstraints(left_foot_constraints,right_foot_constraints)
    K_CoM_Right,k_CoM_Right = right_foot_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]))
    K_CoM_Left,k_CoM_Left = left_foot_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]))

    #K_CoM_Left = kinematicConstraints[0][0]
    #k_CoM_Left = kinematicConstraints[0][1]
    #K_CoM_Right = kinematicConstraints[1][0]
    #k_CoM_Right = kinematicConstraints[1][1]
    #Relative Foot Constraint matrices
    
    #relativeConstraints = genFootRelativeConstraints(right_foot_in_lf_frame_constraints,left_foot_in_rf_frame_constraints)
    Q_rf_in_lf,q_rf_in_lf = right_foot_in_lf_frame_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]))
    Q_lf_in_rf,q_lf_in_rf = left_foot_in_rf_frame_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]))

    #Q_rf_in_lf = relativeConstraints[0][0] #named lf in rf, but representing rf in lf
    #q_rf_in_lf = relativeConstraints[0][1] #named lf in rf, but representing rf in lf
    #Q_lf_in_rf = relativeConstraints[1][0] #named rf in lf, but representing lf in rf
    #q_lf_in_rf = relativeConstraints[1][1] #named rf in lf, but representing lf in rf
    #-----------------------------------------------------------------------------------------------------------------------
    #Define Initial and Terminal Conditions
    #x_init = ca.SX.sym('x_init')
    #y_init = ca.SX.sym('y_init')
    #z_init = ca.SX.sym('z_init')

    x_init = ParameterList["x_init"]
    y_init = ParameterList["y_init"]
    z_init = ParameterList["z_init"]

    xdot_init = ParameterList["xdot_init"]
    ydot_init = ParameterList["ydot_init"]
    zdot_init = ParameterList["zdot_init"]

    #Lx_init = 0
    #Ly_init = 0
    #Lz_init = 0

    #Ldotx_init = 0
    #Ldoty_init = 0
    #Ldotz_init = 0

    PLx_init = ParameterList["PLx_init"]
    PLy_init = ParameterList["PLy_init"]
    PLz_init = ParameterList["PLz_init"]
    PL_init = ca.vertcat(PLx_init,PLy_init,PLz_init)

    PRx_init = ParameterList["PRx_init"]
    PRy_init = ParameterList["PRy_init"]
    PRz_init = ParameterList["PRz_init"]
    PR_init = ca.vertcat(PRx_init,PRy_init,PRz_init)

    x_end = ParameterList["x_end"]
    y_end = ParameterList["y_end"]
    z_end = ParameterList["z_end"]

    xdot_end = ParameterList["xdot_end"]
    ydot_end = ParameterList["ydot_end"]
    zdot_end = ParameterList["zdot_end"]

    #Lx_end = 0
    #Ly_end = 0
    #Lz_end = 0

    #Ldotx_end = 0
    #Ldoty_end = 0
    #Ldotz_end = 0

    #Flags for Swing Legs (Defined as Parameters)
    ParaLeftSwingFlag = ParameterList["LeftSwingFlag"]
    ParaRightSwingFlag = ParameterList["RightSwingFlag"]

    #Surfaces (Only the Second One)
    #Surface Patches
    SurfParas = ParameterList["SurfParas"]

    #Tangents and Norms
    #Initial Contact Norm and Tangents
    PL_init_Norm = ParameterList["PL_init_Norm"]
    PL_init_TangentX = ParameterList["PL_init_TangentX"]
    PL_init_TangentY = ParameterList["PL_init_TangentY"]
    PR_init_Norm = ParameterList["PR_init_Norm"]
    PR_init_TangentX = ParameterList["PR_init_TangentX"]
    PR_init_TangentY = ParameterList["PR_init_TangentY"]
    
    #Future Contact Norm and Tangents
    SurfNorms = ParameterList["SurfNorms"]                
    SurfTangentsX = ParameterList["SurfTangentsX"]
    SurfTangentsY = ParameterList["SurfTangentsY"]

    #-----------------------------------------------------------------------------------------------------------------------
    #Define Variables and Bounds, Parameters
    #   CoM Position x-axis
    x = ca.SX.sym('x',N_K)
    x_lb = np.array([[0]*(x.shape[0]*x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    x_ub = np.array([[30]*(x.shape[0]*x.shape[1])])
    #   CoM Position y-axis
    y = ca.SX.sym('y',N_K)
    y_lb = np.array([[-1]*(y.shape[0]*y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    y_ub = np.array([[1]*(y.shape[0]*y.shape[1])])
    #   CoM Position z-axis
    z = ca.SX.sym('z',N_K)
    z_lb = np.array([[z_lowest]*(z.shape[0]*z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    z_ub = np.array([[z_highest]*(z.shape[0]*z.shape[1])])
    #   CoM Velocity x-axis
    xdot = ca.SX.sym('xdot',N_K)
    xdot_lb = np.array([[-1.5]*(xdot.shape[0]*xdot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    xdot_ub = np.array([[1.5]*(xdot.shape[0]*xdot.shape[1])])
    #   CoM Velocity y-axis
    ydot = ca.SX.sym('ydot',N_K)
    ydot_lb = np.array([[-1.5]*(ydot.shape[0]*ydot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    ydot_ub = np.array([[1.5]*(ydot.shape[0]*ydot.shape[1])])
    #   CoM Velocity z-axis
    zdot = ca.SX.sym('zdot',N_K)
    zdot_lb = np.array([[-1.5]*(zdot.shape[0]*zdot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    zdot_ub = np.array([[1.5]*(zdot.shape[0]*zdot.shape[1])])
    #   Angular Momentum x-axis
    #Lx = ca.SX.sym('Lx',N_K)
    #Lx_lb = np.array([[Llb]*(Lx.shape[0]*Lx.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #Lx_ub = np.array([[Lub]*(Lx.shape[0]*Lx.shape[1])])
    ##   Angular Momentum y-axis
    #Ly = ca.SX.sym('Ly',N_K)
    #Ly_lb = np.array([[Llb]*(Ly.shape[0]*Ly.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #Ly_ub = np.array([[Lub]*(Ly.shape[0]*Ly.shape[1])])
    ##   Angular Momntum y-axis
    #Lz = ca.SX.sym('Lz',N_K)
    #Lz_lb = np.array([[Llb]*(Lz.shape[0]*Lz.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #Lz_ub = np.array([[Lub]*(Lz.shape[0]*Lz.shape[1])])
    ##   Angular Momentum rate x-axis
    #Ldotx = ca.SX.sym('Ldotx',N_K)
    #Ldotx_lb = np.array([[Ldotlb]*(Ldotx.shape[0]*Ldotx.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #Ldotx_ub = np.array([[Ldotub]*(Ldotx.shape[0]*Ldotx.shape[1])])
    ##   Angular Momentum y-axis
    #Ldoty = ca.SX.sym('Ldoty',N_K)
    #Ldoty_lb = np.array([[Ldotlb]*(Ldoty.shape[0]*Ldoty.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #Ldoty_ub = np.array([[Ldotub]*(Ldoty.shape[0]*Ldoty.shape[1])])
    ##   Angular Momntum z-axis
    #Ldotz = ca.SX.sym('Ldotz',N_K)
    #Ldotz_lb = np.array([[Ldotlb]*(Ldotz.shape[0]*Ldotz.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #Ldotz_ub = np.array([[Ldotub]*(Ldotz.shape[0]*Ldotz.shape[1])])
    #left Foot Forces
    #Left Foot Contact Point 1 x-axis
    FL1x = ca.SX.sym('FL1x',N_K)
    FL1x_lb = np.array([[Fxlb]*(FL1x.shape[0]*FL1x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL1x_ub = np.array([[Fxub]*(FL1x.shape[0]*FL1x.shape[1])])
    #Left Foot Contact Point 1 y-axis
    FL1y = ca.SX.sym('FL1y',N_K)
    FL1y_lb = np.array([[Fylb]*(FL1y.shape[0]*FL1y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL1y_ub = np.array([[Fyub]*(FL1y.shape[0]*FL1y.shape[1])])
    #Left Foot Contact Point 1 z-axis
    FL1z = ca.SX.sym('FL1z',N_K)
    FL1z_lb = np.array([[Fzlb]*(FL1z.shape[0]*FL1z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL1z_ub = np.array([[Fzub]*(FL1z.shape[0]*FL1z.shape[1])])
    #Left Foot Contact Point 2 x-axis
    FL2x = ca.SX.sym('FL2x',N_K)
    FL2x_lb = np.array([[Fxlb]*(FL2x.shape[0]*FL2x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL2x_ub = np.array([[Fxub]*(FL2x.shape[0]*FL2x.shape[1])])
    #Left Foot Contact Point 2 y-axis
    FL2y = ca.SX.sym('FL2y',N_K)
    FL2y_lb = np.array([[Fylb]*(FL2y.shape[0]*FL2y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL2y_ub = np.array([[Fyub]*(FL2y.shape[0]*FL2y.shape[1])])
    #Left Foot Contact Point 2 z-axis
    FL2z = ca.SX.sym('FL2z',N_K)
    FL2z_lb = np.array([[Fzlb]*(FL2z.shape[0]*FL2z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL2z_ub = np.array([[Fzub]*(FL2z.shape[0]*FL2z.shape[1])])
    #Left Foot Contact Point 3 x-axis
    FL3x = ca.SX.sym('FL3x',N_K)
    FL3x_lb = np.array([[Fxlb]*(FL3x.shape[0]*FL3x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL3x_ub = np.array([[Fxub]*(FL3x.shape[0]*FL3x.shape[1])])
    #Left Foot Contact Point 3 y-axis
    FL3y = ca.SX.sym('FL3y',N_K)
    FL3y_lb = np.array([[Fylb]*(FL3y.shape[0]*FL3y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL3y_ub = np.array([[Fyub]*(FL3y.shape[0]*FL3y.shape[1])])
    #Left Foot Contact Point 3 z-axis
    FL3z = ca.SX.sym('FL3z',N_K)
    FL3z_lb = np.array([[Fzlb]*(FL3z.shape[0]*FL3z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL3z_ub = np.array([[Fzub]*(FL3z.shape[0]*FL3z.shape[1])])
    #Left Foot Contact Point 4 x-axis
    FL4x = ca.SX.sym('FL4x',N_K)
    FL4x_lb = np.array([[Fxlb]*(FL4x.shape[0]*FL4x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL4x_ub = np.array([[Fxub]*(FL4x.shape[0]*FL4x.shape[1])])
    #Left Foot Contact Point 4 y-axis
    FL4y = ca.SX.sym('FL4y',N_K)
    FL4y_lb = np.array([[Fylb]*(FL4y.shape[0]*FL4y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL4y_ub = np.array([[Fyub]*(FL4y.shape[0]*FL4y.shape[1])])
    #Left Foot Contact Point 4 z-axis
    FL4z = ca.SX.sym('FL4z',N_K)
    FL4z_lb = np.array([[Fzlb]*(FL4z.shape[0]*FL4z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL4z_ub = np.array([[Fzub]*(FL4z.shape[0]*FL4z.shape[1])])

    #Right Contact Force x-axis
    #Right Foot Contact Point 1 x-axis
    FR1x = ca.SX.sym('FR1x',N_K)
    FR1x_lb = np.array([[Fxlb]*(FR1x.shape[0]*FR1x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR1x_ub = np.array([[Fxub]*(FR1x.shape[0]*FR1x.shape[1])])
    #Right Foot Contact Point 1 y-axis
    FR1y = ca.SX.sym('FR1y',N_K)
    FR1y_lb = np.array([[Fylb]*(FR1y.shape[0]*FR1y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR1y_ub = np.array([[Fyub]*(FR1y.shape[0]*FR1y.shape[1])])
    #Right Foot Contact Point 1 z-axis
    FR1z = ca.SX.sym('FR1z',N_K)
    FR1z_lb = np.array([[Fzlb]*(FR1z.shape[0]*FR1z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR1z_ub = np.array([[Fzub]*(FR1z.shape[0]*FR1z.shape[1])])
    #Right Foot Contact Point 2 x-axis
    FR2x = ca.SX.sym('FR2x',N_K)
    FR2x_lb = np.array([[Fxlb]*(FR2x.shape[0]*FR2x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR2x_ub = np.array([[Fxub]*(FR2x.shape[0]*FR2x.shape[1])])
    #Right Foot Contact Point 2 y-axis
    FR2y = ca.SX.sym('FR2y',N_K)
    FR2y_lb = np.array([[Fylb]*(FR2y.shape[0]*FR2y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR2y_ub = np.array([[Fyub]*(FR2y.shape[0]*FR2y.shape[1])])
    #Right Foot Contact Point 2 z-axis
    FR2z = ca.SX.sym('FR2z',N_K)
    FR2z_lb = np.array([[Fzlb]*(FR2z.shape[0]*FR2z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR2z_ub = np.array([[Fzub]*(FR2z.shape[0]*FR2z.shape[1])])
    #Right Foot Contact Point 3 x-axis
    FR3x = ca.SX.sym('FR3x',N_K)
    FR3x_lb = np.array([[Fxlb]*(FR3x.shape[0]*FR3x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR3x_ub = np.array([[Fxub]*(FR3x.shape[0]*FR3x.shape[1])])
    #Right Foot Contact Point 3 y-axis
    FR3y = ca.SX.sym('FR3y',N_K)
    FR3y_lb = np.array([[Fylb]*(FR3y.shape[0]*FR3y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR3y_ub = np.array([[Fyub]*(FR3y.shape[0]*FR3y.shape[1])])
    #Right Foot Contact Point 3 z-axis
    FR3z = ca.SX.sym('FR3z',N_K)
    FR3z_lb = np.array([[Fzlb]*(FR3z.shape[0]*FR3z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR3z_ub = np.array([[Fzub]*(FR3z.shape[0]*FR3z.shape[1])])
    #Right Foot Contact Point 4 x-axis
    FR4x = ca.SX.sym('FR4x',N_K)
    FR4x_lb = np.array([[Fxlb]*(FR4x.shape[0]*FR4x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR4x_ub = np.array([[Fxub]*(FR4x.shape[0]*FR4x.shape[1])])
    #Right Foot Contact Point 4 y-axis
    FR4y = ca.SX.sym('FR4y',N_K)
    FR4y_lb = np.array([[Fylb]*(FR4y.shape[0]*FR4y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR4y_ub = np.array([[Fyub]*(FR4y.shape[0]*FR4y.shape[1])])
    #Right Foot Contact Point 4 z-axis
    FR4z = ca.SX.sym('FR4z',N_K)
    FR4z_lb = np.array([[Fzlb]*(FR4z.shape[0]*FR4z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR4z_ub = np.array([[Fzub]*(FR4z.shape[0]*FR4z.shape[1])])

    #Initial Contact Location (need to connect to the first level)
    #   Plx
    px_init = ca.SX.sym('px_init')
    px_init_lb = np.array([-1])
    px_init_ub = np.array([30])

    #   py
    py_init = ca.SX.sym('py_init')
    py_init_lb = np.array([-1])
    py_init_ub = np.array([1])

    #   pz
    pz_init = ca.SX.sym('pz_init')
    pz_init_lb = np.array([-5])
    pz_init_ub = np.array([5])

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
        pxtemp = ca.SX.sym('px'+str(stepIdx)) #0 + 1
        px.append(pxtemp)
        px_lb.append(np.array([-1]))
        px_ub.append(np.array([30]))

        pytemp = ca.SX.sym('py'+str(stepIdx))
        py.append(pytemp)
        py_lb.append(np.array([-1]))
        py_ub.append(np.array([1]))

        #   Foot steps are all staying on the ground
        pztemp = ca.SX.sym('pz'+str(stepIdx))
        pz.append(pztemp)
        pz_lb.append(np.array([-5]))
        pz_ub.append(np.array([5]))

    #Switching Time Vector
    Ts = []
    Ts_lb = []
    Ts_ub = []
    for n_phase in range(Nphase):
        Tstemp = ca.SX.sym('Ts'+str(n_phase+1)) #0 + 1 + ....
        Ts.append(Tstemp)
        Ts_lb.append(np.array([0.05]))
        Ts_ub.append(np.array([2.0*(Nphase+1)]))

    #   Collect all Decision Variables
    #DecisionVars = ca.vertcat(x,y,z,xdot,ydot,zdot,Lx,Ly,Lz,Ldotx,Ldoty,Ldotz,FL1x,FL1y,FL1z,FL2x,FL2y,FL2z,FL3x,FL3y,FL3z,FL4x,FL4y,FL4z,FR1x,FR1y,FR1z,FR2x,FR2y,FR2z,FR3x,FR3y,FR3z,FR4x,FR4y,FR4z,px_init,py_init,pz_init,*px,*py,*pz,*Ts)
    DecisionVars = ca.vertcat(x,y,z,xdot,ydot,zdot,FL1x,FL1y,FL1z,FL2x,FL2y,FL2z,FL3x,FL3y,FL3z,FL4x,FL4y,FL4z,FR1x,FR1y,FR1z,FR2x,FR2y,FR2z,FR3x,FR3y,FR3z,FR4x,FR4y,FR4z,px_init,py_init,pz_init,*px,*py,*pz,*Ts)
    #print(DecisionVars)
    DecisionVarsShape = DecisionVars.shape

    #   Collect all lower bound and upper bound
    #DecisionVars_lb = np.concatenate(((x_lb,y_lb,z_lb,xdot_lb,ydot_lb,zdot_lb,Lx_lb,Ly_lb,Lz_lb,Ldotx_lb,Ldoty_lb,Ldotz_lb,FL1x_lb,FL1y_lb,FL1z_lb,FL2x_lb,FL2y_lb,FL2z_lb,FL3x_lb,FL3y_lb,FL3z_lb,FL4x_lb,FL4y_lb,FL4z_lb,FR1x_lb,FR1y_lb,FR1z_lb,FR2x_lb,FR2y_lb,FR2z_lb,FR3x_lb,FR3y_lb,FR3z_lb,FR4x_lb,FR4y_lb,FR4z_lb,px_init_lb,py_init_lb,pz_init_lb,px_lb,py_lb,pz_lb,Ts_lb)),axis=None)
    #DecisionVars_ub = np.concatenate(((x_ub,y_ub,z_ub,xdot_ub,ydot_ub,zdot_ub,Lx_ub,Ly_ub,Lz_ub,Ldotx_ub,Ldoty_ub,Ldotz_ub,FL1x_ub,FL1y_ub,FL1z_ub,FL2x_ub,FL2y_ub,FL2z_ub,FL3x_ub,FL3y_ub,FL3z_ub,FL4x_ub,FL4y_ub,FL4z_ub,FR1x_ub,FR1y_ub,FR1z_ub,FR2x_ub,FR2y_ub,FR2z_ub,FR3x_ub,FR3y_ub,FR3z_ub,FR4x_ub,FR4y_ub,FR4z_ub,px_init_ub,py_init_ub,pz_init_ub,px_ub,py_ub,pz_ub,Ts_ub)),axis=None)
    DecisionVars_lb = np.concatenate(((x_lb,y_lb,z_lb,xdot_lb,ydot_lb,zdot_lb,FL1x_lb,FL1y_lb,FL1z_lb,FL2x_lb,FL2y_lb,FL2z_lb,FL3x_lb,FL3y_lb,FL3z_lb,FL4x_lb,FL4y_lb,FL4z_lb,FR1x_lb,FR1y_lb,FR1z_lb,FR2x_lb,FR2y_lb,FR2z_lb,FR3x_lb,FR3y_lb,FR3z_lb,FR4x_lb,FR4y_lb,FR4z_lb,px_init_lb,py_init_lb,pz_init_lb,px_lb,py_lb,pz_lb,Ts_lb)),axis=None)
    DecisionVars_ub = np.concatenate(((x_ub,y_ub,z_ub,xdot_ub,ydot_ub,zdot_ub,FL1x_ub,FL1y_ub,FL1z_ub,FL2x_ub,FL2y_ub,FL2z_ub,FL3x_ub,FL3y_ub,FL3z_ub,FL4x_ub,FL4y_ub,FL4z_ub,FR1x_ub,FR1y_ub,FR1z_ub,FR2x_ub,FR2y_ub,FR2z_ub,FR3x_ub,FR3y_ub,FR3z_ub,FR4x_ub,FR4y_ub,FR4z_ub,px_init_ub,py_init_ub,pz_init_ub,px_ub,py_ub,pz_ub,Ts_ub)),axis=None)

    #Time Span Setup
    tau_upper_limit = 1
    tauStepLength = tau_upper_limit/(N_K-1) #Get the interval length, total number of knots - 1

    #-----------------------------------------------------------------------------------------------------------------------
    #Define Constrains and Running Cost
    g = []
    glb = []
    gub = []
    J = 0

    #Initial and Terminal Condition

    ##   Terminal CoM y-axis
    #g.append(y[-1]-y_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    ##   Terminal CoM z-axis
    #g.append(z[-1]-z_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    #if StaticStop == True:
    #    #   Terminal Zero CoM velocity x-axis
    #    g.append(xdot[-1])
    #    glb.append(np.array([0]))
    #    gub.append(np.array([0]))

    #    #   Terminal Zero CoM velocity y-axis
    #    g.append(ydot[-1])
    #    glb.append(np.array([0]))
    #    gub.append(np.array([0]))

    #    #   Terminal Zero CoM velocity z-axis
    #    g.append(zdot[-1])
    #    glb.append(np.array([0]))
    #    gub.append(np.array([0]))

    #   Terminal Angular Momentum x-axis
    #g.append(Lx[-1]-Lx_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    #   Terminal Angular Momentum y-axis
    #g.append(Ly[-1]-Ly_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    #   Terminal Angular Momentum z-axis
    #g.append(Lz[-1]-Lz_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    #   Terminal Angular Momentum Rate x-axis
    #g.append(Ldotx[-1]-Ldotx_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    #   Terminal Angular Momentum Rate y-axis
    #g.append(Ldoty[-1]-Ldoty_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    #   Terminal Angular Momentum Rate z-axis
    #g.append(Ldotz[-1]-Ldotz_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    #Loop over all Phases (Knots)
    for Nph in range(Nphase):
        #Decide Number of Knots
        if Nph == Nphase-1:  #The last Knot belongs to the Last Phase
            Nk_ThisPhase = Nk_Local+1
        else:
            Nk_ThisPhase = Nk_Local       

        #Fixed Time Step
        h = (PhaseDurationVec[Nph])/Nk_Local

        for Local_k_Count in range(Nk_ThisPhase):
            #Get knot number across the entire time line
            k = Nph*Nk_Local + Local_k_Count
            #print(k)

            #------------------------------------------
            #Build useful vectors
            #   Forces
            FL1_k = ca.vertcat(FL1x[k],FL1y[k],FL1z[k])
            FL2_k = ca.vertcat(FL2x[k],FL2y[k],FL2z[k])
            FL3_k = ca.vertcat(FL3x[k],FL3y[k],FL3z[k])
            FL4_k = ca.vertcat(FL4x[k],FL4y[k],FL4z[k])

            FR1_k = ca.vertcat(FR1x[k],FR1y[k],FR1z[k])
            FR2_k = ca.vertcat(FR2x[k],FR2y[k],FR2z[k])
            FR3_k = ca.vertcat(FR3x[k],FR3y[k],FR3z[k])
            FR4_k = ca.vertcat(FR4x[k],FR4y[k],FR4z[k])
            #   CoM
            CoM_k = ca.vertcat(x[k],y[k],z[k])
            ##   Angular Momentum
            #if k<N_K-1: #N_K-1 the enumeration of the last knot, k<N_K-1 the one before the last knot
            #    Ldot_current = ca.vertcat(Ldotx[k],Ldoty[k],Ldotz[k])
            #    Ldot_next = ca.vertcat(Ldotz[k+1],Ldotz[k+1],Ldotz[k+1])
            #-------------------------------------------

            #-------------------------------------------
            #Phase dependent Constraints (CoM Kinematics and Angular Dynamics)
            
            #Get Step Counter
            StepCnt = Nph//3

            #NOTE: The first phase (Initial Double) --- Needs special care
            if Nph == 0 and GaitPattern[Nph]=='InitialDouble':

                #initial support foot (the landing foot from the first phase)
                p_init = ca.vertcat(px_init,py_init,pz_init)
                p_init_TangentX = SurfTangentsX[0:3]
                p_init_TangentY = SurfTangentsY[0:3]
                p_init_Norm = SurfNorms[0:3]

                #Case 1
                #If First Level Swing the Left, the the 0 phase (InitDouble) has p_init as the left support, PR_init as the right support
                #Kinematics Constraint
                #CoM in Left (p_init)
                g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_init)
                #CoM in Right (PR_init)
                g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = PR_init)
                ##Angular Dynamics
                #if k<N_K-1:
                #    g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, Ldot_next = Ldot_next, Ldot_current = Ldot_current, h = h, PL = p_init, PL_TangentX = p_init_TangentX, PL_TangentY = p_init_TangentY, PR = PR_init, PR_TangentX = PR_init_TangentX, PR_TangentY = PR_init_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                #Unilateral Constraint
                #If the first level swing the Left foot first, then the Left foot is the landing foot (p_init), Left foot obey the unilateral constraint on p_init
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainNorm = p_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainNorm = p_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainNorm = p_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainNorm = p_init_Norm)
                #then the Right foot is obey the unilateral constraint on the PR_init
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainNorm = PR_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainNorm = PR_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainNorm = PR_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainNorm = PR_init_Norm)
                #Friction Cone Constraint
                #If the first level swing the Left foot first, then the Left foot is the landing foot (p_init), Left foot obey the friction cone constraint on p_init
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu)
                #then the right foot obeys the friction cone constraints on the PR_init
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu)
                
                #Case 2
                #If First Level Swing the Right, the the 0 phase (InitDouble) has p_init as the Right support, PL_init as the Left support
                #Kinematics Constraint
                #CoM in the Left foot
                g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = PL_init)
                #CoM in the Right foot
                g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_init)
                ##Agnular Dynamics
                #if k<N_K-1:
                #    g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, Ldot_next = Ldot_next, Ldot_current = Ldot_current, h = h, PL = PL_init, PL_TangentX = PL_init_TangentX, PL_TangentY = PL_init_TangentY, PR = p_init, PR_TangentX = p_init_TangentX, PR_TangentY = p_init_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                #Unilateral Constraint
                #If the first level swings the Right foot first, then the right foot is the landing foot (p_init), Right foot obeys the unilateral constraint on p_init
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainNorm = p_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainNorm = p_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainNorm = p_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainNorm = p_init_Norm)
                #then the Left foot obeys the unilateral constrint on the PL_init
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainNorm = PL_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainNorm = PL_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainNorm = PL_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainNorm = PL_init_Norm)                
                #Friction Cone Constraint
                #if the first level swing the right foot first, then the Right foot is the landing foot (p_init), Right foot obey the friction cone constraints on p_init
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu)
                #then the left foot obeys the friction cone constraint of PL_init
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu)
            
            #All other phases
            else:
                if GaitPattern[Nph]=='InitialDouble':
                    #Get contact location
                    if StepCnt == 1: #Step 1 needs special care (NOTE: Step Count Start from 0)
                        p_previous = ca.vertcat(px_init,py_init,pz_init)
                        p_previous_TangentX = SurfTangentsX[0:3]
                        p_previous_TangentY = SurfTangentsY[0:3]
                        p_previous_Norm = SurfNorms[0:3]

                        #In second level, Surfaces index is Step Vector Index (fpr px, py, pz, here is StepCnt-1) + 1
                        p_current = ca.vertcat(px[StepCnt-1],py[StepCnt-1],pz[StepCnt-1])
                        p_current_TangentX = SurfTangentsX[StepCnt*3:StepCnt*3+3]
                        p_current_TangentY = SurfTangentsY[StepCnt*3:StepCnt*3+3]
                        p_current_Norm = SurfNorms[StepCnt*3:StepCnt*3+3]

                    else: #Like Step 2, 3, 4 .....
                        p_previous = ca.vertcat(px[StepCnt-2],py[StepCnt-2],pz[StepCnt-2])
                        p_previous_TangentX = SurfTangentsX[(StepCnt-1)*3:(StepCnt-1)*3+3]
                        p_previous_TangentY = SurfTangentsY[(StepCnt-1)*3:(StepCnt-1)*3+3]
                        p_previous_Norm = SurfNorms[(StepCnt-1)*3:(StepCnt-1)*3+3]

                        p_current = ca.vertcat(px[StepCnt-1],py[StepCnt-1],pz[StepCnt-1])
                        p_current_TangentX = SurfTangentsX[StepCnt*3:StepCnt*3+3]
                        p_current_TangentY = SurfTangentsY[StepCnt*3:StepCnt*3+3]
                        p_current_Norm = SurfNorms[StepCnt*3:StepCnt*3+3]

                    #Give Constraint according to even and odd steps
                    if StepCnt%2 == 0: #Even Numbers of Footsteps
                        #Case 1
                        #If the first level swing the Left, then the Even Number of Steps in the Intial Double support phase have p_current as Left Support (Landed), p_previous as Right Support (Stationary)
                        #CoM in the Left (p_current)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_current)
                        #CoM in the Right (p_previous)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_previous)
                        ##Angular Dynamics
                        #if k<N_K-1:
                        #    g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, Ldot_next = Ldot_next, Ldot_current = Ldot_current, h = h, PL = p_current, PL_TangentX = p_current_TangentX, PL_TangentY = p_current_TangentY, PR = p_previous, PR_TangentX = p_previous_TangentX, PR_TangentY = p_previous_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                        #Unilateral Constraint
                        #If the first level swing the Left foot first, then the Left foot is the landing foot (p_current), Left foot obey the unilateral constraint on p_current
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainNorm = p_current_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainNorm = p_current_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainNorm = p_current_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainNorm = p_current_Norm)
                        #then the Right foot is obey the unilateral constraint on the Stationary foot p_previous
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainNorm = p_previous_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainNorm = p_previous_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainNorm = p_previous_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainNorm = p_previous_Norm)
                        #Friction Cone Constraint
                        #If the first level swing the Left foot first, then the Left foot is the landing foot (p_current), Left foot obey the friction cone constraint on p_current
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        #then the right foot obeys the friction cone constraints on the on the Stationary foot p_previous
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        
                        #Case 2
                        #If the first level swing the Right, then the Even Number of Steps in the Intial Double support phase have p_current as Right Support (Landed), 
                        #p_previous as Left Support (Stationary)
                        #CoM in the Left (p_previous)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_previous)
                        #CoM in the Right (p_current)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_current)
                        ##Angular Dynamics
                        #if k<N_K-1:
                        #    g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, Ldot_next = Ldot_next, Ldot_current = Ldot_current, h = h, PL = p_previous, PL_TangentX = p_previous_TangentX, PL_TangentY = p_previous_TangentY, PR = p_current, PR_TangentX = p_current_TangentX, PR_TangentY = p_current_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                        #Unilateral Constraint
                        #Left foot obey the unilateral constraint on p_previous
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainNorm = p_previous_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainNorm = p_previous_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainNorm = p_previous_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainNorm = p_previous_Norm)
                        #Right foot is obey the unilateral constraint on p_current
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainNorm = p_current_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainNorm = p_current_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainNorm = p_current_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainNorm = p_current_Norm)
                        #Friction Cone Constraint
                        #Left foot obey the friction cone constraint on p_previous
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        #then the right foot obeys the friction cone constraints on the on the Stationary foot p_previous
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)

                    elif StepCnt%2 == 1: #Odd Number of Steps
                        #Case 1
                        #If the first level swing the Left, then the Odd Number of Steps in the Intial Double support phase have p_current as Right Support (Landed), p_previous as Left Support (Stationary)
                        #CoM in the Left (p_previous)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_previous)
                        #CoM in the Right (p_current)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_current)
                        ##Angular Dynamics
                        #if k<N_K-1:
                        #    g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, Ldot_next = Ldot_next, Ldot_current = Ldot_current, h = h, PL = p_previous, PL_TangentX = p_previous_TangentX, PL_TangentY = p_previous_TangentY, PR = p_current, PR_TangentX = p_current_TangentX, PR_TangentY = p_current_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                        #Unilateral Constraint
                        #Left foot obey the unilateral constraint on p_previous
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainNorm = p_previous_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainNorm = p_previous_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainNorm = p_previous_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainNorm = p_previous_Norm)
                        #Right foot is obey the unilateral constraint on p_current
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainNorm = p_current_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainNorm = p_current_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainNorm = p_current_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainNorm = p_current_Norm)
                        #Friction Cone Constraint
                        #Left foot obey the friction cone constraint on p_previous
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        #right foot obeys the friction cone constraints on p_current
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        
                        #Case 2
                        #If the first level swing the Right, then the Odd Number of Steps in the Intial Double support phase have p_current as Left Support (Landed), 
                        #p_previous as Right Support (Stationary)
                        #CoM in the Left (p_current)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_current)
                        #CoM in the Right (p_previous)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_previous)
                        ##Angular Dynamics
                        #if k<N_K-1:
                        #    g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, Ldot_next = Ldot_next, Ldot_current = Ldot_current, h = h, PL = p_current, PL_TangentX = p_current_TangentX, PL_TangentY = p_current_TangentY, PR = p_previous, PR_TangentX = p_previous_TangentX, PR_TangentY = p_previous_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                        #Unilateral Constraint
                        #Left foot obey the unilateral constraint on p_current
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainNorm = p_current_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainNorm = p_current_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainNorm = p_current_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainNorm = p_current_Norm)
                        #Right foot is obey the unilateral constraint on p_previous
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainNorm = p_previous_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainNorm = p_previous_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainNorm = p_previous_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainNorm = p_previous_Norm)
                        #Friction Cone Constraint
                        #Left foot obey the friction cone constraint on p_current
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        #then the right foot obeys the friction cone constraints on the p_previous
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)

                elif GaitPattern[Nph]== 'Swing':
                    #Get contact location
                    if StepCnt == 0:#Special Case for the First Step (NOTE:Step 0)
                        p_stance = ca.vertcat(px_init,py_init,pz_init)
                        p_stance_TangentX = SurfTangentsX[0:3]
                        p_stance_TangentY = SurfTangentsY[0:3]
                        p_stance_Norm = SurfNorms[0:3]

                    else: #For other Steps, indexed as 1,2,3,4
                        p_stance = ca.vertcat(px[StepCnt-1],py[StepCnt-1],pz[StepCnt-1])
                        p_stance_TangentX = SurfTangentsX[StepCnt*3:StepCnt*3+3]
                        p_stance_TangentY = SurfTangentsX[StepCnt*3:StepCnt*3+3]
                        p_stance_Norm = SurfTangentsX[StepCnt*3:StepCnt*3+3]

                    #Give Constraint according to even and odd steps
                    if StepCnt%2 == 0: #Even Number of Steps
                        #Case 1
                        #If First Level Swing the Left, then the second level Even Number Phases (the first Phase) Swing the Right
                        #Left foot is the stance foot
                        #Right foot is floating
                        #Kinematics Constraint
                        #CoM in the Left (p_stance)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_stance)
                        ##Angular Dynamics (Left Stance)
                        #if k<N_K-1:
                        #    g, glb, gub = Angular_Momentum_Rate_Swing(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, Ldot_next = Ldot_next, Ldot_current = Ldot_current, h = h, P = p_stance, P_TangentX = p_stance_TangentX, P_TangentY = p_stance_TangentY, CoM_k = CoM_k, F1_k = FL1_k, F2_k = FL2_k, F3_k = FL3_k, F4_k = FL4_k)
                        #Zero Forces (Right Foot)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k)
                        #Unilateral Constraints on Left Foot p_stance
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainNorm = p_stance_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainNorm = p_stance_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainNorm = p_stance_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainNorm = p_stance_Norm)
                        #Friction Cone Constraint on Left Foot p_stance
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)

                        #Case 2
                        #If First Level Swing the Right, then the second level Even Number Phases (the first Phase) Swing the Left
                        #Right foot is the stance foot
                        #Left foot is floating
                        #Kinematics Constraint
                        #CoM in the Right (p_stance)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_stance)
                        ##Angular Dynamics(Right Stance)
                        #if k<N_K-1:
                        #    g, glb, gub = Angular_Momentum_Rate_Swing(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, Ldot_next = Ldot_next, Ldot_current = Ldot_current, h = h, P = p_stance, P_TangentX = p_stance_TangentX, P_TangentY = p_stance_TangentY, CoM_k = CoM_k, F1_k = FR1_k, F2_k = FR2_k, F3_k = FR3_k, F4_k = FR4_k)
                        #Zero Forces (Left Foot)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k)
                        #Unilateral Constraints on Right Foot p_stance
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainNorm = p_stance_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainNorm = p_stance_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainNorm = p_stance_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainNorm = p_stance_Norm)
                        #Friction Cone Constraint on Right Foot p_stance
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)

                    elif StepCnt%2 == 1: #Odd Number of Steps
                        #Case 1
                        #If First Level Swing the Left, then the second level Odd Number Steps Swing the Left
                        #Right foot is the stance foot
                        #Left foot is floating
                        #Kinematics Constraint
                        #CoM in the Right (p_stance)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_stance)
                        ##Angular Dynamics (Right Stance)
                        #if k<N_K-1:
                        #    g, glb, gub = Angular_Momentum_Rate_Swing(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, Ldot_next = Ldot_next, Ldot_current = Ldot_current, h = h, P = p_stance, P_TangentX = p_stance_TangentX, P_TangentY = p_stance_TangentY, CoM_k = CoM_k, F1_k = FR1_k, F2_k = FR2_k, F3_k = FR3_k, F4_k = FR4_k)
                        #Zero Forces (Left Foot)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k)
                        #Unilateral Constraints on Right Foot p_stance
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainNorm = p_stance_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainNorm = p_stance_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainNorm = p_stance_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainNorm = p_stance_Norm)
                        #Friction Cone Constraint on Right Foot p_stance
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)

                        #Case 2
                        #If First Level Swing the Right, then the second level Odd Number Steps Swing the Right
                        #Left foot is the stance foot
                        #Right foot is floating
                        #Kinematics Constraint
                        #CoM in the Left (p_stance)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_stance)
                        ##Angular Dynamics (Left Stance)
                        #if k<N_K-1:
                        #    g, glb, gub = Angular_Momentum_Rate_Swing(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, Ldot_next = Ldot_next, Ldot_current = Ldot_current, h = h, P = p_stance, P_TangentX = p_stance_TangentX, P_TangentY = p_stance_TangentY, CoM_k = CoM_k, F1_k = FL1_k, F2_k = FL2_k, F3_k = FL3_k, F4_k = FL4_k)
                        #Zero Forces (Right Foot)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k)
                        #Unilateral Constraints on Left Foot p_stance
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainNorm = p_stance_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainNorm = p_stance_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainNorm = p_stance_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainNorm = p_stance_Norm)
                        #Friction Cone Constraint on Left Foot p_stance
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)

                elif GaitPattern[Nph]=='DoubleSupport':
                    #Get contact location
                    if StepCnt == 0: #Special Case for the First Step (NOTE: Step 0)
                        p_stationary = ca.vertcat(px_init,py_init,pz_init)
                        p_stationary_TangentX = SurfTangentsX[0:3]
                        p_stationary_TangentY = SurfTangentsY[0:3]
                        p_stationary_Norm = SurfNorms[0:3]

                        p_land = ca.vertcat(px[StepCnt],py[StepCnt],pz[StepCnt])
                        p_land_TangentX = SurfTangentsX[(StepCnt+1)*3:(StepCnt+1)*3+3]
                        p_land_TangentY = SurfTangentsX[(StepCnt+1)*3:(StepCnt+1)*3+3]
                        p_land_Norm = SurfTangentsX[(StepCnt+1)*3:(StepCnt+1)*3+3]
                
                    else: #For other steps, indexed as 1,2,3,4
                        p_stationary = ca.vertcat(px[StepCnt-1],py[StepCnt-1],pz[StepCnt-1])
                        p_stationary_TangentX = SurfTangentsX[StepCnt*3:StepCnt*3+3]
                        p_stationary_TangentY = SurfTangentsX[StepCnt*3:StepCnt*3+3]
                        p_stationary_Norm = SurfTangentsX[StepCnt*3:StepCnt*3+3]

                        p_land = ca.vertcat(px[StepCnt],py[StepCnt],pz[StepCnt])
                        p_land_TangentX = SurfTangentsX[(StepCnt+1)*3:(StepCnt+1)*3+3]
                        p_land_TangentY = SurfTangentsX[(StepCnt+1)*3:(StepCnt+1)*3+3]
                        p_land_Norm = SurfTangentsX[(StepCnt+1)*3:(StepCnt+1)*3+3]
                
                    #Give Constraint according to even and odd steps
                    if StepCnt%2 == 0: #Even Number of Steps
                        #Case 1
                        #If First Level Swing the Left, then the second level Even Steps Swing the Right
                        #In Double Support Phase
                        #Left Foot is the Stationary
                        #Right Foot is the Land
                        #Kinemactics Constraint
                        #CoM in the Left (p_stationary)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_stationary)
                        #CoM in the Right (p_land)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_land)
                        ##Angular Dynamics
                        #if k<N_K-1:
                        #    g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, Ldot_next = Ldot_next, Ldot_current = Ldot_current, h = h, PL = p_stationary, PL_TangentX = p_stationary_TangentX, PL_TangentY = p_stationary_TangentY, PR = p_land, PR_TangentX = p_land_TangentX, PR_TangentY = p_land_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                        #Unilateral Constraint
                        #Left foot obey the unilateral constraint on p_stationary
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainNorm = p_stationary_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainNorm = p_stationary_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainNorm = p_stationary_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainNorm = p_stationary_Norm)
                        #then the Right foot is obey the unilateral constraint on the Stationary foot p_land
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainNorm = p_land_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainNorm = p_land_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainNorm = p_land_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainNorm = p_land_Norm)
                        #Friction Cone Constraint
                        #Left foot obey the friction cone constraint on p_stationary
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        #then the right foot obeys the friction cone constraints on the on p_land
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        
                        #Case 2
                        #If First Level Swing the Right, then the second level Even Steps Swing the Left
                        #In Double Support Phase
                        #Right Foot is the Stationary
                        #Left Foot is the Land
                        #Kinemactics Constraint
                        #CoM in the Left (p_land)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_land)
                        #CoM in the Right (p_stationary)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_stationary)
                        ##Angular Dynamics
                        #if k<N_K-1:
                        #    g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, Ldot_next = Ldot_next, Ldot_current = Ldot_current, h = h, PL = p_land, PL_TangentX = p_land_TangentX, PL_TangentY = p_land_TangentY, PR = p_stationary, PR_TangentX = p_stationary_TangentX, PR_TangentY = p_stationary_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                        #Unilateral Constraint
                        #Left foot obey the unilateral constraint on p_land
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainNorm = p_land_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainNorm = p_land_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainNorm = p_land_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainNorm = p_land_Norm)
                        #then the Right foot is obey the unilateral constraint on the Stationary foot p_stationary
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainNorm = p_stationary_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainNorm = p_stationary_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainNorm = p_stationary_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainNorm = p_stationary_Norm)
                        #Friction Cone Constraint
                        #Left foot obey the friction cone constraint on p_land
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        #then the right foot obeys the friction cone constraints on the on p_stationary
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        
                    elif StepCnt%2 == 1:#Odd Number of Steps
                        #Case 1
                        #If First Level Swing the Left, then the second level Odd Steps Swing the Left
                        #In Double Support Phase
                        #Right Foot is the Stationary
                        #Left Foot is the Land
                        #Kinemactics Constraint
                        #CoM in the Left (p_land)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_land)
                        #CoM in the Right (p_stationary)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_stationary)
                        ##Angular Dynamics
                        #if k<N_K-1:
                        #    g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, Ldot_next = Ldot_next, Ldot_current = Ldot_current, h = h, PL = p_land, PL_TangentX = p_land_TangentX, PL_TangentY = p_land_TangentY, PR = p_stationary, PR_TangentX = p_stationary_TangentX, PR_TangentY = p_stationary_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                        #Unilateral Constraint
                        #Left foot obey the unilateral constraint on p_land
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainNorm = p_land_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainNorm = p_land_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainNorm = p_land_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainNorm = p_land_Norm)
                        #then the Right foot is obey the unilateral constraint on the Stationary foot p_stationary
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainNorm = p_stationary_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainNorm = p_stationary_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainNorm = p_stationary_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainNorm = p_stationary_Norm)
                        #Friction Cone Constraint
                        #Left foot obey the friction cone constraint on p_land
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        #then the right foot obeys the friction cone constraints on the on p_stationary
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        
                        #Case 2
                        #If First Level Swing the Right, then the second level Odd Steps Swing the Right
                        #In Double Support Phase
                        #Left Foot is the Stationary
                        #Right Foot is the Land
                        #Kinematics Constraint
                        #CoM in the Left (p_stationary)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_stationary)
                        #CoM in the Right (p_land)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_land)
                        ##Angular Dynamics
                        #if k<N_K-1:
                        #    g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, Ldot_next = Ldot_next, Ldot_current = Ldot_current, h = h, PL = p_stationary, PL_TangentX = p_stationary_TangentX, PL_TangentY = p_stationary_TangentY, PR = p_land, PR_TangentX = p_land_TangentX, PR_TangentY = p_land_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                        #Unilateral Constraint
                        #Left foot obey the unilateral constraint on p_stationary
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainNorm = p_stationary_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainNorm = p_stationary_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainNorm = p_stationary_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainNorm = p_stationary_Norm)
                        #then the Right foot is obey the unilateral constraint on the Stationary foot p_land
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainNorm = p_land_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainNorm = p_land_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainNorm = p_land_Norm)
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainNorm = p_land_Norm)
                        #Friction Cone Constraint
                        #Left foot obey the friction cone constraint on p_stationary
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        #then the right foot obeys the friction cone constraints on the on p_land
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        
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

                ##First-order Angular Momentum Dynamics x-axis
                #g.append(Lx[k+1] - Lx[k] - h*Ldotx[k])
                #glb.append(np.array([0]))
                #gub.append(np.array([0]))

                ##First-order Angular Momentum Dynamics y-axis
                #g.append(Ly[k+1] - Ly[k] - h*Ldoty[k])
                #glb.append(np.array([0]))
                #gub.append(np.array([0]))

                ##First-order Angular Momentum Dynamics z-axis
                #g.append(Lz[k+1] - Lz[k] - h*Ldotz[k])
                #glb.append(np.array([0]))
                #gub.append(np.array([0]))

                #Second-order Dynamics x-axis
                g.append(xdot[k+1] - xdot[k] - h*(FL1x[k]/m+FL2x[k]/m+FL3x[k]/m+FL4x[k]/m+FR1x[k]/m+FR2x[k]/m+FR3x[k]/m+FR4x[k]/m))
                glb.append(np.array([0]))
                gub.append(np.array([0]))

                #Second-order Dynamics y-axis
                g.append(ydot[k+1] - ydot[k] - h*(FL1y[k]/m+FL2y[k]/m+FL3y[k]/m+FL4y[k]/m+FR1y[k]/m+FR2y[k]/m+FR3y[k]/m+FR4y[k]/m))
                glb.append(np.array([0]))
                gub.append(np.array([0]))

                #Second-order Dynamics z-axis
                g.append(zdot[k+1] - zdot[k] - h*(FL1z[k]/m+FL2z[k]/m+FL3z[k]/m+FL4z[k]/m+FR1z[k]/m+FR2z[k]/m+FR3z[k]/m+FR4z[k]/m - G))
                glb.append(np.array([0]))
                gub.append(np.array([0]))
            
            #Add Cost Terms
            if k < N_K - 1:
                #Acc Only
                J = J + h*(FL1x[k]/m+FL2x[k]/m+FL3x[k]/m+FL4x[k]/m+FR1x[k]/m+FR2x[k]/m+FR3x[k]/m+FR4x[k]/m)**2 + h*(FL1y[k]/m+FL2y[k]/m+FL3y[k]/m+FL4y[k]/m+FR1y[k]/m+FR2y[k]/m+FR3y[k]/m+FR4y[k]/m)**2 + h*(FL1z[k]/m+FL2z[k]/m+FL3z[k]/m+FL4z[k]/m+FR1z[k]/m+FR2z[k]/m+FR3z[k]/m+FR4z[k]/m - G)**2

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
            #Also, the stationary foot Left should stay in the polytope of the landed swing foot - RIGHT
            #NOTE: current - next now
            g.append(ca.if_else(ParaLeftSwingFlag,Q_lf_in_rf@(P_k_current-P_k_next)-ca.DM(q_lf_in_rf),np.full((len(q_lf_in_rf),),-1)))
            glb.append(np.full((len(q_lf_in_rf),),-np.inf))
            gub.append(np.full((len(q_lf_in_rf),),0))

            #If First Levvel Swing the Right, then the second level Even Number Phases (the first phase) Swing the Left -> Right -> Left
            #Right foot in contact for p_current, left foot is going to land at p_next
            #Relative Swing Foot Location (lf in rf)
            g.append(ca.if_else(ParaRightSwingFlag,Q_lf_in_rf@(P_k_next-P_k_current)-ca.DM(q_lf_in_rf),np.full((len(q_lf_in_rf),),-1)))
            glb.append(np.full((len(q_lf_in_rf),),-np.inf))
            gub.append(np.full((len(q_lf_in_rf),),0))
            #Also, the stationary foot Rigth should stay in the polytope of the landed swing foot - Left
            #NOTE: current - next now
            g.append(ca.if_else(ParaRightSwingFlag,Q_rf_in_lf@(P_k_current-P_k_next)-ca.DM(q_rf_in_lf),np.full((len(q_rf_in_lf),),-1)))
            glb.append(np.full((len(q_rf_in_lf),),-np.inf))
            gub.append(np.full((len(q_rf_in_lf),),0))

        elif step_cnt%2 == 1: #odd number steps
            #If the first level swings the Left, then the second level for Odd Number Phases (the second phase) swings left -> Right -> Left
            #Right foot in contact for p_current, left foot is going to land at p_next
            #Relative Swing Foot Location (lf in rf)
            g.append(ca.if_else(ParaLeftSwingFlag,Q_lf_in_rf@(P_k_next-P_k_current)-ca.DM(q_lf_in_rf),np.full((len(q_lf_in_rf),),-1)))
            glb.append(np.full((len(q_lf_in_rf),),-np.inf))
            gub.append(np.full((len(q_lf_in_rf),),0))
            #Also, the stationary foot Rigth should stay in the polytope of the landed swing foot - Left
            #NOTE: current - next now
            g.append(ca.if_else(ParaLeftSwingFlag,Q_rf_in_lf@(P_k_current-P_k_next)-ca.DM(q_rf_in_lf),np.full((len(q_rf_in_lf),),-1)))
            glb.append(np.full((len(q_rf_in_lf),),-np.inf))
            gub.append(np.full((len(q_rf_in_lf),),0))

            #If the first level swings the Right, then the second level for Odd Number Phases (the second phase) swings Right -> Left -> Right
            #Left foot in contact for p_current, right foot is going to land as p_next
            #Relative Swing Foot Location (rf in lf)
            g.append(ca.if_else(ParaRightSwingFlag,Q_rf_in_lf@(P_k_next-P_k_current)-ca.DM(q_rf_in_lf),np.full((len(q_rf_in_lf),),-1)))
            glb.append(np.full((len(q_rf_in_lf),),-np.inf))
            gub.append(np.full((len(q_rf_in_lf),),0))
            #Also, the stationary foot LEFT should stay in the polytope of the landed swing foot - RIGHT
            #NOTE: current - next now
            g.append(ca.if_else(ParaRightSwingFlag,Q_lf_in_rf@(P_k_current-P_k_next)-ca.DM(q_lf_in_rf),np.full((len(q_lf_in_rf),),-1)))
            glb.append(np.full((len(q_lf_in_rf),),-np.inf))
            gub.append(np.full((len(q_lf_in_rf),),0))

    #Foot Step Constraint
    #FootStep Constraint
    #P3----------------P1
    #|                  |
    #|                  |
    #|                  |
    #P4----------------P2
    for PatchNum in range(Nsteps):
        #Get Footstep Vector
        P_vector = ca.vertcat(px[PatchNum],py[PatchNum],pz[PatchNum])

        #Get Half Space Representation 
        #NOTE: In the second level, the terrain patch start from the second patch, indexed as 1
        SurfParaTemp = SurfParas[20+PatchNum*20:19+(PatchNum+1)*20+1]
        #print(SurfParaTemp)
        SurfK = SurfParaTemp[0:11+1]
        SurfK = ca.reshape(SurfK,3,4)
        SurfK = SurfK.T #NOTE: This process is due to casadi naming convention to have first row to be x1,x2,x3
        SurfE = SurfParaTemp[11+1:11+3+1]
        Surfk = SurfParaTemp[14+1:14+4+1]
        Surfe = SurfParaTemp[-1]

        #Terrain Tangent and Norms
        P_vector_TangentX = SurfTangentsX[(PatchNum+1)*3:(PatchNum+1)*3+3]
        P_vector_TangentY = SurfTangentsY[(PatchNum+1)*3:(PatchNum+1)*3+3]

        #Contact Point 1
        #Inequality
        g.append(SurfK@(P_vector + 0.11*P_vector_TangentX + 0.06*P_vector_TangentY) - Surfk)
        glb.append(np.full((4,),-np.inf))
        gub.append(np.full((4,),0))
        #Equality
        g.append(SurfE.T@(P_vector + 0.11*P_vector_TangentX + 0.06*P_vector_TangentY) - Surfe)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        #Contact Point 2
        #Inequality
        g.append(SurfK@(P_vector + 0.11*P_vector_TangentX - 0.06*P_vector_TangentY) - Surfk)
        glb.append(np.full((4,),-np.inf))
        gub.append(np.full((4,),0))
        #Equality
        g.append(SurfE.T@(P_vector + 0.11*P_vector_TangentX - 0.06*P_vector_TangentY) - Surfe)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        #Contact Point 3
        #Inequality
        g.append(SurfK@(P_vector - 0.11*P_vector_TangentX + 0.06*P_vector_TangentY) - Surfk)
        glb.append(np.full((4,),-np.inf))
        gub.append(np.full((4,),0))
        #Equality
        g.append(SurfE.T@(P_vector - 0.11*P_vector_TangentX + 0.06*P_vector_TangentY) - Surfe)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        #Contact Point 4
        #Inequality
        g.append(SurfK@(P_vector + 0.11*P_vector_TangentX + 0.06*P_vector_TangentY) - Surfk)
        glb.append(np.full((4,),-np.inf))
        gub.append(np.full((4,),0))
        #Equality
        g.append(SurfE.T@(P_vector + 0.11*P_vector_TangentX + 0.06*P_vector_TangentY) - Surfe)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        ##FootStep Constraint
        ##Inequality
        #g.append(SurfK@P_vector - Surfk)
        #glb.append(np.full((4,),-np.inf))
        #gub.append(np.full((4,),0))
        #print(FirstSurfK@p_next - FirstSurfk)

        ##Equality
        #g.append(SurfE.T@P_vector - Surfe)
        #glb.append(np.array([0]))
        #gub.append(np.array([0]))

    #Approximate Kinematics Constraint --- Disallow over-crossing of footsteps from y =0

    if CentralY == True:

        for step_cnt in range(Nsteps):
            P_k_next = ca.vertcat(px[step_cnt],py[step_cnt],pz[step_cnt])
    
            if step_cnt%2 == 0: #even number steps
                #If First Level Swing the Left, then the second level Even Number Phases (the first Phase) Swing the Right -> Left -> Right
                #Left foot in contact for p_current, right foot is going to land as p_next
                g.append(ca.if_else(ParaLeftSwingFlag,P_k_next[1],np.array([-1])))
                glb.append(np.array([-np.inf]))
                gub.append(np.array([-py_lower_limit]))
    
                #If First Levvel Swing the Right, then the second level Even Number Phases (the first phase) Swing the Left -> Right -> Left
                #Right foot in contact for p_current, left foot is going to land at p_next
                g.append(ca.if_else(ParaRightSwingFlag,P_k_next[1],np.array([1])))
                glb.append(np.array([py_lower_limit]))
                gub.append(np.array([np.inf]))
    
            elif step_cnt%2 == 1: #odd number steps
                #If the first level swings the Left, then the second level for Odd Number Phases (the second phase) swings left -> Right -> Left
                #Right foot in contact for p_current, left foot is going to land at p_next
                g.append(ca.if_else(ParaLeftSwingFlag,P_k_next[1],np.array([1])))
                glb.append(np.array([py_lower_limit]))
                gub.append(np.array([np.inf]))
    
                #If the first level swings the Right, then the second level for Odd Number Phases (the second phase) swings Right -> Left -> Right
                #Left foot in contact for p_current, right foot is going to land as p_next
                g.append(ca.if_else(ParaRightSwingFlag,P_k_next[1],np.array([-1])))
                glb.append(np.array([-np.inf]))
                gub.append(np.array([-py_lower_limit]))

    #Switching Time Constraint
    for phase_cnt in range(Nphase):
        if GaitPattern[phase_cnt] == 'InitialDouble':
            if phase_cnt == 0:
                g.append(Ts[phase_cnt] - 0)
                glb.append(np.array([0.1])) #0.1-0.3
                gub.append(np.array([0.3]))
            else:
                g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
                glb.append(np.array([0.1]))
                gub.append(np.array([0.3]))

        elif GaitPattern[phase_cnt] == 'Swing':
            g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
            glb.append(np.array([0.5]))
            gub.append(np.array([0.7]))

            #if phase_cnt == 0:
            #    g.append(Ts[phase_cnt]-0)#0.6-1
            #    glb.append(np.array([0.5]))#0.5 for NLP success
            #    gub.append(np.array([0.7]))
            #else:
            #    g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
            #    glb.append(np.array([0.5]))
            #    gub.append(np.array([0.7]))

        elif GaitPattern[phase_cnt] == 'DoubleSupport':
            g.append(Ts[phase_cnt]-Ts[phase_cnt-1]) #0.1-0.9
            glb.append(np.array([0.1]))
            gub.append(np.array([0.3])) #0.1-0.3

    #-----------------------------------------------------------------------------------------------------------------------
    #Get Variable Index - !!!This is the pure Index, when try to get the array using other routines, we need to add "+1" at the last index due to Python indexing conventions
    x_index = (0,N_K-1) #First set of variables start counting from 0, The enumeration of the last knot is N_K-1
    y_index = (x_index[1]+1,x_index[1]+N_K)
    z_index = (y_index[1]+1,y_index[1]+N_K)
    xdot_index = (z_index[1]+1,z_index[1]+N_K)
    ydot_index = (xdot_index[1]+1,xdot_index[1]+N_K)
    zdot_index = (ydot_index[1]+1,ydot_index[1]+N_K)

    #Lx_index = (zdot_index[1]+1,zdot_index[1]+N_K)
    #Ly_index = (Lx_index[1]+1,Lx_index[1]+N_K)
    #Lz_index = (Ly_index[1]+1,Ly_index[1]+N_K)
    #Ldotx_index = (Lz_index[1]+1,Lz_index[1]+N_K)
    #Ldoty_index = (Ldotx_index[1]+1,Ldotx_index[1]+N_K)
    #Ldotz_index = (Ldoty_index[1]+1,Ldoty_index[1]+N_K)
    
    FL1x_index = (zdot_index[1]+1,zdot_index[1]+N_K)
    FL1y_index = (FL1x_index[1]+1,FL1x_index[1]+N_K)
    FL1z_index = (FL1y_index[1]+1,FL1y_index[1]+N_K)
    FL2x_index = (FL1z_index[1]+1,FL1z_index[1]+N_K)
    FL2y_index = (FL2x_index[1]+1,FL2x_index[1]+N_K)
    FL2z_index = (FL2y_index[1]+1,FL2y_index[1]+N_K)
    FL3x_index = (FL2z_index[1]+1,FL2z_index[1]+N_K)
    FL3y_index = (FL3x_index[1]+1,FL3x_index[1]+N_K)
    FL3z_index = (FL3y_index[1]+1,FL3y_index[1]+N_K)
    FL4x_index = (FL3z_index[1]+1,FL3z_index[1]+N_K)
    FL4y_index = (FL4x_index[1]+1,FL4x_index[1]+N_K)
    FL4z_index = (FL4y_index[1]+1,FL4y_index[1]+N_K)
    FR1x_index = (FL4z_index[1]+1,FL4z_index[1]+N_K)
    FR1y_index = (FR1x_index[1]+1,FR1x_index[1]+N_K)
    FR1z_index = (FR1y_index[1]+1,FR1y_index[1]+N_K)
    FR2x_index = (FR1z_index[1]+1,FR1z_index[1]+N_K)
    FR2y_index = (FR2x_index[1]+1,FR2x_index[1]+N_K)
    FR2z_index = (FR2y_index[1]+1,FR2y_index[1]+N_K)
    FR3x_index = (FR2z_index[1]+1,FR2z_index[1]+N_K)
    FR3y_index = (FR3x_index[1]+1,FR3x_index[1]+N_K)
    FR3z_index = (FR3y_index[1]+1,FR3y_index[1]+N_K)
    FR4x_index = (FR3z_index[1]+1,FR3z_index[1]+N_K)
    FR4y_index = (FR4x_index[1]+1,FR4x_index[1]+N_K)
    FR4z_index = (FR4y_index[1]+1,FR4y_index[1]+N_K)
    px_init_index = (FR4z_index[1]+1,FR4z_index[1]+1)
    py_init_index = (px_init_index[1]+1,px_init_index[1]+1)
    pz_init_index = (py_init_index[1]+1,py_init_index[1]+1)
    px_index = (pz_init_index[1]+1,pz_init_index[1]+Nsteps)
    py_index = (px_index[1]+1,px_index[1]+Nsteps)
    pz_index = (py_index[1]+1,py_index[1]+Nsteps)
    Ts_index = (pz_index[1]+1,pz_index[1]+Nphase)

    var_index = {"x":x_index,
                 "y":y_index,
                 "z":z_index,
                 "xdot":xdot_index,
                 "ydot":ydot_index,
                 "zdot":zdot_index,
                 #"Lx":Lx_index,
                 #"Ly":Ly_index,
                 #"Lz":Lz_index,
                 #"Ldotx":Ldotx_index,
                 #"Ldoty":Ldoty_index,
                 #"Ldotz":Ldotz_index,
                 "FL1x":FL1x_index,
                 "FL1y":FL1y_index,
                 "FL1z":FL1z_index,
                 "FL2x":FL2x_index,
                 "FL2y":FL2y_index,
                 "FL2z":FL2z_index,
                 "FL3x":FL3x_index,
                 "FL3y":FL3y_index,
                 "FL3z":FL3z_index,
                 "FL4x":FL4x_index,
                 "FL4y":FL4y_index,
                 "FL4z":FL4z_index,
                 "FR1x":FR1x_index,
                 "FR1y":FR1y_index,
                 "FR1z":FR1z_index,
                 "FR2x":FR2x_index,
                 "FR2y":FR2y_index,
                 "FR2z":FR2z_index,
                 "FR3x":FR3x_index,
                 "FR3y":FR3y_index,
                 "FR3z":FR3z_index,
                 "FR4x":FR4x_index,
                 "FR4y":FR4y_index,
                 "FR4z":FR4z_index,
                 "px_init":px_init_index,
                 "py_init":py_init_index,
                 "pz_init":pz_init_index,
                 "px":px_index,
                 "py":py_index,
                 "pz":pz_index,
                 "Ts":Ts_index,
    }

    #print(DecisionVars[var_index["px_init"][0]:var_index["px_init"][1]+1])

    return DecisionVars, DecisionVars_lb, DecisionVars_ub, J, g, glb, gub, var_index

def CoM_Dynamics_SinglePoint(m = 95, Nk_Local = 7, Nsteps = 1, StandAlong = True, StaticStop = False, ParameterList = None, CentralY = False):
    #-----------------------------------------------------------------------------------------------------------------------
    #Define Parameters
    #   Gait Pattern, Each action is followed up by a double support phase
    GaitPattern = ["InitialDouble","Swing","DoubleSupport"] + ["InitialDouble", "Swing","DoubleSupport"]*(Nsteps-1) #,'RightSupport','DoubleSupport','LeftSupport','DoubleSupport'

    PhaseDurationVec = [0.3, 0.5, 0.3]*(Nsteps)

    #print(PhaseDurationVec)

    #   Number of Phases
    Nphase = len(GaitPattern)
    #   Number of Steps
    #Nstep = 1
    #   Number of Knots per Phase - how many intervals do we have for a single phase; 10 intervals need 11 knots/ticks
    #Nk_Local= 5
    #   Compute Number of Total knots/ticks, but the enumeration start from 0 to N_K-1
    N_K = Nk_Local*Nphase + 1 #+1 the last knots to finalize the plan
    #   Robot mass
    #m = 95 #kg
    G = 9.80665 #kg/m^2
    #   Terrain Model
    #       Flat Terrain
    #TerrainNorm = [0,0,1] 
    #TerrainTangentX = [1,0,0]
    #TerrainTangentY = [0,1,0]
    miu = 0.3
    #   Force Limits
    F_bound = 400
    Fxlb = -F_bound*4
    Fxub = F_bound*4
    Fylb = -F_bound*4
    Fyub = F_bound*4
    Fzlb = -F_bound*4
    Fzub = F_bound*4
    #Minimum y-axis foot location
    py_lower_limit = 0.04
    #Lowest Z
    z_lowest = 0.7
    z_highest = 0.85
    #-----------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------
    #Kinematics Constraint for Talos
    #kinematicConstraints = genKinematicConstraints(left_foot_constraints,right_foot_constraints)
    K_CoM_Right,k_CoM_Right = right_foot_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]))
    K_CoM_Left,k_CoM_Left = left_foot_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]))

    #K_CoM_Left = kinematicConstraints[0][0]
    #k_CoM_Left = kinematicConstraints[0][1]
    #K_CoM_Right = kinematicConstraints[1][0]
    #k_CoM_Right = kinematicConstraints[1][1]
    #Relative Foot Constraint matrices
    
    #relativeConstraints = genFootRelativeConstraints(right_foot_in_lf_frame_constraints,left_foot_in_rf_frame_constraints)
    Q_rf_in_lf,q_rf_in_lf = right_foot_in_lf_frame_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]))
    Q_lf_in_rf,q_lf_in_rf = left_foot_in_rf_frame_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]))

    #Q_rf_in_lf = relativeConstraints[0][0] #named lf in rf, but representing rf in lf
    #q_rf_in_lf = relativeConstraints[0][1] #named lf in rf, but representing rf in lf
    #Q_lf_in_rf = relativeConstraints[1][0] #named rf in lf, but representing lf in rf
    #q_lf_in_rf = relativeConstraints[1][1] #named rf in lf, but representing lf in rf
    #-----------------------------------------------------------------------------------------------------------------------
    #Define Initial and Terminal Conditions
    #x_init = ca.SX.sym('x_init')
    #y_init = ca.SX.sym('y_init')
    #z_init = ca.SX.sym('z_init')

    x_init = ParameterList["x_init"]
    y_init = ParameterList["y_init"]
    z_init = ParameterList["z_init"]

    xdot_init = ParameterList["xdot_init"]
    ydot_init = ParameterList["ydot_init"]
    zdot_init = ParameterList["zdot_init"]

    #Lx_init = 0
    #Ly_init = 0
    #Lz_init = 0

    #Ldotx_init = 0
    #Ldoty_init = 0
    #Ldotz_init = 0

    PLx_init = ParameterList["PLx_init"]
    PLy_init = ParameterList["PLy_init"]
    PLz_init = ParameterList["PLz_init"]
    PL_init = ca.vertcat(PLx_init,PLy_init,PLz_init)

    PRx_init = ParameterList["PRx_init"]
    PRy_init = ParameterList["PRy_init"]
    PRz_init = ParameterList["PRz_init"]
    PR_init = ca.vertcat(PRx_init,PRy_init,PRz_init)

    x_end = ParameterList["x_end"]
    y_end = ParameterList["y_end"]
    z_end = ParameterList["z_end"]

    xdot_end = ParameterList["xdot_end"]
    ydot_end = ParameterList["ydot_end"]
    zdot_end = ParameterList["zdot_end"]

    #Lx_end = 0
    #Ly_end = 0
    #Lz_end = 0

    #Ldotx_end = 0
    #Ldoty_end = 0
    #Ldotz_end = 0

    #Flags for Swing Legs (Defined as Parameters)
    ParaLeftSwingFlag = ParameterList["LeftSwingFlag"]
    ParaRightSwingFlag = ParameterList["RightSwingFlag"]

    #Surfaces (Only the Second One)
    #Surface Patches
    SurfParas = ParameterList["SurfParas"]

    #Tangents and Norms
    #Initial Contact Norm and Tangents
    PL_init_Norm = ParameterList["PL_init_Norm"]
    PL_init_TangentX = ParameterList["PL_init_TangentX"]
    PL_init_TangentY = ParameterList["PL_init_TangentY"]
    PR_init_Norm = ParameterList["PR_init_Norm"]
    PR_init_TangentX = ParameterList["PR_init_TangentX"]
    PR_init_TangentY = ParameterList["PR_init_TangentY"]
    
    #Future Contact Norm and Tangents
    SurfNorms = ParameterList["SurfNorms"]                
    SurfTangentsX = ParameterList["SurfTangentsX"]
    SurfTangentsY = ParameterList["SurfTangentsY"]

    #Refrence Trajectories
    x_ref = ParameterList["x_ref"]
    y_ref = ParameterList["y_ref"]
    z_ref = ParameterList["z_ref"]
    xdot_ref = ParameterList["xdot_ref"]
    ydot_ref = ParameterList["ydot_ref"]
    zdot_ref = ParameterList["zdot_ref"]
    FLx_ref = ParameterList["FLx_ref"]
    FLy_ref = ParameterList["FLy_ref"]
    FLz_ref = ParameterList["FLz_ref"]
    FRx_ref = ParameterList["FRx_ref"]
    FRy_ref = ParameterList["FRy_ref"]
    FRz_ref = ParameterList["FRz_ref"]
    SwitchingTimeVec_ref = ParameterList["SwitchingTimeVec_ref"]
    Px_seq_ref = ParameterList["Px_seq_ref"]
    Py_seq_ref = ParameterList["Py_seq_ref"]
    Pz_seq_ref = ParameterList["Pz_seq_ref"]

    #-----------------------------------------------------------------------------------------------------------------------
    #Define Variables and Bounds, Parameters
    #   CoM Position x-axis
    x = ca.SX.sym('x',N_K)
    x_lb = np.array([[0]*(x.shape[0]*x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    x_ub = np.array([[30]*(x.shape[0]*x.shape[1])])
    #   CoM Position y-axis
    y = ca.SX.sym('y',N_K)
    y_lb = np.array([[-1]*(y.shape[0]*y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    y_ub = np.array([[1]*(y.shape[0]*y.shape[1])])
    #   CoM Position z-axis
    z = ca.SX.sym('z',N_K)
    z_lb = np.array([[z_lowest]*(z.shape[0]*z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    z_ub = np.array([[z_highest]*(z.shape[0]*z.shape[1])])
    #   CoM Velocity x-axis
    xdot = ca.SX.sym('xdot',N_K)
    xdot_lb = np.array([[-1.5]*(xdot.shape[0]*xdot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    xdot_ub = np.array([[1.5]*(xdot.shape[0]*xdot.shape[1])])
    #   CoM Velocity y-axis
    ydot = ca.SX.sym('ydot',N_K)
    ydot_lb = np.array([[-1.5]*(ydot.shape[0]*ydot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    ydot_ub = np.array([[1.5]*(ydot.shape[0]*ydot.shape[1])])
    #   CoM Velocity z-axis
    zdot = ca.SX.sym('zdot',N_K)
    zdot_lb = np.array([[-1.5]*(zdot.shape[0]*zdot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    zdot_ub = np.array([[1.5]*(zdot.shape[0]*zdot.shape[1])])

    #left Foot Forces
    #Left Foot Contact Point 1 x-axis
    FLx = ca.SX.sym('FLx',N_K)
    FLx_lb = np.array([[Fxlb]*(FLx.shape[0]*FLx.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FLx_ub = np.array([[Fxub]*(FLx.shape[0]*FLx.shape[1])])
    #Left Foot Contact Point 1 y-axis
    FLy = ca.SX.sym('FLy',N_K)
    FLy_lb = np.array([[Fylb]*(FLy.shape[0]*FLy.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FLy_ub = np.array([[Fyub]*(FLy.shape[0]*FLy.shape[1])])
    #Left Foot Contact Point 1 z-axis
    FLz = ca.SX.sym('FLz',N_K)
    FLz_lb = np.array([[Fzlb]*(FLz.shape[0]*FLz.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FLz_ub = np.array([[Fzub]*(FLz.shape[0]*FLz.shape[1])])

    #Right Contact Force x-axis
    #Right Foot Contact Point 1 x-axis
    FRx = ca.SX.sym('FRx',N_K)
    FRx_lb = np.array([[Fxlb]*(FRx.shape[0]*FRx.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FRx_ub = np.array([[Fxub]*(FRx.shape[0]*FRx.shape[1])])
    #Right Foot Contact Point 1 y-axis
    FRy = ca.SX.sym('FRy',N_K)
    FRy_lb = np.array([[Fylb]*(FRy.shape[0]*FRy.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FRy_ub = np.array([[Fyub]*(FRy.shape[0]*FRy.shape[1])])
    #Right Foot Contact Point 1 z-axis
    FRz = ca.SX.sym('FRz',N_K)
    FRz_lb = np.array([[Fzlb]*(FRz.shape[0]*FRz.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FRz_ub = np.array([[Fzub]*(FRz.shape[0]*FRz.shape[1])])

    #Initial Contact Location (need to connect to the first level)
    #   Plx
    px_init = ca.SX.sym('px_init')
    px_init_lb = np.array([-1])
    px_init_ub = np.array([30])

    #   py
    py_init = ca.SX.sym('py_init')
    py_init_lb = np.array([-1])
    py_init_ub = np.array([1])

    #   pz
    pz_init = ca.SX.sym('pz_init')
    pz_init_lb = np.array([-5])
    pz_init_ub = np.array([5])

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
        pxtemp = ca.SX.sym('px'+str(stepIdx)) #0 + 1
        px.append(pxtemp)
        px_lb.append(np.array([-1]))
        px_ub.append(np.array([30]))

        pytemp = ca.SX.sym('py'+str(stepIdx))
        py.append(pytemp)
        py_lb.append(np.array([-1]))
        py_ub.append(np.array([1]))

        #   Foot steps are all staying on the ground
        pztemp = ca.SX.sym('pz'+str(stepIdx))
        pz.append(pztemp)
        pz_lb.append(np.array([-5]))
        pz_ub.append(np.array([5]))

    #Switching Time Vector
    Ts = []
    Ts_lb = []
    Ts_ub = []
    for n_phase in range(Nphase):
        Tstemp = ca.SX.sym('Ts'+str(n_phase+1)) #0 + 1 + ....
        Ts.append(Tstemp)
        Ts_lb.append(np.array([0.05]))
        Ts_ub.append(np.array([2.0*(Nphase+1)]))

    #   Collect all Decision Variables
    #DecisionVars = ca.vertcat(x,y,z,xdot,ydot,zdot,Lx,Ly,Lz,Ldotx,Ldoty,Ldotz,FL1x,FL1y,FL1z,FL2x,FL2y,FL2z,FL3x,FL3y,FL3z,FL4x,FL4y,FL4z,FR1x,FR1y,FR1z,FR2x,FR2y,FR2z,FR3x,FR3y,FR3z,FR4x,FR4y,FR4z,px_init,py_init,pz_init,*px,*py,*pz,*Ts)
    DecisionVars = ca.vertcat(x,y,z,xdot,ydot,zdot,FLx,FLy,FLz,FRx,FRy,FRz,px_init,py_init,pz_init,*px,*py,*pz,*Ts)
    #print(DecisionVars)
    DecisionVarsShape = DecisionVars.shape

    #   Collect all lower bound and upper bound
    #DecisionVars_lb = np.concatenate(((x_lb,y_lb,z_lb,xdot_lb,ydot_lb,zdot_lb,Lx_lb,Ly_lb,Lz_lb,Ldotx_lb,Ldoty_lb,Ldotz_lb,FL1x_lb,FL1y_lb,FL1z_lb,FL2x_lb,FL2y_lb,FL2z_lb,FL3x_lb,FL3y_lb,FL3z_lb,FL4x_lb,FL4y_lb,FL4z_lb,FR1x_lb,FR1y_lb,FR1z_lb,FR2x_lb,FR2y_lb,FR2z_lb,FR3x_lb,FR3y_lb,FR3z_lb,FR4x_lb,FR4y_lb,FR4z_lb,px_init_lb,py_init_lb,pz_init_lb,px_lb,py_lb,pz_lb,Ts_lb)),axis=None)
    #DecisionVars_ub = np.concatenate(((x_ub,y_ub,z_ub,xdot_ub,ydot_ub,zdot_ub,Lx_ub,Ly_ub,Lz_ub,Ldotx_ub,Ldoty_ub,Ldotz_ub,FL1x_ub,FL1y_ub,FL1z_ub,FL2x_ub,FL2y_ub,FL2z_ub,FL3x_ub,FL3y_ub,FL3z_ub,FL4x_ub,FL4y_ub,FL4z_ub,FR1x_ub,FR1y_ub,FR1z_ub,FR2x_ub,FR2y_ub,FR2z_ub,FR3x_ub,FR3y_ub,FR3z_ub,FR4x_ub,FR4y_ub,FR4z_ub,px_init_ub,py_init_ub,pz_init_ub,px_ub,py_ub,pz_ub,Ts_ub)),axis=None)
    DecisionVars_lb = np.concatenate(((x_lb,y_lb,z_lb,xdot_lb,ydot_lb,zdot_lb,FLx_lb,FLy_lb,FLz_lb,FRx_lb,FRy_lb,FRz_lb,px_init_lb,py_init_lb,pz_init_lb,px_lb,py_lb,pz_lb,Ts_lb)),axis=None)
    DecisionVars_ub = np.concatenate(((x_ub,y_ub,z_ub,xdot_ub,ydot_ub,zdot_ub,FLx_ub,FLy_ub,FLz_ub,FRx_ub,FRy_ub,FRz_ub,px_init_ub,py_init_ub,pz_init_ub,px_ub,py_ub,pz_ub,Ts_ub)),axis=None)

    #Time Span Setup
    tau_upper_limit = 1
    tauStepLength = tau_upper_limit/(N_K-1) #Get the interval length, total number of knots - 1

    #-----------------------------------------------------------------------------------------------------------------------
    #Define Constrains and Running Cost
    g = []
    glb = []
    gub = []
    J = 0

    #Initial and Terminal Condition

    ##   Terminal CoM y-axis
    #g.append(y[-1]-y_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    ##   Terminal CoM z-axis
    #g.append(z[-1]-z_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    #if StaticStop == True:
    #    #   Terminal Zero CoM velocity x-axis
    #    g.append(xdot[-1])
    #    glb.append(np.array([0]))
    #    gub.append(np.array([0]))

    #    #   Terminal Zero CoM velocity y-axis
    #    g.append(ydot[-1])
    #    glb.append(np.array([0]))
    #    gub.append(np.array([0]))

    #    #   Terminal Zero CoM velocity z-axis
    #    g.append(zdot[-1])
    #    glb.append(np.array([0]))
    #    gub.append(np.array([0]))

    #   Terminal Angular Momentum x-axis
    #g.append(Lx[-1]-Lx_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    #   Terminal Angular Momentum y-axis
    #g.append(Ly[-1]-Ly_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    #   Terminal Angular Momentum z-axis
    #g.append(Lz[-1]-Lz_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    #   Terminal Angular Momentum Rate x-axis
    #g.append(Ldotx[-1]-Ldotx_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    #   Terminal Angular Momentum Rate y-axis
    #g.append(Ldoty[-1]-Ldoty_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    #   Terminal Angular Momentum Rate z-axis
    #g.append(Ldotz[-1]-Ldotz_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))


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

        #Fixed Time Step
        #h = (PhaseDurationVec[Nph])/Nk_Local
        h = (SwitchingTimeVec_ref[Nph])/Nk_Local
        
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
            ##   Angular Momentum
            #if k<N_K-1: #N_K-1 the enumeration of the last knot, k<N_K-1 the one before the last knot
            #    Ldot_current = ca.vertcat(Ldotx[k],Ldoty[k],Ldotz[k])
            #    Ldot_next = ca.vertcat(Ldotz[k+1],Ldotz[k+1],Ldotz[k+1])
            #-------------------------------------------

            #-------------------------------------------
            #Phase dependent Constraints (CoM Kinematics and Angular Dynamics)
            
            #Get Step Counter
            StepCnt = Nph//3
            #print("Knot ", k)
            #print("Belongs to Phase", Nph)
            #print("Phase Type ", GaitPattern[Nph])
            #print("Belongs to Step", StepCnt)

            #NOTE: The first phase (Initial Double) --- Needs special care
            if Nph == 0 and GaitPattern[Nph]=='InitialDouble':

                print("Knot ", k)
                print("Belongs to Phase", Nph)
                print("Phase Type ", GaitPattern[Nph])
                print("Belongs to Step", StepCnt)

                #initial support foot (the landing foot from the first phase)
                p_init = ca.vertcat(px_init,py_init,pz_init)
                p_init_TangentX = SurfTangentsX[0:3]
                p_init_TangentY = SurfTangentsY[0:3]
                p_init_Norm = SurfNorms[0:3]

                #Case 1
                #If First Level Swing the Left, the the 0 phase (InitDouble) has p_init as the left support, PR_init as the right support
                #Kinematics Constraint
                #CoM in Left (p_init)
                g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_init)
                #CoM in Right (PR_init)
                g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = PR_init)
                ##Angular Dynamics
                #if k<N_K-1:
                #    g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, Ldot_next = Ldot_next, Ldot_current = Ldot_current, h = h, PL = p_init, PL_TangentX = p_init_TangentX, PL_TangentY = p_init_TangentY, PR = PR_init, PR_TangentX = PR_init_TangentX, PR_TangentY = PR_init_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                #Unilateral Constraint
                #If the first level swing the Left foot first, then the Left foot is the landing foot (p_init), Left foot obey the unilateral constraint on p_init
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL_k, TerrainNorm = p_init_Norm)
                #then the Right foot is obey the unilateral constraint on the PR_init
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR_k, TerrainNorm = PR_init_Norm)
                #Friction Cone Constraint
                #If the first level swing the Left foot first, then the Left foot is the landing foot (p_init), Left foot obey the friction cone constraint on p_init
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu)
                #then the right foot obeys the friction cone constraints on the PR_init
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu)
                
                #Case 2
                #If First Level Swing the Right, the the 0 phase (InitDouble) has p_init as the Right support, PL_init as the Left support
                #Kinematics Constraint
                #CoM in the Left foot
                g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = PL_init)
                #CoM in the Right foot
                g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_init)
                ##Agnular Dynamics
                #if k<N_K-1:
                #    g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, Ldot_next = Ldot_next, Ldot_current = Ldot_current, h = h, PL = PL_init, PL_TangentX = PL_init_TangentX, PL_TangentY = PL_init_TangentY, PR = p_init, PR_TangentX = p_init_TangentX, PR_TangentY = p_init_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                #Unilateral Constraint
                #If the first level swings the Right foot first, then the right foot is the landing foot (p_init), Right foot obeys the unilateral constraint on p_init
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR_k, TerrainNorm = p_init_Norm)
                #then the Left foot obeys the unilateral constrint on the PL_init
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL_k, TerrainNorm = PL_init_Norm)
                #Friction Cone Constraint
                #if the first level swing the right foot first, then the Right foot is the landing foot (p_init), Right foot obey the friction cone constraints on p_init
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu)
                #then the left foot obeys the friction cone constraint of PL_init
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu)
            
            #All other phases
            else:
                if GaitPattern[Nph]=='InitialDouble':

                    print("Knot ", k)
                    print("Belongs to Phase", Nph)
                    print("Phase Type ", GaitPattern[Nph])
                    print("Belongs to Step", StepCnt)

                    #Get contact location
                    if StepCnt == 1: #Step 1 needs special care (NOTE: Step Count Start from 0)
                        p_previous = ca.vertcat(px_init,py_init,pz_init)
                        p_previous_TangentX = SurfTangentsX[0:3]
                        p_previous_TangentY = SurfTangentsY[0:3]
                        p_previous_Norm = SurfNorms[0:3]

                        #In second level, Surfaces index is Step Vector Index (fpr px, py, pz, here is StepCnt-1) + 1
                        p_current = ca.vertcat(px[StepCnt-1],py[StepCnt-1],pz[StepCnt-1])
                        p_current_TangentX = SurfTangentsX[StepCnt*3:StepCnt*3+3]
                        p_current_TangentY = SurfTangentsY[StepCnt*3:StepCnt*3+3]
                        p_current_Norm = SurfNorms[StepCnt*3:StepCnt*3+3]

                    else: #Like Step 2, 3, 4 .....
                        p_previous = ca.vertcat(px[StepCnt-2],py[StepCnt-2],pz[StepCnt-2])
                        p_previous_TangentX = SurfTangentsX[(StepCnt-1)*3:(StepCnt-1)*3+3]
                        p_previous_TangentY = SurfTangentsY[(StepCnt-1)*3:(StepCnt-1)*3+3]
                        p_previous_Norm = SurfNorms[(StepCnt-1)*3:(StepCnt-1)*3+3]

                        p_current = ca.vertcat(px[StepCnt-1],py[StepCnt-1],pz[StepCnt-1])
                        p_current_TangentX = SurfTangentsX[StepCnt*3:StepCnt*3+3]
                        p_current_TangentY = SurfTangentsY[StepCnt*3:StepCnt*3+3]
                        p_current_Norm = SurfNorms[StepCnt*3:StepCnt*3+3]

                    #Give Constraint according to even and odd steps
                    if StepCnt%2 == 0: #Even Numbers of Footsteps
                        #Case 1
                        #If the first level swing the Left, then the Even Number of Steps in the Intial Double support phase have p_current as Left Support (Landed), p_previous as Right Support (Stationary)
                        #CoM in the Left (p_current)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_current)
                        #CoM in the Right (p_previous)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_previous)
                        ##Angular Dynamics
                        #if k<N_K-1:
                        #    g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, Ldot_next = Ldot_next, Ldot_current = Ldot_current, h = h, PL = p_current, PL_TangentX = p_current_TangentX, PL_TangentY = p_current_TangentY, PR = p_previous, PR_TangentX = p_previous_TangentX, PR_TangentY = p_previous_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                        #Unilateral Constraint
                        #If the first level swing the Left foot first, then the Left foot is the landing foot (p_current), Left foot obey the unilateral constraint on p_current
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL_k, TerrainNorm = p_current_Norm)
                        #then the Right foot is obey the unilateral constraint on the Stationary foot p_previous
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR_k, TerrainNorm = p_previous_Norm)
                        #Friction Cone Constraint
                        #If the first level swing the Left foot first, then the Left foot is the landing foot (p_current), Left foot obey the friction cone constraint on p_current
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        #then the right foot obeys the friction cone constraints on the on the Stationary foot p_previous
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        
                        #Case 2
                        #If the first level swing the Right, then the Even Number of Steps in the Intial Double support phase have p_current as Right Support (Landed), 
                        #p_previous as Left Support (Stationary)
                        #CoM in the Left (p_previous)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_previous)
                        #CoM in the Right (p_current)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_current)
                        ##Angular Dynamics
                        #if k<N_K-1:
                        #    g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, Ldot_next = Ldot_next, Ldot_current = Ldot_current, h = h, PL = p_previous, PL_TangentX = p_previous_TangentX, PL_TangentY = p_previous_TangentY, PR = p_current, PR_TangentX = p_current_TangentX, PR_TangentY = p_current_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                        #Unilateral Constraint
                        #Left foot obey the unilateral constraint on p_previous
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL_k, TerrainNorm = p_previous_Norm)
                        #Right foot is obey the unilateral constraint on p_current
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR_k, TerrainNorm = p_current_Norm)
                        #Friction Cone Constraint
                        #Left foot obey the friction cone constraint on p_previous
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        #then the right foot obeys the friction cone constraints on the on the Stationary foot p_previous
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)

                    elif StepCnt%2 == 1: #Odd Number of Steps
                        #Case 1
                        #If the first level swing the Left, then the Odd Number of Steps in the Intial Double support phase have p_current as Right Support (Landed), p_previous as Left Support (Stationary)
                        #CoM in the Left (p_previous)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_previous)
                        #CoM in the Right (p_current)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_current)
                        ##Angular Dynamics
                        #if k<N_K-1:
                        #    g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, Ldot_next = Ldot_next, Ldot_current = Ldot_current, h = h, PL = p_previous, PL_TangentX = p_previous_TangentX, PL_TangentY = p_previous_TangentY, PR = p_current, PR_TangentX = p_current_TangentX, PR_TangentY = p_current_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                        #Unilateral Constraint
                        #Left foot obey the unilateral constraint on p_previous
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL_k, TerrainNorm = p_previous_Norm)
                        #Right foot is obey the unilateral constraint on p_current
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR_k, TerrainNorm = p_current_Norm)
                        #Friction Cone Constraint
                        #Left foot obey the friction cone constraint on p_previous
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)
                        #right foot obeys the friction cone constraints on p_current
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        
                        #Case 2
                        #If the first level swing the Right, then the Odd Number of Steps in the Intial Double support phase have p_current as Left Support (Landed), 
                        #p_previous as Right Support (Stationary)
                        #CoM in the Left (p_current)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_current)
                        #CoM in the Right (p_previous)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_previous)
                        ##Angular Dynamics
                        #if k<N_K-1:
                        #    g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, Ldot_next = Ldot_next, Ldot_current = Ldot_current, h = h, PL = p_current, PL_TangentX = p_current_TangentX, PL_TangentY = p_current_TangentY, PR = p_previous, PR_TangentX = p_previous_TangentX, PR_TangentY = p_previous_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                        #Unilateral Constraint
                        #Left foot obey the unilateral constraint on p_current
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL_k, TerrainNorm = p_current_Norm)
                        #Right foot is obey the unilateral constraint on p_previous
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR_k, TerrainNorm = p_previous_Norm)
                        #Friction Cone Constraint
                        #Left foot obey the friction cone constraint on p_current
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu)
                        #then the right foot obeys the friction cone constraints on the p_previous
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu)

                elif GaitPattern[Nph]== 'Swing':
                    print("Knot ", k)
                    print("Belongs to Phase", Nph)
                    print("Phase Type ", GaitPattern[Nph])
                    print("Belongs to Step", StepCnt)

                    #Get contact location
                    if StepCnt == 0:#Special Case for the First Step (NOTE:Step 0)
                        p_stance = ca.vertcat(px_init,py_init,pz_init)
                        p_stance_TangentX = SurfTangentsX[0:3]
                        p_stance_TangentY = SurfTangentsY[0:3]
                        p_stance_Norm = SurfNorms[0:3]

                    else: #For other Steps, indexed as 1,2,3,4
                        p_stance = ca.vertcat(px[StepCnt-1],py[StepCnt-1],pz[StepCnt-1])
                        p_stance_TangentX = SurfTangentsX[StepCnt*3:StepCnt*3+3]
                        p_stance_TangentY = SurfTangentsY[StepCnt*3:StepCnt*3+3]
                        p_stance_Norm = SurfNorms[StepCnt*3:StepCnt*3+3]

                    #Give Constraint according to even and odd steps
                    if StepCnt%2 == 0: #Even Number of Steps
                        #Case 1
                        #If First Level Swing the Left, then the second level Even Number Phases (the first Phase) Swing the Right
                        #Left foot is the stance foot
                        #Right foot is floating
                        #Kinematics Constraint
                        #CoM in the Left (p_stance)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_stance)
                        ##Angular Dynamics (Left Stance)
                        #if k<N_K-1:
                        #    g, glb, gub = Angular_Momentum_Rate_Swing(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, Ldot_next = Ldot_next, Ldot_current = Ldot_current, h = h, P = p_stance, P_TangentX = p_stance_TangentX, P_TangentY = p_stance_TangentY, CoM_k = CoM_k, F1_k = FL1_k, F2_k = FL2_k, F3_k = FL3_k, F4_k = FL4_k)
                        #Zero Forces (Right Foot)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR_k)
                        #Unilateral Constraints on Left Foot p_stance
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL_k, TerrainNorm = p_stance_Norm)
                        #Friction Cone Constraint on Left Foot p_stance
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)

                        #Case 2
                        #If First Level Swing the Right, then the second level Even Number Phases (the first Phase) Swing the Left
                        #Right foot is the stance foot
                        #Left foot is floating
                        #Kinematics Constraint
                        #CoM in the Right (p_stance)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_stance)
                        ##Angular Dynamics(Right Stance)
                        #if k<N_K-1:
                        #    g, glb, gub = Angular_Momentum_Rate_Swing(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, Ldot_next = Ldot_next, Ldot_current = Ldot_current, h = h, P = p_stance, P_TangentX = p_stance_TangentX, P_TangentY = p_stance_TangentY, CoM_k = CoM_k, F1_k = FR1_k, F2_k = FR2_k, F3_k = FR3_k, F4_k = FR4_k)
                        #Zero Forces (Left Foot)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL_k)
                        #Unilateral Constraints on Right Foot p_stance
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR_k, TerrainNorm = p_stance_Norm)
                        #Friction Cone Constraint on Right Foot p_stance
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)

                    elif StepCnt%2 == 1: #Odd Number of Steps
                        #Case 1
                        #If First Level Swing the Left, then the second level Odd Number Steps Swing the Left
                        #Right foot is the stance foot
                        #Left foot is floating
                        #Kinematics Constraint
                        #CoM in the Right (p_stance)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_stance)
                        ##Angular Dynamics (Right Stance)
                        #if k<N_K-1:
                        #    g, glb, gub = Angular_Momentum_Rate_Swing(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, Ldot_next = Ldot_next, Ldot_current = Ldot_current, h = h, P = p_stance, P_TangentX = p_stance_TangentX, P_TangentY = p_stance_TangentY, CoM_k = CoM_k, F1_k = FR1_k, F2_k = FR2_k, F3_k = FR3_k, F4_k = FR4_k)
                        #Zero Forces (Left Foot)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL_k)
                        #Unilateral Constraints on Right Foot p_stance
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR_k, TerrainNorm = p_stance_Norm)
                        #Friction Cone Constraint on Right Foot p_stance
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)

                        #Case 2
                        #If First Level Swing the Right, then the second level Odd Number Steps Swing the Right
                        #Left foot is the stance foot
                        #Right foot is floating
                        #Kinematics Constraint
                        #CoM in the Left (p_stance)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_stance)
                        ##Angular Dynamics (Left Stance)
                        #if k<N_K-1:
                        #    g, glb, gub = Angular_Momentum_Rate_Swing(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, Ldot_next = Ldot_next, Ldot_current = Ldot_current, h = h, P = p_stance, P_TangentX = p_stance_TangentX, P_TangentY = p_stance_TangentY, CoM_k = CoM_k, F1_k = FL1_k, F2_k = FL2_k, F3_k = FL3_k, F4_k = FL4_k)
                        #Zero Forces (Right Foot)
                        g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR_k)
                        #Unilateral Constraints on Left Foot p_stance
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL_k, TerrainNorm = p_stance_Norm)
                        #Friction Cone Constraint on Left Foot p_stance
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu)

                elif GaitPattern[Nph]=='DoubleSupport':
                    print("Knot ", k)
                    print("Belongs to Phase", Nph)
                    print("Phase Type ", GaitPattern[Nph])
                    print("Belongs to Step", StepCnt)
                    #Get contact location
                    if StepCnt == 0: #Special Case for the First Step (NOTE: Step 0)
                        p_stationary = ca.vertcat(px_init,py_init,pz_init)
                        p_stationary_TangentX = SurfTangentsX[0:3]
                        p_stationary_TangentY = SurfTangentsY[0:3]
                        p_stationary_Norm = SurfNorms[0:3]

                        p_land = ca.vertcat(px[StepCnt],py[StepCnt],pz[StepCnt])
                        p_land_TangentX = SurfTangentsX[(StepCnt+1)*3:(StepCnt+1)*3+3]
                        p_land_TangentY = SurfTangentsY[(StepCnt+1)*3:(StepCnt+1)*3+3]
                        p_land_Norm = SurfNorms[(StepCnt+1)*3:(StepCnt+1)*3+3]
                
                    else: #For other steps, indexed as 1,2,3,4
                        p_stationary = ca.vertcat(px[StepCnt-1],py[StepCnt-1],pz[StepCnt-1])
                        p_stationary_TangentX = SurfTangentsX[StepCnt*3:StepCnt*3+3]
                        p_stationary_TangentY = SurfTangentsY[StepCnt*3:StepCnt*3+3]
                        p_stationary_Norm = SurfNorms[StepCnt*3:StepCnt*3+3]

                        p_land = ca.vertcat(px[StepCnt],py[StepCnt],pz[StepCnt])
                        p_land_TangentX = SurfTangentsX[(StepCnt+1)*3:(StepCnt+1)*3+3]
                        p_land_TangentY = SurfTangentsY[(StepCnt+1)*3:(StepCnt+1)*3+3]
                        p_land_Norm = SurfNorms[(StepCnt+1)*3:(StepCnt+1)*3+3]
                
                    #Give Constraint according to even and odd steps
                    if StepCnt%2 == 0: #Even Number of Steps
                        #Case 1
                        #If First Level Swing the Left, then the second level Even Steps Swing the Right
                        #In Double Support Phase
                        #Left Foot is the Stationary
                        #Right Foot is the Land
                        #Kinemactics Constraint
                        #CoM in the Left (p_stationary)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_stationary)
                        #CoM in the Right (p_land)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_land)
                        ##Angular Dynamics
                        #if k<N_K-1:
                        #    g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, Ldot_next = Ldot_next, Ldot_current = Ldot_current, h = h, PL = p_stationary, PL_TangentX = p_stationary_TangentX, PL_TangentY = p_stationary_TangentY, PR = p_land, PR_TangentX = p_land_TangentX, PR_TangentY = p_land_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                        #Unilateral Constraint
                        #Left foot obey the unilateral constraint on p_stationary
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL_k, TerrainNorm = p_stationary_Norm)
                        #then the Right foot is obey the unilateral constraint on the Stationary foot p_land
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR_k, TerrainNorm = p_land_Norm)
                        #Friction Cone Constraint
                        #Left foot obey the friction cone constraint on p_stationary
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        #then the right foot obeys the friction cone constraints on the on p_land
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        
                        #Case 2
                        #If First Level Swing the Right, then the second level Even Steps Swing the Left
                        #In Double Support Phase
                        #Right Foot is the Stationary
                        #Left Foot is the Land
                        #Kinemactics Constraint
                        #CoM in the Left (p_land)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_land)
                        #CoM in the Right (p_stationary)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_stationary)
                        ##Angular Dynamics
                        #if k<N_K-1:
                        #    g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, Ldot_next = Ldot_next, Ldot_current = Ldot_current, h = h, PL = p_land, PL_TangentX = p_land_TangentX, PL_TangentY = p_land_TangentY, PR = p_stationary, PR_TangentX = p_stationary_TangentX, PR_TangentY = p_stationary_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                        #Unilateral Constraint
                        #Left foot obey the unilateral constraint on p_land
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL_k, TerrainNorm = p_land_Norm)
                        #then the Right foot is obey the unilateral constraint on the Stationary foot p_stationary
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR_k, TerrainNorm = p_stationary_Norm)
                        #Friction Cone Constraint
                        #Left foot obey the friction cone constraint on p_land
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        #then the right foot obeys the friction cone constraints on the on p_stationary
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        
                    elif StepCnt%2 == 1:#Odd Number of Steps
                        #Case 1
                        #If First Level Swing the Left, then the second level Odd Steps Swing the Left
                        #In Double Support Phase
                        #Right Foot is the Stationary
                        #Left Foot is the Land
                        #Kinemactics Constraint
                        #CoM in the Left (p_land)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_land)
                        #CoM in the Right (p_stationary)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_stationary)
                        ##Angular Dynamics
                        #if k<N_K-1:
                        #    g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, Ldot_next = Ldot_next, Ldot_current = Ldot_current, h = h, PL = p_land, PL_TangentX = p_land_TangentX, PL_TangentY = p_land_TangentY, PR = p_stationary, PR_TangentX = p_stationary_TangentX, PR_TangentY = p_stationary_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                        #Unilateral Constraint
                        #Left foot obey the unilateral constraint on p_land
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL_k, TerrainNorm = p_land_Norm)
                        #then the Right foot is obey the unilateral constraint on the Stationary foot p_stationary
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR_k, TerrainNorm = p_stationary_Norm)
                        #Friction Cone Constraint
                        #Left foot obey the friction cone constraint on p_land
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        #then the right foot obeys the friction cone constraints on the on p_stationary
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        
                        #Case 2
                        #If First Level Swing the Right, then the second level Odd Steps Swing the Right
                        #In Double Support Phase
                        #Left Foot is the Stationary
                        #Right Foot is the Land
                        #Kinematics Constraint
                        #CoM in the Left (p_stationary)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, CoM_k = CoM_k, p = p_stationary)
                        #CoM in the Right (p_land)
                        g, glb, gub = CoM_Kinematics(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, CoM_k = CoM_k, p = p_land)
                        ##Angular Dynamics
                        #if k<N_K-1:
                        #    g, glb, gub = Angular_Momentum_Rate_DoubleSupport(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, Ldot_next = Ldot_next, Ldot_current = Ldot_current, h = h, PL = p_stationary, PL_TangentX = p_stationary_TangentX, PL_TangentY = p_stationary_TangentY, PR = p_land, PR_TangentX = p_land_TangentX, PR_TangentY = p_land_TangentY, CoM_k = CoM_k, FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k)
                        #Unilateral Constraint
                        #Left foot obey the unilateral constraint on p_stationary
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL_k, TerrainNorm = p_stationary_Norm)
                        #then the Right foot is obey the unilateral constraint on the Stationary foot p_land
                        g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR_k, TerrainNorm = p_land_Norm)
                        #Friction Cone Constraint
                        #Left foot obey the friction cone constraint on p_stationary
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu)
                        #then the right foot obeys the friction cone constraints on the on p_land
                        g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu)
                        
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

                ##First-order Angular Momentum Dynamics x-axis
                #g.append(Lx[k+1] - Lx[k] - h*Ldotx[k])
                #glb.append(np.array([0]))
                #gub.append(np.array([0]))

                ##First-order Angular Momentum Dynamics y-axis
                #g.append(Ly[k+1] - Ly[k] - h*Ldoty[k])
                #glb.append(np.array([0]))
                #gub.append(np.array([0]))

                ##First-order Angular Momentum Dynamics z-axis
                #g.append(Lz[k+1] - Lz[k] - h*Ldotz[k])
                #glb.append(np.array([0]))
                #gub.append(np.array([0]))

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
            
            #----------------------
            #Add Cost Terms
            #Original Cost
            #if k < N_K - 1:
            #    #Acc Only
            #    J = J + h*(FLx[k]/m+FRx[k]/m)**2 + h*(FLy[k]/m+FRy[k]/m)**2 + h*(FLz[k]/m+FRz[k]/m - G)**2

            #-------------------
            #Tracking Traj Cost
            #for x position
            J = J + (x[k]-x_ref[k])**2
            #for y position
            J = J + (y[k]-y_ref[k])**2
            #for z position
            J = J + (z[k]-z_ref[k])**2
            #for xdot 
            J = J + (xdot[k]-xdot_ref[k])**2
            #for ydot
            J = J + (ydot[k]-ydot_ref[k])**2
            #for zdot
            J = J + (zdot[k]-zdot_ref[k])**2
            # ##for FLx
            # J = J + (FLx[k]-FLx_ref[k])**2
            # ##for FLy
            # J = J + (FLy[k]-FLy_ref[k])**2
            # ##for FLz
            # J = J + (FLz[k]-FLz_ref[k])**2
            # ##for FRx
            # J = J + (FRx[k]-FRx_ref[k])**2
            # ##for FRy
            # J = J + (FRy[k]-FRy_ref[k])**2
            # ##for FRz
            # J = J + (FRz[k]-FRz_ref[k])**2
    #----------------------------------
    #Cost Term for Tracking Constact Locations
    for step_Count in range(len(px)):
        #For Px
        J = J + (px[step_Count]-Px_seq_ref[step_Count])**2
        #For py
        J = J + (py[step_Count]-Py_seq_ref[step_Count])**2
        #For pz
        J = J + (pz[step_Count]-Pz_seq_ref[step_Count])**2

    # # #--------------
    # # #Initial Condition Constraint (Align with Second Level)
    # # #for x position
    # g.append(x[0]-x_ref[0])
    # glb.append([0])
    # gub.append([0])
    # # #for y position
    # g.append(y[0]-y_ref[0])
    # glb.append([0])
    # gub.append([0])    
    # # #for z position
    # g.append(z[0]-z_ref[0])
    # glb.append([0])
    # gub.append([0])    
    # # #for xdot 
    # g.append(xdot[0]-xdot_ref[0])
    # glb.append([0])
    # gub.append([0]) 
    # #for ydot
    # g.append(ydot[0]-ydot_ref[0])
    # glb.append([0])
    # gub.append([0]) 
    # # #for zdot
    # g.append(zdot[0]-zdot_ref[0])
    # glb.append([0])
    # gub.append([0]) 

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
            #Also, the stationary foot Left should stay in the polytope of the landed swing foot - RIGHT
            #NOTE: current - next now
            g.append(ca.if_else(ParaLeftSwingFlag,Q_lf_in_rf@(P_k_current-P_k_next)-ca.DM(q_lf_in_rf),np.full((len(q_lf_in_rf),),-1)))
            glb.append(np.full((len(q_lf_in_rf),),-np.inf))
            gub.append(np.full((len(q_lf_in_rf),),0))

            #If First Levvel Swing the Right, then the second level Even Number Phases (the first phase) Swing the Left -> Right -> Left
            #Right foot in contact for p_current, left foot is going to land at p_next
            #Relative Swing Foot Location (lf in rf)
            g.append(ca.if_else(ParaRightSwingFlag,Q_lf_in_rf@(P_k_next-P_k_current)-ca.DM(q_lf_in_rf),np.full((len(q_lf_in_rf),),-1)))
            glb.append(np.full((len(q_lf_in_rf),),-np.inf))
            gub.append(np.full((len(q_lf_in_rf),),0))
            #Also, the stationary foot Rigth should stay in the polytope of the landed swing foot - Left
            #NOTE: current - next now
            g.append(ca.if_else(ParaRightSwingFlag,Q_rf_in_lf@(P_k_current-P_k_next)-ca.DM(q_rf_in_lf),np.full((len(q_rf_in_lf),),-1)))
            glb.append(np.full((len(q_rf_in_lf),),-np.inf))
            gub.append(np.full((len(q_rf_in_lf),),0))

        elif step_cnt%2 == 1: #odd number steps
            #If the first level swings the Left, then the second level for Odd Number Phases (the second phase) swings left -> Right -> Left
            #Right foot in contact for p_current, left foot is going to land at p_next
            #Relative Swing Foot Location (lf in rf)
            g.append(ca.if_else(ParaLeftSwingFlag,Q_lf_in_rf@(P_k_next-P_k_current)-ca.DM(q_lf_in_rf),np.full((len(q_lf_in_rf),),-1)))
            glb.append(np.full((len(q_lf_in_rf),),-np.inf))
            gub.append(np.full((len(q_lf_in_rf),),0))
            #Also, the stationary foot Rigth should stay in the polytope of the landed swing foot - Left
            #NOTE: current - next now
            g.append(ca.if_else(ParaLeftSwingFlag,Q_rf_in_lf@(P_k_current-P_k_next)-ca.DM(q_rf_in_lf),np.full((len(q_rf_in_lf),),-1)))
            glb.append(np.full((len(q_rf_in_lf),),-np.inf))
            gub.append(np.full((len(q_rf_in_lf),),0))

            #If the first level swings the Right, then the second level for Odd Number Phases (the second phase) swings Right -> Left -> Right
            #Left foot in contact for p_current, right foot is going to land as p_next
            #Relative Swing Foot Location (rf in lf)
            g.append(ca.if_else(ParaRightSwingFlag,Q_rf_in_lf@(P_k_next-P_k_current)-ca.DM(q_rf_in_lf),np.full((len(q_rf_in_lf),),-1)))
            glb.append(np.full((len(q_rf_in_lf),),-np.inf))
            gub.append(np.full((len(q_rf_in_lf),),0))
            #Also, the stationary foot LEFT should stay in the polytope of the landed swing foot - RIGHT
            #NOTE: current - next now
            g.append(ca.if_else(ParaRightSwingFlag,Q_lf_in_rf@(P_k_current-P_k_next)-ca.DM(q_lf_in_rf),np.full((len(q_lf_in_rf),),-1)))
            glb.append(np.full((len(q_lf_in_rf),),-np.inf))
            gub.append(np.full((len(q_lf_in_rf),),0))

    #Foot Step Constraint
    #FootStep Constraint
    #P3----------------P1
    #|                  |
    #|                  |
    #|                  |
    #P4----------------P2
    for PatchNum in range(Nsteps):
        #Get Footstep Vector
        P_vector = ca.vertcat(px[PatchNum],py[PatchNum],pz[PatchNum])

        #Get Half Space Representation 
        #NOTE: In the second level, the terrain patch start from the second patch, indexed as 1
        SurfParaTemp = SurfParas[20+PatchNum*20:19+(PatchNum+1)*20+1]
        #print(SurfParaTemp)
        SurfK = SurfParaTemp[0:11+1]
        SurfK = ca.reshape(SurfK,3,4)
        SurfK = SurfK.T #NOTE: This process is due to casadi naming convention to have first row to be x1,x2,x3
        SurfE = SurfParaTemp[11+1:11+3+1]
        Surfk = SurfParaTemp[14+1:14+4+1]
        Surfe = SurfParaTemp[-1]

        #Terrain Tangent and Norms
        P_vector_TangentX = SurfTangentsX[(PatchNum+1)*3:(PatchNum+1)*3+3]
        P_vector_TangentY = SurfTangentsY[(PatchNum+1)*3:(PatchNum+1)*3+3]

        #Contact Point 1
        #Inequality
        g.append(SurfK@(P_vector + 0.11*P_vector_TangentX + 0.06*P_vector_TangentY) - Surfk)
        glb.append(np.full((4,),-np.inf))
        gub.append(np.full((4,),0))
        #Equality
        g.append(SurfE.T@(P_vector + 0.11*P_vector_TangentX + 0.06*P_vector_TangentY) - Surfe)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        #Contact Point 2
        #Inequality
        g.append(SurfK@(P_vector + 0.11*P_vector_TangentX - 0.06*P_vector_TangentY) - Surfk)
        glb.append(np.full((4,),-np.inf))
        gub.append(np.full((4,),0))
        #Equality
        g.append(SurfE.T@(P_vector + 0.11*P_vector_TangentX - 0.06*P_vector_TangentY) - Surfe)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        #Contact Point 3
        #Inequality
        g.append(SurfK@(P_vector - 0.11*P_vector_TangentX + 0.06*P_vector_TangentY) - Surfk)
        glb.append(np.full((4,),-np.inf))
        gub.append(np.full((4,),0))
        #Equality
        g.append(SurfE.T@(P_vector - 0.11*P_vector_TangentX + 0.06*P_vector_TangentY) - Surfe)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        #Contact Point 4
        #Inequality
        g.append(SurfK@(P_vector + 0.11*P_vector_TangentX + 0.06*P_vector_TangentY) - Surfk)
        glb.append(np.full((4,),-np.inf))
        gub.append(np.full((4,),0))
        #Equality
        g.append(SurfE.T@(P_vector + 0.11*P_vector_TangentX + 0.06*P_vector_TangentY) - Surfe)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        ##FootStep Constraint
        ##Inequality
        #g.append(SurfK@P_vector - Surfk)
        #glb.append(np.full((4,),-np.inf))
        #gub.append(np.full((4,),0))
        #print(FirstSurfK@p_next - FirstSurfk)

        ##Equality
        #g.append(SurfE.T@P_vector - Surfe)
        #glb.append(np.array([0]))
        #gub.append(np.array([0]))

    #Approximate Kinematics Constraint --- Disallow over-crossing of footsteps from y =0

    if CentralY == True:

        for step_cnt in range(Nsteps):
            P_k_next = ca.vertcat(px[step_cnt],py[step_cnt],pz[step_cnt])
    
            if step_cnt%2 == 0: #even number steps
                #If First Level Swing the Left, then the second level Even Number Phases (the first Phase) Swing the Right -> Left -> Right
                #Left foot in contact for p_current, right foot is going to land as p_next
                g.append(ca.if_else(ParaLeftSwingFlag,P_k_next[1],np.array([-1])))
                glb.append(np.array([-np.inf]))
                gub.append(np.array([-py_lower_limit]))
    
                #If First Levvel Swing the Right, then the second level Even Number Phases (the first phase) Swing the Left -> Right -> Left
                #Right foot in contact for p_current, left foot is going to land at p_next
                g.append(ca.if_else(ParaRightSwingFlag,P_k_next[1],np.array([1])))
                glb.append(np.array([py_lower_limit]))
                gub.append(np.array([np.inf]))
    
            elif step_cnt%2 == 1: #odd number steps
                #If the first level swings the Left, then the second level for Odd Number Phases (the second phase) swings left -> Right -> Left
                #Right foot in contact for p_current, left foot is going to land at p_next
                g.append(ca.if_else(ParaLeftSwingFlag,P_k_next[1],np.array([1])))
                glb.append(np.array([py_lower_limit]))
                gub.append(np.array([np.inf]))
    
                #If the first level swings the Right, then the second level for Odd Number Phases (the second phase) swings Right -> Left -> Right
                #Left foot in contact for p_current, right foot is going to land as p_next
                g.append(ca.if_else(ParaRightSwingFlag,P_k_next[1],np.array([-1])))
                glb.append(np.array([-np.inf]))
                gub.append(np.array([-py_lower_limit]))

    #Switching Time Constraint
    for phase_cnt in range(Nphase):
        if GaitPattern[phase_cnt] == 'InitialDouble':
            if phase_cnt == 0:
                g.append(Ts[phase_cnt] - 0)
                glb.append(np.array([0.3])) #0.1-0.3
                gub.append(np.array([0.3]))
            else:
                g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
                glb.append(np.array([0.3]))
                gub.append(np.array([0.3]))

        elif GaitPattern[phase_cnt] == 'Swing':
            g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
            glb.append(np.array([0.5]))
            gub.append(np.array([0.5]))

            #if phase_cnt == 0:
            #    g.append(Ts[phase_cnt]-0)#0.6-1
            #    glb.append(np.array([0.5]))#0.5 for NLP success
            #    gub.append(np.array([0.7]))
            #else:
            #    g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
            #    glb.append(np.array([0.5]))
            #    gub.append(np.array([0.7]))

        elif GaitPattern[phase_cnt] == 'DoubleSupport':
            g.append(Ts[phase_cnt]-Ts[phase_cnt-1]) #0.1-0.9
            glb.append(np.array([0.3]))
            gub.append(np.array([0.3])) #0.1-0.3

    #-----------------------------------------------------------------------------------------------------------------------
    #Get Variable Index - !!!This is the pure Index, when try to get the array using other routines, we need to add "+1" at the last index due to Python indexing conventions
    x_index = (0,N_K-1) #First set of variables start counting from 0, The enumeration of the last knot is N_K-1
    y_index = (x_index[1]+1,x_index[1]+N_K)
    z_index = (y_index[1]+1,y_index[1]+N_K)
    xdot_index = (z_index[1]+1,z_index[1]+N_K)
    ydot_index = (xdot_index[1]+1,xdot_index[1]+N_K)
    zdot_index = (ydot_index[1]+1,ydot_index[1]+N_K)

    #Lx_index = (zdot_index[1]+1,zdot_index[1]+N_K)
    #Ly_index = (Lx_index[1]+1,Lx_index[1]+N_K)
    #Lz_index = (Ly_index[1]+1,Ly_index[1]+N_K)
    #Ldotx_index = (Lz_index[1]+1,Lz_index[1]+N_K)
    #Ldoty_index = (Ldotx_index[1]+1,Ldotx_index[1]+N_K)
    #Ldotz_index = (Ldoty_index[1]+1,Ldoty_index[1]+N_K)
    
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
    Ts_index = (pz_index[1]+1,pz_index[1]+Nphase)

    var_index = {"x":x_index,
                 "y":y_index,
                 "z":z_index,
                 "xdot":xdot_index,
                 "ydot":ydot_index,
                 "zdot":zdot_index,
                 #"Lx":Lx_index,
                 #"Ly":Ly_index,
                 #"Lz":Lz_index,
                 #"Ldotx":Ldotx_index,
                 #"Ldoty":Ldoty_index,
                 #"Ldotz":Ldotz_index,
                 "FLx":FLx_index,
                 "FLy":FLy_index,
                 "FLz":FLz_index,
                 #"FL2x":FL2x_index,
                 #"FL2y":FL2y_index,
                 #"FL2z":FL2z_index,
                 #"FL3x":FL3x_index,
                 #"FL3y":FL3y_index,
                 #"FL3z":FL3z_index,
                 #"FL4x":FL4x_index,
                 #"FL4y":FL4y_index,
                 #"FL4z":FL4z_index,
                 "FRx":FRx_index,
                 "FRy":FRy_index,
                 "FRz":FRz_index,
                 #"FR2x":FR2x_index,
                 #"FR2y":FR2y_index,
                 #"FR2z":FR2z_index,
                 #"FR3x":FR3x_index,
                 #"FR3y":FR3y_index,
                 #"FR3z":FR3z_index,
                 #"FR4x":FR4x_index,
                 #"FR4y":FR4y_index,
                 #"FR4z":FR4z_index,
                 "px_init":px_init_index,
                 "py_init":py_init_index,
                 "pz_init":pz_init_index,
                 "px":px_index,
                 "py":py_index,
                 "pz":pz_index,
                 "Ts":Ts_index,
    }

    #print(DecisionVars[var_index["px_init"][0]:var_index["px_init"][1]+1])

    return DecisionVars, DecisionVars_lb, DecisionVars_ub, J, g, glb, gub, var_index

def CoM_Dynamics_Ponton_Cost(m = 95, Nk_Local = 7, Nsteps = 1, ParameterList = None, StaticStop = False, NumPatches = None, CentralY = False):
    
    print("Using Ponton's to approximate Angular Momentum Rate")
    #-----------------------------------------------------------------------------------------------------------------------
    #Define Parameters
    #   Gait Pattern, Each action is followed up by a double support phase
    GaitPattern = ["InitialDouble","Swing","DoubleSupport"] + ["Swing","DoubleSupport"]*(Nsteps-1) #,'RightSupport','DoubleSupport','LeftSupport','DoubleSupport'

    #ProblemType = ["NLP", "NLP", "NLP"] + ["CoM"]*50

    #ProblemType = ["CoM", "CoM", "CoM"] + ["CoM"]*50

    #   Number of Phases
    Nphase = len(GaitPattern)
    #   Number of Steps
    #Nstep = 1
    #   Number of Knots per Phase - how many intervals do we have for a single phase; 10 intervals need 11 knots/ticks
    #Nk_Local= 5
    #   Compute Number of Total knots/ticks, but the enumeration start from 0 to N_K-1
    N_K = Nk_Local*Nphase + 1 #+1 the last knots to finalize the plan
    #   Robot mass
    #m = 95 #kg
    G = 9.80665 #kg/m^2
    #   Terrain Model
    #       Flat Terrain
    #TerrainNorm = [0,0,1] 
    #TerrainTangentX = [1,0,0]
    #TerrainTangentY = [0,1,0]
    miu = 0.3
    #   Force Limits
    F_bound = 400
    Fxlb = -F_bound
    Fxub = F_bound
    Fylb = -F_bound
    Fyub = F_bound
    Fzlb = -F_bound
    Fzub = F_bound
    #Minimum y-axis foot location
    py_lower_limit = 0.04
    #Lowest Z
    z_lowest = 0.7
    z_highest = 0.8
    #Phase duration
    h_initialdouble = 0.3
    h_doublesupport = 0.4
    h_swing = 0.5
    #Ponton's Variables
    p_lb = -0.6
    p_ub = 0.6
    q_lb = -0.6
    q_ub = 0.6
    #-----------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------
    #Kinematics Constraint for Talos
    #kinematicConstraints = genKinematicConstraints(left_foot_constraints,right_foot_constraints)
    K_CoM_Right,k_CoM_Right = right_foot_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]))
    K_CoM_Left,k_CoM_Left = left_foot_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]))

    #K_CoM_Left = kinematicConstraints[0][0]
    #k_CoM_Left = kinematicConstraints[0][1]
    #K_CoM_Right = kinematicConstraints[1][0]
    #k_CoM_Right = kinematicConstraints[1][1]
    #Relative Foot Constraint matrices
    
    #relativeConstraints = genFootRelativeConstraints(right_foot_in_lf_frame_constraints,left_foot_in_rf_frame_constraints)
    Q_rf_in_lf,q_rf_in_lf = right_foot_in_lf_frame_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]))
    Q_lf_in_rf,q_lf_in_rf = left_foot_in_rf_frame_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]))

    #Q_rf_in_lf = relativeConstraints[0][0] #named lf in rf, but representing rf in lf
    #q_rf_in_lf = relativeConstraints[0][1] #named lf in rf, but representing rf in lf
    #Q_lf_in_rf = relativeConstraints[1][0] #named rf in lf, but representing lf in rf
    #q_lf_in_rf = relativeConstraints[1][1] #named rf in lf, but representing lf in rf
    #-----------------------------------------------------------------------------------------------------------------------
    #Define Initial and Terminal Conditions
    #x_init = ca.SX.sym('x_init')
    #y_init = ca.SX.sym('y_init')
    #z_init = ca.SX.sym('z_init')

    x_init = ParameterList["x_init"]
    y_init = ParameterList["y_init"]
    z_init = ParameterList["z_init"]

    xdot_init = ParameterList["xdot_init"]
    ydot_init = ParameterList["ydot_init"]
    zdot_init = ParameterList["zdot_init"]

    #Lx_init = 0
    #Ly_init = 0
    #Lz_init = 0

    #Ldotx_init = 0
    #Ldoty_init = 0
    #Ldotz_init = 0

    PLx_init = ParameterList["PLx_init"]
    PLy_init = ParameterList["PLy_init"]
    PLz_init = ParameterList["PLz_init"]
    PL_init = ca.vertcat(PLx_init,PLy_init,PLz_init)

    PRx_init = ParameterList["PRx_init"]
    PRy_init = ParameterList["PRy_init"]
    PRz_init = ParameterList["PRz_init"]
    PR_init = ca.vertcat(PRx_init,PRy_init,PRz_init)

    x_end = ParameterList["x_end"]
    y_end = ParameterList["y_end"]
    z_end = ParameterList["z_end"]

    xdot_end = ParameterList["xdot_end"]
    ydot_end = ParameterList["ydot_end"]
    zdot_end = ParameterList["zdot_end"]

    #Lx_end = 0
    #Ly_end = 0
    #Lz_end = 0

    #Ldotx_end = 0
    #Ldoty_end = 0
    #Ldotz_end = 0

    #Flags for Swing Legs (Defined as Parameters)
    ParaLeftSwingFlag = ParameterList["LeftSwingFlag"]
    ParaRightSwingFlag = ParameterList["RightSwingFlag"]

    #Surfaces (Only the Second One)
    #Surface Patches
    SurfParas = ParameterList["SurfParas"]

    #Tangents and Norms
    #Initial Contact Norm and Tangents
    PL_init_Norm = ParameterList["PL_init_Norm"]
    PL_init_TangentX = ParameterList["PL_init_TangentX"]
    PL_init_TangentY = ParameterList["PL_init_TangentY"]
    PR_init_Norm = ParameterList["PR_init_Norm"]
    PR_init_TangentX = ParameterList["PR_init_TangentX"]
    PR_init_TangentY = ParameterList["PR_init_TangentY"]
    
    #Future Contact Norm and Tangents
    SurfNorms = ParameterList["SurfNorms"]                
    SurfTangentsX = ParameterList["SurfTangentsX"]
    SurfTangentsY = ParameterList["SurfTangentsY"]

    #-----------------------------------------------------------------------------------------------------------------------
    #Define Variables and Bounds, Parameters
    #   CoM Position x-axis
    x = ca.SX.sym('x',N_K)
    x_lb = np.array([[0]*(x.shape[0]*x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    x_ub = np.array([[30]*(x.shape[0]*x.shape[1])])
    #   CoM Position y-axis
    y = ca.SX.sym('y',N_K)
    y_lb = np.array([[-1]*(y.shape[0]*y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    y_ub = np.array([[1]*(y.shape[0]*y.shape[1])])
    #   CoM Position z-axis
    z = ca.SX.sym('z',N_K)
    z_lb = np.array([[z_lowest]*(z.shape[0]*z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    z_ub = np.array([[z_highest]*(z.shape[0]*z.shape[1])])
    #   CoM Velocity x-axis
    xdot = ca.SX.sym('xdot',N_K)
    xdot_lb = np.array([[-1]*(xdot.shape[0]*xdot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    xdot_ub = np.array([[1]*(xdot.shape[0]*xdot.shape[1])])
    #   CoM Velocity y-axis
    ydot = ca.SX.sym('ydot',N_K)
    ydot_lb = np.array([[-1]*(ydot.shape[0]*ydot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    ydot_ub = np.array([[1]*(ydot.shape[0]*ydot.shape[1])])
    #   CoM Velocity z-axis
    zdot = ca.SX.sym('zdot',N_K)
    zdot_lb = np.array([[-1]*(zdot.shape[0]*zdot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    zdot_ub = np.array([[1]*(zdot.shape[0]*zdot.shape[1])])
    #left Foot Forces
    #Left Foot Contact Point 1 x-axis
    FL1x = ca.SX.sym('FL1x',N_K)
    FL1x_lb = np.array([[Fxlb]*(FL1x.shape[0]*FL1x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL1x_ub = np.array([[Fxub]*(FL1x.shape[0]*FL1x.shape[1])])
    #Left Foot Contact Point 1 y-axis
    FL1y = ca.SX.sym('FL1y',N_K)
    FL1y_lb = np.array([[Fylb]*(FL1y.shape[0]*FL1y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL1y_ub = np.array([[Fyub]*(FL1y.shape[0]*FL1y.shape[1])])
    #Left Foot Contact Point 1 z-axis
    FL1z = ca.SX.sym('FL1z',N_K)
    FL1z_lb = np.array([[Fzlb]*(FL1z.shape[0]*FL1z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL1z_ub = np.array([[Fzub]*(FL1z.shape[0]*FL1z.shape[1])])
    #Left Foot Contact Point 2 x-axis
    FL2x = ca.SX.sym('FL2x',N_K)
    FL2x_lb = np.array([[Fxlb]*(FL2x.shape[0]*FL2x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL2x_ub = np.array([[Fxub]*(FL2x.shape[0]*FL2x.shape[1])])
    #Left Foot Contact Point 2 y-axis
    FL2y = ca.SX.sym('FL2y',N_K)
    FL2y_lb = np.array([[Fylb]*(FL2y.shape[0]*FL2y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL2y_ub = np.array([[Fyub]*(FL2y.shape[0]*FL2y.shape[1])])
    #Left Foot Contact Point 2 z-axis
    FL2z = ca.SX.sym('FL2z',N_K)
    FL2z_lb = np.array([[Fzlb]*(FL2z.shape[0]*FL2z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL2z_ub = np.array([[Fzub]*(FL2z.shape[0]*FL2z.shape[1])])
    #Left Foot Contact Point 3 x-axis
    FL3x = ca.SX.sym('FL3x',N_K)
    FL3x_lb = np.array([[Fxlb]*(FL3x.shape[0]*FL3x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL3x_ub = np.array([[Fxub]*(FL3x.shape[0]*FL3x.shape[1])])
    #Left Foot Contact Point 3 y-axis
    FL3y = ca.SX.sym('FL3y',N_K)
    FL3y_lb = np.array([[Fylb]*(FL3y.shape[0]*FL3y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL3y_ub = np.array([[Fyub]*(FL3y.shape[0]*FL3y.shape[1])])
    #Left Foot Contact Point 3 z-axis
    FL3z = ca.SX.sym('FL3z',N_K)
    FL3z_lb = np.array([[Fzlb]*(FL3z.shape[0]*FL3z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL3z_ub = np.array([[Fzub]*(FL3z.shape[0]*FL3z.shape[1])])
    #Left Foot Contact Point 4 x-axis
    FL4x = ca.SX.sym('FL4x',N_K)
    FL4x_lb = np.array([[Fxlb]*(FL4x.shape[0]*FL4x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL4x_ub = np.array([[Fxub]*(FL4x.shape[0]*FL4x.shape[1])])
    #Left Foot Contact Point 4 y-axis
    FL4y = ca.SX.sym('FL4y',N_K)
    FL4y_lb = np.array([[Fylb]*(FL4y.shape[0]*FL4y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL4y_ub = np.array([[Fyub]*(FL4y.shape[0]*FL4y.shape[1])])
    #Left Foot Contact Point 4 z-axis
    FL4z = ca.SX.sym('FL4z',N_K)
    FL4z_lb = np.array([[Fzlb]*(FL4z.shape[0]*FL4z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL4z_ub = np.array([[Fzub]*(FL4z.shape[0]*FL4z.shape[1])])

    #Right Contact Force x-axis
    #Right Foot Contact Point 1 x-axis
    FR1x = ca.SX.sym('FR1x',N_K)
    FR1x_lb = np.array([[Fxlb]*(FR1x.shape[0]*FR1x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR1x_ub = np.array([[Fxub]*(FR1x.shape[0]*FR1x.shape[1])])
    #Right Foot Contact Point 1 y-axis
    FR1y = ca.SX.sym('FR1y',N_K)
    FR1y_lb = np.array([[Fylb]*(FR1y.shape[0]*FR1y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR1y_ub = np.array([[Fyub]*(FR1y.shape[0]*FR1y.shape[1])])
    #Right Foot Contact Point 1 z-axis
    FR1z = ca.SX.sym('FR1z',N_K)
    FR1z_lb = np.array([[Fzlb]*(FR1z.shape[0]*FR1z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR1z_ub = np.array([[Fzub]*(FR1z.shape[0]*FR1z.shape[1])])
    #Right Foot Contact Point 2 x-axis
    FR2x = ca.SX.sym('FR2x',N_K)
    FR2x_lb = np.array([[Fxlb]*(FR2x.shape[0]*FR2x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR2x_ub = np.array([[Fxub]*(FR2x.shape[0]*FR2x.shape[1])])
    #Right Foot Contact Point 2 y-axis
    FR2y = ca.SX.sym('FR2y',N_K)
    FR2y_lb = np.array([[Fylb]*(FR2y.shape[0]*FR2y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR2y_ub = np.array([[Fyub]*(FR2y.shape[0]*FR2y.shape[1])])
    #Right Foot Contact Point 2 z-axis
    FR2z = ca.SX.sym('FR2z',N_K)
    FR2z_lb = np.array([[Fzlb]*(FR2z.shape[0]*FR2z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR2z_ub = np.array([[Fzub]*(FR2z.shape[0]*FR2z.shape[1])])
    #Right Foot Contact Point 3 x-axis
    FR3x = ca.SX.sym('FR3x',N_K)
    FR3x_lb = np.array([[Fxlb]*(FR3x.shape[0]*FR3x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR3x_ub = np.array([[Fxub]*(FR3x.shape[0]*FR3x.shape[1])])
    #Right Foot Contact Point 3 y-axis
    FR3y = ca.SX.sym('FR3y',N_K)
    FR3y_lb = np.array([[Fylb]*(FR3y.shape[0]*FR3y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR3y_ub = np.array([[Fyub]*(FR3y.shape[0]*FR3y.shape[1])])
    #Right Foot Contact Point 3 z-axis
    FR3z = ca.SX.sym('FR3z',N_K)
    FR3z_lb = np.array([[Fzlb]*(FR3z.shape[0]*FR3z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR3z_ub = np.array([[Fzub]*(FR3z.shape[0]*FR3z.shape[1])])
    #Right Foot Contact Point 4 x-axis
    FR4x = ca.SX.sym('FR4x',N_K)
    FR4x_lb = np.array([[Fxlb]*(FR4x.shape[0]*FR4x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR4x_ub = np.array([[Fxub]*(FR4x.shape[0]*FR4x.shape[1])])
    #Right Foot Contact Point 4 y-axis
    FR4y = ca.SX.sym('FR4y',N_K)
    FR4y_lb = np.array([[Fylb]*(FR4y.shape[0]*FR4y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR4y_ub = np.array([[Fyub]*(FR4y.shape[0]*FR4y.shape[1])])
    #Right Foot Contact Point 4 z-axis
    FR4z = ca.SX.sym('FR4z',N_K)
    FR4z_lb = np.array([[Fzlb]*(FR4z.shape[0]*FR4z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR4z_ub = np.array([[Fzub]*(FR4z.shape[0]*FR4z.shape[1])])

    #Initial Contact Location (need to connect to the first level)
    #   Plx
    px_init = ca.SX.sym('px_init')
    px_init_lb = np.array([-1])
    px_init_ub = np.array([30])

    #   py
    py_init = ca.SX.sym('py_init')
    py_init_lb = np.array([-1])
    py_init_ub = np.array([1])

    #   pz
    pz_init = ca.SX.sym('pz_init')
    pz_init_lb = np.array([-5])
    pz_init_ub = np.array([5])

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
        px_lb.append(np.array([-1]))
        px_ub.append(np.array([30]))

        pytemp = ca.SX.sym('py'+str(stepIdx+1))
        py.append(pytemp)
        py_lb.append(np.array([-1]))
        py_ub.append(np.array([1]))

        #   Foot steps are all staying on the ground
        pztemp = ca.SX.sym('pz'+str(stepIdx+1))
        pz.append(pztemp)
        pz_lb.append(np.array([-5]))
        pz_ub.append(np.array([5]))

    #Switching Time Vector
    Ts = []
    Ts_lb = []
    Ts_ub = []
    for n_phase in range(Nphase):
        Tstemp = ca.SX.sym('Ts'+str(n_phase+1)) #0 + 1 + ....
        Ts.append(Tstemp)
        Ts_lb.append(np.array([0.05]))
        Ts_ub.append(np.array([2.0*(Nphase+1)]))

    cL1x_p = ca.SX.sym('cL1x_p',N_K)
    cL1x_p_lb = np.array([[p_lb]*(cL1x_p.shape[0]*cL1x_p.shape[1])])
    cL1x_p_ub = np.array([[p_ub]*(cL1x_p.shape[0]*cL1x_p.shape[1])])

    cL1x_q = ca.SX.sym('cL1x_q',N_K)
    cL1x_q_lb = np.array([[q_lb]*(cL1x_q.shape[0]*cL1x_q.shape[1])])
    cL1x_q_ub = np.array([[q_ub]*(cL1x_q.shape[0]*cL1x_q.shape[1])])

    cL1y_p = ca.SX.sym('cL1y_p',N_K)
    cL1y_p_lb = np.array([[p_lb]*(cL1y_p.shape[0]*cL1y_p.shape[1])])
    cL1y_p_ub = np.array([[p_ub]*(cL1y_p.shape[0]*cL1y_p.shape[1])])

    cL1y_q = ca.SX.sym('cL1y_q',N_K)
    cL1y_q_lb = np.array([[q_lb]*(cL1y_q.shape[0]*cL1y_q.shape[1])])
    cL1y_q_ub = np.array([[q_ub]*(cL1y_q.shape[0]*cL1y_q.shape[1])])

    cL1z_p = ca.SX.sym('cL1z_p',N_K)
    cL1z_p_lb = np.array([[p_lb]*(cL1z_p.shape[0]*cL1z_p.shape[1])])
    cL1z_p_ub = np.array([[p_ub]*(cL1z_p.shape[0]*cL1z_p.shape[1])])

    cL1z_q = ca.SX.sym('cL1z_q',N_K)
    cL1z_q_lb = np.array([[q_lb]*(cL1z_q.shape[0]*cL1z_q.shape[1])])
    cL1z_q_ub = np.array([[q_ub]*(cL1z_q.shape[0]*cL1z_q.shape[1])])
    
    cL2x_p = ca.SX.sym('cL2x_p',N_K)
    cL2x_p_lb = np.array([[p_lb]*(cL2x_p.shape[0]*cL2x_p.shape[1])])
    cL2x_p_ub = np.array([[p_ub]*(cL2x_p.shape[0]*cL2x_p.shape[1])])

    cL2x_q = ca.SX.sym('cL2x_q',N_K)
    cL2x_q_lb = np.array([[q_lb]*(cL2x_q.shape[0]*cL2x_q.shape[1])])
    cL2x_q_ub = np.array([[q_ub]*(cL2x_q.shape[0]*cL2x_q.shape[1])])

    cL2y_p = ca.SX.sym('cL2y_p',N_K)
    cL2y_p_lb = np.array([[p_lb]*(cL2y_p.shape[0]*cL2y_p.shape[1])])
    cL2y_p_ub = np.array([[p_ub]*(cL2y_p.shape[0]*cL2y_p.shape[1])])

    cL2y_q = ca.SX.sym('cL2y_q',N_K)
    cL2y_q_lb = np.array([[q_lb]*(cL2y_q.shape[0]*cL2y_q.shape[1])])
    cL2y_q_ub = np.array([[q_ub]*(cL2y_q.shape[0]*cL2y_q.shape[1])])

    cL2z_p = ca.SX.sym('cL2z_p',N_K)
    cL2z_p_lb = np.array([[p_lb]*(cL2z_p.shape[0]*cL2z_p.shape[1])])
    cL2z_p_ub = np.array([[p_ub]*(cL2z_p.shape[0]*cL2z_p.shape[1])])

    cL2z_q = ca.SX.sym('cL2z_q',N_K)
    cL2z_q_lb = np.array([[q_lb]*(cL2z_q.shape[0]*cL2z_q.shape[1])])
    cL2z_q_ub = np.array([[q_ub]*(cL2z_q.shape[0]*cL2z_q.shape[1])])
    
    cL3x_p = ca.SX.sym('cL3x_p',N_K)
    cL3x_p_lb = np.array([[p_lb]*(cL3x_p.shape[0]*cL3x_p.shape[1])])
    cL3x_p_ub = np.array([[p_ub]*(cL3x_p.shape[0]*cL3x_p.shape[1])])

    cL3x_q = ca.SX.sym('cL3x_q',N_K)
    cL3x_q_lb = np.array([[q_lb]*(cL3x_q.shape[0]*cL3x_q.shape[1])])
    cL3x_q_ub = np.array([[q_ub]*(cL3x_q.shape[0]*cL3x_q.shape[1])])

    cL3y_p = ca.SX.sym('cL3y_p',N_K)
    cL3y_p_lb = np.array([[p_lb]*(cL3y_p.shape[0]*cL3y_p.shape[1])])
    cL3y_p_ub = np.array([[p_ub]*(cL3y_p.shape[0]*cL3y_p.shape[1])])

    cL3y_q = ca.SX.sym('cL3y_q',N_K)
    cL3y_q_lb = np.array([[q_lb]*(cL3y_q.shape[0]*cL3y_q.shape[1])])
    cL3y_q_ub = np.array([[q_ub]*(cL3y_q.shape[0]*cL3y_q.shape[1])])

    cL3z_p = ca.SX.sym('cL3z_p',N_K)
    cL3z_p_lb = np.array([[p_lb]*(cL3z_p.shape[0]*cL3z_p.shape[1])])
    cL3z_p_ub = np.array([[p_ub]*(cL3z_p.shape[0]*cL3z_p.shape[1])])

    cL3z_q = ca.SX.sym('cL3z_q',N_K)
    cL3z_q_lb = np.array([[q_lb]*(cL3z_q.shape[0]*cL3z_q.shape[1])])
    cL3z_q_ub = np.array([[q_ub]*(cL3z_q.shape[0]*cL3z_q.shape[1])])

    cL4x_p = ca.SX.sym('cL4x_p',N_K)
    cL4x_p_lb = np.array([[p_lb]*(cL4x_p.shape[0]*cL4x_p.shape[1])])
    cL4x_p_ub = np.array([[p_ub]*(cL4x_p.shape[0]*cL4x_p.shape[1])])

    cL4x_q = ca.SX.sym('cL4x_q',N_K)
    cL4x_q_lb = np.array([[q_lb]*(cL4x_q.shape[0]*cL4x_q.shape[1])])
    cL4x_q_ub = np.array([[q_ub]*(cL4x_q.shape[0]*cL4x_q.shape[1])])

    cL4y_p = ca.SX.sym('cL4y_p',N_K)
    cL4y_p_lb = np.array([[p_lb]*(cL4y_p.shape[0]*cL4y_p.shape[1])])
    cL4y_p_ub = np.array([[p_ub]*(cL4y_p.shape[0]*cL4y_p.shape[1])])

    cL4y_q = ca.SX.sym('cL4y_q',N_K)
    cL4y_q_lb = np.array([[q_lb]*(cL4y_q.shape[0]*cL4y_q.shape[1])])
    cL4y_q_ub = np.array([[q_ub]*(cL4y_q.shape[0]*cL4y_q.shape[1])])

    cL4z_p = ca.SX.sym('cL4z_p',N_K)
    cL4z_p_lb = np.array([[p_lb]*(cL4z_p.shape[0]*cL4z_p.shape[1])])
    cL4z_p_ub = np.array([[p_ub]*(cL4z_p.shape[0]*cL4z_p.shape[1])])

    cL4z_q = ca.SX.sym('cL4z_q',N_K)
    cL4z_q_lb = np.array([[q_lb]*(cL4z_q.shape[0]*cL4z_q.shape[1])])
    cL4z_q_ub = np.array([[q_ub]*(cL4z_q.shape[0]*cL4z_q.shape[1])])    

    cR1x_p = ca.SX.sym('cR1x_p',N_K)
    cR1x_p_lb = np.array([[p_lb]*(cR1x_p.shape[0]*cR1x_p.shape[1])])
    cR1x_p_ub = np.array([[p_ub]*(cR1x_p.shape[0]*cR1x_p.shape[1])])

    cR1x_q = ca.SX.sym('cR1x_q',N_K)
    cR1x_q_lb = np.array([[q_lb]*(cR1x_q.shape[0]*cR1x_q.shape[1])])
    cR1x_q_ub = np.array([[q_ub]*(cR1x_q.shape[0]*cR1x_q.shape[1])])

    cR1y_p = ca.SX.sym('cR1y_p',N_K)
    cR1y_p_lb = np.array([[p_lb]*(cR1y_p.shape[0]*cR1y_p.shape[1])])
    cR1y_p_ub = np.array([[p_ub]*(cR1y_p.shape[0]*cR1y_p.shape[1])])

    cR1y_q = ca.SX.sym('cR1y_q',N_K)
    cR1y_q_lb = np.array([[q_lb]*(cR1y_q.shape[0]*cR1y_q.shape[1])])
    cR1y_q_ub = np.array([[q_ub]*(cR1y_q.shape[0]*cR1y_q.shape[1])])

    cR1z_p = ca.SX.sym('cR1z_p',N_K)
    cR1z_p_lb = np.array([[p_lb]*(cR1z_p.shape[0]*cR1z_p.shape[1])])
    cR1z_p_ub = np.array([[p_ub]*(cR1z_p.shape[0]*cR1z_p.shape[1])])

    cR1z_q = ca.SX.sym('cR1z_q',N_K)
    cR1z_q_lb = np.array([[q_lb]*(cR1z_q.shape[0]*cR1z_q.shape[1])])
    cR1z_q_ub = np.array([[q_ub]*(cR1z_q.shape[0]*cR1z_q.shape[1])])

    cR2x_p = ca.SX.sym('cR2x_p',N_K)
    cR2x_p_lb = np.array([[p_lb]*(cR2x_p.shape[0]*cR2x_p.shape[1])])
    cR2x_p_ub = np.array([[p_ub]*(cR2x_p.shape[0]*cR2x_p.shape[1])])

    cR2x_q = ca.SX.sym('cR2x_q',N_K)
    cR2x_q_lb = np.array([[q_lb]*(cR2x_q.shape[0]*cR2x_q.shape[1])])
    cR2x_q_ub = np.array([[q_ub]*(cR2x_q.shape[0]*cR2x_q.shape[1])])

    cR2y_p = ca.SX.sym('cR2y_p',N_K)
    cR2y_p_lb = np.array([[p_lb]*(cR2y_p.shape[0]*cR2y_p.shape[1])])
    cR2y_p_ub = np.array([[p_ub]*(cR2y_p.shape[0]*cR2y_p.shape[1])])

    cR2y_q = ca.SX.sym('cR2y_q',N_K)
    cR2y_q_lb = np.array([[q_lb]*(cR2y_q.shape[0]*cR2y_q.shape[1])])
    cR2y_q_ub = np.array([[q_ub]*(cR2y_q.shape[0]*cR2y_q.shape[1])])

    cR2z_p = ca.SX.sym('cR2z_p',N_K)
    cR2z_p_lb = np.array([[p_lb]*(cR2z_p.shape[0]*cR2z_p.shape[1])])
    cR2z_p_ub = np.array([[p_ub]*(cR2z_p.shape[0]*cR2z_p.shape[1])])

    cR2z_q = ca.SX.sym('cR2z_q',N_K)
    cR2z_q_lb = np.array([[q_lb]*(cR2z_q.shape[0]*cR2z_q.shape[1])])
    cR2z_q_ub = np.array([[q_ub]*(cR2z_q.shape[0]*cR2z_q.shape[1])])

    cR3x_p = ca.SX.sym('cR3x_p',N_K)
    cR3x_p_lb = np.array([[p_lb]*(cR3x_p.shape[0]*cR3x_p.shape[1])])
    cR3x_p_ub = np.array([[p_ub]*(cR3x_p.shape[0]*cR3x_p.shape[1])])

    cR3x_q = ca.SX.sym('cR3x_q',N_K)
    cR3x_q_lb = np.array([[q_lb]*(cR3x_q.shape[0]*cR3x_q.shape[1])])
    cR3x_q_ub = np.array([[q_ub]*(cR3x_q.shape[0]*cR3x_q.shape[1])])

    cR3y_p = ca.SX.sym('cR3y_p',N_K)
    cR3y_p_lb = np.array([[p_lb]*(cR3y_p.shape[0]*cR3y_p.shape[1])])
    cR3y_p_ub = np.array([[p_ub]*(cR3y_p.shape[0]*cR3y_p.shape[1])])

    cR3y_q = ca.SX.sym('cR3y_q',N_K)
    cR3y_q_lb = np.array([[q_lb]*(cR3y_q.shape[0]*cR3y_q.shape[1])])
    cR3y_q_ub = np.array([[q_ub]*(cR3y_q.shape[0]*cR3y_q.shape[1])])

    cR3z_p = ca.SX.sym('cR3z_p',N_K)
    cR3z_p_lb = np.array([[p_lb]*(cR3z_p.shape[0]*cR3z_p.shape[1])])
    cR3z_p_ub = np.array([[p_ub]*(cR3z_p.shape[0]*cR3z_p.shape[1])])

    cR3z_q = ca.SX.sym('cR3z_q',N_K)
    cR3z_q_lb = np.array([[q_lb]*(cR3z_q.shape[0]*cR3z_q.shape[1])])
    cR3z_q_ub = np.array([[q_ub]*(cR3z_q.shape[0]*cR3z_q.shape[1])])
    #--------------
    cR4x_p = ca.SX.sym('cR4x_p',N_K)
    cR4x_p_lb = np.array([[p_lb]*(cR4x_p.shape[0]*cR4x_p.shape[1])])
    cR4x_p_ub = np.array([[p_ub]*(cR4x_p.shape[0]*cR4x_p.shape[1])])

    cR4x_q = ca.SX.sym('cR4x_q',N_K)
    cR4x_q_lb = np.array([[q_lb]*(cR4x_q.shape[0]*cR4x_q.shape[1])])
    cR4x_q_ub = np.array([[q_ub]*(cR4x_q.shape[0]*cR4x_q.shape[1])])

    cR4y_p = ca.SX.sym('cR4y_p',N_K)
    cR4y_p_lb = np.array([[p_lb]*(cR4y_p.shape[0]*cR4y_p.shape[1])])
    cR4y_p_ub = np.array([[p_ub]*(cR4y_p.shape[0]*cR4y_p.shape[1])])

    cR4y_q = ca.SX.sym('cR4y_q',N_K)
    cR4y_q_lb = np.array([[q_lb]*(cR4y_q.shape[0]*cR4y_q.shape[1])])
    cR4y_q_ub = np.array([[q_ub]*(cR4y_q.shape[0]*cR4y_q.shape[1])])

    cR4z_p = ca.SX.sym('cR4z_p',N_K)
    cR4z_p_lb = np.array([[p_lb]*(cR4z_p.shape[0]*cR4z_p.shape[1])])
    cR4z_p_ub = np.array([[p_ub]*(cR4z_p.shape[0]*cR4z_p.shape[1])])

    cR4z_q = ca.SX.sym('cR4z_q',N_K)
    cR4z_q_lb = np.array([[q_lb]*(cR4z_q.shape[0]*cR4z_q.shape[1])])
    cR4z_q_ub = np.array([[q_ub]*(cR4z_q.shape[0]*cR4z_q.shape[1])])

    #   Collect all Decision Variables
    DecisionVars = ca.vertcat(x,y,z,xdot,ydot,zdot,FL1x,FL1y,FL1z,FL2x,FL2y,FL2z,FL3x,FL3y,FL3z,FL4x,FL4y,FL4z,FR1x,FR1y,FR1z,FR2x,FR2y,FR2z,FR3x,FR3y,FR3z,FR4x,FR4y,FR4z,px_init,py_init,pz_init,*px,*py,*pz,*Ts,cL1x_p,cL1x_q,cL1y_p,cL1y_q,cL1z_p,cL1z_q,cL2x_p,cL2x_q,cL2y_p,cL2y_q,cL2z_p,cL2z_q,cL3x_p,cL3x_q,cL3y_p,cL3y_q,cL3z_p,cL3z_q,cL4x_p,cL4x_q,cL4y_p,cL4y_q,cL4z_p,cL4z_q,cR1x_p,cR1x_q,cR1y_p,cR1y_q,cR1z_p,cR1z_q,cR2x_p,cR2x_q,cR2y_p,cR2y_q,cR2z_p,cR2z_q,cR3x_p,cR3x_q,cR3y_p,cR3y_q,cR3z_p,cR3z_q,cR4x_p,cR4x_q,cR4y_p,cR4y_q,cR4z_p,cR4z_q)
    #print(DecisionVars)
    DecisionVarsShape = DecisionVars.shape

    #   Collect all lower bound and upper bound
    DecisionVars_lb = np.concatenate(((x_lb,y_lb,z_lb,xdot_lb,ydot_lb,zdot_lb,FL1x_lb,FL1y_lb,FL1z_lb,FL2x_lb,FL2y_lb,FL2z_lb,FL3x_lb,FL3y_lb,FL3z_lb,FL4x_lb,FL4y_lb,FL4z_lb,FR1x_lb,FR1y_lb,FR1z_lb,FR2x_lb,FR2y_lb,FR2z_lb,FR3x_lb,FR3y_lb,FR3z_lb,FR4x_lb,FR4y_lb,FR4z_lb,px_init_lb,py_init_lb,pz_init_lb,px_lb,py_lb,pz_lb,Ts_lb,cL1x_p_lb,cL1x_q_lb,cL1y_p_lb,cL1y_q_lb,cL1z_p_lb,cL1z_q_lb,cL2x_p_lb,cL2x_q_lb,cL2y_p_lb,cL2y_q_lb,cL2z_p_lb,cL2z_q_lb,cL3x_p_lb,cL3x_q_lb,cL3y_p_lb,cL3y_q_lb,cL3z_p_lb,cL3z_q_lb,cL4x_p_lb,cL4x_q_lb,cL4y_p_lb,cL4y_q_lb,cL4z_p_lb,cL4z_q_lb,cR1x_p_lb,cR1x_q_lb,cR1y_p_lb,cR1y_q_lb,cR1z_p_lb,cR1z_q_lb,cR2x_p_lb,cR2x_q_lb,cR2y_p_lb,cR2y_q_lb,cR2z_p_lb,cR2z_q_lb,cR3x_p_lb,cR3x_q_lb,cR3y_p_lb,cR3y_q_lb,cR3z_p_lb,cR3z_q_lb,cR4x_p_lb,cR4x_q_lb,cR4y_p_lb,cR4y_q_lb,cR4z_p_lb,cR4z_q_lb)),axis=None)
    DecisionVars_ub = np.concatenate(((x_ub,y_ub,z_ub,xdot_ub,ydot_ub,zdot_ub,FL1x_ub,FL1y_ub,FL1z_ub,FL2x_ub,FL2y_ub,FL2z_ub,FL3x_ub,FL3y_ub,FL3z_ub,FL4x_ub,FL4y_ub,FL4z_ub,FR1x_ub,FR1y_ub,FR1z_ub,FR2x_ub,FR2y_ub,FR2z_ub,FR3x_ub,FR3y_ub,FR3z_ub,FR4x_ub,FR4y_ub,FR4z_ub,px_init_ub,py_init_ub,pz_init_ub,px_ub,py_ub,pz_ub,Ts_ub,cL1x_p_ub,cL1x_q_ub,cL1y_p_ub,cL1y_q_ub,cL1z_p_ub,cL1z_q_ub,cL2x_p_ub,cL2x_q_ub,cL2y_p_ub,cL2y_q_ub,cL2z_p_ub,cL2z_q_ub,cL3x_p_ub,cL3x_q_ub,cL3y_p_ub,cL3y_q_ub,cL3z_p_ub,cL3z_q_ub,cL4x_p_ub,cL4x_q_ub,cL4y_p_ub,cL4y_q_ub,cL4z_p_ub,cL4z_q_ub,cR1x_p_ub,cR1x_q_ub,cR1y_p_ub,cR1y_q_ub,cR1z_p_ub,cR1z_q_ub,cR2x_p_ub,cR2x_q_ub,cR2y_p_ub,cR2y_q_ub,cR2z_p_ub,cR2z_q_ub,cR3x_p_ub,cR3x_q_ub,cR3y_p_ub,cR3y_q_ub,cR3z_p_ub,cR3z_q_ub,cR4x_p_ub,cR4x_q_ub,cR4y_p_ub,cR4y_q_ub,cR4z_p_ub,cR4z_q_ub)),axis=None)

    #Time Span Setup
    tau_upper_limit = 1
    tauStepLength = tau_upper_limit/(N_K-1) #Get the interval length, total number of knots - 1

    #-----------------------------------------------------------------------------------------------------------------------
    #Define Constrains and Running Cost
    g = []
    glb = []
    gub = []
    J = 0

    #Initial and Terminal Condition

    ##   Terminal CoM y-axis
    #g.append(y[-1]-y_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    ##   Terminal CoM z-axis
    #g.append(z[-1]-z_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    #if StaticStop == True:
    #    #   Terminal Zero CoM velocity x-axis
    #    g.append(xdot[-1])
    #    glb.append(np.array([0]))
    #    gub.append(np.array([0]))

    #    #   Terminal Zero CoM velocity y-axis
    #    g.append(ydot[-1])
    #    glb.append(np.array([0]))
    #    gub.append(np.array([0]))

    #    #   Terminal Zero CoM velocity z-axis
    #    g.append(zdot[-1])
    #    glb.append(np.array([0]))
    #    gub.append(np.array([0]))

    #   Terminal Angular Momentum x-axis
    #g.append(Lx[-1]-Lx_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    #   Terminal Angular Momentum y-axis
    #g.append(Ly[-1]-Ly_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    #   Terminal Angular Momentum z-axis
    #g.append(Lz[-1]-Lz_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    #   Terminal Angular Momentum Rate x-axis
    #g.append(Ldotx[-1]-Ldotx_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    #   Terminal Angular Momentum Rate y-axis
    #g.append(Ldoty[-1]-Ldoty_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    #   Terminal Angular Momentum Rate z-axis
    #g.append(Ldotz[-1]-Ldotz_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    #Loop over all Phases (Knots)
    for Nph in range(Nphase):
        #Decide Number of Knots
        if Nph == Nphase-1:  #The last Knot belongs to the Last Phase
            Nk_ThisPhase = Nk_Local+1
        else:
            Nk_ThisPhase = Nk_Local       

        #Decide Time Vector
        #if Nph == 0: #first phase
        #    h = tauStepLength*Nphase*(Ts[Nph]-0)
        #else: #other phases
        #    h = tauStepLength*Nphase*(Ts[Nph]-Ts[Nph-1]) 

        #Fixed Time Step
        if GaitPattern[Nph]=='InitialDouble':
            h = h_initialdouble/Nk_Local
        elif GaitPattern[Nph]=='Swing':
            h = h_swing/Nk_Local
        elif GaitPattern[Nph]=='DoubleSupport':
            h = h_doublesupport/Nk_Local

        for Local_k_Count in range(Nk_ThisPhase):
            #Get knot number across the entire time line
            k = Nph*Nk_Local + Local_k_Count
            #print(k)

            print(h)
            #------------------------------------------
            #Build useful vectors
            #   Forces
            FL1_k = ca.vertcat(FL1x[k],FL1y[k],FL1z[k])
            FL2_k = ca.vertcat(FL2x[k],FL2y[k],FL2z[k])
            FL3_k = ca.vertcat(FL3x[k],FL3y[k],FL3z[k])
            FL4_k = ca.vertcat(FL4x[k],FL4y[k],FL4z[k])

            FR1_k = ca.vertcat(FR1x[k],FR1y[k],FR1z[k])
            FR2_k = ca.vertcat(FR2x[k],FR2y[k],FR2z[k])
            FR3_k = ca.vertcat(FR3x[k],FR3y[k],FR3z[k])
            FR4_k = ca.vertcat(FR4x[k],FR4y[k],FR4z[k])
            #   CoM
            CoM_k = ca.vertcat(x[k],y[k],z[k])
            #   Angular Momentum
            #if k<N_K-1: #N_K-1 the enumeration of the last knot, k<N_K-1 the one before the last knot
            #    Ldot_current = ca.vertcat(Ldotx[k],Ldoty[k],Ldotz[k])
            #    Ldot_next = ca.vertcat(Ldotz[k+1],Ldotz[k+1],Ldotz[k+1])
            #-------------------------------------------

            #-------------------------------------------
            #Phase dependent Constraints (CoM Kinematics and Angular Dynamics)
            if GaitPattern[Nph]=='InitialDouble':
                #initial support foot (the landing foot from the first phase)
                p_init = ca.vertcat(px_init,py_init,pz_init)
                p_init_TangentX = SurfTangentsX[0:3]
                p_init_TangentY = SurfTangentsY[0:3]
                p_init_Norm = SurfNorms[0:3]
                
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

                #Angular Dynamics Cost
                #if k<N_K-1: #double check the knot number is valid
                #    g.append(ca.if_else(ParaLeftSwingFlag,Ldot_next - Ldot_current - h*(ca.cross((p_init+0.11*p_init_TangentX+0.06*p_init_TangentY-CoM_k),FL1_k) + 
                #                                                                        ca.cross((p_init+0.11*p_init_TangentX-0.06*p_init_TangentY-CoM_k),FL2_k) + 
                #                                                                        ca.cross((p_init-0.11*p_init_TangentX+0.06*p_init_TangentY-CoM_k),FL3_k) + 
                #                                                                        ca.cross((p_init-0.11*p_init_TangentX-0.06*p_init_TangentY-CoM_k),FL4_k) + 
                #                                                                        ca.cross((PR_init+0.11*PR_init_TangentX+0.06*PR_init_TangentY-CoM_k),FR1_k) + 
                #                                                                        ca.cross((PR_init+0.11*PR_init_TangentX-0.06*PR_init_TangentY-CoM_k),FR2_k) + 
                #                                                                        ca.cross((PR_init-0.11*PR_init_TangentX+0.06*PR_init_TangentY-CoM_k),FR3_k) + 
                #                                                                        ca.cross((PR_init-0.11*PR_init_TangentX-0.06*PR_init_TangentY-CoM_k),FR4_k)),np.array([0,0,0])))
                #    #g.append(ca.if_else(ParaLeftSwingFlag,Ldot_next-Ldot_current-h*(ca.cross((p_init+np.array([0.11,0.06,0])-CoM_k),FL1_k)+ca.cross((p_init+np.array([0.11,-0.06,0])-CoM_k),FL2_k)+ca.cross((p_init+np.array([-0.11,0.06,0])-CoM_k),FL3_k)+ca.cross((p_init+np.array([-0.11,-0.06,0])-CoM_k),FL4_k)+ca.cross((PR_init+np.array([0.11,0.06,0])-CoM_k),FR1_k)+ca.cross((PR_init+np.array([0.11,-0.06,0])-CoM_k),FR2_k)+ca.cross((PR_init+np.array([-0.11,0.06,0])-CoM_k),FR3_k)+ca.cross((PR_init+np.array([-0.11,-0.06,0])-CoM_k),FR4_k)),np.array([0,0,0])))
                #    glb.append(np.array([0,0,0]))
                #    gub.append(np.array([0,0,0]))
                if k<N_K-1: #double check the knot number is valid
                    Leg_vec = p_init+0.11*p_init_TangentX+0.06*p_init_TangentY-CoM_k
                    forcevec = FL1_k
                    g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cL1x_p[k],x_q_bar = cL1x_q[k], y_p_bar = cL1y_p[k], y_q_bar = cL1y_q[k], z_p_bar = cL1z_p[k], z_q_bar = cL1z_q[k], l = Leg_vec,f = forcevec)

                    Leg_vec = p_init+0.11*p_init_TangentX-0.06*p_init_TangentY-CoM_k
                    forcevec = FL2_k
                    g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cL2x_p[k],x_q_bar = cL2x_q[k], y_p_bar = cL2y_p[k], y_q_bar = cL2y_q[k], z_p_bar = cL2z_p[k], z_q_bar = cL2z_q[k], l = Leg_vec,f = forcevec)

                    Leg_vec = p_init-0.11*p_init_TangentX+0.06*p_init_TangentY-CoM_k
                    forcevec = FL3_k
                    g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cL3x_p[k],x_q_bar = cL3x_q[k], y_p_bar = cL3y_p[k], y_q_bar = cL3y_q[k], z_p_bar = cL3z_p[k], z_q_bar = cL3z_q[k], l = Leg_vec,f = forcevec)

                    Leg_vec = p_init-0.11*p_init_TangentX-0.06*p_init_TangentY-CoM_k
                    forcevec = FL4_k
                    g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cL4x_p[k],x_q_bar = cL4x_q[k], y_p_bar = cL4y_p[k], y_q_bar = cL4y_q[k], z_p_bar = cL4z_p[k], z_q_bar = cL4z_q[k], l = Leg_vec,f = forcevec)

                    Leg_vec = PR_init+0.11*PR_init_TangentX+0.06*PR_init_TangentY-CoM_k
                    forcevec = FR1_k
                    g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cR1x_p[k],x_q_bar = cR1x_q[k], y_p_bar = cR1y_p[k], y_q_bar = cR1y_q[k], z_p_bar = cR1z_p[k], z_q_bar = cR1z_q[k], l = Leg_vec,f = forcevec)

                    Leg_vec = PR_init+0.11*PR_init_TangentX-0.06*PR_init_TangentY-CoM_k
                    forcevec = FR2_k
                    g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cR2x_p[k],x_q_bar = cR2x_q[k], y_p_bar = cR2y_p[k], y_q_bar = cR2y_q[k], z_p_bar = cR2z_p[k], z_q_bar = cR2z_q[k], l = Leg_vec,f = forcevec)

                    Leg_vec = PR_init-0.11*PR_init_TangentX+0.06*PR_init_TangentY-CoM_k
                    forcevec = FR3_k
                    g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cR3x_p[k],x_q_bar = cR3x_q[k], y_p_bar = cR3y_p[k], y_q_bar = cR3y_q[k], z_p_bar = cR3z_p[k], z_q_bar = cR3z_q[k], l = Leg_vec,f = forcevec)

                    Leg_vec = PR_init-0.11*PR_init_TangentX-0.06*PR_init_TangentY-CoM_k
                    forcevec = FR4_k
                    g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cR4x_p[k],x_q_bar = cR4x_q[k], y_p_bar = cR4y_p[k], y_q_bar = cR4y_q[k], z_p_bar = cR4z_p[k], z_q_bar = cR4z_q[k], l = Leg_vec,f = forcevec)

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

                #Angular Dynamics Cost

                #if k<N_K-1: #double check the knot number is valid
                #    g.append(ca.if_else(ParaRightSwingFlag,Ldot_next - Ldot_current - h*(ca.cross((PL_init+0.11*PL_init_TangentX+0.06*PL_init_TangentY-CoM_k),FL1_k) + 
                #                                                                        ca.cross((PL_init+0.11*PL_init_TangentX-0.06*PL_init_TangentY-CoM_k),FL2_k) + 
                #                                                                        ca.cross((PL_init-0.11*PL_init_TangentX+0.06*PL_init_TangentY-CoM_k),FL3_k) + 
                #                                                                        ca.cross((PL_init-0.11*PL_init_TangentX-0.06*PL_init_TangentY-CoM_k),FL4_k) + 
                #                                                                        ca.cross((p_init+0.11*p_init_TangentX+0.06*p_init_TangentY-CoM_k),FR1_k) + 
                #                                                                        ca.cross((p_init+0.11*p_init_TangentX-0.06*p_init_TangentY-CoM_k),FR2_k) + 
                #                                                                        ca.cross((p_init-0.11*p_init_TangentX+0.06*p_init_TangentY-CoM_k),FR3_k) + 
                #                                                                        ca.cross((p_init-0.11*p_init_TangentX-0.06*p_init_TangentY-CoM_k),FR4_k)),np.array([0,0,0])))
                #    #g.append(ca.if_else(ParaRightSwingFlag,Ldot_next-Ldot_current-h*(ca.cross((PL_init+np.array([0.11,0.06,0])-CoM_k),FL1_k)+ca.cross((PL_init+np.array([0.11,-0.06,0])-CoM_k),FL2_k)+ca.cross((PL_init+np.array([-0.11,0.06,0])-CoM_k),FL3_k)+ca.cross((PL_init+np.array([-0.11,-0.06,0])-CoM_k),FL4_k)+ca.cross((p_init+np.array([0.11,0.06,0])-CoM_k),FR1_k)+ca.cross((p_init+np.array([0.11,-0.06,0])-CoM_k),FR2_k)+ca.cross((p_init+np.array([-0.11,0.06,0])-CoM_k),FR3_k)+ca.cross((p_init+np.array([-0.11,-0.06,0])-CoM_k),FR4_k)),np.array([0,0,0])))
                #    glb.append(np.array([0,0,0]))
                #    gub.append(np.array([0,0,0]))
                if k<N_K-1: #double check the knot number is valid
                    Leg_vec = PL_init+0.11*PL_init_TangentX+0.06*PL_init_TangentY-CoM_k
                    forcevec = FL1_k
                    g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cL1x_p[k],x_q_bar = cL1x_q[k], y_p_bar = cL1y_p[k], y_q_bar = cL1y_q[k], z_p_bar = cL1z_p[k], z_q_bar = cL1z_q[k], l = Leg_vec,f = forcevec)

                    Leg_vec = PL_init+0.11*PL_init_TangentX-0.06*PL_init_TangentY-CoM_k
                    forcevec = FL2_k
                    g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cL2x_p[k],x_q_bar = cL2x_q[k], y_p_bar = cL2y_p[k], y_q_bar = cL2y_q[k], z_p_bar = cL2z_p[k], z_q_bar = cL2z_q[k], l = Leg_vec,f = forcevec)

                    Leg_vec = PL_init-0.11*PL_init_TangentX+0.06*PL_init_TangentY-CoM_k
                    forcevec = FL3_k
                    g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cL3x_p[k],x_q_bar = cL3x_q[k], y_p_bar = cL3y_p[k], y_q_bar = cL3y_q[k], z_p_bar = cL3z_p[k], z_q_bar = cL3z_q[k], l = Leg_vec,f = forcevec)

                    Leg_vec = PL_init-0.11*PL_init_TangentX-0.06*PL_init_TangentY-CoM_k
                    forcevec = FL4_k
                    g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cL4x_p[k],x_q_bar = cL4x_q[k], y_p_bar = cL4y_p[k], y_q_bar = cL4y_q[k], z_p_bar = cL4z_p[k], z_q_bar = cL4z_q[k], l = Leg_vec,f = forcevec)

                    Leg_vec = p_init+0.11*p_init_TangentX+0.06*p_init_TangentY-CoM_k
                    forcevec = FR1_k
                    g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cR1x_p[k],x_q_bar = cR1x_q[k], y_p_bar = cR1y_p[k], y_q_bar = cR1y_q[k], z_p_bar = cR1z_p[k], z_q_bar = cR1z_q[k], l = Leg_vec,f = forcevec)

                    Leg_vec = p_init+0.11*p_init_TangentX-0.06*p_init_TangentY-CoM_k
                    forcevec = FR2_k
                    g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cR2x_p[k],x_q_bar = cR2x_q[k], y_p_bar = cR2y_p[k], y_q_bar = cR2y_q[k], z_p_bar = cR2z_p[k], z_q_bar = cR2z_q[k], l = Leg_vec,f = forcevec)

                    Leg_vec = p_init-0.11*p_init_TangentX+0.06*p_init_TangentY-CoM_k
                    forcevec = FR3_k
                    g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cR3x_p[k],x_q_bar = cR3x_q[k], y_p_bar = cR3y_p[k], y_q_bar = cR3y_q[k], z_p_bar = cR3z_p[k], z_q_bar = cR3z_q[k], l = Leg_vec,f = forcevec)

                    Leg_vec = p_init-0.11*p_init_TangentX-0.06*p_init_TangentY-CoM_k
                    forcevec = FR4_k
                    g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cR4x_p[k],x_q_bar = cR4x_q[k], y_p_bar = cR4y_p[k], y_q_bar = cR4y_q[k], z_p_bar = cR4z_p[k], z_q_bar = cR4z_q[k], l = Leg_vec,f = forcevec)

                #Unilateral Constraint
                #Case 1
                #If the first level swing the Left foot first, then the Left foot is the landing foot (p_init), Left foot obey the unilateral constraint on p_init
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainNorm = p_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainNorm = p_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainNorm = p_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainNorm = p_init_Norm)
                #then the Right foot is obey the unilateral constraint on the PR_init
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainNorm = PR_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainNorm = PR_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainNorm = PR_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainNorm = PR_init_Norm)

                #Case 2
                #If the first level swings the Right foot first, then the right foot is the landing foot (p_init), Right foot obeys the unilateral constraint on p_init
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainNorm = p_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainNorm = p_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainNorm = p_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainNorm = p_init_Norm)
                #then the Left foot obeys the unilateral constrint on the PL_init
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainNorm = PL_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainNorm = PL_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainNorm = PL_init_Norm)
                g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainNorm = PL_init_Norm)                
                
                #Friction Cone Constraint
                #Case 1
                #If the first level swing the Left foot first, then the Left foot is the landing foot (p_init), Left foot obey the friction cone constraint on p_init
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu)
                #then the right foot obeys the friction cone constraints on the PR_init
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu)
                
                #Case 2
                #if the first level swing the right foot first, then the Right foot is the landing foot (p_init), Right foot obey the friction cone constraints on p_init
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu)
                #then the left foot obeys the friction cone constraint of PL_init
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu)
                g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu)

            elif GaitPattern[Nph]== 'Swing':

                if (Nph-1)//2 == 0:
                    #!!!!!!Pass from the first Level!!!!!!
                    P_k_current = ca.vertcat(px_init,py_init,pz_init) #ca.vertcat(px[-1],py[-1],pz[-1])
                    P_k_current_TangentX = SurfTangentsX[0:3]
                    P_k_current_TangentY = SurfTangentsY[0:3]
                    P_k_current_Norm = SurfNorms[0:3]
                    #!!!!!!
                    #P_k_next = ca.vertcat(px[Nph//2],py[Nph//2],pz[Nph//2])
                    #P_k_next_TangentX = SurfTangentsX[0:3]
                    #P_k_next_TangentX = SurfTangentsY[0:3]
                    #P_k_next_Norm = SurfNorms[0:3]
                else:
                    #Note: there are two lists, the first is the contact sequence (with total number of the steps of the whole framwork - 1)
                    P_k_current = ca.vertcat(px[Nph//2-1],py[Nph//2-1],pz[Nph//2-1])
                    #Note: and the terrain tangents and norm vector( with total number of the steps of the whole framwork)
                    #Note the length difference
                    P_k_current_TangentX = SurfTangentsX[(Nph//2)*3:(Nph//2)*3+3]
                    P_k_current_TangentY = SurfTangentsY[(Nph//2)*3:(Nph//2)*3+3]
                    P_k_current_Norm = SurfNorms[(Nph//2)*3:(Nph//2)*3+3]

                    #P_k_next = ca.vertcat(px[Nph//2],py[Nph//2],pz[Nph//2])

                if ((Nph-1)//2)%2 == 0: #even number steps

                    #Kinematics Constraint and Angular Dynamics Constraint
                    #If First Level Swing the Left, then the second level Even Number Phases (the first Phase) Swing the Right -> Left -> Right
                    #Left foot in contact for p_current, right foot is going to land as p_next
                    #   CoM in the Left Foot
                    g.append(ca.if_else(ParaLeftSwingFlag, K_CoM_Left@(CoM_k-P_k_current)-ca.DM(k_CoM_Left), np.full((len(k_CoM_Left),),-1)))
                    glb.append(np.full((len(k_CoM_Left),),-np.inf))
                    gub.append(np.full((len(k_CoM_Left),),0))

                    #Angular Dynamics (Left Support) Cost
                    
                    #if k<N_K-1:
                    #    g.append(ca.if_else(ParaLeftSwingFlag,Ldot_next - Ldot_current - h*(ca.cross((P_k_current+0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k),FL1_k) + 
                    #                                                                        ca.cross((P_k_current+0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k),FL2_k) + 
                    #                                                                        ca.cross((P_k_current-0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k),FL3_k) + 
                    #                                                                        ca.cross((P_k_current-0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k),FL4_k)),np.array([0,0,0])))
                    #    #g.append(ca.if_else(ParaLeftSwingFlag, Ldot_next-Ldot_current-h*(ca.cross((P_k_current+np.array([0.11,0.06,0])-CoM_k),FL1_k)+ca.cross((P_k_current+np.array([0.11,-0.06,0])-CoM_k),FL2_k)+ca.cross((P_k_current+np.array([-0.11,0.06,0])-CoM_k),FL3_k)+ca.cross((P_k_current+np.array([-0.11,-0.06,0])-CoM_k),FL4_k)), np.array([0,0,0])))
                    #    glb.append(np.array([0,0,0]))
                    #    gub.append(np.array([0,0,0]))
                    if k<N_K-1: #double check the knot number is valid

                        Leg_vec = P_k_current+0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k
                        forcevec = FL1_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cL1x_p[k],x_q_bar = cL1x_q[k], y_p_bar = cL1y_p[k], y_q_bar = cL1y_q[k], z_p_bar = cL1z_p[k], z_q_bar = cL1z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_current+0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k
                        forcevec = FL2_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cL2x_p[k],x_q_bar = cL2x_q[k], y_p_bar = cL2y_p[k], y_q_bar = cL2y_q[k], z_p_bar = cL2z_p[k], z_q_bar = cL2z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_current-0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k
                        forcevec = FL3_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cL3x_p[k],x_q_bar = cL3x_q[k], y_p_bar = cL3y_p[k], y_q_bar = cL3y_q[k], z_p_bar = cL3z_p[k], z_q_bar = cL3z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_current-0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k
                        forcevec = FL4_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cL4x_p[k],x_q_bar = cL4x_q[k], y_p_bar = cL4y_p[k], y_q_bar = cL4y_q[k], z_p_bar = cL4z_p[k], z_q_bar = cL4z_q[k], l = Leg_vec,f = forcevec)


                        #Zero Ponton's term due to swing
                        #Right Contact 1
                        ponton_term_vec = ca.vertcat(cR1x_p[k],cR1x_q[k],cR1y_p[k],cR1y_q[k],cR1z_p[k],cR1z_q[k])
                        g.append(ca.if_else(ParaLeftSwingFlag,ponton_term_vec,np.array([0,0,0,0,0,0])))
                        glb.append(np.array([0,0,0,0,0,0]))
                        gub.append(np.array([0,0,0,0,0,0]))

                        #Right Contact 2
                        ponton_term_vec = ca.vertcat(cR2x_p[k],cR2x_q[k],cR2y_p[k],cR2y_q[k],cR2z_p[k],cR2z_q[k])
                        g.append(ca.if_else(ParaLeftSwingFlag,ponton_term_vec,np.array([0,0,0,0,0,0])))
                        glb.append(np.array([0,0,0,0,0,0]))
                        gub.append(np.array([0,0,0,0,0,0]))

                        #Right Contact 3
                        ponton_term_vec = ca.vertcat(cR3x_p[k],cR3x_q[k],cR3y_p[k],cR3y_q[k],cR3z_p[k],cR3z_q[k])
                        g.append(ca.if_else(ParaLeftSwingFlag,ponton_term_vec,np.array([0,0,0,0,0,0])))
                        glb.append(np.array([0,0,0,0,0,0]))
                        gub.append(np.array([0,0,0,0,0,0]))

                        #Right Contact 4  
                        ponton_term_vec = ca.vertcat(cR4x_p[k],cR4x_q[k],cR4y_p[k],cR4y_q[k],cR4z_p[k],cR4z_q[k])
                        g.append(ca.if_else(ParaLeftSwingFlag,ponton_term_vec,np.array([0,0,0,0,0,0])))
                        glb.append(np.array([0,0,0,0,0,0]))
                        gub.append(np.array([0,0,0,0,0,0]))


                    #If First Levvel Swing the Right, then the second level Even Number Phases (the first phase) Swing the Left -> Right -> Left
                    #Right foot in contact for p_current, left foot is going to land at p_next
                    #   CoM in the Right foot
                    g.append(ca.if_else(ParaRightSwingFlag, K_CoM_Right@(CoM_k-P_k_current)-ca.DM(k_CoM_Right), np.full((len(k_CoM_Right),),-1)))
                    glb.append(np.full((len(k_CoM_Right),),-np.inf))
                    gub.append(np.full((len(k_CoM_Right),),0))

                    #Angular Dynamics (Right Support) Cost
                    #if k<N_K-1:
                    #    g.append(ca.if_else(ParaRightSwingFlag,Ldot_next - Ldot_current - h*(ca.cross((P_k_current+0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k),FR1_k) + 
                    #                                                                        ca.cross((P_k_current+0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k),FR2_k) + 
                    #                                                                        ca.cross((P_k_current-0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k),FR3_k) + 
                    #                                                                        ca.cross((P_k_current-0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k),FR4_k)),np.array([0,0,0])))                        
                    #    #g.append(ca.if_else(ParaRightSwingFlag,Ldot_next-Ldot_current-h*(ca.cross((P_k_current+np.array([0.11,0.06,0])-CoM_k),FR1_k)+ca.cross((P_k_current+np.array([0.11,-0.06,0])-CoM_k),FR2_k)+ca.cross((P_k_current+np.array([-0.11,0.06,0])-CoM_k),FR3_k)+ca.cross((P_k_current+np.array([-0.11,-0.06,0])-CoM_k),FR4_k)), np.array([0,0,0])))
                    #    glb.append(np.array([0,0,0]))
                    #    gub.append(np.array([0,0,0]))

                    if k<N_K-1: #double check the knot number is valid

                        Leg_vec = P_k_current+0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k
                        forcevec = FR1_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cR1x_p[k],x_q_bar = cR1x_q[k], y_p_bar = cR1y_p[k], y_q_bar = cR1y_q[k], z_p_bar = cR1z_p[k], z_q_bar = cR1z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_current+0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k
                        forcevec = FR2_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cR2x_p[k],x_q_bar = cR2x_q[k], y_p_bar = cR2y_p[k], y_q_bar = cR2y_q[k], z_p_bar = cR2z_p[k], z_q_bar = cR2z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_current-0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k
                        forcevec = FR3_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cR3x_p[k],x_q_bar = cR3x_q[k], y_p_bar = cR3y_p[k], y_q_bar = cR3y_q[k], z_p_bar = cR3z_p[k], z_q_bar = cR3z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_current-0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k
                        forcevec = FR4_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cR4x_p[k],x_q_bar = cR4x_q[k], y_p_bar = cR4y_p[k], y_q_bar = cR4y_q[k], z_p_bar = cR4z_p[k], z_q_bar = cR4z_q[k], l = Leg_vec,f = forcevec)


                        #Zero Ponton's term due to swing
                        #Right Contact 1
                        ponton_term_vec = ca.vertcat(cL1x_p[k],cL1x_q[k],cL1y_p[k],cL1y_q[k],cL1z_p[k],cL1z_q[k])
                        g.append(ca.if_else(ParaRightSwingFlag,ponton_term_vec,np.array([0,0,0,0,0,0])))
                        glb.append(np.array([0,0,0,0,0,0]))
                        gub.append(np.array([0,0,0,0,0,0]))

                        #Right Contact 2
                        ponton_term_vec = ca.vertcat(cL2x_p[k],cL2x_q[k],cL2y_p[k],cL2y_q[k],cL2z_p[k],cL2z_q[k])
                        g.append(ca.if_else(ParaRightSwingFlag,ponton_term_vec,np.array([0,0,0,0,0,0])))
                        glb.append(np.array([0,0,0,0,0,0]))
                        gub.append(np.array([0,0,0,0,0,0]))

                        #Right Contact 3
                        ponton_term_vec = ca.vertcat(cL3x_p[k],cL3x_q[k],cL3y_p[k],cL3y_q[k],cL3z_p[k],cL3z_q[k])
                        g.append(ca.if_else(ParaRightSwingFlag,ponton_term_vec,np.array([0,0,0,0,0,0])))
                        glb.append(np.array([0,0,0,0,0,0]))
                        gub.append(np.array([0,0,0,0,0,0]))

                        #Right Contact 4  
                        ponton_term_vec = ca.vertcat(cL4x_p[k],cL4x_q[k],cL4y_p[k],cL4y_q[k],cL4z_p[k],cL4z_q[k])
                        g.append(ca.if_else(ParaRightSwingFlag,ponton_term_vec,np.array([0,0,0,0,0,0])))
                        glb.append(np.array([0,0,0,0,0,0]))
                        gub.append(np.array([0,0,0,0,0,0]))

                    #Zero Forces
                    
                    #Case 1
                    #If the First Level swing the Left, then the second level Even Number Phase (the first phase) Swing the Right -> Left -> Right
                    #RIGHT FOOT has zero forces
                    g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k)
                    g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k)
                    g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k)
                    g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k)

                    #Case 2
                    #If First Levvel Swing the Right, then the second level Even Number Phases (the first phase) Swing the Left -> Right -> Left
                    #LEFT FOOT has zero Forces
                    g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k)
                    g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k)
                    g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k)
                    g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k)

                    #Unilateral Constraints

                    #Case 1
                    #If the First Level swing the Left, then the second level Even Number Phase (the first phase) Swing the Right -> Left -> Right
                    #Then the Left Foot obey unilateral constraints on the p_current
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainNorm = P_k_current_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainNorm = P_k_current_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainNorm = P_k_current_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainNorm = P_k_current_Norm)

                    #Case 2
                    #If First Levvel Swing the Right, then the second level Even Number Phases (the first phase) Swing the Left -> Right -> Left
                    #Then the Right foot obey unilateral constraints on the p_current
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainNorm = P_k_current_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainNorm = P_k_current_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainNorm = P_k_current_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainNorm = P_k_current_Norm)

                    #Friction Cone Constraint

                    #Case 1
                    #If the First Level swing the Left, then the second level Even Number Phase (the first phase) Swing the Right -> Left -> Right
                    #Then the Left Foot obey friction cone constraints on the p_current
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)

                    #Case 2
                    #If First Levvel Swing the Right, then the second level Even Number Phases (the first phase) Swing the Left -> Right -> Left
                    #Then the Right foot obey unilateral constraints on the p_current
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)


                elif ((Nph-1)//2)%2 == 1: #odd number steps
                    
                    #------------------------------------
                    #CoM Kinematics
                    #If the first level swings the Left, then the second level for Odd Number Phases (the second phase) swings left -> Right -> Left
                    #Right foot in contact for p_current, left foot is going to land at p_next
                    #   CoM in the Right
                    g.append(ca.if_else(ParaLeftSwingFlag, K_CoM_Right@(CoM_k-P_k_current)-ca.DM(k_CoM_Right), np.full((len(k_CoM_Right),),-1)))
                    glb.append(np.full((len(k_CoM_Right),),-np.inf))
                    gub.append(np.full((len(k_CoM_Right),),0))
                    
                    #Angular Dynamics (Right Support) Cost
                    #if k<N_K-1:
                    #    g.append(ca.if_else(ParaLeftSwingFlag,Ldot_next - Ldot_current - h*(ca.cross((P_k_current+0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k),FR1_k) + 
                    #                                                                        ca.cross((P_k_current+0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k),FR2_k) + 
                    #                                                                        ca.cross((P_k_current-0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k),FR3_k) + 
                    #                                                                        ca.cross((P_k_current-0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k),FR4_k)),np.array([0,0,0])))
                    #    #g.append(ca.if_else(ParaLeftSwingFlag,Ldot_next-Ldot_current-h*(ca.cross((P_k_current+np.array([0.11,0.06,0])-CoM_k),FR1_k)+ca.cross((P_k_current+np.array([0.11,-0.06,0])-CoM_k),FR2_k)+ca.cross((P_k_current+np.array([-0.11,0.06,0])-CoM_k),FR3_k)+ca.cross((P_k_current+np.array([-0.11,-0.06,0])-CoM_k),FR4_k)), np.array([0,0,0])))
                    #    glb.append(np.array([0,0,0]))
                    #    gub.append(np.array([0,0,0]))

                    if k<N_K-1: #double check the knot number is valid

                        Leg_vec = P_k_current+0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k
                        forcevec = FR1_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cR1x_p[k],x_q_bar = cR1x_q[k], y_p_bar = cR1y_p[k], y_q_bar = cR1y_q[k], z_p_bar = cR1z_p[k], z_q_bar = cR1z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_current+0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k
                        forcevec = FR2_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cR2x_p[k],x_q_bar = cR2x_q[k], y_p_bar = cR2y_p[k], y_q_bar = cR2y_q[k], z_p_bar = cR2z_p[k], z_q_bar = cR2z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_current-0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k
                        forcevec = FR3_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cR3x_p[k],x_q_bar = cR3x_q[k], y_p_bar = cR3y_p[k], y_q_bar = cR3y_q[k], z_p_bar = cR3z_p[k], z_q_bar = cR3z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_current-0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k
                        forcevec = FR4_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cR4x_p[k],x_q_bar = cR4x_q[k], y_p_bar = cR4y_p[k], y_q_bar = cR4y_q[k], z_p_bar = cR4z_p[k], z_q_bar = cR4z_q[k], l = Leg_vec,f = forcevec)


                        #Zero Ponton's term due to swing
                        #Right Contact 1
                        ponton_term_vec = ca.vertcat(cL1x_p[k],cL1x_q[k],cL1y_p[k],cL1y_q[k],cL1z_p[k],cL1z_q[k])
                        g.append(ca.if_else(ParaLeftSwingFlag,ponton_term_vec,np.array([0,0,0,0,0,0])))
                        glb.append(np.array([0,0,0,0,0,0]))
                        gub.append(np.array([0,0,0,0,0,0]))

                        #Right Contact 2
                        ponton_term_vec = ca.vertcat(cL2x_p[k],cL2x_q[k],cL2y_p[k],cL2y_q[k],cL2z_p[k],cL2z_q[k])
                        g.append(ca.if_else(ParaLeftSwingFlag,ponton_term_vec,np.array([0,0,0,0,0,0])))
                        glb.append(np.array([0,0,0,0,0,0]))
                        gub.append(np.array([0,0,0,0,0,0]))

                        #Right Contact 3
                        ponton_term_vec = ca.vertcat(cL3x_p[k],cL3x_q[k],cL3y_p[k],cL3y_q[k],cL3z_p[k],cL3z_q[k])
                        g.append(ca.if_else(ParaLeftSwingFlag,ponton_term_vec,np.array([0,0,0,0,0,0])))
                        glb.append(np.array([0,0,0,0,0,0]))
                        gub.append(np.array([0,0,0,0,0,0]))

                        #Right Contact 4  
                        ponton_term_vec = ca.vertcat(cL4x_p[k],cL4x_q[k],cL4y_p[k],cL4y_q[k],cL4z_p[k],cL4z_q[k])
                        g.append(ca.if_else(ParaLeftSwingFlag,ponton_term_vec,np.array([0,0,0,0,0,0])))
                        glb.append(np.array([0,0,0,0,0,0]))
                        gub.append(np.array([0,0,0,0,0,0]))

                    #If the first level swings the Right, then the second level for Odd Number Phases (the second phase) swings Right -> Left -> Right
                    #Left foot in contact for p_current, right foot is going to land as p_next
                    #   CoM in the Left
                    g.append(ca.if_else(ParaRightSwingFlag, K_CoM_Left@(CoM_k-P_k_current)-ca.DM(k_CoM_Left), np.full((len(k_CoM_Left),),-1)))
                    glb.append(np.full((len(k_CoM_Left),),-np.inf))
                    gub.append(np.full((len(k_CoM_Left),),0))
                    
                    #Angular Dynamics (Left Support) Cost
                    #if k<N_K-1:
                    #    g.append(ca.if_else(ParaRightSwingFlag,Ldot_next - Ldot_current - h*(ca.cross((P_k_current+0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k),FL1_k) + 
                    #                                                                        ca.cross((P_k_current+0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k),FL2_k) + 
                    #                                                                        ca.cross((P_k_current-0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k),FL3_k) + 
                    #                                                                        ca.cross((P_k_current-0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k),FL4_k)),np.array([0,0,0])))
                    #    #g.append(ca.if_else(ParaRightSwingFlag, Ldot_next-Ldot_current-h*(ca.cross((P_k_current+np.array([0.11,0.06,0])-CoM_k),FL1_k)+ca.cross((P_k_current+np.array([0.11,-0.06,0])-CoM_k),FL2_k)+ca.cross((P_k_current+np.array([-0.11,0.06,0])-CoM_k),FL3_k)+ca.cross((P_k_current+np.array([-0.11,-0.06,0])-CoM_k),FL4_k)), np.array([0,0,0])))
                    #    glb.append(np.array([0,0,0]))
                    #    gub.append(np.array([0,0,0]))

                    if k<N_K-1: #double check the knot number is valid

                        Leg_vec = P_k_current+0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k
                        forcevec = FL1_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cL1x_p[k],x_q_bar = cL1x_q[k], y_p_bar = cL1y_p[k], y_q_bar = cL1y_q[k], z_p_bar = cL1z_p[k], z_q_bar = cL1z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_current+0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k
                        forcevec = FL2_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cL2x_p[k],x_q_bar = cL2x_q[k], y_p_bar = cL2y_p[k], y_q_bar = cL2y_q[k], z_p_bar = cL2z_p[k], z_q_bar = cL2z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_current-0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k
                        forcevec = FL3_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cL3x_p[k],x_q_bar = cL3x_q[k], y_p_bar = cL3y_p[k], y_q_bar = cL3y_q[k], z_p_bar = cL3z_p[k], z_q_bar = cL3z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_current-0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k
                        forcevec = FL4_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cL4x_p[k],x_q_bar = cL4x_q[k], y_p_bar = cL4y_p[k], y_q_bar = cL4y_q[k], z_p_bar = cL4z_p[k], z_q_bar = cL4z_q[k], l = Leg_vec,f = forcevec)


                        #Zero Ponton's term due to swing
                        #Right Contact 1
                        ponton_term_vec = ca.vertcat(cR1x_p[k],cR1x_q[k],cR1y_p[k],cR1y_q[k],cR1z_p[k],cR1z_q[k])
                        g.append(ca.if_else(ParaRightSwingFlag,ponton_term_vec,np.array([0,0,0,0,0,0])))
                        glb.append(np.array([0,0,0,0,0,0]))
                        gub.append(np.array([0,0,0,0,0,0]))

                        #Right Contact 2
                        ponton_term_vec = ca.vertcat(cR2x_p[k],cR2x_q[k],cR2y_p[k],cR2y_q[k],cR2z_p[k],cR2z_q[k])
                        g.append(ca.if_else(ParaRightSwingFlag,ponton_term_vec,np.array([0,0,0,0,0,0])))
                        glb.append(np.array([0,0,0,0,0,0]))
                        gub.append(np.array([0,0,0,0,0,0]))

                        #Right Contact 3
                        ponton_term_vec = ca.vertcat(cR3x_p[k],cR3x_q[k],cR3y_p[k],cR3y_q[k],cR3z_p[k],cR3z_q[k])
                        g.append(ca.if_else(ParaRightSwingFlag,ponton_term_vec,np.array([0,0,0,0,0,0])))
                        glb.append(np.array([0,0,0,0,0,0]))
                        gub.append(np.array([0,0,0,0,0,0]))

                        #Right Contact 4  
                        ponton_term_vec = ca.vertcat(cR4x_p[k],cR4x_q[k],cR4y_p[k],cR4y_q[k],cR4z_p[k],cR4z_q[k])
                        g.append(ca.if_else(ParaRightSwingFlag,ponton_term_vec,np.array([0,0,0,0,0,0])))
                        glb.append(np.array([0,0,0,0,0,0]))
                        gub.append(np.array([0,0,0,0,0,0]))

                    #Zero Forces
                    
                    #Case 1
                    #If the first level swings the Left, then the second level for Odd Number Phases (the second phase) swings left -> Right -> Left
                    #LEFT FOOT has zero forces
                    g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k)
                    g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k)
                    g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k)
                    g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k)

                    #Case 2
                    #If the first level swings the Right, then the second level for Odd Number Phases (the second phase) swings Right -> Left -> Right
                    #RIGHT FOOT has zero forces
                    g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k)
                    g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k)
                    g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k)
                    g, glb, gub = ZeroForces(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k)

                    #Unilateral Constraints

                    #Case 1
                    #If the first level swings the Left, then the second level for Odd Number Phases (the second phase) swings left -> Right -> Left
                    #Then the Right Foot obey unilateral constraints on the p_current
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainNorm = P_k_current_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainNorm = P_k_current_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainNorm = P_k_current_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainNorm = P_k_current_Norm)

                    #Case 2
                    #If the first level swings the Right, then the second level for Odd Number Phases (the second phase) swings Right -> Left -> Right
                    #Then the Left foot obey unilateral constraints on the p_current
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainNorm = P_k_current_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainNorm = P_k_current_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainNorm = P_k_current_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainNorm = P_k_current_Norm)

                    #Friction Cone Constraint

                    #Case 1
                    #If the first level swings the Left, then the second level for Odd Number Phases (the second phase) swings left -> Right -> Left
                    #Then the Right Foot obey friction cone constraints on the p_current
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)

                    #Case 2
                    #If the first level swings the Right, then the second level for Odd Number Phases (the second phase) swings Right -> Left -> Right
                    #Then the Left foot obey unilateral constraints on the p_current
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)


            elif GaitPattern[Nph]=='DoubleSupport':

                #CoM Kinematic Constraint
                if (Nph-1-1)//2 == 0:
                    #!!!!!!Pass from the first Level!!!!!!
                    P_k_current = ca.vertcat(px_init,py_init,pz_init) #ca.vertcat(px[-1],py[-1],pz[-1])
                    P_k_current_TangentX = SurfTangentsX[0:3]
                    P_k_current_TangentY = SurfTangentsY[0:3]
                    P_k_current_Norm = SurfNorms[0:3]

                    #!!!!!!
                    P_k_next = ca.vertcat(px[(Nph-1)//2],py[(Nph-1)//2],pz[(Nph-1)//2])
                    P_k_next_TangentX = SurfTangentsX[3:6]
                    P_k_next_TangentY = SurfTangentsY[3:6]
                    P_k_next_Norm = SurfNorms[3:6]

                else:

                    P_k_current = ca.vertcat(px[(Nph-1)//2-1],py[(Nph-1)//2-1],pz[(Nph-1)//2-1])
                    P_k_current_TangentX = SurfTangentsX[((Nph-1)//2)*3:((Nph-1)//2)*3+3]
                    P_k_current_TangentY = SurfTangentsY[((Nph-1)//2)*3:((Nph-1)//2)*3+3]
                    P_k_current_Norm = SurfNorms[((Nph-1)//2)*3:((Nph-1)//2)*3+3]

                    P_k_next = ca.vertcat(px[(Nph-1)//2],py[(Nph-1)//2],pz[(Nph-1)//2])
                    P_k_next_TangentX = SurfTangentsX[((Nph-1)//2 + 1)*3:((Nph-1)//2 + 1)*3+3]
                    P_k_next_TangentY = SurfTangentsY[((Nph-1)//2 + 1)*3:((Nph-1)//2 + 1)*3+3]
                    P_k_next_Norm = SurfNorms[((Nph-1)//2 + 1)*3:((Nph-1)//2 + 1)*3+3]

                    #P_k_current = ca.vertcat(px[(Nph-1)//2-1],py[(Nph-1)//2-1],pz[(Nph-1)//2-1])
                    #P_k_current_TangentX = SurfTangentsX[((Nph-1)//2-1)*3:((Nph-1)//2-1)*3+3]
                    #P_k_current_TangentY = SurfTangentsY[((Nph-1)//2-1)*3:((Nph-1)//2-1)*3+3]
                    #P_k_current_Norm = SurfNorms[((Nph-1)//2-1)*3:((Nph-1)//2-1)*3+3]

                    #P_k_next = ca.vertcat(px[(Nph-1)//2],py[(Nph-1)//2],pz[(Nph-1)//2])
                    #P_k_next_TangentX = SurfTangentsX[((Nph-1)//2)*3:((Nph-1)//2)*3+3]
                    #P_k_next_TangentY = SurfTangentsY[((Nph-1)//2)*3:((Nph-1)//2)*3+3]
                    #P_k_next_Norm = SurfNorms[((Nph-1)//2)*3:((Nph-1)//2)*3+3]



                if ((Nph-1-1)//2)%2 == 0: #even number steps
                    #If First Level Swing the Left, then the second level Even Number Phases (the first Phase) Swing the Right -> Left -> Right
                    #Left foot in contact for p_current, right foot is going to land as p_next
                    g.append(ca.if_else(ParaLeftSwingFlag, K_CoM_Left@(CoM_k-P_k_current)-ca.DM(k_CoM_Left), np.full((len(k_CoM_Left),),-1)))
                    glb.append(np.full((len(k_CoM_Left),),-np.inf))
                    gub.append(np.full((len(k_CoM_Left),),0))

                    g.append(ca.if_else(ParaLeftSwingFlag, K_CoM_Right@(CoM_k-P_k_next)-ca.DM(k_CoM_Right), np.full((len(k_CoM_Right),),-1)))
                    glb.append(np.full((len(k_CoM_Right),),-np.inf))
                    gub.append(np.full((len(k_CoM_Right),),0))

                    #Angular Dynamics Cost, P_current as Left foot, P_next as Right
                    #if k<N_K-1:
                    #    g.append(ca.if_else(ParaLeftSwingFlag, Ldot_next - Ldot_current - h*(ca.cross((P_k_current+0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k),FL1_k) + 
                    #                                                                        ca.cross((P_k_current+0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k),FL2_k) + 
                    #                                                                        ca.cross((P_k_current-0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k),FL3_k) + 
                    #                                                                        ca.cross((P_k_current-0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k),FL4_k) +
                    #                                                                        ca.cross((P_k_next+0.11*P_k_next_TangentX+0.06*P_k_next_TangentY-CoM_k),FR1_k) + 
                    #                                                                        ca.cross((P_k_next+0.11*P_k_next_TangentX-0.06*P_k_next_TangentY-CoM_k),FR2_k) + 
                    #                                                                        ca.cross((P_k_next-0.11*P_k_next_TangentX+0.06*P_k_next_TangentY-CoM_k),FR3_k) + 
                    #                                                                        ca.cross((P_k_next-0.11*P_k_next_TangentX-0.06*P_k_next_TangentY-CoM_k),FR4_k)), np.array([0,0,0])))
                    #    #g.append(ca.if_else(ParaLeftSwingFlag,Ldot_next-Ldot_current-h*(ca.cross((P_k_current+np.array([0.11,0.06,0])-CoM_k),FL1_k)+ca.cross((P_k_current+np.array([0.11,-0.06,0])-CoM_k),FL2_k)+ca.cross((P_k_current+np.array([-0.11,0.06,0])-CoM_k),FL3_k)+ca.cross((P_k_current+np.array([-0.11,-0.06,0])-CoM_k),FL4_k)+ca.cross((P_k_next+np.array([0.11,0.06,0])-CoM_k),FR1_k)+ca.cross((P_k_next+np.array([0.11,-0.06,0])-CoM_k),FR2_k)+ca.cross((P_k_next+np.array([-0.11,0.06,0])-CoM_k),FR3_k)+ca.cross((P_k_next+np.array([-0.11,-0.06,0])-CoM_k),FR4_k)),np.array([0,0,0])))
                    #    glb.append(np.array([0,0,0]))
                    #    gub.append(np.array([0,0,0]))
                    if k<N_K-1: #double check the knot number is valid

                        Leg_vec = P_k_current+0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k
                        forcevec = FL1_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cL1x_p[k],x_q_bar = cL1x_q[k], y_p_bar = cL1y_p[k], y_q_bar = cL1y_q[k], z_p_bar = cL1z_p[k], z_q_bar = cL1z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_current+0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k
                        forcevec = FL2_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cL2x_p[k],x_q_bar = cL2x_q[k], y_p_bar = cL2y_p[k], y_q_bar = cL2y_q[k], z_p_bar = cL2z_p[k], z_q_bar = cL2z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_current-0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k
                        forcevec = FL3_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cL3x_p[k],x_q_bar = cL3x_q[k], y_p_bar = cL3y_p[k], y_q_bar = cL3y_q[k], z_p_bar = cL3z_p[k], z_q_bar = cL3z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_current-0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k
                        forcevec = FL4_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cL4x_p[k],x_q_bar = cL4x_q[k], y_p_bar = cL4y_p[k], y_q_bar = cL4y_q[k], z_p_bar = cL4z_p[k], z_q_bar = cL4z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_next+0.11*P_k_next_TangentX+0.06*P_k_next_TangentY-CoM_k
                        forcevec = FR1_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cR1x_p[k],x_q_bar = cR1x_q[k], y_p_bar = cR1y_p[k], y_q_bar = cR1y_q[k], z_p_bar = cR1z_p[k], z_q_bar = cR1z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_next+0.11*P_k_next_TangentX-0.06*P_k_next_TangentY-CoM_k
                        forcevec = FR2_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cR2x_p[k],x_q_bar = cR2x_q[k], y_p_bar = cR2y_p[k], y_q_bar = cR2y_q[k], z_p_bar = cR2z_p[k], z_q_bar = cR2z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_next-0.11*P_k_next_TangentX+0.06*P_k_next_TangentY-CoM_k
                        forcevec = FR3_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cR3x_p[k],x_q_bar = cR3x_q[k], y_p_bar = cR3y_p[k], y_q_bar = cR3y_q[k], z_p_bar = cR3z_p[k], z_q_bar = cR3z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_next-0.11*P_k_next_TangentX-0.06*P_k_next_TangentY-CoM_k
                        forcevec = FR4_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cR4x_p[k],x_q_bar = cR4x_q[k], y_p_bar = cR4y_p[k], y_q_bar = cR4y_q[k], z_p_bar = cR4z_p[k], z_q_bar = cR4z_q[k], l = Leg_vec,f = forcevec)


                    #If First Levvel Swing the Right, then the second level Even Number Phases (the first phase) Swing the Left -> Right -> Left
                    #Right foot in contact for p_current, left foot is going to land at p_next
                    g.append(ca.if_else(ParaRightSwingFlag, K_CoM_Right@(CoM_k-P_k_current)-ca.DM(k_CoM_Right), np.full((len(k_CoM_Right),),-1)))
                    glb.append(np.full((len(k_CoM_Right),),-np.inf))
                    gub.append(np.full((len(k_CoM_Right),),0))

                    g.append(ca.if_else(ParaRightSwingFlag, K_CoM_Left@(CoM_k-P_k_next)-ca.DM(k_CoM_Left), np.full((len(k_CoM_Left),),-1)))
                    glb.append(np.full((len(k_CoM_Left),),-np.inf))
                    gub.append(np.full((len(k_CoM_Left),),0))

                    #Angular Dynamics (Double Support) P_current as Right foot, P_next as Left NOTE:The Flippped FR and FL
                    #if k<N_K-1:
                    #    g.append(ca.if_else(ParaRightSwingFlag, Ldot_next - Ldot_current - h*(ca.cross((P_k_current+0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k),FR1_k) + 
                    #                                                                        ca.cross((P_k_current+0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k),FR2_k) + 
                    #                                                                        ca.cross((P_k_current-0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k),FR3_k) + 
                    #                                                                        ca.cross((P_k_current-0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k),FR4_k) +
                    #                                                                        ca.cross((P_k_next+0.11*P_k_next_TangentX+0.06*P_k_next_TangentY-CoM_k),FL1_k) + 
                    #                                                                        ca.cross((P_k_next+0.11*P_k_next_TangentX-0.06*P_k_next_TangentY-CoM_k),FL2_k) + 
                    #                                                                        ca.cross((P_k_next-0.11*P_k_next_TangentX+0.06*P_k_next_TangentY-CoM_k),FL3_k) + 
                    #                                                                        ca.cross((P_k_next-0.11*P_k_next_TangentX-0.06*P_k_next_TangentY-CoM_k),FL4_k)), np.array([0,0,0])))  
                    #    #g.append(ca.if_else(ParaRightSwingFlag,Ldot_next-Ldot_current-h*(ca.cross((P_k_current+np.array([0.11,0.06,0])-CoM_k),FR1_k)+ca.cross((P_k_current+np.array([0.11,-0.06,0])-CoM_k),FR2_k)+ca.cross((P_k_current+np.array([-0.11,0.06,0])-CoM_k),FR3_k)+ca.cross((P_k_current+np.array([-0.11,-0.06,0])-CoM_k),FR4_k)+ca.cross((P_k_next+np.array([0.11,0.06,0])-CoM_k),FL1_k)+ca.cross((P_k_next+np.array([0.11,-0.06,0])-CoM_k),FL2_k)+ca.cross((P_k_next+np.array([-0.11,0.06,0])-CoM_k),FL3_k)+ca.cross((P_k_next+np.array([-0.11,-0.06,0])-CoM_k),FL4_k)),np.array([0,0,0])))
                    #    glb.append(np.array([0,0,0]))
                    #    gub.append(np.array([0,0,0]))
                    if k<N_K-1: #double check the knot number is valid

                        Leg_vec = P_k_current+0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k
                        forcevec = FR1_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cR1x_p[k],x_q_bar = cR1x_q[k], y_p_bar = cR1y_p[k], y_q_bar = cR1y_q[k], z_p_bar = cR1z_p[k], z_q_bar = cR1z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_current+0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k
                        forcevec = FR2_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cR2x_p[k],x_q_bar = cR2x_q[k], y_p_bar = cR2y_p[k], y_q_bar = cR2y_q[k], z_p_bar = cR2z_p[k], z_q_bar = cR2z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_current-0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k
                        forcevec = FR3_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cR3x_p[k],x_q_bar = cR3x_q[k], y_p_bar = cR3y_p[k], y_q_bar = cR3y_q[k], z_p_bar = cR3z_p[k], z_q_bar = cR3z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_current-0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k
                        forcevec = FR4_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cR4x_p[k],x_q_bar = cR4x_q[k], y_p_bar = cR4y_p[k], y_q_bar = cR4y_q[k], z_p_bar = cR4z_p[k], z_q_bar = cR4z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_next+0.11*P_k_next_TangentX+0.06*P_k_next_TangentY-CoM_k
                        forcevec = FL1_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cL1x_p[k],x_q_bar = cL1x_q[k], y_p_bar = cL1y_p[k], y_q_bar = cL1y_q[k], z_p_bar = cL1z_p[k], z_q_bar = cL1z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_next+0.11*P_k_next_TangentX-0.06*P_k_next_TangentY-CoM_k
                        forcevec = FL2_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cL2x_p[k],x_q_bar = cL2x_q[k], y_p_bar = cL2y_p[k], y_q_bar = cL2y_q[k], z_p_bar = cL2z_p[k], z_q_bar = cL2z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_next-0.11*P_k_next_TangentX+0.06*P_k_next_TangentY-CoM_k
                        forcevec = FL3_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cL3x_p[k],x_q_bar = cL3x_q[k], y_p_bar = cL3y_p[k], y_q_bar = cL3y_q[k], z_p_bar = cL3z_p[k], z_q_bar = cL3z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_next-0.11*P_k_next_TangentX-0.06*P_k_next_TangentY-CoM_k
                        forcevec = FL4_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cL4x_p[k],x_q_bar = cL4x_q[k], y_p_bar = cL4y_p[k], y_q_bar = cL4y_q[k], z_p_bar = cL4z_p[k], z_q_bar = cL4z_q[k], l = Leg_vec,f = forcevec)


                    #Unilateral constraints

                    #Case 1
                    #If First Level Swing the Left, then the second level Even Number Phases (the first Phase) Swing the Right -> Left -> Right
                    #Left foot in contact for p_current
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainNorm = P_k_current_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainNorm = P_k_current_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainNorm = P_k_current_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainNorm = P_k_current_Norm)
                    # right foot is going to land as p_next
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainNorm = P_k_next_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainNorm = P_k_next_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainNorm = P_k_next_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainNorm = P_k_next_Norm)

                    #Case 2
                    #If First Levvel Swing the Right, then the second level Even Number Phases (the first phase) Swing the Left -> Right -> Left
                    #Right foot in contact for p_current
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainNorm = P_k_current_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainNorm = P_k_current_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainNorm = P_k_current_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainNorm = P_k_current_Norm)
                    #Left foot is going to land at p_next
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainNorm = P_k_next_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainNorm = P_k_next_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainNorm = P_k_next_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainNorm = P_k_next_Norm)

                    #Friction Cone Constraint

                    #Case 1
                    #If First Level Swing the Left, then the second level Even Number Phases (the first Phase) Swing the Right -> Left -> Right
                    #Left foot in contact for p_current
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)
                    # right foot is going to land as p_next
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainTangentX = P_k_next_TangentX, TerrainTangentY = P_k_next_TangentY, TerrainNorm = P_k_next_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainTangentX = P_k_next_TangentX, TerrainTangentY = P_k_next_TangentY, TerrainNorm = P_k_next_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainTangentX = P_k_next_TangentX, TerrainTangentY = P_k_next_TangentY, TerrainNorm = P_k_next_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainTangentX = P_k_next_TangentX, TerrainTangentY = P_k_next_TangentY, TerrainNorm = P_k_next_Norm, miu = miu)

                    #Case 2
                    #If First Levvel Swing the Right, then the second level Even Number Phases (the first phase) Swing the Left -> Right -> Left
                    #Right foot in contact for p_current
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)
                    #Left foot is going to land at p_next
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainTangentX = P_k_next_TangentX, TerrainTangentY = P_k_next_TangentY, TerrainNorm = P_k_next_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainTangentX = P_k_next_TangentX, TerrainTangentY = P_k_next_TangentY, TerrainNorm = P_k_next_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainTangentX = P_k_next_TangentX, TerrainTangentY = P_k_next_TangentY, TerrainNorm = P_k_next_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainTangentX = P_k_next_TangentX, TerrainTangentY = P_k_next_TangentY, TerrainNorm = P_k_next_Norm, miu = miu)


                elif ((Nph-1-1)//2)%2 == 1: #odd number steps
                    #If the first level swings the Left, then the second level for Odd Number Phases (the second phase) swings left -> Right -> Left
                    #Right foot in contact for p_current, left foot is going to land at p_next
                    g.append(ca.if_else(ParaLeftSwingFlag, K_CoM_Right@(CoM_k-P_k_current)-ca.DM(k_CoM_Right), np.full((len(k_CoM_Right),),-1)))
                    glb.append(np.full((len(k_CoM_Right),),-np.inf))
                    gub.append(np.full((len(k_CoM_Right),),0))

                    g.append(ca.if_else(ParaLeftSwingFlag, K_CoM_Left@(CoM_k-P_k_next)-ca.DM(k_CoM_Left), np.full((len(k_CoM_Left),),-1)))
                    glb.append(np.full((len(k_CoM_Left),),-np.inf))
                    gub.append(np.full((len(k_CoM_Left),),0))

                    #Angular Dynamics Cost, P_current as Right, P_next as Left
                    #if k<N_K-1:
                    #    g.append(ca.if_else(ParaLeftSwingFlag, Ldot_next - Ldot_current - h*(ca.cross((P_k_current+0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k),FR1_k) + 
                    #                                                                        ca.cross((P_k_current+0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k),FR2_k) + 
                    #                                                                        ca.cross((P_k_current-0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k),FR3_k) + 
                    #                                                                        ca.cross((P_k_current-0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k),FR4_k) +
                    #                                                                        ca.cross((P_k_next+0.11*P_k_next_TangentX+0.06*P_k_next_TangentY-CoM_k),FL1_k) + 
                    #                                                                        ca.cross((P_k_next+0.11*P_k_next_TangentX-0.06*P_k_next_TangentY-CoM_k),FL2_k) + 
                    #                                                                        ca.cross((P_k_next-0.11*P_k_next_TangentX+0.06*P_k_next_TangentY-CoM_k),FL3_k) + 
                    #                                                                        ca.cross((P_k_next-0.11*P_k_next_TangentX-0.06*P_k_next_TangentY-CoM_k),FL4_k)), np.array([0,0,0])))
                    #    #g.append(ca.if_else(ParaLeftSwingFlag,Ldot_next-Ldot_current-h*(ca.cross((P_k_current+np.array([0.11,0.06,0])-CoM_k),FR1_k)+ca.cross((P_k_current+np.array([0.11,-0.06,0])-CoM_k),FR2_k)+ca.cross((P_k_current+np.array([-0.11,0.06,0])-CoM_k),FR3_k)+ca.cross((P_k_current+np.array([-0.11,-0.06,0])-CoM_k),FR4_k)+ca.cross((P_k_next+np.array([0.11,0.06,0])-CoM_k),FL1_k)+ca.cross((P_k_next+np.array([0.11,-0.06,0])-CoM_k),FL2_k)+ca.cross((P_k_next+np.array([-0.11,0.06,0])-CoM_k),FL3_k)+ca.cross((P_k_next+np.array([-0.11,-0.06,0])-CoM_k),FL4_k)),np.array([0,0,0])))
                    #    glb.append(np.array([0,0,0]))
                    #    gub.append(np.array([0,0,0]))

                    if k<N_K-1: #double check the knot number is valid

                        Leg_vec = P_k_current+0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k
                        forcevec = FR1_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cR1x_p[k],x_q_bar = cR1x_q[k], y_p_bar = cR1y_p[k], y_q_bar = cR1y_q[k], z_p_bar = cR1z_p[k], z_q_bar = cR1z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_current+0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k
                        forcevec = FR2_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cR2x_p[k],x_q_bar = cR2x_q[k], y_p_bar = cR2y_p[k], y_q_bar = cR2y_q[k], z_p_bar = cR2z_p[k], z_q_bar = cR2z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_current-0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k
                        forcevec = FR3_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cR3x_p[k],x_q_bar = cR3x_q[k], y_p_bar = cR3y_p[k], y_q_bar = cR3y_q[k], z_p_bar = cR3z_p[k], z_q_bar = cR3z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_current-0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k
                        forcevec = FR4_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cR4x_p[k],x_q_bar = cR4x_q[k], y_p_bar = cR4y_p[k], y_q_bar = cR4y_q[k], z_p_bar = cR4z_p[k], z_q_bar = cR4z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_next+0.11*P_k_next_TangentX+0.06*P_k_next_TangentY-CoM_k
                        forcevec = FL1_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cL1x_p[k],x_q_bar = cL1x_q[k], y_p_bar = cL1y_p[k], y_q_bar = cL1y_q[k], z_p_bar = cL1z_p[k], z_q_bar = cL1z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_next+0.11*P_k_next_TangentX-0.06*P_k_next_TangentY-CoM_k
                        forcevec = FL2_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cL2x_p[k],x_q_bar = cL2x_q[k], y_p_bar = cL2y_p[k], y_q_bar = cL2y_q[k], z_p_bar = cL2z_p[k], z_q_bar = cL2z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_next-0.11*P_k_next_TangentX+0.06*P_k_next_TangentY-CoM_k
                        forcevec = FL3_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cL3x_p[k],x_q_bar = cL3x_q[k], y_p_bar = cL3y_p[k], y_q_bar = cL3y_q[k], z_p_bar = cL3z_p[k], z_q_bar = cL3z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_next-0.11*P_k_next_TangentX-0.06*P_k_next_TangentY-CoM_k
                        forcevec = FL4_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, x_p_bar = cL4x_p[k],x_q_bar = cL4x_q[k], y_p_bar = cL4y_p[k], y_q_bar = cL4y_q[k], z_p_bar = cL4z_p[k], z_q_bar = cL4z_q[k], l = Leg_vec,f = forcevec)


                    #If the first level swings the Right, then the second level for Odd Number Phases (the second phase) swings Right -> Left -> Right
                    #Left foot in contact for p_current, right foot is going to land as p_next
                    g.append(ca.if_else(ParaRightSwingFlag, K_CoM_Left@(CoM_k-P_k_current)-ca.DM(k_CoM_Left), np.full((len(k_CoM_Left),),-1)))
                    glb.append(np.full((len(k_CoM_Left),),-np.inf))
                    gub.append(np.full((len(k_CoM_Left),),0))

                    g.append(ca.if_else(ParaRightSwingFlag, K_CoM_Right@(CoM_k-P_k_next)-ca.DM(k_CoM_Right), np.full((len(k_CoM_Right),),-1)))
                    glb.append(np.full((len(k_CoM_Right),),-np.inf))
                    gub.append(np.full((len(k_CoM_Right),),0))

                    #Angular Dynamics Cost, P_current as Left, P_next as Right
                    #if k<N_K-1:
                    #    g.append(ca.if_else(ParaRightSwingFlag, Ldot_next - Ldot_current - h*(ca.cross((P_k_current+0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k),FL1_k) + 
                    #                                                                        ca.cross((P_k_current+0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k),FL2_k) + 
                    #                                                                        ca.cross((P_k_current-0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k),FL3_k) + 
                    #                                                                        ca.cross((P_k_current-0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k),FL4_k) +
                    #                                                                        ca.cross((P_k_next+0.11*P_k_next_TangentX+0.06*P_k_next_TangentY-CoM_k),FR1_k) + 
                    #                                                                        ca.cross((P_k_next+0.11*P_k_next_TangentX-0.06*P_k_next_TangentY-CoM_k),FR2_k) + 
                    #                                                                        ca.cross((P_k_next-0.11*P_k_next_TangentX+0.06*P_k_next_TangentY-CoM_k),FR3_k) + 
                    #                                                                        ca.cross((P_k_next-0.11*P_k_next_TangentX-0.06*P_k_next_TangentY-CoM_k),FR4_k)), np.array([0,0,0])))
                    #    #g.append(ca.if_else(ParaRightSwingFlag,Ldot_next-Ldot_current-h*(ca.cross((P_k_current+np.array([0.11,0.06,0])-CoM_k),FL1_k)+ca.cross((P_k_current+np.array([0.11,-0.06,0])-CoM_k),FL2_k)+ca.cross((P_k_current+np.array([-0.11,0.06,0])-CoM_k),FL3_k)+ca.cross((P_k_current+np.array([-0.11,-0.06,0])-CoM_k),FL4_k)+ca.cross((P_k_next+np.array([0.11,0.06,0])-CoM_k),FR1_k)+ca.cross((P_k_next+np.array([0.11,-0.06,0])-CoM_k),FR2_k)+ca.cross((P_k_next+np.array([-0.11,0.06,0])-CoM_k),FR3_k)+ca.cross((P_k_next+np.array([-0.11,-0.06,0])-CoM_k),FR4_k)),np.array([0,0,0])))
                    #    glb.append(np.array([0,0,0]))
                    #    gub.append(np.array([0,0,0]))
                    if k<N_K-1: #double check the knot number is valid

                        Leg_vec = P_k_current+0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k
                        forcevec = FL1_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cL1x_p[k],x_q_bar = cL1x_q[k], y_p_bar = cL1y_p[k], y_q_bar = cL1y_q[k], z_p_bar = cL1z_p[k], z_q_bar = cL1z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_current+0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k
                        forcevec = FL2_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cL2x_p[k],x_q_bar = cL2x_q[k], y_p_bar = cL2y_p[k], y_q_bar = cL2y_q[k], z_p_bar = cL2z_p[k], z_q_bar = cL2z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_current-0.11*P_k_current_TangentX+0.06*P_k_current_TangentY-CoM_k
                        forcevec = FL3_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cL3x_p[k],x_q_bar = cL3x_q[k], y_p_bar = cL3y_p[k], y_q_bar = cL3y_q[k], z_p_bar = cL3z_p[k], z_q_bar = cL3z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_current-0.11*P_k_current_TangentX-0.06*P_k_current_TangentY-CoM_k
                        forcevec = FL4_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cL4x_p[k],x_q_bar = cL4x_q[k], y_p_bar = cL4y_p[k], y_q_bar = cL4y_q[k], z_p_bar = cL4z_p[k], z_q_bar = cL4z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_next+0.11*P_k_next_TangentX+0.06*P_k_next_TangentY-CoM_k
                        forcevec = FR1_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cR1x_p[k],x_q_bar = cR1x_q[k], y_p_bar = cR1y_p[k], y_q_bar = cR1y_q[k], z_p_bar = cR1z_p[k], z_q_bar = cR1z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_next+0.11*P_k_next_TangentX-0.06*P_k_next_TangentY-CoM_k
                        forcevec = FR2_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cR2x_p[k],x_q_bar = cR2x_q[k], y_p_bar = cR2y_p[k], y_q_bar = cR2y_q[k], z_p_bar = cR2z_p[k], z_q_bar = cR2z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_next-0.11*P_k_next_TangentX+0.06*P_k_next_TangentY-CoM_k
                        forcevec = FR3_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cR3x_p[k],x_q_bar = cR3x_q[k], y_p_bar = cR3y_p[k], y_q_bar = cR3y_q[k], z_p_bar = cR3z_p[k], z_q_bar = cR3z_q[k], l = Leg_vec,f = forcevec)

                        Leg_vec = P_k_next-0.11*P_k_next_TangentX-0.06*P_k_next_TangentY-CoM_k
                        forcevec = FR4_k
                        g,glb,gub = Ponton_Concex_Constraint(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, x_p_bar = cR4x_p[k],x_q_bar = cR4x_q[k], y_p_bar = cR4y_p[k], y_q_bar = cR4y_q[k], z_p_bar = cR4z_p[k], z_q_bar = cR4z_q[k], l = Leg_vec,f = forcevec)

                    #Unilateral constraints

                    #Case 1
                    #If the first level swings the Left, then the second level for Odd Number Phases (the second phase) swings left -> Right -> Left
                    #Right foot in contact for p_current
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainNorm = P_k_current_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainNorm = P_k_current_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainNorm = P_k_current_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainNorm = P_k_current_Norm)
                    #Left foot is going to land at p_next
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainNorm = P_k_next_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainNorm = P_k_next_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainNorm = P_k_next_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainNorm = P_k_next_Norm)

                    #Case 2
                    #If the first level swings the Right, then the second level for Odd Number Phases (the second phase) swings Right -> Left -> Right
                    #Left foot in contact for p_current
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainNorm = P_k_current_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainNorm = P_k_current_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainNorm = P_k_current_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainNorm = P_k_current_Norm)
                    #Right foot is going to land as p_next
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainNorm = P_k_next_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainNorm = P_k_next_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainNorm = P_k_next_Norm)
                    g, glb, gub = Unilateral_Constraints(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainNorm = P_k_next_Norm)

                    #Friction Cone Constraint

                    #Case 1
                    #If the first level swings the Left, then the second level for Odd Number Phases (the second phase) swings left -> Right -> Left
                    #Right foot in contact for p_current
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)
                    #Left foot is going to land at p_next
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL1_k, TerrainTangentX = P_k_next_TangentX, TerrainTangentY = P_k_next_TangentY, TerrainNorm = P_k_next_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL2_k, TerrainTangentX = P_k_next_TangentX, TerrainTangentY = P_k_next_TangentY, TerrainNorm = P_k_next_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL3_k, TerrainTangentX = P_k_next_TangentX, TerrainTangentY = P_k_next_TangentY, TerrainNorm = P_k_next_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaLeftSwingFlag, F_k = FL4_k, TerrainTangentX = P_k_next_TangentX, TerrainTangentY = P_k_next_TangentY, TerrainNorm = P_k_next_Norm, miu = miu)

                    #Case 2
                    #If the first level swings the Right, then the second level for Odd Number Phases (the second phase) swings Right -> Left -> Right
                    #Left foot in contact for p_current
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL1_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL2_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL3_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FL4_k, TerrainTangentX = P_k_current_TangentX, TerrainTangentY = P_k_current_TangentY, TerrainNorm = P_k_current_Norm, miu = miu)
                    #Right foot is going to land as p_next
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR1_k, TerrainTangentX = P_k_next_TangentX, TerrainTangentY = P_k_next_TangentY, TerrainNorm = P_k_next_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR2_k, TerrainTangentX = P_k_next_TangentX, TerrainTangentY = P_k_next_TangentY, TerrainNorm = P_k_next_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR3_k, TerrainTangentX = P_k_next_TangentX, TerrainTangentY = P_k_next_TangentY, TerrainNorm = P_k_next_Norm, miu = miu)
                    g, glb, gub = FrictionCone(g = g, glb = glb, gub = gub, SwingLegIndicator = ParaRightSwingFlag, F_k = FR4_k, TerrainTangentX = P_k_next_TangentX, TerrainTangentY = P_k_next_TangentY, TerrainNorm = P_k_next_Norm, miu = miu)

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

                ##First-order Angular Momentum Dynamics x-axis
                #g.append(Lx[k+1] - Lx[k] - h*Ldotx[k])
                #glb.append(np.array([0]))
                #gub.append(np.array([0]))

                ##First-order Angular Momentum Dynamics y-axis
                #g.append(Ly[k+1] - Ly[k] - h*Ldoty[k])
                #glb.append(np.array([0]))
                #gub.append(np.array([0]))

                ##First-order Angular Momentum Dynamics z-axis
                #g.append(Lz[k+1] - Lz[k] - h*Ldotz[k])
                #glb.append(np.array([0]))
                #gub.append(np.array([0]))

                #Second-order Dynamics x-axis
                g.append(xdot[k+1] - xdot[k] - h*(FL1x[k]/m+FL2x[k]/m+FL3x[k]/m+FL4x[k]/m+FR1x[k]/m+FR2x[k]/m+FR3x[k]/m+FR4x[k]/m))
                glb.append(np.array([0]))
                gub.append(np.array([0]))

                #Second-order Dynamics y-axis
                g.append(ydot[k+1] - ydot[k] - h*(FL1y[k]/m+FL2y[k]/m+FL3y[k]/m+FL4y[k]/m+FR1y[k]/m+FR2y[k]/m+FR3y[k]/m+FR4y[k]/m))
                glb.append(np.array([0]))
                gub.append(np.array([0]))

                #Second-order Dynamics z-axis
                g.append(zdot[k+1] - zdot[k] - h*(FL1z[k]/m+FL2z[k]/m+FL3z[k]/m+FL4z[k]/m+FR1z[k]/m+FR2z[k]/m+FR3z[k]/m+FR4z[k]/m - G))
                glb.append(np.array([0]))
                gub.append(np.array([0]))
            
            #Add Cost Terms
            if k < N_K - 1:
                #J = J + h*(FL1x[k]/m+FL2x[k]/m+FL3x[k]/m+FL4x[k]/m+FR1x[k]/m+FR2x[k]/m+FR3x[k]/m+FR4x[k]/m)**2 + h*(FL1y[k]/m+FL2y[k]/m+FL3y[k]/m+FL4y[k]/m+FR1y[k]/m+FR2y[k]/m+FR3y[k]/m+FR4y[k]/m)**2 + h*(FL1z[k]/m+FL2z[k]/m+FL3z[k]/m+FL4z[k]/m+FR1z[k]/m+FR2z[k]/m+FR3z[k]/m+FR4z[k]/m - G)**2 + 1*(h*(1/4*(cL1x_p[k]-cL1x_q[k]))**2 + h*(1/4*(cL1y_p[k]-cL1y_q[k]))**2 + h*(1/4*(cL1z_p[k]-cL1z_q[k]))**2 + h*(1/4*(cL2x_p[k]-cL2x_q[k]))**2 + h*(1/4*(cL2y_p[k]-cL2y_q[k]))**2 + h*(1/4*(cL2z_p[k]-cL2z_q[k]))**2 + h*(1/4*(cL3x_p[k]-cL3x_q[k]))**2 + h*(1/4*(cL3y_p[k]-cL3y_q[k]))**2 + h*(1/4*(cL3z_p[k]-cL3z_q[k]))**2 + h*(1/4*(cL4x_p[k]-cL4x_q[k]))**2 + h*(1/4*(cL4y_p[k]-cL4y_q[k]))**2 + h*(1/4*(cL4z_p[k]-cL4z_q[k]))**2 + h*(1/4*(cR1x_p[k]-cR1x_q[k]))**2 + h*(1/4*(cR1y_p[k]-cR1y_q[k]))**2 + h*(1/4*(cR1z_p[k]-cR1z_q[k]))**2 + h*(1/4*(cR2x_p[k]-cR2x_q[k]))**2 + h*(1/4*(cR2y_p[k]-cR2y_q[k]))**2 + h*(1/4*(cR2z_p[k]-cR2z_q[k]))**2 + h*(1/4*(cR3x_p[k]-cR3x_q[k]))**2 + h*(1/4*(cR3y_p[k]-cR3y_q[k]))**2 + h*(1/4*(cR3z_p[k]-cR3z_q[k]))**2 + h*(1/4*(cR4x_p[k]-cR4x_q[k]))**2 + h*(1/4*(cR4y_p[k]-cR4y_q[k]))**2 + h*(1/4*(cR4z_p[k]-cR4z_q[k]))**2)
                #J = J + h*(FL1x[k]/m+FL2x[k]/m+FL3x[k]/m+FL4x[k]/m+FR1x[k]/m+FR2x[k]/m+FR3x[k]/m+FR4x[k]/m)**2 + h*(FL1y[k]/m+FL2y[k]/m+FL3y[k]/m+FL4y[k]/m+FR1y[k]/m+FR2y[k]/m+FR3y[k]/m+FR4y[k]/m)**2 + h*(FL1z[k]/m+FL2z[k]/m+FL3z[k]/m+FL4z[k]/m+FR1z[k]/m+FR2z[k]/m+FR3z[k]/m+FR4z[k]/m - G)**2 + h*(1/4*(cL1x_p[k]-cL1x_q[k]) + 1/4*(cL2x_p[k]-cL2x_q[k]) + 1/4*(cL3x_p[k]-cL3x_q[k]) + 1/4*(cL4x_p[k]-cL4x_q[k]) + 1/4*(cR1x_p[k]-cR1x_q[k]) + 1/4*(cR2x_p[k]-cR2x_q[k]) + 1/4*(cR3x_p[k]-cR3x_q[k]) + 1/4*(cR4x_p[k]-cR4x_q[k]))**2 + h*(1/4*(cL1y_p[k]-cL1y_q[k]) + 1/4*(cL2y_p[k]-cL2y_q[k]) + 1/4*(cL3y_p[k]-cL3y_q[k]) + 1/4*(cL4y_p[k]-cL4y_q[k]) + 1/4*(cR1y_p[k]-cR1y_q[k]) + 1/4*(cR2y_p[k]-cR2y_q[k]) + 1/4*(cR3y_p[k]-cR3y_q[k]) + 1/4*(cR4y_p[k]-cR4y_q[k]))**2 + h*(1/4*(cL1z_p[k]-cL1z_q[k]) + 1/4*(cL2z_p[k]-cL2z_q[k]) + 1/4*(cL3z_p[k]-cL3z_q[k]) + 1/4*(cL4z_p[k]-cL4z_q[k]) + 1/4*(cR1z_p[k]-cR1z_q[k]) + 1/4*(cR2z_p[k]-cR2z_q[k]) + 1/4*(cR3z_p[k]-cR3z_q[k]) + 1/4*(cR4z_p[k]-cR4z_q[k]))**2
                J = J + h*(FL1x[k]/m+FL2x[k]/m+FL3x[k]/m+FL4x[k]/m+FR1x[k]/m+FR2x[k]/m+FR3x[k]/m+FR4x[k]/m)**2 + h*(FL1y[k]/m+FL2y[k]/m+FL3y[k]/m+FL4y[k]/m+FR1y[k]/m+FR2y[k]/m+FR3y[k]/m+FR4y[k]/m)**2 + h*(FL1z[k]/m+FL2z[k]/m+FL3z[k]/m+FL4z[k]/m+FR1z[k]/m+FR2z[k]/m+FR3z[k]/m+FR4z[k]/m - G)**2 + h*(cL1x_p[k]**2 + cL1x_q[k]**2 + cL1y_p[k]**2 + cL1y_q[k]**2 + cL1z_p[k]**2 + cL1z_q[k]**2 + cL2x_p[k]**2 + cL2x_q[k]**2 + cL2y_p[k]**2 + cL2y_q[k]**2 + cL2z_p[k]**2 + cL2z_q[k]**2 + cL3x_p[k]**2 + cL3x_q[k]**2 + cL3y_p[k]**2 + cL3y_q[k]**2 + cL3z_p[k]**2 + cL3z_q[k]**2 + cL4x_p[k]**2 + cL4x_q[k]**2 + cL4y_p[k]**2 + cL4y_q[k]**2 + cL4z_p[k]**2 + cL4z_q[k]**2 + cR1x_p[k]**2 + cR1x_q[k]**2 + cR1y_p[k]**2 + cR1y_q[k]**2 + cR1z_p[k]**2 + cR1z_q[k]**2 + cR2x_p[k]**2 + cR2x_q[k]**2 + cR2y_p[k]**2 + cR2y_q[k]**2 + cR2z_p[k]**2 + cR2z_q[k]**2 + cR3x_p[k]**2 + cR3x_q[k]**2 + cR3y_p[k]**2 + cR3y_q[k]**2 + cR3z_p[k]**2 + cR3z_q[k]**2 + cR4x_p[k]**2 + cR4x_q[k]**2 + cR4y_p[k]**2 + cR4y_q[k]**2 + cR4z_p[k]**2 + cR4z_q[k]**2)
                #J = J + h*((FL1x[k]/m)**2+(FL2x[k]/m)**2+(FL3x[k]/m)**2+(FL4x[k]/m)**2+(FR1x[k]/m)**2+(FR2x[k]/m)**2+(FR3x[k]/m)**2+(FR4x[k]/m)**2) + h*((FL1y[k]/m)**2+(FL2y[k]/m)**2+(FL3y[k]/m)**2+(FL4y[k]/m)**2+(FR1y[k]/m)**2+(FR2y[k]/m)**2+(FR3y[k]/m)**2+(FR4y[k]/m)**2) + h*((FL1z[k]/m)**2+(FL2z[k]/m)**2+(FL3z[k]/m)**2+(FL4z[k]/m)**2+(FR1z[k]/m)**2+(FR2z[k]/m)**2+(FR3z[k]/m)**2+(FR4z[k]/m)**2) + h*(cL1x_p[k]**2 + cL1x_q[k]**2 + cL1y_p[k]**2 + cL1y_q[k]**2 + cL1z_p[k]**2 + cL1z_q[k]**2 + cL2x_p[k]**2 + cL2x_q[k]**2 + cL2y_p[k]**2 + cL2y_q[k]**2 + cL2z_p[k]**2 + cL2z_q[k]**2 + cL3x_p[k]**2 + cL3x_q[k]**2 + cL3y_p[k]**2 + cL3y_q[k]**2 + cL3z_p[k]**2 + cL3z_q[k]**2 + cL4x_p[k]**2 + cL4x_q[k]**2 + cL4y_p[k]**2 + cL4y_q[k]**2 + cL4z_p[k]**2 + cL4z_q[k]**2 + cR1x_p[k]**2 + cR1x_q[k]**2 + cR1y_p[k]**2 + cR1y_q[k]**2 + cR1z_p[k]**2 + cR1z_q[k]**2 + cR2x_p[k]**2 + cR2x_q[k]**2 + cR2y_p[k]**2 + cR2y_q[k]**2 + cR2z_p[k]**2 + cR2z_q[k]**2 + cR3x_p[k]**2 + cR3x_q[k]**2 + cR3y_p[k]**2 + cR3y_q[k]**2 + cR3z_p[k]**2 + cR3z_q[k]**2 + cR4x_p[k]**2 + cR4x_q[k]**2 + cR4y_p[k]**2 + cR4y_q[k]**2 + cR4z_p[k]**2 + cR4z_q[k]**2)

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
            #Also, the stationary foot Left should stay in the polytope of the landed swing foot - RIGHT
            #NOTE: current - next now
            g.append(ca.if_else(ParaLeftSwingFlag,Q_lf_in_rf@(P_k_current-P_k_next)-ca.DM(q_lf_in_rf),np.full((len(q_lf_in_rf),),-1)))
            glb.append(np.full((len(q_lf_in_rf),),-np.inf))
            gub.append(np.full((len(q_lf_in_rf),),0))

            #If First Levvel Swing the Right, then the second level Even Number Phases (the first phase) Swing the Left -> Right -> Left
            #Right foot in contact for p_current, left foot is going to land at p_next
            #Relative Swing Foot Location (lf in rf)
            g.append(ca.if_else(ParaRightSwingFlag,Q_lf_in_rf@(P_k_next-P_k_current)-ca.DM(q_lf_in_rf),np.full((len(q_lf_in_rf),),-1)))
            glb.append(np.full((len(q_lf_in_rf),),-np.inf))
            gub.append(np.full((len(q_lf_in_rf),),0))
            #Also, the stationary foot Rigth should stay in the polytope of the landed swing foot - Left
            #NOTE: current - next now
            g.append(ca.if_else(ParaRightSwingFlag,Q_rf_in_lf@(P_k_current-P_k_next)-ca.DM(q_rf_in_lf),np.full((len(q_rf_in_lf),),-1)))
            glb.append(np.full((len(q_rf_in_lf),),-np.inf))
            gub.append(np.full((len(q_rf_in_lf),),0))

        elif step_cnt%2 == 1: #odd number steps
            #If the first level swings the Left, then the second level for Odd Number Phases (the second phase) swings left -> Right -> Left
            #Right foot in contact for p_current, left foot is going to land at p_next
            #Relative Swing Foot Location (lf in rf)
            g.append(ca.if_else(ParaLeftSwingFlag,Q_lf_in_rf@(P_k_next-P_k_current)-ca.DM(q_lf_in_rf),np.full((len(q_lf_in_rf),),-1)))
            glb.append(np.full((len(q_lf_in_rf),),-np.inf))
            gub.append(np.full((len(q_lf_in_rf),),0))
            #Also, the stationary foot Rigth should stay in the polytope of the landed swing foot - Left
            #NOTE: current - next now
            g.append(ca.if_else(ParaLeftSwingFlag,Q_rf_in_lf@(P_k_current-P_k_next)-ca.DM(q_rf_in_lf),np.full((len(q_rf_in_lf),),-1)))
            glb.append(np.full((len(q_rf_in_lf),),-np.inf))
            gub.append(np.full((len(q_rf_in_lf),),0))

            #If the first level swings the Right, then the second level for Odd Number Phases (the second phase) swings Right -> Left -> Right
            #Left foot in contact for p_current, right foot is going to land as p_next
            #Relative Swing Foot Location (rf in lf)
            g.append(ca.if_else(ParaRightSwingFlag,Q_rf_in_lf@(P_k_next-P_k_current)-ca.DM(q_rf_in_lf),np.full((len(q_rf_in_lf),),-1)))
            glb.append(np.full((len(q_rf_in_lf),),-np.inf))
            gub.append(np.full((len(q_rf_in_lf),),0))
            #Also, the stationary foot LEFT should stay in the polytope of the landed swing foot - RIGHT
            #NOTE: current - next now
            g.append(ca.if_else(ParaRightSwingFlag,Q_lf_in_rf@(P_k_current-P_k_next)-ca.DM(q_lf_in_rf),np.full((len(q_lf_in_rf),),-1)))
            glb.append(np.full((len(q_lf_in_rf),),-np.inf))
            gub.append(np.full((len(q_lf_in_rf),),0))

    #Foot Step Constraint
    #FootStep Constraint
    #P3----------------P1
    #|                  |
    #|                  |
    #|                  |
    #P4----------------P2
    for PatchNum in range(Nsteps):
        #Get Footstep Vector
        P_vector = ca.vertcat(px[PatchNum],py[PatchNum],pz[PatchNum])

        #Get Half Space Representation 
        #NOTE: In the second level, the terrain patch start from the second patch, indexed as 1
        SurfParaTemp = SurfParas[20+PatchNum*20:19+(PatchNum+1)*20+1]
        #print(SurfParaTemp)
        SurfK = SurfParaTemp[0:11+1]
        SurfK = ca.reshape(SurfK,3,4)
        SurfK = SurfK.T #NOTE: This process is due to casadi naming convention to have first row to be x1,x2,x3
        SurfE = SurfParaTemp[11+1:11+3+1]
        Surfk = SurfParaTemp[14+1:14+4+1]
        Surfe = SurfParaTemp[-1]

        #Terrain Tangent and Norms
        P_vector_TangentX = SurfTangentsX[(PatchNum+1)*3:(PatchNum+1)*3+3]
        P_vector_TangentY = SurfTangentsY[(PatchNum+1)*3:(PatchNum+1)*3+3]

        #Contact Point 1
        #Inequality
        g.append(SurfK@(P_vector + 0.11*P_vector_TangentX + 0.06*P_vector_TangentY) - Surfk)
        glb.append(np.full((4,),-np.inf))
        gub.append(np.full((4,),0))
        #Equality
        g.append(SurfE.T@(P_vector + 0.11*P_vector_TangentX + 0.06*P_vector_TangentY) - Surfe)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        #Contact Point 2
        #Inequality
        g.append(SurfK@(P_vector + 0.11*P_vector_TangentX - 0.06*P_vector_TangentY) - Surfk)
        glb.append(np.full((4,),-np.inf))
        gub.append(np.full((4,),0))
        #Equality
        g.append(SurfE.T@(P_vector + 0.11*P_vector_TangentX - 0.06*P_vector_TangentY) - Surfe)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        #Contact Point 3
        #Inequality
        g.append(SurfK@(P_vector - 0.11*P_vector_TangentX + 0.06*P_vector_TangentY) - Surfk)
        glb.append(np.full((4,),-np.inf))
        gub.append(np.full((4,),0))
        #Equality
        g.append(SurfE.T@(P_vector - 0.11*P_vector_TangentX + 0.06*P_vector_TangentY) - Surfe)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        #Contact Point 4
        #Inequality
        g.append(SurfK@(P_vector + 0.11*P_vector_TangentX + 0.06*P_vector_TangentY) - Surfk)
        glb.append(np.full((4,),-np.inf))
        gub.append(np.full((4,),0))
        #Equality
        g.append(SurfE.T@(P_vector + 0.11*P_vector_TangentX + 0.06*P_vector_TangentY) - Surfe)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        ##FootStep Constraint
        ##Inequality
        #g.append(SurfK@P_vector - Surfk)
        #glb.append(np.full((4,),-np.inf))
        #gub.append(np.full((4,),0))
        #print(FirstSurfK@p_next - FirstSurfk)

        ##Equality
        #g.append(SurfE.T@P_vector - Surfe)
        #glb.append(np.array([0]))
        #gub.append(np.array([0]))

    #Approximate Kinematics Constraint --- Disallow over-crossing of footsteps from y =0

    if CentralY == True:

        for step_cnt in range(Nsteps):
            P_k_next = ca.vertcat(px[step_cnt],py[step_cnt],pz[step_cnt])
    
            if step_cnt%2 == 0: #even number steps
                #If First Level Swing the Left, then the second level Even Number Phases (the first Phase) Swing the Right -> Left -> Right
                #Left foot in contact for p_current, right foot is going to land as p_next
                g.append(ca.if_else(ParaLeftSwingFlag,P_k_next[1],np.array([-1])))
                glb.append(np.array([-np.inf]))
                gub.append(np.array([-py_lower_limit]))
    
                #If First Levvel Swing the Right, then the second level Even Number Phases (the first phase) Swing the Left -> Right -> Left
                #Right foot in contact for p_current, left foot is going to land at p_next
                g.append(ca.if_else(ParaRightSwingFlag,P_k_next[1],np.array([1])))
                glb.append(np.array([py_lower_limit]))
                gub.append(np.array([np.inf]))
    
            elif step_cnt%2 == 1: #odd number steps
                #If the first level swings the Left, then the second level for Odd Number Phases (the second phase) swings left -> Right -> Left
                #Right foot in contact for p_current, left foot is going to land at p_next
                g.append(ca.if_else(ParaLeftSwingFlag,P_k_next[1],np.array([1])))
                glb.append(np.array([py_lower_limit]))
                gub.append(np.array([np.inf]))
    
                #If the first level swings the Right, then the second level for Odd Number Phases (the second phase) swings Right -> Left -> Right
                #Left foot in contact for p_current, right foot is going to land as p_next
                g.append(ca.if_else(ParaRightSwingFlag,P_k_next[1],np.array([-1])))
                glb.append(np.array([-np.inf]))
                gub.append(np.array([-py_lower_limit]))

    #Switching Time Constraint
    for phase_cnt in range(Nphase):
        if GaitPattern[phase_cnt] == 'InitialDouble':
            g.append(Ts[phase_cnt])
            glb.append(np.array([0.05])) #0.05 0.6
            gub.append(np.array([0.3])) #o.3
        elif GaitPattern[phase_cnt] == 'Swing':
            if phase_cnt == 0:
                g.append(Ts[phase_cnt]-0)#0.6-1
                glb.append(np.array([0.5]))#0.5 for NLP success
                gub.append(np.array([0.7]))
            else:
                g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
                glb.append(np.array([0.5]))
                gub.append(np.array([0.7]))
        elif GaitPattern[phase_cnt] == 'DoubleSupport':
            g.append(Ts[phase_cnt]-Ts[phase_cnt-1]) #0.1-0.9
            glb.append(np.array([0.1]))
            gub.append(np.array([0.4])) #0.1-0.3

    #-----------------------------------------------------------------------------------------------------------------------
    #Get Variable Index - !!!This is the pure Index, when try to get the array using other routines, we need to add "+1" at the last index due to Python indexing conventions
    x_index = (0,N_K-1) #First set of variables start counting from 0, The enumeration of the last knot is N_K-1
    y_index = (x_index[1]+1,x_index[1]+N_K)
    z_index = (y_index[1]+1,y_index[1]+N_K)
    xdot_index = (z_index[1]+1,z_index[1]+N_K)
    ydot_index = (xdot_index[1]+1,xdot_index[1]+N_K)
    zdot_index = (ydot_index[1]+1,ydot_index[1]+N_K)
    FL1x_index = (zdot_index[1]+1,zdot_index[1]+N_K)
    FL1y_index = (FL1x_index[1]+1,FL1x_index[1]+N_K)
    FL1z_index = (FL1y_index[1]+1,FL1y_index[1]+N_K)
    FL2x_index = (FL1z_index[1]+1,FL1z_index[1]+N_K)
    FL2y_index = (FL2x_index[1]+1,FL2x_index[1]+N_K)
    FL2z_index = (FL2y_index[1]+1,FL2y_index[1]+N_K)
    FL3x_index = (FL2z_index[1]+1,FL2z_index[1]+N_K)
    FL3y_index = (FL3x_index[1]+1,FL3x_index[1]+N_K)
    FL3z_index = (FL3y_index[1]+1,FL3y_index[1]+N_K)
    FL4x_index = (FL3z_index[1]+1,FL3z_index[1]+N_K)
    FL4y_index = (FL4x_index[1]+1,FL4x_index[1]+N_K)
    FL4z_index = (FL4y_index[1]+1,FL4y_index[1]+N_K)
    FR1x_index = (FL4z_index[1]+1,FL4z_index[1]+N_K)
    FR1y_index = (FR1x_index[1]+1,FR1x_index[1]+N_K)
    FR1z_index = (FR1y_index[1]+1,FR1y_index[1]+N_K)
    FR2x_index = (FR1z_index[1]+1,FR1z_index[1]+N_K)
    FR2y_index = (FR2x_index[1]+1,FR2x_index[1]+N_K)
    FR2z_index = (FR2y_index[1]+1,FR2y_index[1]+N_K)
    FR3x_index = (FR2z_index[1]+1,FR2z_index[1]+N_K)
    FR3y_index = (FR3x_index[1]+1,FR3x_index[1]+N_K)
    FR3z_index = (FR3y_index[1]+1,FR3y_index[1]+N_K)
    FR4x_index = (FR3z_index[1]+1,FR3z_index[1]+N_K)
    FR4y_index = (FR4x_index[1]+1,FR4x_index[1]+N_K)
    FR4z_index = (FR4y_index[1]+1,FR4y_index[1]+N_K)
    px_init_index = (FR4z_index[1]+1,FR4z_index[1]+1)
    py_init_index = (px_init_index[1]+1,px_init_index[1]+1)
    pz_init_index = (py_init_index[1]+1,py_init_index[1]+1)
    px_index = (pz_init_index[1]+1,pz_init_index[1]+Nsteps)
    py_index = (px_index[1]+1,px_index[1]+Nsteps)
    pz_index = (py_index[1]+1,py_index[1]+Nsteps)
    Ts_index = (pz_index[1]+1,pz_index[1]+Nphase)

    var_index = {"x":x_index,
                 "y":y_index,
                 "z":z_index,
                 "xdot":xdot_index,
                 "ydot":ydot_index,
                 "zdot":zdot_index,
                 "FL1x":FL1x_index,
                 "FL1y":FL1y_index,
                 "FL1z":FL1z_index,
                 "FL2x":FL2x_index,
                 "FL2y":FL2y_index,
                 "FL2z":FL2z_index,
                 "FL3x":FL3x_index,
                 "FL3y":FL3y_index,
                 "FL3z":FL3z_index,
                 "FL4x":FL4x_index,
                 "FL4y":FL4y_index,
                 "FL4z":FL4z_index,
                 "FR1x":FR1x_index,
                 "FR1y":FR1y_index,
                 "FR1z":FR1z_index,
                 "FR2x":FR2x_index,
                 "FR2y":FR2y_index,
                 "FR2z":FR2z_index,
                 "FR3x":FR3x_index,
                 "FR3y":FR3y_index,
                 "FR3z":FR3z_index,
                 "FR4x":FR4x_index,
                 "FR4y":FR4y_index,
                 "FR4z":FR4z_index,
                 "px_init":px_init_index,
                 "py_init":py_init_index,
                 "pz_init":pz_init_index,
                 "px":px_index,
                 "py":py_index,
                 "pz":pz_index,
                 "Ts":Ts_index,
    }

    #print(DecisionVars[var_index["px_init"][0]:var_index["px_init"][1]+1])

    return DecisionVars, DecisionVars_lb, DecisionVars_ub, J, g, glb, gub, var_index

#Build Solver in accordance to the set up of first level second levels
def BuildSolver(FirstLevel = None, ConservativeFirstStep = True, SecondLevel = None, m = 95, NumSurfaces = None):

    #Check if the First Level is selected properly
    assert FirstLevel != None, "First Level is Not Selected."
    assert NumSurfaces != None, "No Surface Vector Defined."

    #-------------------------------------------
    #Define Solver Parameter Vector
    #   Initial Position
    x_init = ca.SX.sym('x_init')
    y_init = ca.SX.sym('y_init')
    z_init = ca.SX.sym('z_init')
    #   Initial Velocity
    xdot_init = ca.SX.sym('xdot_init')
    ydot_init = ca.SX.sym('ydot_init')
    zdot_init = ca.SX.sym('zdot_init')
    #   Initial Anglular Momentum
    Lx_init = ca.SX.sym('Lx_init')
    Ly_init = ca.SX.sym('Ly_init')
    Lz_init = ca.SX.sym('Lz_init')
    #   Initial Anglular Momentum Rate
    Ldotx_init = ca.SX.sym('Ldotx_init')
    Ldoty_init = ca.SX.sym('Ldoty_init')
    Ldotz_init = ca.SX.sym('Ldotz_init')
    #   Initial Left Foot Position
    PLx_init = ca.SX.sym('PLx_init')
    PLy_init = ca.SX.sym('PLy_init')
    PLz_init = ca.SX.sym('PLz_init')
    #   Initial Right Foot Position
    PRx_init = ca.SX.sym('PRx_init')
    PRy_init = ca.SX.sym('PRy_init')
    PRz_init = ca.SX.sym('PRz_init')
    #   Terminal Position
    x_end = ca.SX.sym('x_end')
    y_end = ca.SX.sym('y_end')
    z_end = ca.SX.sym('z_end')
    #   Terminal Velocity
    xdot_end = ca.SX.sym('xdot_end')
    ydot_end = ca.SX.sym('ydot_end')
    zdot_end = ca.SX.sym('zdot_end')
    #   Foot Swing Indicators
    ParaLeftSwingFlag = ca.SX.sym('LeftSwingFlag')
    ParaRightSwingFlag = ca.SX.sym('RightSwingFlag')
    #   Surface Parameters
    #       Surface patch
    SurfParas = []
    for surfNum in range(NumSurfaces):
        SurfTemp = ca.SX.sym('S'+str(surfNum),3*4+3+5)
        SurfParas.append(SurfTemp)
    SurfParas = ca.vertcat(*SurfParas)
    #print(SurfParas)
    #       Surface TangentsX
    SurfTangentsX = []
    for surfNum in range(NumSurfaces):
        SurfTangentX_temp = ca.SX.sym('SurfTengentX'+str(surfNum),3)
        SurfTangentsX.append(SurfTangentX_temp)
    SurfTangentsX = ca.vertcat(*SurfTangentsX)
    #       Surface TangentsY
    SurfTangentsY = []
    for surfNum in range(NumSurfaces):
        SurfTangentY_temp = ca.SX.sym('SurfTengentY'+str(surfNum),3)
        SurfTangentsY.append(SurfTangentY_temp)
    SurfTangentsY = ca.vertcat(*SurfTangentsY)
    #       Surface Norm
    SurfNorms = []
    for surfNum in range(NumSurfaces):
        SurfNorm_temp = ca.SX.sym('SurfNorm'+str(surfNum),3)
        SurfNorms.append(SurfNorm_temp)
    SurfNorms = ca.vertcat(*SurfNorms)

    #   Initial Contact Tangents and Norm
    PL_init_TangentX = ca.SX.sym('PL_init_TangentX',3)
    PL_init_TangentY = ca.SX.sym('PL_init_TangentY',3)
    PL_init_Norm = ca.SX.sym('PL_init_Norm',3)
    PR_init_TangentX = ca.SX.sym('PR_init_TangentX',3)
    PR_init_TangentY = ca.SX.sym('PR_init_TangentY',3)
    PR_init_Norm = ca.SX.sym('PR_init_Norm',3)

    #Ref Trajectories
    LookAhead_Num_SecondLevel = NumSurfaces - 1
    LocalKnotNum_SecondLevel = 7
    TotalPhaseNum_SecondLevel = 3*LookAhead_Num_SecondLevel #3 phases per step
    TotalKnotsNum_SecondLevel = TotalPhaseNum_SecondLevel*LocalKnotNum_SecondLevel + 1
    
    x_ref = ca.SX.sym('x_ref',TotalKnotsNum_SecondLevel)
    y_ref = ca.SX.sym('y_ref',TotalKnotsNum_SecondLevel)
    z_ref = ca.SX.sym('z_ref',TotalKnotsNum_SecondLevel)

    xdot_ref = ca.SX.sym('xdot_ref',TotalKnotsNum_SecondLevel)
    ydot_ref = ca.SX.sym('ydot_ref',TotalKnotsNum_SecondLevel)
    zdot_ref = ca.SX.sym('zdot_ref',TotalKnotsNum_SecondLevel)

    FLx_ref = ca.SX.sym('FLx_ref',TotalKnotsNum_SecondLevel)
    FLy_ref = ca.SX.sym('FLy_ref',TotalKnotsNum_SecondLevel)
    FLz_ref = ca.SX.sym('FLz_ref',TotalKnotsNum_SecondLevel)

    FRx_ref = ca.SX.sym('FRx_ref',TotalKnotsNum_SecondLevel)
    FRy_ref = ca.SX.sym('FRy_ref',TotalKnotsNum_SecondLevel)
    FRz_ref = ca.SX.sym('FRz_ref',TotalKnotsNum_SecondLevel)

    SwitchingTimeVec_ref = ca.SX.sym('SwitchingTimeVec_ref',TotalPhaseNum_SecondLevel)

    Px_seq_ref = ca.SX.sym('Px_seq_ref',LookAhead_Num_SecondLevel)
    Py_seq_ref = ca.SX.sym('Py_seq_ref',LookAhead_Num_SecondLevel)
    Pz_seq_ref = ca.SX.sym('Pz_seq_ref',LookAhead_Num_SecondLevel)



    ##   FirstRound Indicators (if yes, we have an initial double support phase, if not, then we dont have an initial double support phase)
    #FirstRoundFlag = ca.SX.sym('FirstRoundFlag')
    #   Collect all Parameters

    if SecondLevel == "CoM_Dynamics":

        ParaList = {"LeftSwingFlag":ParaLeftSwingFlag,
                    "RightSwingFlag":ParaRightSwingFlag,
                    "x_init":x_init,
                    "y_init":y_init,
                    "z_init":z_init,
                    "xdot_init":xdot_init,
                    "ydot_init":ydot_init,
                    "zdot_init":zdot_init,
                    "Lx_init":Lx_init,
                    "Ly_init":Ly_init,
                    "Lz_init":Lz_init,
                    "Ldotx_init":Ldotx_init,
                    "Ldoty_init":Ldoty_init,
                    "Ldotz_init":Ldotz_init,
                    "PLx_init":PLx_init,
                    "PLy_init":PLy_init,
                    "PLz_init":PLz_init,
                    "PRx_init":PRx_init,
                    "PRy_init":PRy_init,
                    "PRz_init":PRz_init,
                    "x_end":x_end,
                    "y_end":y_end,
                    "z_end":z_end,
                    "xdot_end":xdot_end,
                    "ydot_end":ydot_end,
                    "zdot_end":zdot_end,
                    "SurfParas":SurfParas,
                    "SurfTangentsX":SurfTangentsX,
                    "SurfTangentsY":SurfTangentsY,
                    "SurfNorms":SurfNorms,
                    "PL_init_TangentX":PL_init_TangentX,
                    "PL_init_TangentY":PL_init_TangentY,
                    "PL_init_Norm":PL_init_Norm,
                    "PR_init_TangentX":PR_init_TangentX,
                    "PR_init_TangentY":PR_init_TangentY,
                    "PR_init_Norm":PR_init_Norm,
                    "x_ref":x_ref,
                    "y_ref":y_ref,
                    "z_ref":z_ref,
                    "xdot_ref":xdot_ref,
                    "ydot_ref":ydot_ref,
                    "zdot_ref":zdot_ref,
                    "FLx_ref":FLx_ref,
                    "FLy_ref":FLy_ref,
                    "FLz_ref":FLz_ref,
                    "FRx_ref":FRx_ref,
                    "FRy_ref":FRy_ref,
                    "FRz_ref":FRz_ref,
                    "SwitchingTimeVec_ref":SwitchingTimeVec_ref,
                    "Px_seq_ref":Px_seq_ref,
                    "Py_seq_ref":Py_seq_ref,
                    "Pz_seq_ref":Pz_seq_ref,
        }
        #            "FirstRoundFlag":FirstRoundFlag,
        #Collect all Parameters
        paras = ca.vertcat(ParaLeftSwingFlag,ParaRightSwingFlag,
                        x_init,y_init,z_init,
                        xdot_init,ydot_init,zdot_init,
                        Lx_init,Ly_init,Lz_init,
                        Ldotx_init,Ldoty_init,Ldotz_init,
                        PLx_init,PLy_init,PLz_init,
                        PRx_init,PRy_init,PRz_init,
                        x_end,y_end,z_end,
                        xdot_end,ydot_end,zdot_end,
                        SurfParas,SurfTangentsX,SurfTangentsY,SurfNorms,
                        PL_init_TangentX,PL_init_TangentY,PL_init_Norm,
                        PR_init_TangentX,PR_init_TangentY,PR_init_Norm,
                        x_ref,y_ref,z_ref,
                        xdot_ref,ydot_ref,zdot_ref,
                        FLx_ref,FLy_ref,FLz_ref,
                        FRx_ref,FRy_ref,FRz_ref,
                        SwitchingTimeVec_ref,
                        Px_seq_ref,Py_seq_ref,Pz_seq_ref)
    else: 
        ParaList = {"LeftSwingFlag":ParaLeftSwingFlag,
                    "RightSwingFlag":ParaRightSwingFlag,
                    "x_init":x_init,
                    "y_init":y_init,
                    "z_init":z_init,
                    "xdot_init":xdot_init,
                    "ydot_init":ydot_init,
                    "zdot_init":zdot_init,
                    "Lx_init":Lx_init,
                    "Ly_init":Ly_init,
                    "Lz_init":Lz_init,
                    "Ldotx_init":Ldotx_init,
                    "Ldoty_init":Ldoty_init,
                    "Ldotz_init":Ldotz_init,
                    "PLx_init":PLx_init,
                    "PLy_init":PLy_init,
                    "PLz_init":PLz_init,
                    "PRx_init":PRx_init,
                    "PRy_init":PRy_init,
                    "PRz_init":PRz_init,
                    "x_end":x_end,
                    "y_end":y_end,
                    "z_end":z_end,
                    "xdot_end":xdot_end,
                    "ydot_end":ydot_end,
                    "zdot_end":zdot_end,
                    "SurfParas":SurfParas,
                    "SurfTangentsX":SurfTangentsX,
                    "SurfTangentsY":SurfTangentsY,
                    "SurfNorms":SurfNorms,
                    "PL_init_TangentX":PL_init_TangentX,
                    "PL_init_TangentY":PL_init_TangentY,
                    "PL_init_Norm":PL_init_Norm,
                    "PR_init_TangentX":PR_init_TangentX,
                    "PR_init_TangentY":PR_init_TangentY,
                    "PR_init_Norm":PR_init_Norm
        }
        #            "FirstRoundFlag":FirstRoundFlag,
        #Collect all Parameters
        paras = ca.vertcat(ParaLeftSwingFlag,ParaRightSwingFlag,
                        x_init,y_init,z_init,
                        xdot_init,ydot_init,zdot_init,
                        Lx_init,Ly_init,Lz_init,
                        Ldotx_init,Ldoty_init,Ldotz_init,
                        PLx_init,PLy_init,PLz_init,
                        PRx_init,PRy_init,PRz_init,
                        x_end,y_end,z_end,
                        xdot_end,ydot_end,zdot_end,
                        SurfParas,SurfTangentsX,SurfTangentsY,SurfNorms,
                        PL_init_TangentX,PL_init_TangentY,PL_init_Norm,
                        PR_init_TangentX,PR_init_TangentY,PR_init_Norm)

    #-----------------------------------------------------------------------------------------------------------------
    #Identify the Fidelity Type of the whole framework, Used to tell the First Level to set Constraints Accordingly
    if SecondLevel == None:
        SingleFidelity = True
    else:
        SingleFidelity = False

    #Bulding the First Level
    if FirstLevel == "NLP_SingleStep":
        var_Level1, var_lb_Level1, var_ub_Level1, J_Level1, g_Level1, glb_Level1, gub_Level1, var_index_Level1 = NLP_SingleStep(m = m, StandAlong = SingleFidelity, ParameterList = ParaList)
    elif FirstLevel == "Pure_Kinematics_Check":
        var_Level1, var_lb_Level1, var_ub_Level1, J_Level1, g_Level1, glb_Level1, gub_Level1, var_index_Level1 = Pure_Kinematics_Check(StandAlong = SingleFidelity, ParameterList = ParaList)
    else:
        print("Print Not implemented or Wrong Solver Build Enumeration")        
    
    #!!!!!Building the Second Level
    if SecondLevel == None:
        print("No Second Level")
        var_index_Level2 = []
    elif SecondLevel == "CoM_Dynamics":
        #var_Level2, var_lb_Level2, var_ub_Level2, J_Level2, g_Level2, glb_Level2, gub_Level2, var_index_Level2 = CoM_Dynamics_Ponton_Cost(m = m,  ParameterList = ParaList, Nsteps = NumSurfaces-1)#Here is the total number of steps
        var_Level2, var_lb_Level2, var_ub_Level2, J_Level2, g_Level2, glb_Level2, gub_Level2, var_index_Level2 = CoM_Dynamics_SinglePoint(m = m,  ParameterList = ParaList, Nsteps = NumSurfaces-1)#Here is the total number of steps
        #var_Level2, var_lb_Level2, var_ub_Level2, J_Level2, g_Level2, glb_Level2, gub_Level2, var_index_Level2 = CoM_Dynamics_Four_Points(m = m,  ParameterList = ParaList, Nsteps = NumSurfaces-1)#Here is the total number of steps
    elif SecondLevel == "NLP_SecondLevel":
        var_Level2, var_lb_Level2, var_ub_Level2, J_Level2, g_Level2, glb_Level2, gub_Level2, var_index_Level2 = NLP_SecondLevel(m = m, ParameterList = ParaList, Nsteps = NumSurfaces-1)
    elif SecondLevel == "Mixure":
        var_Level2, var_lb_Level2, var_ub_Level2, J_Level2, g_Level2, glb_Level2, gub_Level2, var_index_Level2 = Mixure(m = m, ParameterList = ParaList, Nsteps = NumSurfaces-1)
    #!!!!!Connect the Terminal state of the first Level with the Second Level

    #Set-up Terminal Cost Here and Sum over all costs
    if SecondLevel == None: #No second Level
        #   Collect the variables, terminal cost set as the end of the single first level
        J = J_Level1
        x_Level1 = var_Level1[var_index_Level1["x"][0]:var_index_Level1["x"][1]+1]
        y_Level1 = var_Level1[var_index_Level1["y"][0]:var_index_Level1["y"][1]+1]
        z_Level1 = var_Level1[var_index_Level1["z"][0]:var_index_Level1["z"][1]+1]
        #xdot_Level1 = var_Level1[var_index_Level1["xdot"][0]:var_index_Level1["xdot"][1]+1]
        #ydot_Level1 = var_Level1[var_index_Level1["ydot"][0]:var_index_Level1["ydot"][1]+1]
        #zdot_Level1 = var_Level1[var_index_Level1["zdot"][0]:var_index_Level1["zdot"][1]+1]
        
        #------
        #Terminal Cost
        #J = J + 10*(x_Level1[-1]-x_end)**2 + 10*(y_Level1[-1]-y_end)**2 + 10*(z_Level1[-1]-z_end)**2
        #---------

    else:#With Second level
        #Summation of the all running cost
        J = J_Level1 + J_Level2
        x_Level2 = var_Level2[var_index_Level2["x"][0]:var_index_Level2["x"][1]+1]
        y_Level2 = var_Level2[var_index_Level2["y"][0]:var_index_Level2["y"][1]+1]
        z_Level2 = var_Level2[var_index_Level2["z"][0]:var_index_Level2["z"][1]+1]
        #if SecondLevel == "CoM_Dynamics" or SecondLevel == "NLP_SecondLevel":
        #    xdot_Level2 = var_Level2[var_index_Level2["xdot"][0]:var_index_Level2["xdot"][1]+1]
        #    ydot_Level2 = var_Level2[var_index_Level2["ydot"][0]:var_index_Level2["ydot"][1]+1]
        #    zdot_Level2 = var_Level2[var_index_Level2["zdot"][0]:var_index_Level2["zdot"][1]+1]
        #Add terminal cost
        #J = J + 100*(x_Level2[-1]-x_end)**2 + 100*(y_Level2[-1]-y_end)**2 + 100*(z_Level2[-1]-z_end)**2 + 100*(xdot_Level2[-1])**2 + 100*(ydot_Level2[-1])**2 + 100*(zdot_Level2[-1])**2
        
        #---------
        #Terminal Cost
        #J = J + 10*(x_Level2[-1]-x_end)**2 + 10*(y_Level2[-1]-y_end)**2 + 10*(z_Level2[-1]-z_end)**2
        #----------

        #J = J + 100*(x_Level2[-1]-x_end)**2
        
        #Deal with Connections between the first level and the second level
        gConnect = []
        gConnect_lb = []
        gConnect_ub = []

        #   Initial Condition x-axis (Connect to the First Level)
        gConnect.append(var_Level2[var_index_Level2["x"][0]]-var_Level1[var_index_Level1["x"][-1]])
        gConnect_lb.append(np.array([0]))
        gConnect_ub.append(np.array([0]))

        #   Initial Condition y-axis (Connect to the First Level)
        gConnect.append(var_Level2[var_index_Level2["y"][0]]-var_Level1[var_index_Level1["y"][-1]])
        gConnect_lb.append(np.array([0]))
        gConnect_ub.append(np.array([0]))

        #   Initial Condition z-axis (Connect to the First Level)
        gConnect.append(var_Level2[var_index_Level2["z"][0]]-var_Level1[var_index_Level1["z"][-1]])
        gConnect_lb.append(np.array([0]))
        gConnect_ub.append(np.array([0]))

        #   Initial Contact Location x-axis (Connect to the First Level)
        gConnect.append(var_Level2[var_index_Level2["px_init"][0]]-var_Level1[var_index_Level1["px"][-1]])
        gConnect_lb.append(np.array([0]))
        gConnect_ub.append(np.array([0]))

        #   Initial Contact Location y-axis (Connect to the First Level)
        gConnect.append(var_Level2[var_index_Level2["py_init"][0]]-var_Level1[var_index_Level1["py"][-1]])
        gConnect_lb.append(np.array([0]))
        gConnect_ub.append(np.array([0]))

        #   Initial Contact Location y-axis (Connect to the First Level)
        gConnect.append(var_Level2[var_index_Level2["pz_init"][0]]-var_Level1[var_index_Level1["pz"][-1]])
        gConnect_lb.append(np.array([0]))
        gConnect_ub.append(np.array([0]))

        #xdot,ydot,zdot_end according to the choice of second level
        if SecondLevel == "CoM_Dynamics" or SecondLevel == "NLP_SecondLevel" or SecondLevel == "Mixure":
            #   xdot
            gConnect.append(var_Level2[var_index_Level2["xdot"][0]]-var_Level1[var_index_Level1["xdot"][-1]])
            gConnect_lb.append(np.array([0]))
            gConnect_ub.append(np.array([0]))

            #   ydot
            gConnect.append(var_Level2[var_index_Level2["ydot"][0]]-var_Level1[var_index_Level1["ydot"][-1]])
            gConnect_lb.append(np.array([0]))
            gConnect_ub.append(np.array([0]))

            #   zdot
            gConnect.append(var_Level2[var_index_Level2["zdot"][0]]-var_Level1[var_index_Level1["zdot"][-1]])
            gConnect_lb.append(np.array([0]))
            gConnect_ub.append(np.array([0]))
        
        if SecondLevel == "NLP_SecondLevel" or SecondLevel == "MixureNLP":
            #Lx
            gConnect.append(var_Level2[var_index_Level2["Lx"][0]]-var_Level1[var_index_Level1["Lx"][-1]])
            gConnect_lb.append(np.array([0]))
            gConnect_ub.append(np.array([0]))

            #Ly
            gConnect.append(var_Level2[var_index_Level2["Ly"][0]]-var_Level1[var_index_Level1["Ly"][-1]])
            gConnect_lb.append(np.array([0]))
            gConnect_ub.append(np.array([0]))

            #Lz
            gConnect.append(var_Level2[var_index_Level2["Lz"][0]]-var_Level1[var_index_Level1["Lz"][-1]])
            gConnect_lb.append(np.array([0]))
            gConnect_ub.append(np.array([0]))

            #Ldotx
            gConnect.append(var_Level2[var_index_Level2["Ldotx"][0]]-var_Level1[var_index_Level1["Ldotx"][-1]])
            gConnect_lb.append(np.array([0]))
            gConnect_ub.append(np.array([0]))

            #Ldoty
            gConnect.append(var_Level2[var_index_Level2["Ldoty"][0]]-var_Level1[var_index_Level1["Ldoty"][-1]])
            gConnect_lb.append(np.array([0]))
            gConnect_ub.append(np.array([0]))

            #Ldotz
            gConnect.append(var_Level2[var_index_Level2["Ldotz"][0]]-var_Level1[var_index_Level1["Ldotz"][-1]])
            gConnect_lb.append(np.array([0]))
            gConnect_ub.append(np.array([0]))

    #Lamp all Levels
    #   No Second Level
    if SecondLevel == None:
        DecisionVars = var_Level1
        DecisionVars_lb = var_lb_Level1
        DecisionVars_ub = var_ub_Level1
        #need to reshape constraints
        g = ca.vertcat(*g_Level1)
        glb = np.concatenate(glb_Level1)
        gub = np.concatenate(gub_Level1)
        #var_index = {"Level1_Var_Index": var_index_Level1}
    #   With Second Level
    else:
        DecisionVars = ca.vertcat(var_Level1,var_Level2)
        DecisionVars_lb  = np.concatenate((var_lb_Level1,var_lb_Level2),axis=None)
        DecisionVars_ub = np.concatenate((var_ub_Level1,var_ub_Level2),axis=None)
        g = ca.vertcat(*g_Level1,*g_Level2,*gConnect)
        #   Convert shape of glb and gub
        glb_Level1 = np.concatenate((glb_Level1))
        gub_Level1 = np.concatenate((gub_Level1))

        glb_Level2 = np.concatenate((glb_Level2))
        gub_Level2 = np.concatenate((gub_Level2))

        gConnect_lb = np.concatenate((gConnect_lb))
        gConnect_ub = np.concatenate((gConnect_ub))
        #Get all constraint bounds
        glb = np.concatenate((glb_Level1,glb_Level2,gConnect_lb))
        gub = np.concatenate((gub_Level1,gub_Level2,gConnect_ub))
        
    #Collect all Variable Index
    var_index = {"Level1_Var_Index": var_index_Level1,
                 "Level2_Var_Index": var_index_Level2,
    }
    
    #-----------------------------------------------------------------------------------------------------------------------
    #Reshape Constraints
    #g = ca.vertcat(*g)
    #glb = np.concatenate(glb)
    #gub = np.concatenate(gub)

    #-----------------------------------------------------------------------------------------------------------------------
    #   Build Solvers
    #Build Solver
    #opts = {}
    prob = {'x': DecisionVars, 'f': J, 'g': g, 'p': paras}
    #solver = ca.nlpsol('solver', 'knitro', prob,opts)
    solver = ca.nlpsol('solver', 'knitro', prob)

    return solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index

    #print("Not Implemented")    


