#Description:
#   Functions for Building Problem Descriptions for Humanoid with Multi-fiedelity Planning Framework

# Import Important Modules
import numpy as np #Numpy
import casadi as ca #Casadi
import matplotlib.pyplot as plt #Matplotlib
from mpl_toolkits.mplot3d import Axes3D
# Import SL1M modules
from sl1m.constants_and_tools import *
from sl1m.planner import *
from constraints import *

#FUNCTION: Build a single step NLP problem
#Parameters:
#   m: robot mass, default value set as the one of Talos
def NLP_SingleStep(m = 95, StandAlong = True, ConservativeEnd = True, ParameterList = None):
    #-----------------------------------------------------------------------------------------------------------------------
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
    G = 9.80665 #kg/m^2
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

    Lx_init = 0
    Ly_init = 0
    Lz_init = 0

    Ldotx_init = 0
    Ldoty_init = 0
    Ldotz_init = 0

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

    Lx_end = 0
    Ly_end = 0
    Lz_end = 0

    Ldotx_end = 0
    Ldoty_end = 0
    Ldotz_end = 0

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

    #First Round Flag (If yes, we have an initial double support phase, if not, we don't have an initial double support phase)
    ParaFirstRoundFlag = ParameterList["FirstRoundFlag"]

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
    z_lb = np.array([[[0.55]]*(z.shape[0]*z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    z_ub = np.array([[1]*(z.shape[0]*z.shape[1])])
    #z_lb = np.array([[0.55]*(z.shape[0]*z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #z_ub = np.array([[0.65]*(z.shape[0]*z.shape[1])])
    #   CoM Velocity x-axis
    xdot = ca.SX.sym('xdot',N_K)
    xdot_lb = np.array([[-0.5]*(xdot.shape[0]*xdot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    xdot_ub = np.array([[0.5]*(xdot.shape[0]*xdot.shape[1])])
    #   CoM Velocity y-axis
    ydot = ca.SX.sym('ydot',N_K)
    ydot_lb = np.array([[-0.5]*(ydot.shape[0]*ydot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    ydot_ub = np.array([[0.5]*(ydot.shape[0]*ydot.shape[1])])
    #   CoM Velocity z-axis
    zdot = ca.SX.sym('zdot',N_K)
    zdot_lb = np.array([[-0.5]*(zdot.shape[0]*zdot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    zdot_ub = np.array([[0.5]*(zdot.shape[0]*zdot.shape[1])])
    #   Angular Momentum x-axis
    Lx = ca.SX.sym('Lx',N_K)
    Lx_lb = np.array([[-0.5]*(Lx.shape[0]*Lx.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Lx_ub = np.array([[0.5]*(Lx.shape[0]*Lx.shape[1])])
    #   Angular Momentum y-axis
    Ly = ca.SX.sym('Ly',N_K)
    Ly_lb = np.array([[-0.5]*(Ly.shape[0]*Ly.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Ly_ub = np.array([[0.5]*(Ly.shape[0]*Ly.shape[1])])
    #   Angular Momntum y-axis
    Lz = ca.SX.sym('Lz',N_K)
    Lz_lb = np.array([[-0.5]*(Lz.shape[0]*Lz.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Lz_ub = np.array([[0.5]*(Lz.shape[0]*Lz.shape[1])])
    #   Angular Momentum rate x-axis
    Ldotx = ca.SX.sym('Ldotx',N_K)
    Ldotx_lb = np.array([[-1]*(Ldotx.shape[0]*Ldotx.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Ldotx_ub = np.array([[1]*(Ldotx.shape[0]*Ldotx.shape[1])])
    #   Angular Momentum y-axis
    Ldoty = ca.SX.sym('Ldoty',N_K)
    Ldoty_lb = np.array([[-1]*(Ldoty.shape[0]*Ldoty.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Ldoty_ub = np.array([[1]*(Ldoty.shape[0]*Ldoty.shape[1])])
    #   Angular Momntum z-axis
    Ldotz = ca.SX.sym('Ldotz',N_K)
    Ldotz_lb = np.array([[-1]*(Ldotz.shape[0]*Ldotz.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Ldotz_ub = np.array([[1]*(Ldotz.shape[0]*Ldotz.shape[1])])
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
        px_lb.append(np.array([-0.5]))
        px_ub.append(np.array([10]))

        pytemp = ca.SX.sym('py'+str(stepIdx+1))
        py.append(pytemp)
        py_lb.append(np.array([-1]))
        py_ub.append(np.array([1]))

        #   Foot steps are all staying on the ground
        pztemp = ca.SX.sym('pz'+str(stepIdx+1))
        pz.append(pztemp)
        pz_lb.append(np.array([-1]))
        pz_ub.append(np.array([1]))

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

    #   Initial Angular Momentum rate x-axis
    g.append(Ldotx[0]-Ldotx_init)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

    #   Initial Angular Momentum rate y-axis
    g.append(Ldoty[0]-Ldoty_init)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

    #   Initial Angular Momentum rate z-axis
    g.append(Ldotz[0]-Ldotz_init)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

    if StandAlong == True:
        #   Terminal CoM y-axis
        g.append(y[-1]-y_end)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        #   Terminal CoM z-axis
        g.append(z[-1]-z_end)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

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

    #   Terminal Angular Momentum x-axis
    g.append(Lx[-1]-Lx_end)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

    #   Terminal Angular Momentum y-axis
    g.append(Ly[-1]-Ly_end)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

    #   Terminal Angular Momentum z-axis
    g.append(Lz[-1]-Lz_end)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

    #   Terminal Angular Momentum Rate x-axis
    g.append(Ldotx[-1]-Ldotx_end)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

    #   Terminal Angular Momentum Rate y-axis
    g.append(Ldoty[-1]-Ldoty_end)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

    #   Terminal Angular Momentum Rate z-axis
    g.append(Ldotz[-1]-Ldotz_end)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

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
                Ldot_next = ca.vertcat(Ldotz[k+1],Ldotz[k+1],Ldotz[k+1])
            #-------------------------------------------

            #-------------------------------------------
            #Phase dependent Constraints (CoM Kinematics and Angular Dynamics)
            if GaitPattern[Nph]=='InitialDouble':
                #Kinematics Constraint
                #   CoM in the Left foot
                g.append(ca.if_else(ParaFirstRoundFlag,K_CoM_Left@(CoM_k-PL_init)-ca.DM(k_CoM_Left),np.full((len(k_CoM_Left),),-1)))
                glb.append(np.full((len(k_CoM_Left),),-np.inf))
                gub.append(np.full((len(k_CoM_Left),),0))

                #   CoM in the Right foot
                g.append(ca.if_else(ParaFirstRoundFlag,K_CoM_Right@(CoM_k-PR_init)-ca.DM(k_CoM_Right),np.full((len(k_CoM_Right),),-1)))
                glb.append(np.full((len(k_CoM_Right),),-np.inf))
                gub.append(np.full((len(k_CoM_Right),),0))
                #Angular Dynamics
                if k<N_K-1: #double check the knot number is valid
                    g.append(ca.if_else(ParaFirstRoundFlag,Ldot_next-Ldot_current-h*(ca.cross((PL_init+np.array([0.11,0.06,0])-CoM_k),FL1_k)+ca.cross((PL_init+np.array([0.11,-0.06,0])-CoM_k),FL2_k)+ca.cross((PL_init+np.array([-0.11,0.06,0])-CoM_k),FL3_k)+ca.cross((PL_init+np.array([-0.11,-0.06,0])-CoM_k),FL4_k)+ca.cross((PR_init+np.array([0.11,0.06,0])-CoM_k),FR1_k)+ca.cross((PR_init+np.array([0.11,-0.06,0])-CoM_k),FR2_k)+ca.cross((PR_init+np.array([-0.11,0.06,0])-CoM_k),FR3_k)+ca.cross((PR_init+np.array([-0.11,-0.06,0])-CoM_k),FR4_k)),np.array([0,0,0])))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))
                else:
                    print("Initial Double Stage - Angular Dynamics Constraint - Knot number exceeds limit")
            elif GaitPattern[Nph]=='Swing':
                #   Complementarity Condition
                #   If LEFT Foot is SWING (RIGHT is STATONARY), then Zero Forces for the LEFT Foot
                g.append(ca.if_else(ParaLeftSwingFlag,FL1_k,np.array([0,0,0])))
                glb.append(np.array([0,0,0]))
                gub.append(np.array([0,0,0]))

                g.append(ca.if_else(ParaLeftSwingFlag,FL2_k,np.array([0,0,0])))
                glb.append(np.array([0,0,0]))
                gub.append(np.array([0,0,0]))

                g.append(ca.if_else(ParaLeftSwingFlag,FL3_k,np.array([0,0,0])))
                glb.append(np.array([0,0,0]))
                gub.append(np.array([0,0,0]))

                g.append(ca.if_else(ParaLeftSwingFlag,FL4_k,np.array([0,0,0])))
                glb.append(np.array([0,0,0]))
                gub.append(np.array([0,0,0]))

                #   If RIGHT Foot is SWING (LEFT is STATIONARY), then Zero Forces for the RIGHT Foot
                g.append(ca.if_else(ParaRightSwingFlag,FR1_k,np.array([0,0,0])))
                glb.append(np.array([0,0,0]))
                gub.append(np.array([0,0,0]))

                g.append(ca.if_else(ParaRightSwingFlag,FR2_k,np.array([0,0,0])))
                glb.append(np.array([0,0,0]))
                gub.append(np.array([0,0,0]))

                g.append(ca.if_else(ParaRightSwingFlag,FR3_k,np.array([0,0,0])))
                glb.append(np.array([0,0,0]))
                gub.append(np.array([0,0,0]))

                g.append(ca.if_else(ParaRightSwingFlag,FR4_k,np.array([0,0,0])))
                glb.append(np.array([0,0,0]))
                gub.append(np.array([0,0,0]))

                #Kinematics Constraint and Angular Dynamics Constraint

                #IF LEFT Foot is SWING (RIGHT FOOT is STATIONARY)
                #Kinematics Constraint
                #   CoM in the RIGHT Foot
                g.append(ca.if_else(ParaLeftSwingFlag, K_CoM_Right@(CoM_k-PR_init)-ca.DM(k_CoM_Right), np.full((len(k_CoM_Right),),-1)))
                glb.append(np.full((len(k_CoM_Right),),-np.inf))
                gub.append(np.full((len(k_CoM_Right),),0))

                #Angular Dynamics (Right Support)
                if k<N_K-1:
                    g.append(ca.if_else(ParaLeftSwingFlag, Ldot_next-Ldot_current-h*(ca.cross((PR_init+np.array([0.11,0.06,0])-CoM_k),FR1_k)+ca.cross((PR_init+np.array([0.11,-0.06,0])-CoM_k),FR2_k)+ca.cross((PR_init+np.array([-0.11,0.06,0])-CoM_k),FR3_k)+ca.cross((PR_init+np.array([-0.11,-0.06,0])-CoM_k),FR4_k)), np.array([0,0,0])))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))

                #If RIGHT foot is SWING (LEFT is STATIONARY), Then LEFT Foot is the Support FOOT
                #Kinematics Constraint
                #   CoM in the Left foot
                g.append(ca.if_else(ParaRightSwingFlag, K_CoM_Left@(CoM_k-PL_init)-ca.DM(k_CoM_Left), np.full((len(k_CoM_Left),),-1)))
                glb.append(np.full((len(k_CoM_Left),),-np.inf))
                gub.append(np.full((len(k_CoM_Left),),0))

                #Angular Dynamics (Left Support)
                if k<N_K-1:
                    g.append(ca.if_else(ParaRightSwingFlag,Ldot_next-Ldot_current-h*(ca.cross((PL_init+np.array([0.11,0.06,0])-CoM_k),FL1_k)+ca.cross((PL_init+np.array([0.11,-0.06,0])-CoM_k),FL2_k)+ca.cross((PL_init+np.array([-0.11,0.06,0])-CoM_k),FL3_k)+ca.cross((PL_init+np.array([-0.11,-0.06,0])-CoM_k),FL4_k)), np.array([0,0,0])))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))

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
                if k<N_K-1:
                    g.append(ca.if_else(ParaLeftSwingFlag,Ldot_next-Ldot_current-h*(ca.cross((PL_k+np.array([0.11,0.06,0])-CoM_k),FL1_k)+ca.cross((PL_k+np.array([0.11,-0.06,0])-CoM_k),FL2_k)+ca.cross((PL_k+np.array([-0.11,0.06,0])-CoM_k),FL3_k)+ca.cross((PL_k+np.array([-0.11,-0.06,0])-CoM_k),FL4_k)+ca.cross((PR_init+np.array([0.11,0.06,0])-CoM_k),FR1_k)+ca.cross((PR_init+np.array([0.11,-0.06,0])-CoM_k),FR2_k)+ca.cross((PR_init+np.array([-0.11,0.06,0])-CoM_k),FR3_k)+ca.cross((PR_init+np.array([-0.11,-0.06,0])-CoM_k),FR4_k)),np.array([0,0,0])))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))
                
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
                if k<N_K-1:
                    g.append(ca.if_else(ParaRightSwingFlag,Ldot_next-Ldot_current-h*(ca.cross((PL_init+np.array([0.11,0.06,0])-CoM_k),FL1_k)+ca.cross((PL_init+np.array([0.11,-0.06,0])-CoM_k),FL2_k)+ca.cross((PL_init+np.array([-0.11,0.06,0])-CoM_k),FL3_k)+ca.cross((PL_init+np.array([-0.11,-0.06,0])-CoM_k),FL4_k)+ca.cross((PR_k+np.array([0.11,0.06,0])-CoM_k),FR1_k)+ca.cross((PR_k+np.array([0.11,-0.06,0])-CoM_k),FR2_k)+ca.cross((PR_k+np.array([-0.11,0.06,0])-CoM_k),FR3_k)+ca.cross((PR_k+np.array([-0.11,-0.06,0])-CoM_k),FR4_k)),np.array([0,0,0])))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))
            
            #-------------------------------------
            #Unilateral Forces and Friction Cone
            
            #Unilateral Forces
            #Left Foot 1
            g.append(FL1_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #Left Foot 2
            g.append(FL2_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #Left Foot 3
            g.append(FL3_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #Left Foot 4
            g.append(FL4_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])

            #Right Foot 1
            g.append(FR1_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #Right Foot 2
            g.append(FR2_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #Right Foot 3
            g.append(FR3_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #Right Foot 4
            g.append(FR4_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])

            #Friction Cone
            #   Left Foot 1 x-axis Set 1
            g.append(FL1_k.T@TerrainTangentX - miu*FL1_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Left Foot 1 x-axis Set 2
            g.append(FL1_k.T@TerrainTangentX + miu*FL1_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #   Left Foot 1 y-axis Set 1
            g.append(FL1_k.T@TerrainTangentY - miu*FL1_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Left Foot 1 Y-axis Set 2
            g.append(FL1_k.T@TerrainTangentY + miu*FL1_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])

            #   Left Foot 2 x-axis Set 1
            g.append(FL2_k.T@TerrainTangentX - miu*FL2_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Left Foot 2 x-axis Set 2
            g.append(FL2_k.T@TerrainTangentX + miu*FL2_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #   Left Foot 2 y-axis Set 1
            g.append(FL2_k.T@TerrainTangentY - miu*FL2_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Left Foot 2 Y-axis Set 2
            g.append(FL2_k.T@TerrainTangentY + miu*FL2_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])

            #   Left Foot 3 x-axis Set 1
            g.append(FL3_k.T@TerrainTangentX - miu*FL3_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Left Foot 3 x-axis Set 2
            g.append(FL3_k.T@TerrainTangentX + miu*FL3_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #   Left Foot 3 y-axis Set 1
            g.append(FL3_k.T@TerrainTangentY - miu*FL3_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Left Foot 3 Y-axis Set 2
            g.append(FL3_k.T@TerrainTangentY + miu*FL3_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])

            #   Left Foot 4 x-axis Set 1
            g.append(FL4_k.T@TerrainTangentX - miu*FL4_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Left Foot 4 x-axis Set 2
            g.append(FL4_k.T@TerrainTangentX + miu*FL4_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #   Left Foot 4 y-axis Set 1
            g.append(FL4_k.T@TerrainTangentY - miu*FL4_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Left Foot 4 Y-axis Set 2
            g.append(FL4_k.T@TerrainTangentY + miu*FL4_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])

            #   Right Foot 1 x-axis Set 1
            g.append(FR1_k.T@TerrainTangentX - miu*FR1_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Right Foot 1 x-axis Set 2
            g.append(FR1_k.T@TerrainTangentX + miu*FR1_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #   Right Foot 1 Y-axis Set 1
            g.append(FR1_k.T@TerrainTangentY - miu*FR1_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Right Foot 1 Y-axis Set 2
            g.append(FR1_k.T@TerrainTangentY + miu*FR1_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])

            #   Right Foot 2 x-axis Set 1
            g.append(FR2_k.T@TerrainTangentX - miu*FR2_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Right Foot 2 x-axis Set 2
            g.append(FR2_k.T@TerrainTangentX + miu*FR2_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #   Right Foot 2 Y-axis Set 1
            g.append(FR2_k.T@TerrainTangentY - miu*FR2_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Right Foot 2 Y-axis Set 2
            g.append(FR2_k.T@TerrainTangentY + miu*FR2_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])

            #   Right Foot 3 x-axis Set 1
            g.append(FR3_k.T@TerrainTangentX - miu*FR3_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Right Foot 3 x-axis Set 2
            g.append(FR3_k.T@TerrainTangentX + miu*FR3_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #   Right Foot 3 Y-axis Set 1
            g.append(FR3_k.T@TerrainTangentY - miu*FR3_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Right Foot 3 Y-axis Set 2
            g.append(FR3_k.T@TerrainTangentY + miu*FR3_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])

            #   Right Foot 4 x-axis Set 1
            g.append(FR4_k.T@TerrainTangentX - miu*FR4_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Right Foot 4 x-axis Set 2
            g.append(FR4_k.T@TerrainTangentX + miu*FR4_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #   Right Foot 4 Y-axis Set 1
            g.append(FR4_k.T@TerrainTangentY - miu*FR4_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Right Foot 4 Y-axis Set 2
            g.append(FR4_k.T@TerrainTangentY + miu*FR4_k.T@TerrainNorm)
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
                J = J + h*Lx[k]**2 + h*Ly[k]**2 + h*Lz[k]**2 + h*(FL1x[k]/m+FL2x[k]/m+FL3x[k]/m+FL4x[k]/m+FR1x[k]/m+FR2x[k]/m+FR3x[k]/m+FR4x[k]/m)**2 + h*(FL1y[k]/m+FL2y[k]/m+FL3y[k]/m+FL4y[k]/m+FR1y[k]/m+FR2y[k]/m+FR3y[k]/m+FR4y[k]/m)**2 + h*(FL1z[k]/m+FL2z[k]/m+FL3z[k]/m+FL4z[k]/m+FR1z[k]/m+FR2z[k]/m+FR3z[k]/m+FR4z[k]/m - G)**2
                #No Angular momentum
                #J = J + h*(FL1x[k]/m+FL2x[k]/m+FL3x[k]/m+FL4x[k]/m+FR1x[k]/m+FR2x[k]/m+FR3x[k]/m+FR4x[k]/m)**2 + h*(FL1y[k]/m+FL2y[k]/m+FL3y[k]/m+FL4y[k]/m+FR1y[k]/m+FR2y[k]/m+FR3y[k]/m+FR4y[k]/m)**2 + h*(FL1z[k]/m+FL2z[k]/m+FL3z[k]/m+FL4z[k]/m+FR1z[k]/m+FR2z[k]/m+FR3z[k]/m+FR4z[k]/m - G)**2

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

    #If RIGHT foot is SWING (LEFT is STATIONARY), Then RIGHT Foot should stay in the polytope of the LEFT Foot
    g.append(ca.if_else(ParaRightSwingFlag,Q_rf_in_lf@(p_next-PL_init)-ca.DM(q_rf_in_lf),np.full((len(q_rf_in_lf),),-1)))
    glb.append(np.full((len(q_rf_in_lf),),-np.inf))
    gub.append(np.full((len(q_rf_in_lf),),0))

    #FootStep Constraint
    #Inequality
    g.append(FirstSurfK@p_next - FirstSurfk)
    glb.append(np.full((4,),-np.inf))
    gub.append(np.full((4,),0))
    #print(FirstSurfK@p_next - FirstSurfk)

    #Equality
    g.append(FirstSurfE.T@p_next - FirstSurfe)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

    #Switching Time Constraint
    #NOTE: For Unconservative First Level, The range of double support is 0.1 to 0.3, The range of swing is 0.3 to 0.9
    #NOTE: For conservative First Level, The range of double support is 0.1 to 0.5, The range of swing is 0.3 to 0.9
    for phase_cnt in range(Nphase):
        if GaitPattern[phase_cnt] == 'InitialDouble':
            g.append(Ts[phase_cnt])
            glb.append(np.array([0.1]))
            gub.append(np.array([0.3]))
        elif GaitPattern[phase_cnt] == 'Swing':
            g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
            glb.append(np.array([0.4])) #0.3 is better
            gub.append(np.array([0.9])) #0.4 - 0.9
        elif GaitPattern[phase_cnt] == 'DoubleSupport':
            g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
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
def NLP_SecondLevel(m = 95, Nk_Local = 5, Nsteps = 1, ParameterList = None, StaticStop = False, NumPatches = None):
    #-----------------------------------------------------------------------------------------------------------------------
    #Define Parameters
    #   Gait Pattern, Each action is followed up by a double support phase
    GaitPattern = ["Swing","DoubleSupport"]*Nsteps #,'RightSupport','DoubleSupport','LeftSupport','DoubleSupport'
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

    Lx_init = 0
    Ly_init = 0
    Lz_init = 0

    Ldotx_init = 0
    Ldoty_init = 0
    Ldotz_init = 0

    PLx_init = ParameterList["PLx_init"]
    PLy_init = ParameterList["PLy_init"]
    PLz_init = ParameterList["PLz_init"]
    #PL_init = ca.vertcat(PLx_init,PLy_init,PLz_init)

    PRx_init = ParameterList["PRx_init"]
    PRy_init = ParameterList["PRy_init"]
    PRz_init = ParameterList["PRz_init"]
    #PR_init = ca.vertcat(PRx_init,PRy_init,PRz_init)

    x_end = ParameterList["x_end"]
    y_end = ParameterList["y_end"]
    z_end = ParameterList["z_end"]

    xdot_end = ParameterList["xdot_end"]
    ydot_end = ParameterList["ydot_end"]
    zdot_end = ParameterList["zdot_end"]

    Lx_end = 0
    Ly_end = 0
    Lz_end = 0

    Ldotx_end = 0
    Ldoty_end = 0
    Ldotz_end = 0

    #Flags for Swing Legs (Defined as Parameters)
    ParaLeftSwingFlag = ParameterList["LeftSwingFlag"]
    ParaRightSwingFlag = ParameterList["RightSwingFlag"]

    #Surfaces (Only the Second One)
    #Surface Patches
    SurfParas = ParameterList["SurfParas"]

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
    xdot = ca.SX.sym('xdot',N_K)
    xdot_lb = np.array([[-0.5]*(xdot.shape[0]*xdot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    xdot_ub = np.array([[0.5]*(xdot.shape[0]*xdot.shape[1])])
    #   CoM Velocity y-axis
    ydot = ca.SX.sym('ydot',N_K)
    ydot_lb = np.array([[-0.5]*(ydot.shape[0]*ydot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    ydot_ub = np.array([[0.5]*(ydot.shape[0]*ydot.shape[1])])
    #   CoM Velocity z-axis
    zdot = ca.SX.sym('zdot',N_K)
    zdot_lb = np.array([[-0.5]*(zdot.shape[0]*zdot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    zdot_ub = np.array([[0.5]*(zdot.shape[0]*zdot.shape[1])])
    #   Angular Momentum x-axis
    Lx = ca.SX.sym('Lx',N_K)
    Lx_lb = np.array([[-0.5]*(Lx.shape[0]*Lx.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Lx_ub = np.array([[0.5]*(Lx.shape[0]*Lx.shape[1])])
    #   Angular Momentum y-axis
    Ly = ca.SX.sym('Ly',N_K)
    Ly_lb = np.array([[-0.5]*(Ly.shape[0]*Ly.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Ly_ub = np.array([[0.5]*(Ly.shape[0]*Ly.shape[1])])
    #   Angular Momntum y-axis
    Lz = ca.SX.sym('Lz',N_K)
    Lz_lb = np.array([[-0.5]*(Lz.shape[0]*Lz.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Lz_ub = np.array([[0.5]*(Lz.shape[0]*Lz.shape[1])])
    #   Angular Momentum rate x-axis
    Ldotx = ca.SX.sym('Ldotx',N_K)
    Ldotx_lb = np.array([[-1]*(Ldotx.shape[0]*Ldotx.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Ldotx_ub = np.array([[1]*(Ldotx.shape[0]*Ldotx.shape[1])])
    #   Angular Momentum y-axis
    Ldoty = ca.SX.sym('Ldoty',N_K)
    Ldoty_lb = np.array([[-1]*(Ldoty.shape[0]*Ldoty.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Ldoty_ub = np.array([[1]*(Ldoty.shape[0]*Ldoty.shape[1])])
    #   Angular Momntum z-axis
    Ldotz = ca.SX.sym('Ldotz',N_K)
    Ldotz_lb = np.array([[-1]*(Ldotz.shape[0]*Ldotz.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Ldotz_ub = np.array([[1]*(Ldotz.shape[0]*Ldotz.shape[1])])
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
        pz_lb.append(np.array([-1]))
        pz_ub.append(np.array([1]))

    #Switching Time Vector
    Ts = []
    Ts_lb = []
    Ts_ub = []
    for n_phase in range(Nphase):
        Tstemp = ca.SX.sym('Ts'+str(n_phase+1)) #0 + 1 + ....
        Ts.append(Tstemp)
        Ts_lb.append(np.array([0.05]))
        Ts_ub.append(np.array([1.5*(Nphase+1)]))

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

    #   Terminal CoM y-axis
    g.append(y[-1]-y_end)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

    #   Terminal CoM z-axis
    g.append(z[-1]-z_end)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

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
    g.append(Lx[-1]-Lx_end)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

    #   Terminal Angular Momentum y-axis
    g.append(Ly[-1]-Ly_end)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

    #   Terminal Angular Momentum z-axis
    g.append(Lz[-1]-Lz_end)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

    #   Terminal Angular Momentum Rate x-axis
    g.append(Ldotx[-1]-Ldotx_end)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

    #   Terminal Angular Momentum Rate y-axis
    g.append(Ldoty[-1]-Ldoty_end)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

    #   Terminal Angular Momentum Rate z-axis
    g.append(Ldotz[-1]-Ldotz_end)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

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
                Ldot_next = ca.vertcat(Ldotz[k+1],Ldotz[k+1],Ldotz[k+1])
            #-------------------------------------------

            #-------------------------------------------
            #Phase dependent Constraints (CoM Kinematics and Angular Dynamics)
            if GaitPattern[Nph]== 'Swing':

                if Nph//2 == 0:
                    #!!!!!!Pass from the first Level!!!!!!
                    P_k_current = ca.vertcat(px_init,py_init,pz_init) #ca.vertcat(px[-1],py[-1],pz[-1])
                    #!!!!!!
                    P_k_next = ca.vertcat(px[Nph//2],py[Nph//2],pz[Nph//2])
                else:
                    P_k_current = ca.vertcat(px[Nph//2-1],py[Nph//2-1],pz[Nph//2-1])
                    P_k_next = ca.vertcat(px[Nph//2],py[Nph//2],pz[Nph//2])

                if (Nph//2)%2 == 0: #even number steps

                    #------------------------------------
                    #Zero Forces

                    #If First Level Swing the Left, then the second level Even Number Phases (the first Phase) Swing the Right -> Left -> Right
                    #RIGHT FOOT has zero forces
                    g.append(ca.if_else(ParaLeftSwingFlag,FR1_k,np.array([0,0,0])))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))

                    g.append(ca.if_else(ParaLeftSwingFlag,FR2_k,np.array([0,0,0])))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))

                    g.append(ca.if_else(ParaLeftSwingFlag,FR3_k,np.array([0,0,0])))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))

                    g.append(ca.if_else(ParaLeftSwingFlag,FR4_k,np.array([0,0,0])))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))

                    #If First Levvel Swing the Right, then the second level Even Number Phases (the first phase) Swing the Left -> Right -> Left
                    #LEFT FOOT has zero Forces

                    g.append(ca.if_else(ParaRightSwingFlag,FL1_k,np.array([0,0,0])))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))

                    g.append(ca.if_else(ParaRightSwingFlag,FL2_k,np.array([0,0,0])))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))

                    g.append(ca.if_else(ParaRightSwingFlag,FL3_k,np.array([0,0,0])))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))

                    g.append(ca.if_else(ParaRightSwingFlag,FL4_k,np.array([0,0,0])))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))

                    #Kinematics Constraint and Angular Dynamics Constraint
                    #If First Level Swing the Left, then the second level Even Number Phases (the first Phase) Swing the Right -> Left -> Right
                    #Left foot in contact for p_current, right foot is going to land as p_next
                    #   CoM in the Left Foot
                    g.append(ca.if_else(ParaLeftSwingFlag, K_CoM_Left@(CoM_k-P_k_current)-ca.DM(k_CoM_Left), np.full((len(k_CoM_Left),),-1)))
                    glb.append(np.full((len(k_CoM_Left),),-np.inf))
                    gub.append(np.full((len(k_CoM_Left),),0))

                    #Angular Dynamics (Left Support)
                    if k<N_K-1:
                        g.append(ca.if_else(ParaLeftSwingFlag, Ldot_next-Ldot_current-h*(ca.cross((P_k_current+np.array([0.11,0.06,0])-CoM_k),FL1_k)+ca.cross((P_k_current+np.array([0.11,-0.06,0])-CoM_k),FL2_k)+ca.cross((P_k_current+np.array([-0.11,0.06,0])-CoM_k),FL3_k)+ca.cross((P_k_current+np.array([-0.11,-0.06,0])-CoM_k),FL4_k)), np.array([0,0,0])))
                        glb.append(np.array([0,0,0]))
                        gub.append(np.array([0,0,0]))

                    #If First Levvel Swing the Right, then the second level Even Number Phases (the first phase) Swing the Left -> Right -> Left
                    #Right foot in contact for p_current, left foot is going to land at p_next
                    #   CoM in the Right foot
                    g.append(ca.if_else(ParaRightSwingFlag, K_CoM_Right@(CoM_k-P_k_current)-ca.DM(k_CoM_Right), np.full((len(k_CoM_Right),),-1)))
                    glb.append(np.full((len(k_CoM_Right),),-np.inf))
                    gub.append(np.full((len(k_CoM_Right),),0))

                    #Angular Dynamics (Right Support)
                    if k<N_K-1:
                        g.append(ca.if_else(ParaRightSwingFlag,Ldot_next-Ldot_current-h*(ca.cross((P_k_current+np.array([0.11,0.06,0])-CoM_k),FR1_k)+ca.cross((P_k_current+np.array([0.11,-0.06,0])-CoM_k),FR2_k)+ca.cross((P_k_current+np.array([-0.11,0.06,0])-CoM_k),FR3_k)+ca.cross((P_k_current+np.array([-0.11,-0.06,0])-CoM_k),FR4_k)), np.array([0,0,0])))
                        glb.append(np.array([0,0,0]))
                        gub.append(np.array([0,0,0]))

                elif (Nph//2)%2 == 1: #odd number steps
                    #------------------------------------
                    #Zero Forces

                    #If the first level swings the Left, then the second level for Odd Number Phases (the second phase) swings left -> Right -> Left
                    #LEFT FOOT has zero forces
                    g.append(ca.if_else(ParaLeftSwingFlag,FL1_k,np.array([0,0,0])))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))

                    g.append(ca.if_else(ParaLeftSwingFlag,FL2_k,np.array([0,0,0])))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))

                    g.append(ca.if_else(ParaLeftSwingFlag,FL3_k,np.array([0,0,0])))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))

                    g.append(ca.if_else(ParaLeftSwingFlag,FL4_k,np.array([0,0,0])))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))

                    #If the first level swings the Right, then the second level for Odd Number Phases (the second phase) swings Right -> Left -> Right
                    #RIGHT FOOT has zero forces
                    g.append(ca.if_else(ParaRightSwingFlag,FR1_k,np.array([0,0,0])))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))

                    g.append(ca.if_else(ParaRightSwingFlag,FR2_k,np.array([0,0,0])))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))

                    g.append(ca.if_else(ParaRightSwingFlag,FR3_k,np.array([0,0,0])))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))

                    g.append(ca.if_else(ParaRightSwingFlag,FR4_k,np.array([0,0,0])))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))

                    #------------------------------------
                    #CoM Kinematics
                    #If the first level swings the Left, then the second level for Odd Number Phases (the second phase) swings left -> Right -> Left
                    #Right foot in contact for p_current, left foot is going to land at p_next
                    #   CoM in the Right
                    g.append(ca.if_else(ParaLeftSwingFlag, K_CoM_Right@(CoM_k-P_k_current)-ca.DM(k_CoM_Right), np.full((len(k_CoM_Right),),-1)))
                    glb.append(np.full((len(k_CoM_Right),),-np.inf))
                    gub.append(np.full((len(k_CoM_Right),),0))
                    
                    #Angular Dynamics (Right Support)
                    if k<N_K-1:
                        g.append(ca.if_else(ParaLeftSwingFlag,Ldot_next-Ldot_current-h*(ca.cross((P_k_current+np.array([0.11,0.06,0])-CoM_k),FR1_k)+ca.cross((P_k_current+np.array([0.11,-0.06,0])-CoM_k),FR2_k)+ca.cross((P_k_current+np.array([-0.11,0.06,0])-CoM_k),FR3_k)+ca.cross((P_k_current+np.array([-0.11,-0.06,0])-CoM_k),FR4_k)), np.array([0,0,0])))
                        glb.append(np.array([0,0,0]))
                        gub.append(np.array([0,0,0]))

                    #If the first level swings the Right, then the second level for Odd Number Phases (the second phase) swings Right -> Left -> Right
                    #Left foot in contact for p_current, right foot is going to land as p_next
                    #   CoM in the Left
                    g.append(ca.if_else(ParaRightSwingFlag, K_CoM_Left@(CoM_k-P_k_current)-ca.DM(k_CoM_Left), np.full((len(k_CoM_Left),),-1)))
                    glb.append(np.full((len(k_CoM_Left),),-np.inf))
                    gub.append(np.full((len(k_CoM_Left),),0))
                    
                    #Angular Dynamics (Left Support)
                    if k<N_K-1:
                        g.append(ca.if_else(ParaRightSwingFlag, Ldot_next-Ldot_current-h*(ca.cross((P_k_current+np.array([0.11,0.06,0])-CoM_k),FL1_k)+ca.cross((P_k_current+np.array([0.11,-0.06,0])-CoM_k),FL2_k)+ca.cross((P_k_current+np.array([-0.11,0.06,0])-CoM_k),FL3_k)+ca.cross((P_k_current+np.array([-0.11,-0.06,0])-CoM_k),FL4_k)), np.array([0,0,0])))
                        glb.append(np.array([0,0,0]))
                        gub.append(np.array([0,0,0]))

            elif GaitPattern[Nph]=='DoubleSupport':

                #CoM Kinematic Constraint
                if (Nph-1)//2 == 0:
                    #!!!!!!Pass from the first Level!!!!!!
                    P_k_current = ca.vertcat(px_init,py_init,pz_init) #ca.vertcat(px[-1],py[-1],pz[-1])
                    #!!!!!!
                    P_k_next = ca.vertcat(px[(Nph-1)//2],py[(Nph-1)//2],pz[(Nph-1)//2])
                else:
                    P_k_current = ca.vertcat(px[(Nph-1)//2-1],py[(Nph-1)//2-1],pz[(Nph-1)//2-1])
                    P_k_next = ca.vertcat(px[(Nph-1)//2],py[(Nph-1)//2],pz[(Nph-1)//2])

                if ((Nph-1)//2)%2 == 0: #even number steps
                    #If First Level Swing the Left, then the second level Even Number Phases (the first Phase) Swing the Right -> Left -> Right
                    #Left foot in contact for p_current, right foot is going to land as p_next
                    g.append(ca.if_else(ParaLeftSwingFlag, K_CoM_Left@(CoM_k-P_k_current)-ca.DM(k_CoM_Left), np.full((len(k_CoM_Left),),-1)))
                    glb.append(np.full((len(k_CoM_Left),),-np.inf))
                    gub.append(np.full((len(k_CoM_Left),),0))

                    g.append(ca.if_else(ParaLeftSwingFlag, K_CoM_Right@(CoM_k-P_k_next)-ca.DM(k_CoM_Right), np.full((len(k_CoM_Right),),-1)))
                    glb.append(np.full((len(k_CoM_Right),),-np.inf))
                    gub.append(np.full((len(k_CoM_Right),),0))

                    #Angular Dynamics, P_current as Left foot, P_next as Right
                    if k<N_K-1:
                        g.append(ca.if_else(ParaLeftSwingFlag,Ldot_next-Ldot_current-h*(ca.cross((P_k_current+np.array([0.11,0.06,0])-CoM_k),FL1_k)+ca.cross((P_k_current+np.array([0.11,-0.06,0])-CoM_k),FL2_k)+ca.cross((P_k_current+np.array([-0.11,0.06,0])-CoM_k),FL3_k)+ca.cross((P_k_current+np.array([-0.11,-0.06,0])-CoM_k),FL4_k)+ca.cross((P_k_next+np.array([0.11,0.06,0])-CoM_k),FR1_k)+ca.cross((P_k_next+np.array([0.11,-0.06,0])-CoM_k),FR2_k)+ca.cross((P_k_next+np.array([-0.11,0.06,0])-CoM_k),FR3_k)+ca.cross((P_k_next+np.array([-0.11,-0.06,0])-CoM_k),FR4_k)),np.array([0,0,0])))
                        glb.append(np.array([0,0,0]))
                        gub.append(np.array([0,0,0]))

                    #If First Levvel Swing the Right, then the second level Even Number Phases (the first phase) Swing the Left -> Right -> Left
                    #Right foot in contact for p_current, left foot is going to land at p_next
                    g.append(ca.if_else(ParaRightSwingFlag, K_CoM_Right@(CoM_k-P_k_current)-ca.DM(k_CoM_Right), np.full((len(k_CoM_Right),),-1)))
                    glb.append(np.full((len(k_CoM_Right),),-np.inf))
                    gub.append(np.full((len(k_CoM_Right),),0))

                    g.append(ca.if_else(ParaRightSwingFlag, K_CoM_Left@(CoM_k-P_k_next)-ca.DM(k_CoM_Left), np.full((len(k_CoM_Left),),-1)))
                    glb.append(np.full((len(k_CoM_Left),),-np.inf))
                    gub.append(np.full((len(k_CoM_Left),),0))

                    #Angular Dynamics (Double Support) P_current as Right foot, P_next as Left NOTE:The Flippped FR and FL
                    if k<N_K-1:
                        g.append(ca.if_else(ParaRightSwingFlag,Ldot_next-Ldot_current-h*(ca.cross((P_k_current+np.array([0.11,0.06,0])-CoM_k),FR1_k)+ca.cross((P_k_current+np.array([0.11,-0.06,0])-CoM_k),FR2_k)+ca.cross((P_k_current+np.array([-0.11,0.06,0])-CoM_k),FR3_k)+ca.cross((P_k_current+np.array([-0.11,-0.06,0])-CoM_k),FR4_k)+ca.cross((P_k_next+np.array([0.11,0.06,0])-CoM_k),FL1_k)+ca.cross((P_k_next+np.array([0.11,-0.06,0])-CoM_k),FL2_k)+ca.cross((P_k_next+np.array([-0.11,0.06,0])-CoM_k),FL3_k)+ca.cross((P_k_next+np.array([-0.11,-0.06,0])-CoM_k),FL4_k)),np.array([0,0,0])))
                        glb.append(np.array([0,0,0]))
                        gub.append(np.array([0,0,0]))

                elif ((Nph-1)//2)%2 == 1: #odd number steps
                    #If the first level swings the Left, then the second level for Odd Number Phases (the second phase) swings left -> Right -> Left
                    #Right foot in contact for p_current, left foot is going to land at p_next
                    g.append(ca.if_else(ParaLeftSwingFlag, K_CoM_Right@(CoM_k-P_k_current)-ca.DM(k_CoM_Right), np.full((len(k_CoM_Right),),-1)))
                    glb.append(np.full((len(k_CoM_Right),),-np.inf))
                    gub.append(np.full((len(k_CoM_Right),),0))

                    g.append(ca.if_else(ParaLeftSwingFlag, K_CoM_Left@(CoM_k-P_k_next)-ca.DM(k_CoM_Left), np.full((len(k_CoM_Left),),-1)))
                    glb.append(np.full((len(k_CoM_Left),),-np.inf))
                    gub.append(np.full((len(k_CoM_Left),),0))

                    #Angular Dynamics, P_current as Right, P_next as Left
                    if k<N_K-1:
                        g.append(ca.if_else(ParaLeftSwingFlag,Ldot_next-Ldot_current-h*(ca.cross((P_k_current+np.array([0.11,0.06,0])-CoM_k),FR1_k)+ca.cross((P_k_current+np.array([0.11,-0.06,0])-CoM_k),FR2_k)+ca.cross((P_k_current+np.array([-0.11,0.06,0])-CoM_k),FR3_k)+ca.cross((P_k_current+np.array([-0.11,-0.06,0])-CoM_k),FR4_k)+ca.cross((P_k_next+np.array([0.11,0.06,0])-CoM_k),FL1_k)+ca.cross((P_k_next+np.array([0.11,-0.06,0])-CoM_k),FL2_k)+ca.cross((P_k_next+np.array([-0.11,0.06,0])-CoM_k),FL3_k)+ca.cross((P_k_next+np.array([-0.11,-0.06,0])-CoM_k),FL4_k)),np.array([0,0,0])))
                        glb.append(np.array([0,0,0]))
                        gub.append(np.array([0,0,0]))

                    #If the first level swings the Right, then the second level for Odd Number Phases (the second phase) swings Right -> Left -> Right
                    #Left foot in contact for p_current, right foot is going to land as p_next
                    g.append(ca.if_else(ParaRightSwingFlag, K_CoM_Left@(CoM_k-P_k_current)-ca.DM(k_CoM_Left), np.full((len(k_CoM_Left),),-1)))
                    glb.append(np.full((len(k_CoM_Left),),-np.inf))
                    gub.append(np.full((len(k_CoM_Left),),0))

                    g.append(ca.if_else(ParaRightSwingFlag, K_CoM_Right@(CoM_k-P_k_current)-ca.DM(k_CoM_Right), np.full((len(k_CoM_Right),),-1)))
                    glb.append(np.full((len(k_CoM_Right),),-np.inf))
                    gub.append(np.full((len(k_CoM_Right),),0))

                    #Angular Dynamics, P_current as Left, P_next as Right
                    if k<N_K-1:
                        g.append(ca.if_else(ParaRightSwingFlag,Ldot_next-Ldot_current-h*(ca.cross((P_k_current+np.array([0.11,0.06,0])-CoM_k),FL1_k)+ca.cross((P_k_current+np.array([0.11,-0.06,0])-CoM_k),FL2_k)+ca.cross((P_k_current+np.array([-0.11,0.06,0])-CoM_k),FL3_k)+ca.cross((P_k_current+np.array([-0.11,-0.06,0])-CoM_k),FL4_k)+ca.cross((P_k_next+np.array([0.11,0.06,0])-CoM_k),FR1_k)+ca.cross((P_k_next+np.array([0.11,-0.06,0])-CoM_k),FR2_k)+ca.cross((P_k_next+np.array([-0.11,0.06,0])-CoM_k),FR3_k)+ca.cross((P_k_next+np.array([-0.11,-0.06,0])-CoM_k),FR4_k)),np.array([0,0,0])))
                        glb.append(np.array([0,0,0]))
                        gub.append(np.array([0,0,0]))
            
            #-------------------------------------
            #Unilateral Forces and Friction Cone
            
            #Unilateral Forces
            #Left Foot 1
            g.append(FL1_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #Left Foot 2
            g.append(FL2_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #Left Foot 3
            g.append(FL3_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #Left Foot 4
            g.append(FL4_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])

            #Right Foot 1
            g.append(FR1_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #Right Foot 2
            g.append(FR2_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #Right Foot 3
            g.append(FR3_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #Right Foot 4
            g.append(FR4_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])

            #Friction Cone
            #   Left Foot 1 x-axis Set 1
            g.append(FL1_k.T@TerrainTangentX - miu*FL1_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Left Foot 1 x-axis Set 2
            g.append(FL1_k.T@TerrainTangentX + miu*FL1_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #   Left Foot 1 y-axis Set 1
            g.append(FL1_k.T@TerrainTangentY - miu*FL1_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Left Foot 1 Y-axis Set 2
            g.append(FL1_k.T@TerrainTangentY + miu*FL1_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])

            #   Left Foot 2 x-axis Set 1
            g.append(FL2_k.T@TerrainTangentX - miu*FL2_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Left Foot 2 x-axis Set 2
            g.append(FL2_k.T@TerrainTangentX + miu*FL2_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #   Left Foot 2 y-axis Set 1
            g.append(FL2_k.T@TerrainTangentY - miu*FL2_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Left Foot 2 Y-axis Set 2
            g.append(FL2_k.T@TerrainTangentY + miu*FL2_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])

            #   Left Foot 3 x-axis Set 1
            g.append(FL3_k.T@TerrainTangentX - miu*FL3_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Left Foot 3 x-axis Set 2
            g.append(FL3_k.T@TerrainTangentX + miu*FL3_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #   Left Foot 3 y-axis Set 1
            g.append(FL3_k.T@TerrainTangentY - miu*FL3_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Left Foot 3 Y-axis Set 2
            g.append(FL3_k.T@TerrainTangentY + miu*FL3_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])

            #   Left Foot 4 x-axis Set 1
            g.append(FL4_k.T@TerrainTangentX - miu*FL4_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Left Foot 4 x-axis Set 2
            g.append(FL4_k.T@TerrainTangentX + miu*FL4_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #   Left Foot 4 y-axis Set 1
            g.append(FL4_k.T@TerrainTangentY - miu*FL4_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Left Foot 4 Y-axis Set 2
            g.append(FL4_k.T@TerrainTangentY + miu*FL4_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])

            #   Right Foot 1 x-axis Set 1
            g.append(FR1_k.T@TerrainTangentX - miu*FR1_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Right Foot 1 x-axis Set 2
            g.append(FR1_k.T@TerrainTangentX + miu*FR1_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #   Right Foot 1 Y-axis Set 1
            g.append(FR1_k.T@TerrainTangentY - miu*FR1_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Right Foot 1 Y-axis Set 2
            g.append(FR1_k.T@TerrainTangentY + miu*FR1_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])

            #   Right Foot 2 x-axis Set 1
            g.append(FR2_k.T@TerrainTangentX - miu*FR2_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Right Foot 2 x-axis Set 2
            g.append(FR2_k.T@TerrainTangentX + miu*FR2_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #   Right Foot 2 Y-axis Set 1
            g.append(FR2_k.T@TerrainTangentY - miu*FR2_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Right Foot 2 Y-axis Set 2
            g.append(FR2_k.T@TerrainTangentY + miu*FR2_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])

            #   Right Foot 3 x-axis Set 1
            g.append(FR3_k.T@TerrainTangentX - miu*FR3_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Right Foot 3 x-axis Set 2
            g.append(FR3_k.T@TerrainTangentX + miu*FR3_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #   Right Foot 3 Y-axis Set 1
            g.append(FR3_k.T@TerrainTangentY - miu*FR3_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Right Foot 3 Y-axis Set 2
            g.append(FR3_k.T@TerrainTangentY + miu*FR3_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])

            #   Right Foot 4 x-axis Set 1
            g.append(FR4_k.T@TerrainTangentX - miu*FR4_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Right Foot 4 x-axis Set 2
            g.append(FR4_k.T@TerrainTangentX + miu*FR4_k.T@TerrainNorm)
            glb.append(np.array([0]))
            gub.append([np.inf])
            #   Right Foot 4 Y-axis Set 1
            g.append(FR4_k.T@TerrainTangentY - miu*FR4_k.T@TerrainNorm)
            glb.append([-np.inf])
            gub.append(np.array([0]))
            #   Right Foot 4 Y-axis Set 2
            g.append(FR4_k.T@TerrainTangentY + miu*FR4_k.T@TerrainNorm)
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
                J = J + h*Lx[k]**2 + h*Ly[k]**2 + h*Lz[k]**2 + h*(FL1x[k]/m+FL2x[k]/m+FL3x[k]/m+FL4x[k]/m+FR1x[k]/m+FR2x[k]/m+FR3x[k]/m+FR4x[k]/m)**2 + h*(FL1y[k]/m+FL2y[k]/m+FL3y[k]/m+FL4y[k]/m+FR1y[k]/m+FR2y[k]/m+FR3y[k]/m+FR4y[k]/m)**2 + h*(FL1z[k]/m+FL2z[k]/m+FL3z[k]/m+FL4z[k]/m+FR1z[k]/m+FR2z[k]/m+FR3z[k]/m+FR4z[k]/m - G)**2
                #No Angular momentum
                #J = J + h*(FL1x[k]/m+FL2x[k]/m+FL3x[k]/m+FL4x[k]/m+FR1x[k]/m+FR2x[k]/m+FR3x[k]/m+FR4x[k]/m)**2 + h*(FL1y[k]/m+FL2y[k]/m+FL3y[k]/m+FL4y[k]/m+FR1y[k]/m+FR2y[k]/m+FR3y[k]/m+FR4y[k]/m)**2 + h*(FL1z[k]/m+FL2z[k]/m+FL3z[k]/m+FL4z[k]/m+FR1z[k]/m+FR2z[k]/m+FR3z[k]/m+FR4z[k]/m - G)**2

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
    for PatchNum in range(Nsteps):
        #Get Footstep Vector
        P_vector = ca.vertcat(px[PatchNum],py[PatchNum],pz[PatchNum])

        #Get Half Space Representation
        SurfParaTemp = SurfParas[20+PatchNum*20:19+(PatchNum+1)*20+1]
        SurfK = SurfParaTemp[0:11+1]
        SurfK = ca.reshape(SurfK,3,4)
        SurfK = SurfK.T #NOTE: This process is due to casadi naming convention to have first row to be x1,x2,x3
        SurfE = SurfParaTemp[11+1:11+3+1]
        Surfk = SurfParaTemp[14+1:14+4+1]
        Surfe = SurfParaTemp[-1]

        #FootStep Constraint
        #Inequality
        g.append(SurfK@P_vector - Surfk)
        glb.append(np.full((4,),-np.inf))
        gub.append(np.full((4,),0))
        #print(FirstSurfK@p_next - FirstSurfk)

        #Equality
        g.append(SurfE.T@P_vector - Surfe)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

    #Switching Time Constraint
    for phase_cnt in range(Nphase):
        if GaitPattern[phase_cnt] == 'Swing':
            if phase_cnt == 0:
                g.append(Ts[phase_cnt]-0)
                glb.append(np.array([0.3]))
                gub.append(np.array([0.9]))
            else:
                g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
                glb.append(np.array([0.4]))
                gub.append(np.array([0.9]))
        elif GaitPattern[phase_cnt] == 'DoubleSupport':
            g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
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
#Pure Kinematics Check
def Pure_Kinematics_Check(Nsteps = 8, StandAlong = False, ParameterList = None):

    #-----------------------------------------------------------------------------
    #Parameter Set-up
    #Nsteps = 8 #Enumeration of the Steps start from 0, so Number of Steps - 1, use mod function to check it is left or right
    #   Kinematics Constraint
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

    #------------------------------------------------------------------------------
    #Extract Casadi Parameters
    x_init = ParameterList["x_init"]
    y_init = ParameterList["y_init"]
    z_init = ParameterList["z_init"]

    PLx_init = ParameterList["PLx_init"]
    PLy_init = ParameterList["PLy_init"]
    PLz_init = ParameterList["PLz_init"]
    PL_init = ca.vertcat(PLx_init,PLy_init,PLz_init)

    PRx_init = ParameterList["PRx_init"]
    PRy_init = ParameterList["PRy_init"]
    PRz_init = ParameterList["PRz_init"]
    PR_init = ca.vertcat(PRx_init,PRy_init,PRz_init)

    #x_end = ParameterList["x_end"]
    y_end = ParameterList["y_end"]
    z_end = ParameterList["z_end"]

    #Flags for Swing Legs (Defined as Parameters)
    ParaLeftSwingFlag = ParameterList["LeftSwingFlag"]
    ParaRightSwingFlag = ParameterList["RightSwingFlag"]

    #-----------------------------------------------------------------------------------------------------------------------
    #Define Variables and Lower and Upper Bounds
    #   Init Contact Location
    #   px
    px_init = ca.SX.sym('px_init')
    px_init_lb = np.array([-1])
    px_init_ub = np.array([10])

    #   py
    py_init = ca.SX.sym('py_init')
    py_init_lb = np.array([-1])
    py_init_ub = np.array([1])

    #   pz
    pz_init = ca.SX.sym('pz_init')
    pz_init_lb = np.array([0])
    pz_init_ub = np.array([0])

    #   CoM Position, 2 knots per step
    x = []
    xlb = []
    xub = []
    y = []
    ylb = []
    yub = []
    z = []
    zlb = []
    zub = []
    #   Contact Location List
    px = []
    pxlb = []
    pxub = []
    py = []
    pylb = []
    pyub = []
    pz = []
    pzlb = []
    pzub = []

    for stepIdx in range(Nsteps):
        #For CoM state, each phase/step has two knots
        xtemp = ca.SX.sym('x'+str(stepIdx+1),2)
        x.append(xtemp)
        xlb.append(np.array([-5,-5]))
        xub.append(np.array([5,5]))

        ytemp = ca.SX.sym('y'+str(stepIdx+1),2)
        y.append(ytemp)
        ylb.append(np.array([-1,-1]))
        yub.append(np.array([1,1]))

        ztemp = ca.SX.sym('z'+str(stepIdx+1),2)
        z.append(ztemp)
        zlb.append(np.array([0.5,0.5]))
        zub.append(np.array([0.7,0.7]))

        pxtemp = ca.SX.sym('px'+str(stepIdx+1))
        px.append(pxtemp)
        pxlb.append(np.array([-5]))
        pxub.append(np.array([5]))

        pytemp = ca.SX.sym('py'+str(stepIdx+1))
        py.append(pytemp)
        pylb.append(np.array([-2]))
        pyub.append(np.array([2]))

        #   Foot steps are all staying on the ground
        pztemp = ca.SX.sym('pz'+str(stepIdx+1))
        pz.append(pztemp)
        pzlb.append(np.array([0]))
        pzub.append(np.array([0]))

    #Reshape Variable List
    x = ca.vertcat(*x)
    y = ca.vertcat(*y)
    z = ca.vertcat(*z)
    px = ca.vertcat(*px)
    py = ca.vertcat(*py)
    pz = ca.vertcat(*pz)

    #--------------------------------
    #Update Decision Vars and Bounds
    #--------------------------------
    DecisionVars = ca.vertcat(x,y,z,px_init,py_init,pz_init,px,py,pz) #treat all elements in the list as single variables
    DecisionVarsShape = DecisionVars.shape #get decision variable list shape, for future use

    DecisionVars_lb = np.concatenate((xlb,ylb,zlb,px_init_lb,py_init_lb,pz_init_lb,pxlb,pylb,pzlb),axis=None)
    DecisionVars_ub = np.concatenate((xub,yub,zub,px_init_ub,py_init_ub,pz_init_ub,pxub,pyub,pzub),axis=None)

    #-----------------------------------------------------------------------------------------------------------------------
    #Define Constrains

    g = []
    glb = []
    gub = []
    J = 0

    #   Loop over to have Kinematics Constraint
    for Nph in range(Nsteps):
        #Get Important Variables
        CoM_0 = ca.vertcat(x[2*Nph],y[2*Nph],z[2*Nph])
        CoM_1 = ca.vertcat(x[2*Nph+1],y[2*Nph+1],z[2*Nph+1])
        if Nph == 0:
            #!!!!!!Pass from the first Level!!!!!!
            P_k_current = ca.vertcat(px_init,py_init,pz_init) #ca.vertcat(px[-1],py[-1],pz[-1])
            #!!!!!!
            P_k_next = ca.vertcat(px[Nph],py[Nph],pz[Nph])
        else:
            P_k_current = ca.vertcat(px[Nph-1],py[Nph-1],pz[Nph-1])
            P_k_next = ca.vertcat(px[Nph],py[Nph],pz[Nph])
            
        #Construct Kinematics Constraints
        if Nph%2 == 0: #even number
            #------------------------------------
            #If First Level Swing the Left, then the second level Even Number Phases (the first Phase) Swing the Right -> Left -> Right
            #Left foot in contact for p_current, right foot is going to land as p_next
            #CoM Kinemactis Constraint for the support foot (CoM_0 in *current* contact -> left)
            g.append(ca.if_else(ParaLeftSwingFlag,K_CoM_Left@(CoM_0-P_k_current)-ca.DM(k_CoM_Left),np.full((len(k_CoM_Left),),-1)))
            glb.append(np.full((len(k_CoM_Left),),-np.inf))
            gub.append(np.full((len(k_CoM_Left),),0))
            #CoM Kinemactis Constraint for the support foot (CoM_0 in *next* contact -> right)
            g.append(ca.if_else(ParaLeftSwingFlag,K_CoM_Right@(CoM_0-P_k_next)-ca.DM(k_CoM_Right),np.full((len(k_CoM_Right),),-1)))
            glb.append(np.full((len(k_CoM_Right),),-np.inf))
            gub.append(np.full((len(k_CoM_Right),),0))
            #CoM Kinemactis Constraint for the swing foot (CoM_1 land but in the *current* contact -> left)
            g.append(ca.if_else(ParaLeftSwingFlag,K_CoM_Left@(CoM_1-P_k_current)-ca.DM(k_CoM_Left),np.full((len(k_CoM_Left),),-1)))
            glb.append(np.full((len(k_CoM_Left),),-np.inf))
            gub.append(np.full((len(k_CoM_Left),),0))
            #CoM Kinemactis Constraint for the swing foot (CoM_1 land but in the *next* contact -> right)
            g.append(ca.if_else(ParaLeftSwingFlag,K_CoM_Right@(CoM_1-P_k_next)-ca.DM(k_CoM_Right),np.full((len(k_CoM_Right),),-1)))
            glb.append(np.full((len(k_CoM_Right),),-np.inf))
            gub.append(np.full((len(k_CoM_Right),),0))
            #Relative Swing Foot Location (rf in lf)
            g.append(ca.if_else(ParaLeftSwingFlag,Q_rf_in_lf@(P_k_next-P_k_current)-ca.DM(q_rf_in_lf),np.full((len(q_rf_in_lf),),-1)))
            glb.append(np.full((len(q_rf_in_lf),),-np.inf))
            gub.append(np.full((len(q_rf_in_lf),),0))
            #-------------------------------------
            #If First Levvel Swing the Right, then the second level Even Number Phases (the first phase) Swing the Left -> Right -> Left
            #Right foot in contact for p_current, left foot is going to land at p_next
            #CoM Kinemactis Constraint for the support foot (CoM_0 in *current* contact -> right)
            g.append(ca.if_else(ParaRightSwingFlag,K_CoM_Right@(CoM_0-P_k_current)-ca.DM(k_CoM_Right),np.full((len(k_CoM_Right),),-1)))
            glb.append(np.full((len(k_CoM_Right),),-np.inf))
            gub.append(np.full((len(k_CoM_Right),),0))
            #CoM Kinemactis Constraint for the support foot (CoM_0 in *next* contact -> left)
            g.append(ca.if_else(ParaRightSwingFlag,K_CoM_Left@(CoM_0-P_k_next)-ca.DM(k_CoM_Left),np.full((len(k_CoM_Left),),-1)))
            glb.append(np.full((len(k_CoM_Left),),-np.inf))
            gub.append(np.full((len(k_CoM_Left),),0))
            #CoM Kinemactis Constraint for the swing foot (CoM_1 land but in the *current contact* -> right)
            g.append(ca.if_else(ParaRightSwingFlag,K_CoM_Right@(CoM_1-P_k_current)-ca.DM(k_CoM_Right),np.full((len(k_CoM_Right),),-1)))
            glb.append(np.full((len(k_CoM_Right),),-np.inf))
            gub.append(np.full((len(k_CoM_Right),),0))
            #CoM Kinemactis Constraint for the swing foot (CoM_1 land but in the *next contact* -> left)
            g.append(ca.if_else(ParaRightSwingFlag,K_CoM_Left@(CoM_1-P_k_next)-ca.DM(k_CoM_Left),np.full((len(k_CoM_Left),),-1)))
            glb.append(np.full((len(k_CoM_Left),),-np.inf))
            gub.append(np.full((len(k_CoM_Left),),0))
            #Relative Swing Foot Location (lf in rf)
            g.append(ca.if_else(ParaRightSwingFlag,Q_lf_in_rf@(P_k_next-P_k_current)-ca.DM(q_lf_in_rf),np.full((len(q_lf_in_rf),),-1)))
            glb.append(np.full((len(q_lf_in_rf),),-np.inf))
            gub.append(np.full((len(q_lf_in_rf),),0))
            
        elif Nph%2 ==1: #odd number, right foot in contact for p_current, left foot is going to land at p_next
            #If the first level swings the Left, then the second level for Odd Number Phases (the second phase) swings left -> Right -> Left
            ##Right foot in contact for p_current, left foot is going to land at p_next
            g.append(ca.if_else(ParaLeftSwingFlag,K_CoM_Right@(CoM_0-P_k_current)-ca.DM(k_CoM_Right),np.full((len(k_CoM_Right),),-1)))
            glb.append(np.full((len(k_CoM_Right),),-np.inf))
            gub.append(np.full((len(k_CoM_Right),),0))
            #CoM Kinemactis Constraint for the support foot (CoM_0 in *next* contact -> left)
            g.append(ca.if_else(ParaLeftSwingFlag,K_CoM_Left@(CoM_0-P_k_next)-ca.DM(k_CoM_Left),np.full((len(k_CoM_Left),),-1)))
            glb.append(np.full((len(k_CoM_Left),),-np.inf))
            gub.append(np.full((len(k_CoM_Left),),0))
            #CoM Kinemactis Constraint for the swing foot (CoM_1 land but in the *current contact* -> right)
            g.append(ca.if_else(ParaLeftSwingFlag,K_CoM_Right@(CoM_1-P_k_current)-ca.DM(k_CoM_Right),np.full((len(k_CoM_Right),),-1)))
            glb.append(np.full((len(k_CoM_Right),),-np.inf))
            gub.append(np.full((len(k_CoM_Right),),0))
            #CoM Kinemactis Constraint for the swing foot (CoM_1 land but in the *next contact* -> left)
            g.append(ca.if_else(ParaLeftSwingFlag,K_CoM_Left@(CoM_1-P_k_next)-ca.DM(k_CoM_Left),np.full((len(k_CoM_Left),),-1)))
            glb.append(np.full((len(k_CoM_Left),),-np.inf))
            gub.append(np.full((len(k_CoM_Left),),0))
            #Relative Swing Foot Location (lf in rf)
            g.append(ca.if_else(ParaLeftSwingFlag,Q_lf_in_rf@(P_k_next-P_k_current)-ca.DM(q_lf_in_rf),np.full((len(q_lf_in_rf),),-1)))
            glb.append(np.full((len(q_lf_in_rf),),-np.inf))
            gub.append(np.full((len(q_lf_in_rf),),0))
                
            #If the first level swings the Right, then the second level for Odd Number Phases (the second phase) swings Right -> Left -> Right
            #Left foot in contact for p_current, right foot is going to land as p_next
            #CoM Kinemactis Constraint for the support foot (CoM_0 in *current* contact -> left)
            g.append(ca.if_else(ParaRightSwingFlag,K_CoM_Left@(CoM_0-P_k_current)-ca.DM(k_CoM_Left),np.full((len(k_CoM_Left),),-1)))
            glb.append(np.full((len(k_CoM_Left),),-np.inf))
            gub.append(np.full((len(k_CoM_Left),),0))
            #CoM Kinemactis Constraint for the support foot (CoM_0 in *next* contact -> right)
            g.append(ca.if_else(ParaRightSwingFlag,K_CoM_Right@(CoM_0-P_k_next)-ca.DM(k_CoM_Right),np.full((len(k_CoM_Right),),-1)))
            glb.append(np.full((len(k_CoM_Right),),-np.inf))
            gub.append(np.full((len(k_CoM_Right),),0))
            #CoM Kinemactis Constraint for the swing foot (CoM_1 land but in the *current* contact -> left)
            g.append(ca.if_else(ParaRightSwingFlag,K_CoM_Left@(CoM_1-P_k_current)-ca.DM(k_CoM_Left),np.full((len(k_CoM_Left),),-1)))
            glb.append(np.full((len(k_CoM_Left),),-np.inf))
            gub.append(np.full((len(k_CoM_Left),),0))
            #CoM Kinemactis Constraint for the swing foot (CoM_1 land but in the *next* contact -> right)
            g.append(ca.if_else(ParaRightSwingFlag,K_CoM_Right@(CoM_1-P_k_next)-ca.DM(k_CoM_Right),np.full((len(k_CoM_Right),),-1)))
            glb.append(np.full((len(k_CoM_Right),),-np.inf))
            gub.append(np.full((len(k_CoM_Right),),0))
            #Relative Swing Foot Location (rf in lf)
            g.append(ca.if_else(ParaRightSwingFlag,Q_rf_in_lf@(P_k_next-P_k_current)-ca.DM(q_rf_in_lf),np.full((len(q_rf_in_lf),),-1)))
            glb.append(np.full((len(q_rf_in_lf),),-np.inf))
            gub.append(np.full((len(q_rf_in_lf),),0))

    #Initial and Terminal Constraints
    #No matter second level or first Level, the terminal constraint for y and z axis should present
    #   Terminal Condition
    #   y-axis
    g.append(y[-1]-y_end)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

    #   z-axis
    g.append(z[-1]-z_end)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

    #If it is playing the first level as a standalong layer, then we need to give initial conditions
    if StandAlong == True:
        #Init x-axis
        g.append(x[0]-x_init)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        #Init y-axis
        g.append(y[0]-y_init)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        #Init z-axis
        g.append(z[0]-z_init)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        #------------------------------------
        #Initial Footstep Locations
        p_init = ca.vertcat(px_init,py_init,pz_init)

        #If First Level Swing the Left, then the second level Even Number Phases (the first Phase) Swing the Right -> Left -> Right
        #Left foot in contact for p_current, right foot is going to land as p_next
        #CoM Kinemactis Constraint for the support foot (CoM_0 in *current* contact -> left)
        g.append(ca.if_else(ParaLeftSwingFlag,p_init-PL_init,np.array([0,0,0])))
        glb.append(np.array([0,0,0]))
        gub.append(np.array([0,0,0]))
        
        #If the first level swings the Right, then the second level for Odd Number Phases (the second phase) swings Right -> Left -> Right
        #Left foot in contact for p_current, right foot is going to land as p_next
        #CoM Kinemactis Constraint for the support foot (CoM_0 in *current* contact -> left)
        g.append(ca.if_else(ParaRightSwingFlag,p_init-PR_init,np.array([0,0,0])))
        glb.append(np.array([0,0,0]))
        gub.append(np.array([0,0,0]))

    #-------------------------------------------
    #Build Variable Index
    #Get Updated variable index
    x_index = (0,Nsteps*2-1)
    y_index = (x_index[1]+1,x_index[1]+Nsteps*2)
    z_index = (y_index[1]+1,y_index[1]+Nsteps*2)
    px_init_index = (z_index[1]+1,z_index[1]+1)
    py_init_index = (px_init_index[1]+1,px_init_index[1]+1)
    pz_init_index = (py_init_index[1]+1,py_init_index[1]+1)
    px_index = (pz_init_index[1]+1,pz_init_index[1]+Nsteps)
    py_index = (px_index[1]+1,px_index[1]+Nsteps)
    pz_index = (py_index[1]+1,py_index[1]+Nsteps)
    #Update var_index dictionary
    var_index = {"x":x_index,
                 "y":y_index,
                 "z":z_index,
                 "px_init":px_init_index,
                 "py_init":py_init_index,
                 "pz_init":pz_init_index,
                 "px":px_index,
                 "py":py_index,
                 "pz":pz_index
        }

    return DecisionVars, DecisionVars_lb, DecisionVars_ub, J, g, glb, gub, var_index

#CoM Dynamics Only
def CoM_Dynamics(m = 95, Nsteps = 1, StandAlong = True, StaticStop = False, ParameterList = None):
    #------------------------------------------------
    #Parameter Setup
    #   Set up Gait pattern
    #GaitPattern = ["InitialDoubleSupport"] + ["Swing","DoubleSupport"]*Nsteps
    GaitPattern = ["Swing","DoubleSupport"]*Nsteps
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
    #Define Initial and Terminal Conditions, Get from casadi Parameters
    x_init = ParameterList["x_init"]
    y_init = ParameterList["y_init"]
    z_init = ParameterList["z_init"]

    xdot_init = ParameterList["xdot_init"]
    ydot_init = ParameterList["ydot_init"]
    zdot_init = ParameterList["zdot_init"]

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

    #Flags for Swing Legs (Defined as Parameters)
    ParaLeftSwingFlag = ParameterList["LeftSwingFlag"]
    ParaRightSwingFlag = ParameterList["RightSwingFlag"]

    #Surface Patches
    SurfParas = ParameterList["SurfParas"]

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
    xdot_lb = np.array([[-0.5]*(xdot.shape[0]*xdot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    xdot_ub = np.array([[0.5]*(xdot.shape[0]*xdot.shape[1])])
    #   CoM Velocity y-axis
    ydot = ca.SX.sym('ydot',N_K)
    ydot_lb = np.array([[-0.5]*(ydot.shape[0]*ydot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    ydot_ub = np.array([[0.5]*(ydot.shape[0]*ydot.shape[1])])
    #   CoM Velocity z-axis
    zdot = ca.SX.sym('zdot',N_K)
    zdot_lb = np.array([[-0.5]*(zdot.shape[0]*zdot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    zdot_ub = np.array([[0.5]*(zdot.shape[0]*zdot.shape[1])])

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
        pz_lb.append(np.array([-1]))
        pz_ub.append(np.array([1]))

    #Switching Time Vector
    Ts = []
    Ts_lb = []
    Ts_ub = []
    for n_phase in range(Nphase):
        Tstemp = ca.SX.sym('Ts'+str(n_phase+1)) #0 + 1 + ....
        Ts.append(Tstemp)
        Ts_lb.append(np.array([0.05]))
        Ts_ub.append(np.array([1.5*(Nphase+1)])) #Consider the First Level

    #   Collect all Decision Variables
    DecisionVars = ca.vertcat(x,y,z,
                              xdot,ydot,zdot,
                              FLx,FLy,FLz,
                              FRx,FRy,FRz,
                              px_init,py_init,pz_init,
                              *px,*py,*pz,*Ts)
    #print(DecisionVars)
    DecisionVarsShape = DecisionVars.shape

    #   Collect all lower bound and upper bound
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

    #Initial and Termianl Conditions
    #   Terminal CoM y-axis
    g.append(y[-1]-y_end)
    glb.append(np.array([0]))
    gub.append(np.array([0]))

    #   Terminal CoM z-axis
    g.append(z[-1]-z_end)
    glb.append(np.array([0]))    
    gub.append(np.array([0]))

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

    h_doublesupport = 0.3
    h_swing = 0.8

    #Loop over all Phases (Knots)
    for Nph in range(Nphase):
        #Decide Number of Knots
        if Nph == Nphase-1:  #The last Knot belongs to the Last Phase
            Nk_ThisPhase = Nk_Local+1
        else:
            Nk_ThisPhase = Nk_Local       

        ##Decide Time Vector
        if Nph == 0: #first phase
            h = tauStepLength*Nphase*(Ts[Nph]-0)
        else: #other phases
            h = tauStepLength*Nphase*(Ts[Nph]-Ts[Nph-1]) 

        #if GaitPattern[Nph]=='Swing':
        #    h = h_swing/Nk_Local
        #elif GaitPattern[Nph]=='DoubleSupport':
        #    h = h_doublesupport/Nk_Local


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
            if GaitPattern[Nph]=='Swing':
                
                if Nph//2 == 0:
                    #!!!!!!Pass from the first Level!!!!!!
                    P_k_current = ca.vertcat(px_init,py_init,pz_init) #ca.vertcat(px[-1],py[-1],pz[-1])
                    #!!!!!!
                    P_k_next = ca.vertcat(px[Nph//2],py[Nph//2],pz[Nph//2])
                else:
                    P_k_current = ca.vertcat(px[Nph//2-1],py[Nph//2-1],pz[Nph//2-1])
                    P_k_next = ca.vertcat(px[Nph//2],py[Nph//2],pz[Nph//2])

                if (Nph//2)%2 == 0: #even number steps
                    
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

                elif (Nph//2)%2 == 1: #odd number steps
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
                if (Nph-1)//2 == 0:
                    #!!!!!!Pass from the first Level!!!!!!
                    P_k_current = ca.vertcat(px_init,py_init,pz_init) #ca.vertcat(px[-1],py[-1],pz[-1])
                    #!!!!!!
                    P_k_next = ca.vertcat(px[(Nph-1)//2],py[(Nph-1)//2],pz[(Nph-1)//2])
                else:
                    P_k_current = ca.vertcat(px[(Nph-1)//2-1],py[(Nph-1)//2-1],pz[(Nph-1)//2-1])
                    P_k_next = ca.vertcat(px[(Nph-1)//2],py[(Nph-1)//2],pz[(Nph-1)//2])

                if ((Nph-1)//2)%2 == 0: #even number steps
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

                elif ((Nph-1)//2)%2 == 1: #odd number steps
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
    for PatchNum in range(Nsteps):
        #Get Footstep Vector
        P_vector = ca.vertcat(px[PatchNum],py[PatchNum],pz[PatchNum])

        #Get Half Space Representation
        SurfParaTemp = SurfParas[20+PatchNum*20:19+(PatchNum+1)*20+1]
        SurfK = SurfParaTemp[0:11+1]
        SurfK = ca.reshape(SurfK,3,4)
        SurfK = SurfK.T #NOTE: This process is due to casadi naming convention to have first row to be x1,x2,x3
        SurfE = SurfParaTemp[11+1:11+3+1]
        Surfk = SurfParaTemp[14+1:14+4+1]
        Surfe = SurfParaTemp[-1]

        #FootStep Constraint
        #Inequality
        g.append(SurfK@P_vector - Surfk)
        glb.append(np.full((4,),-np.inf))
        gub.append(np.full((4,),0))
        #print(FirstSurfK@p_next - FirstSurfk)

        #Equality
        g.append(SurfE.T@P_vector - Surfe)
        glb.append(np.array([0]))
        gub.append(np.array([0]))


    #Switching Time Constraint
    for phase_cnt in range(Nphase):
        if GaitPattern[phase_cnt] == "Swing":
            if phase_cnt == 0:
                g.append(Ts[phase_cnt]-0)
                glb.append(np.array([0.2]))
                gub.append(np.array([0.9]))
            else:
                g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
                glb.append(np.array([0.2]))
                gub.append(np.array([0.9]))
        elif GaitPattern[phase_cnt] == 'DoubleSupport':
            g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
            glb.append(np.array([0.1]))
            gub.append(np.array([0.5]))

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
    Ts_index = (pz_index[1]+1,pz_index[1]+Nphase)

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
                 "Ts":Ts_index,
    }

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
    SurfParas = []
    for surfNum in range(NumSurfaces):
        SurfTemp = ca.SX.sym('S'+str(surfNum),3*4+3+5)
        SurfParas.append(SurfTemp)
    SurfParas = ca.vertcat(*SurfParas)
    #   FirstRound Indicators (if yes, we have an initial double support phase, if not, then we dont have an initial double support phase)
    FirstRoundFlag = ca.SX.sym('FirstRoundFlag')
    #   Collect all Parameters
    ParaList = {"LeftSwingFlag":ParaLeftSwingFlag,
                "RightSwingFlag":ParaRightSwingFlag,
                "x_init":x_init,
                "y_init":y_init,
                "z_init":z_init,
                "xdot_init":xdot_init,
                "ydot_init":ydot_init,
                "zdot_init":zdot_init,
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
                "FirstRoundFlag":FirstRoundFlag,
    }
    #Collect all Parameters
    paras = ca.vertcat(ParaLeftSwingFlag,ParaRightSwingFlag,
                       x_init,y_init,z_init,
                       xdot_init,ydot_init,zdot_init,
                       PLx_init,PLy_init,PLz_init,
                       PRx_init,PRy_init,PRz_init,
                       x_end,y_end,z_end,
                       xdot_end,ydot_end,zdot_end,SurfParas,FirstRoundFlag)

    #-----------------------------------------------------------------------------------------------------------------
    #Identify the Fidelity Type of the whole framework, Used to tell the First Level to set Constraints Accordingly
    if SecondLevel == None:
        SingleFidelity = True
    else:
        SingleFidelity = False

    #Bulding the First Level
    if FirstLevel == "NLP_SingleStep":
        var_Level1, var_lb_Level1, var_ub_Level1, J_Level1, g_Level1, glb_Level1, gub_Level1, var_index_Level1 = NLP_SingleStep(m = m, StandAlong = SingleFidelity, ConservativeEnd = ConservativeFirstStep, ParameterList = ParaList)
    elif FirstLevel == "Pure_Kinematics_Check":
        var_Level1, var_lb_Level1, var_ub_Level1, J_Level1, g_Level1, glb_Level1, gub_Level1, var_index_Level1 = Pure_Kinematics_Check(StandAlong = SingleFidelity, ParameterList = ParaList)
    else:
        print("Print Not implemented or Wrong Solver Build Enumeration")        
    
    #!!!!!Building the Second Level
    if SecondLevel == None:
        print("No Second Level")
        var_index_Level2 = []
    elif SecondLevel == "Pure_Kinematics_Check":
        var_Level2, var_lb_Level2, var_ub_Level2, J_Level2, g_Level2, glb_Level2, gub_Level2, var_index_Level2 = Pure_Kinematics_Check(StandAlong = SingleFidelity, ParameterList = ParaList)
    elif SecondLevel == "CoM_Dynamics":
        var_Level2, var_lb_Level2, var_ub_Level2, J_Level2, g_Level2, glb_Level2, gub_Level2, var_index_Level2 = CoM_Dynamics(m = m, StandAlong = SingleFidelity,  ParameterList = ParaList, Nsteps = NumSurfaces-1)#Here is the total number of steps
    elif SecondLevel == "NLP_SecondLevel":
        var_Level2, var_lb_Level2, var_ub_Level2, J_Level2, g_Level2, glb_Level2, gub_Level2, var_index_Level2 = NLP_SecondLevel(m = m, ParameterList = ParaList, Nsteps = NumSurfaces-1)
    #!!!!!Connect the Terminal state of the first Level with the Second Level

    #Set-up Terminal Cost Here and Sum over all costs
    if SecondLevel == None: #No second Level
        #   Collect the variables, terminal cost set as the end of the single first level
        J = J_Level1
        x_Level1 = var_Level1[var_index_Level1["x"][0]:var_index_Level1["x"][1]+1]
        y_Level1 = var_Level1[var_index_Level1["y"][0]:var_index_Level1["y"][1]+1]
        z_Level1 = var_Level1[var_index_Level1["z"][0]:var_index_Level1["z"][1]+1]
        xdot_Level1 = var_Level1[var_index_Level1["xdot"][0]:var_index_Level1["xdot"][1]+1]
        ydot_Level1 = var_Level1[var_index_Level1["ydot"][0]:var_index_Level1["ydot"][1]+1]
        zdot_Level1 = var_Level1[var_index_Level1["zdot"][0]:var_index_Level1["zdot"][1]+1]
        #J = J + 100*(x_Level1[-1]-x_end)**2 + 100*(y_Level1[-1]-y_end)**2 + 100*(z_Level1[-1]-z_end)**2
        J = J + 100*(x_Level1[-1]-x_end)**2
    else:#With Second level
        #Summation of the all running cost
        J = J_Level1 + J_Level2
        x_Level2 = var_Level2[var_index_Level2["x"][0]:var_index_Level2["x"][1]+1]
        y_Level2 = var_Level2[var_index_Level2["y"][0]:var_index_Level2["y"][1]+1]
        z_Level2 = var_Level2[var_index_Level2["z"][0]:var_index_Level2["z"][1]+1]
        if SecondLevel == "CoM_Dynamics" or SecondLevel == "NLP_SecondLevel":
            xdot_Level2 = var_Level2[var_index_Level2["xdot"][0]:var_index_Level2["xdot"][1]+1]
            ydot_Level2 = var_Level2[var_index_Level2["ydot"][0]:var_index_Level2["ydot"][1]+1]
            zdot_Level2 = var_Level2[var_index_Level2["zdot"][0]:var_index_Level2["zdot"][1]+1]
        #Add terminal cost
        #J = J + 100*(x_Level2[-1]-x_end)**2 + 100*(y_Level2[-1]-y_end)**2 + 100*(z_Level2[-1]-z_end)**2 + 100*(xdot_Level2[-1])**2 + 100*(ydot_Level2[-1])**2 + 100*(zdot_Level2[-1])**2
        J = J + 100*(x_Level2[-1]-x_end)**2
        
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

        #   xdot,ydot,zdot_end according to the choice of second level
        if SecondLevel == "CoM_Dynamics" or SecondLevel == "NLP_SecondLevel":
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
        
        if SecondLevel == "NLP_SecondLevel":
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
    prob = {'x': DecisionVars, 'f': J, 'g': g, 'p': paras}
    solver = ca.nlpsol('solver', 'ipopt', prob)

    return solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index

    #print("Not Implemented")    

def TestSolverBuild():
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
    #   Collect all Parameters
    ParaList = {"LeftSwingFlag":ParaLeftSwingFlag,
                "RightSwingFlag":ParaRightSwingFlag,
                "x_init":x_init,
                "y_init":y_init,
                "z_init":z_init,
                "xdot_init":xdot_init,
                "ydot_init":ydot_init,
                "zdot_init":zdot_init,
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
    }
    #Collect all Parameters
    paras = ca.vertcat(ParaLeftSwingFlag,ParaRightSwingFlag,
                       x_init,y_init,z_init,
                       xdot_init,ydot_init,zdot_init,
                       PLx_init,PLy_init,PLz_init,
                       PRx_init,PRy_init,PRz_init,
                       x_end,y_end,z_end,
                       xdot_end,ydot_end,zdot_end)

    CoM_Dynamics(ParameterList=ParaList)