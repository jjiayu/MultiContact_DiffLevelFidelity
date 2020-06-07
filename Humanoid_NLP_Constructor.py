# Import Important Modules
import numpy as np #Numpy
import casadi as ca #Casadi
import matplotlib.pyplot as plt #Matplotlib
from mpl_toolkits.mplot3d import Axes3D
# Import SL1M modules
from sl1m.constants_and_tools import *
from sl1m.planner import *
from constraints import *

def Humanoid_NLP_MultiFidelity_Constructor(withSecondLevel = True):

    #   Set Decimal Printing Precision
    np.set_printoptions(precision=4)
    
    #First Level NLP
    #-----------------------------------------------------------------------------------------------------------------------
    #Decide Parameters
    #   Gait Pattern, Each action is followed up by a double support phase
    GaitPattern = ['InitialDouble','Swing','DoubleSupport'] #,'RightSupport','DoubleSupport','LeftSupport','DoubleSupport'
    #   Number of Phases
    Nphase = len(GaitPattern)
    #   Number of Steps
    Nstep = 1
    #Nstep = (Nphase-1)//2 #Integer Division to get number of steps from Number of Phases
    #print(Nstep)
    #Nstep = 1 #Enumeration of the Steps start from 0, so Number of Steps - 1, use mod function to check it is left or right
    #   Number of Knots per Phase - how many intervals do we have for a single phase; 10 intervals need 11 knots/ticks
    Nk_Local= 5
    #   Compute Number of Total knots/ticks, but the enumeration start from 0 to N_K-1
    N_K = Nk_Local*Nphase + 1 #+1 the last knots to finalize the plan
    #   Robot mass
    m = 95 #kg
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
    #Q_rf_in_lf = relativeConstraints[0][0]
    #q_rf_in_lf = relativeConstraints[0][1]
    #Q_lf_in_rf = relativeConstraints[1][0]
    #q_lf_in_rf = relativeConstraints[1][1]
    #------
    Q_lf_in_rf = relativeConstraints[0][0] #named rf in lf, but representing lf in rf
    q_lf_in_rf = relativeConstraints[0][1] #named rf in lf, but representing lf in rf
    Q_rf_in_lf = relativeConstraints[1][0] #named lf in rf, but representing rf in lf
    q_rf_in_lf = relativeConstraints[1][1] #named lf in rf, but representing rf in lf
    #-----------------------------------------------------------------------------------------------------------------------

    #-----------------------------------------------------------------------------------------------------------------------
    #Set up Initial Condition, Some are Defined as Parameters
    #-----------------------------------------------------------------------------------------------------------------------
    #x_init = 0
    #y_init = 0
    #z_init = 0.55

    x_init = ca.SX.sym('x_init')
    y_init = ca.SX.sym('y_init')
    z_init = ca.SX.sym('z_init')

    xdot_init = 0
    ydot_init = 0
    zdot_init = 0

    Lx_init = 0
    Ly_init = 0
    Lz_init = 0

    Ldotx_init = 0
    Ldoty_init = 0
    Ldotz_init = 0

    #PLx_init = 0
    #PLy_init = 0.1
    #PLz_init = 0
    #PL_init = np.array([PLx_init,PLy_init,PLz_init])
    PLx_init = ca.SX.sym('PLx_init')
    PLy_init = ca.SX.sym('PLy_init')
    PLz_init = ca.SX.sym('PLz_init')
    PL_init = ca.vertcat(PLx_init,PLy_init,PLz_init)

    #PRx_init = 0
    #PRy_init = -0.1
    #PRz_init = 0
    #PR_init = np.array([PRx_init,PRy_init,PRz_init])
    PRx_init = ca.SX.sym('PRx_init')
    PRy_init = ca.SX.sym('PRy_init')
    PRz_init = ca.SX.sym('PRz_init')
    PR_init = ca.vertcat(PRx_init,PRy_init,PRz_init)

    #x_end = 0.375
    #y_end = 0
    #z_end = 0.55
    x_end = ca.SX.sym('x_end')
    y_end = ca.SX.sym('y_end')
    z_end = ca.SX.sym('z_end')

    xdot_end = 0
    ydot_end = 0
    zdot_end = 0

    Lx_end = 0
    Ly_end = 0
    Lz_end = 0

    Ldotx_end = 0
    Ldoty_end = 0
    Ldotz_end = 0

    #-----------------------------------------------------------------------------------------------------------------------
    #Define Variables and Bounds, Parameters
    #   CoM Position x-axis
    x = ca.SX.sym('x',N_K)
    x_lb = np.array([[0]*(x.shape[0]*x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    x_ub = np.array([[5]*(x.shape[0]*x.shape[1])])
    #   CoM Position y-axis
    y = ca.SX.sym('y',N_K)
    y_lb = np.array([[-1]*(y.shape[0]*y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    y_ub = np.array([[1]*(y.shape[0]*y.shape[1])])
    #   CoM Position z-axis
    z = ca.SX.sym('z',N_K)
    z_lb = np.array([[0.5]*(z.shape[0]*z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    z_ub = np.array([[0.7]*(z.shape[0]*z.shape[1])])
    #   CoM Velocity x-axis
    xdot = ca.SX.sym('xdot',N_K)
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
        px_ub.append(np.array([5]))

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
    Ts = []
    Ts_lb = []
    Ts_ub = []
    for n_phase in range(Nphase):
        Tstemp = ca.SX.sym('Ts'+str(n_phase+1)) #0 + 1 + ....
        Ts.append(Tstemp)
        Ts_lb.append(np.array([0.15]))
        Ts_ub.append(np.array([1.2]))

    #Define Parameters
    ParaLeftSwingFlag = ca.SX.sym('LeftSwingFlag')
    ParaRightSwingFlag = ca.SX.sym('RightSwingFlag')

    paras = ca.vertcat(ParaLeftSwingFlag,ParaRightSwingFlag,x_init,y_init,z_init,PLx_init,PLy_init,PLz_init,PRx_init,PRy_init,PRz_init,x_end,y_end,z_end)

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
    #J = 10*(x[-1]-x_end)**2

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

    #   Terminal CoM x-axis
    #g.append(x[-1]-x_end)
    #glb.append(np.array([0]))
    #gub.append(np.array([0]))

    if withSecondLevel == False:
        #   Terminal CoM y-axis
        g.append(y[-1]-y_end)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        #   Terminal CoM z-axis
        g.append(z[-1]-z_end)
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

            #make vectors
            FL1_k = ca.vertcat(FL1x[k],FL1y[k],FL1z[k])
            FL2_k = ca.vertcat(FL2x[k],FL2y[k],FL2z[k])
            FL3_k = ca.vertcat(FL3x[k],FL3y[k],FL3z[k])
            FL4_k = ca.vertcat(FL4x[k],FL4y[k],FL4z[k])

            FR1_k = ca.vertcat(FR1x[k],FR1y[k],FR1z[k])
            FR2_k = ca.vertcat(FR2x[k],FR2y[k],FR2z[k])
            FR3_k = ca.vertcat(FR3x[k],FR3y[k],FR3z[k])
            FR4_k = ca.vertcat(FR4x[k],FR4y[k],FR4z[k])

            CoM_k = ca.vertcat(x[k],y[k],z[k])

            if k<N_K-1:
                Ldot_current = ca.vertcat(Ldotx[k],Ldoty[k],Ldotz[k])
                Ldot_next = ca.vertcat(Ldotz[k+1],Ldotz[k+1],Ldotz[k+1])

            #Phase dependent Constraints
            if GaitPattern[Nph]=='InitialDouble':
                #Kinematics Constraint
                #   CoM in the Left foot
                #g.append(K_CoM_Left@(CoM_k-ca.DM(PL_init)))
                g.append(K_CoM_Left@(CoM_k-PL_init))
                glb.append(np.full((len(k_CoM_Left),),-np.inf))
                gub.append(k_CoM_Left)
                #   CoM in the Right foot
                #g.append(K_CoM_Right@(CoM_k-ca.DM(PR_init)))
                g.append(K_CoM_Right@(CoM_k-PR_init))
                glb.append(np.full((len(k_CoM_Right),),-np.inf))
                gub.append(k_CoM_Right)
                #Angular Dynamics
                if k<N_K-1:
                    g.append(Ldot_next-Ldot_current-h*(ca.cross((PL_init+np.array([0.11,0.06,0])-CoM_k),FL1_k)+ca.cross((PL_init+np.array([0.11,-0.06,0])-CoM_k),FL2_k)+ca.cross((PL_init+np.array([-0.11,0.06,0])-CoM_k),FL3_k)+ca.cross((PL_init+np.array([-0.11,-0.06,0])-CoM_k),FL4_k)+ca.cross((PR_init+np.array([0.11,0.06,0])-CoM_k),FR1_k)+ca.cross((PR_init+np.array([0.11,-0.06,0])-CoM_k),FR2_k)+ca.cross((PR_init+np.array([-0.11,0.06,0])-CoM_k),FR3_k)+ca.cross((PR_init+np.array([-0.11,-0.06,0])-CoM_k),FR4_k)))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))

                #g.append(ca.cross((PL_init-CoM_k),FL_k)+ca.cross((PR_init-CoM_k),FR_k))
                #glb.append(np.array([-1,-1,-1]))
                #gub.append(np.array([1,1,1]))

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
                StepCnt = np.max((Nph-1),0)//2 #Step Count - Start from zero and negelect phase 0

                #   If LEFT foot is SWING (RIGHT is STATIONARY), Then Right Foot is the SUPPORT FOOT
                if StepCnt == 0:#First Step
                    #print('First Step For Right Support')
                    #Kinematics Constraint
                    #   CoM in the Rigth foot
                    #g.append(ca.if_else(ParaLeftSwingFlag, K_CoM_Right@(CoM_k-ca.DM(PR_init))-ca.DM(k_CoM_Right), np.full((len(k_CoM_Right),),-1)))
                    g.append(ca.if_else(ParaLeftSwingFlag, K_CoM_Right@(CoM_k-PR_init)-ca.DM(k_CoM_Right), np.full((len(k_CoM_Right),),-1)))
                    glb.append(np.full((len(k_CoM_Right),),-np.inf))
                    gub.append(np.full((len(k_CoM_Right),),0))
                    #Angular Dynamics (Right Support)
                    if k<N_K-1:
                        g.append(ca.if_else(ParaLeftSwingFlag, Ldot_next-Ldot_current-h*(ca.cross((PR_init+np.array([0.11,0.06,0])-CoM_k),FR1_k)+ca.cross((PR_init+np.array([0.11,-0.06,0])-CoM_k),FR2_k)+ca.cross((PR_init+np.array([-0.11,0.06,0])-CoM_k),FR3_k)+ca.cross((PR_init+np.array([-0.11,-0.06,0])-CoM_k),FR4_k)), np.array([0,0,0])))
                        glb.append(np.array([0,0,0]))
                        gub.append(np.array([0,0,0]))
                else:
                    print('No implementation yet')

                #   If RIGHT foot is SWING (LEFT is STATIONARY), Then LEFT Foot is the Support FOOT
                if StepCnt == 0:#First Step
                    #print('First Step For Left Support')
                    #Kinematics Constraint
                    #   CoM in the Left foot
                    #g.append(ca.if_else(ParaRightSwingFlag, K_CoM_Left@(CoM_k-ca.DM(PL_init))-ca.DM(k_CoM_Left), np.full((len(k_CoM_Left),),-1)))
                    g.append(ca.if_else(ParaRightSwingFlag, K_CoM_Left@(CoM_k-PL_init)-ca.DM(k_CoM_Left), np.full((len(k_CoM_Left),),-1)))
                    glb.append(np.full((len(k_CoM_Left),),-np.inf))
                    gub.append(np.full((len(k_CoM_Left),),0))
                    #Angular Dynamics (Left Support)
                    if k<N_K-1:
                        g.append(ca.if_else(ParaRightSwingFlag,Ldot_next-Ldot_current-h*(ca.cross((PL_init+np.array([0.11,0.06,0])-CoM_k),FL1_k)+ca.cross((PL_init+np.array([0.11,-0.06,0])-CoM_k),FL2_k)+ca.cross((PL_init+np.array([-0.11,0.06,0])-CoM_k),FL3_k)+ca.cross((PL_init+np.array([-0.11,-0.06,0])-CoM_k),FL4_k)), np.array([0,0,0])))
                        glb.append(np.array([0,0,0]))
                        gub.append(np.array([0,0,0]))
                else:
                    print('No implementation yet')

            elif GaitPattern[Nph]=='DoubleSupport':
                #print('Double Support: Knot',str(k))

                #Kinematic Constraint and Angular Dynamics
                StepCnt = np.max((Nph-1),0)//2 #Step Count - Start from zero and negelect phase 0
                #print('StepCount'+str(StepCnt))
                if StepCnt == 0:#First Step
                    #print('First Step for Double Support')
                    
                    #IF LEFT Foot is SWING (RIGHT FOOT is STATIONARY)
                    #Kinematics Constraint
                    #   CoM in the RIGHT Foot
                    #g.append(ca.if_else(ParaLeftSwingFlag, K_CoM_Right@(CoM_k-ca.DM(PR_init))-ca.DM(k_CoM_Right), np.full((len(k_CoM_Right),),-1)))
                    g.append(ca.if_else(ParaLeftSwingFlag, K_CoM_Right@(CoM_k-PR_init)-ca.DM(k_CoM_Right), np.full((len(k_CoM_Right),),-1)))
                    glb.append(np.full((len(k_CoM_Right),),-np.inf))
                    gub.append(np.full((len(k_CoM_Right),),0))
                    #   CoM in the LEFT foot
                    PL_k = ca.vertcat(px[StepCnt],py[StepCnt],pz[StepCnt])
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
                    #   CoM in the Left foot
                        #g.append(ca.if_else(ParaRightSwingFlag, K_CoM_Left@(CoM_k-ca.DM(PL_init))-ca.DM(k_CoM_Left),np.full((len(k_CoM_Left),),-1)))
                        g.append(ca.if_else(ParaRightSwingFlag, K_CoM_Left@(CoM_k-PL_init)-ca.DM(k_CoM_Left),np.full((len(k_CoM_Left),),-1)))
                        glb.append(np.full((len(k_CoM_Left),),-np.inf))
                        gub.append(np.full((len(k_CoM_Left),),0))
                    #   CoM in the Right foot
                        PR_k = ca.vertcat(px[StepCnt],py[StepCnt],pz[StepCnt])
                        g.append(ca.if_else(ParaRightSwingFlag,K_CoM_Right@(CoM_k-PR_k)-ca.DM(k_CoM_Right),np.full((len(k_CoM_Right),),-1)))
                        glb.append(np.full((len(k_CoM_Right),),-np.inf))
                        gub.append(np.full((len(k_CoM_Right),),0))
                    #Angular Dynamics (Double Support)
                        if k<N_K-1:
                            g.append(ca.if_else(ParaRightSwingFlag,Ldot_next-Ldot_current-h*(ca.cross((PL_init+np.array([0.11,0.06,0])-CoM_k),FL1_k)+ca.cross((PL_init+np.array([0.11,-0.06,0])-CoM_k),FL2_k)+ca.cross((PL_init+np.array([-0.11,0.06,0])-CoM_k),FL3_k)+ca.cross((PL_init+np.array([-0.11,-0.06,0])-CoM_k),FL4_k)+ca.cross((PR_k+np.array([0.11,0.06,0])-CoM_k),FR1_k)+ca.cross((PR_k+np.array([0.11,-0.06,0])-CoM_k),FR2_k)+ca.cross((PR_k+np.array([-0.11,0.06,0])-CoM_k),FR3_k)+ca.cross((PR_k+np.array([-0.11,-0.06,0])-CoM_k),FR4_k)),np.array([0,0,0])))
                            glb.append(np.array([0,0,0]))
                            gub.append(np.array([0,0,0]))
                else:
                    print('No implementation yet')
                
            #Add Cost Terms
            if k < N_K - 1:
                J = J + h*Lx[k]**2 + h*Ly[k]**2 + h*Lz[k]**2 + h*(FL1x[k]/m+FL2x[k]/m+FL3x[k]/m+FL4x[k]/m+FR1x[k]/m+FR2x[k]/m+FR3x[k]/m+FR4x[k]/m)**2 + h*(FL1y[k]/m+FL2y[k]/m+FL3y[k]/m+FL4y[k]/m+FR1y[k]/m+FR2y[k]/m+FR3y[k]/m+FR4y[k]/m)**2 + h*(FL1z[k]/m+FL2z[k]/m+FL3z[k]/m+FL4z[k]/m+FR1z[k]/m+FR2z[k]/m+FR3z[k]/m+FR4z[k]/m - G)**2
            
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

            if k <= N_K - 1 - 1: #N_k - 1 the enumeration of the last knot, -1 the knot before the last knot
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

    #Relative Footstep Constraints
    #print('Relative Footstep Constraints')
    #   For init phase
    g.append(Q_rf_in_lf@(PR_init-PL_init))
    glb.append(np.full((len(q_rf_in_lf),),-np.inf))
    gub.append(q_rf_in_lf)
    #   For other phases
    for PhaseCnt in range(Nphase):
        if GaitPattern[PhaseCnt]=='DoubleSupport':
            StepCnt = np.max((PhaseCnt-1),0)//2 #Step Count - Start from zero and negelect phase 0
            #print('StepCount'+str(StepCnt))
            if StepCnt == 0:#First Step
                    p_next = ca.vertcat(px[StepCnt],py[StepCnt],pz[StepCnt])

                    #If LEFT foot is SWING (RIGHT is STATIONARY), Then LEFT Foot should Stay in the polytpe of the RIGHT FOOT
                    g.append(ca.if_else(ParaLeftSwingFlag,Q_lf_in_rf@(p_next-PR_init)-ca.DM(q_lf_in_rf),np.full((len(q_lf_in_rf),),-1)))
                    glb.append(np.full((len(q_lf_in_rf),),-np.inf))
                    gub.append(np.full((len(q_lf_in_rf),),0))

                    #If RIGHT foot is SWING (LEFT is STATIONARY), Then RIGHT Foot should stay in the polytope of the LEFT Foot
                    g.append(ca.if_else(ParaRightSwingFlag,Q_rf_in_lf@(p_next-PL_init)-ca.DM(q_rf_in_lf),np.full((len(q_rf_in_lf),),-1)))
                    glb.append(np.full((len(q_rf_in_lf),),-np.inf))
                    gub.append(np.full((len(q_rf_in_lf),),0))
            else:
                print('No implementation yet')

    #Align Terminal Footstep location with Terminal x position
    #g.append(px[-1]-x[-1])
    #glb.append(np.array([0]))
    #gub.append(np.array([np.inf]))

    #Switching Time Constraint
    for phase_cnt in range(Nphase):
        if GaitPattern[phase_cnt] == 'InitialDouble':
            g.append(Ts[phase_cnt])
            glb.append(np.array([0.1]))
            gub.append(np.array([0.25]))
        elif GaitPattern[phase_cnt] == 'Swing':
            g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
            glb.append(np.array([0.5]))
            gub.append(np.array([0.9]))
        elif GaitPattern[phase_cnt] == 'DoubleSupport':
            g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
            glb.append(np.array([0.1]))
            gub.append(np.array([0.25]))
    
    #-----------------------------------------------------------------------------------------------------------------------
    #Add Second Level: a Kinematics Check
    if withSecondLevel == True:

        Kine_Nsteps = 8 #Enumeration of the Steps start from 0, so Number of Steps - 1, use mod function to check it is left or right

        #-----------------------------------------------------------------------------------------------------------------------
        #Define Variables and Lower and Upper Bounds
        #   CoM Position, 2 knots per step
        Kine_x = []
        Kine_xlb = []
        Kine_xub = []
        Kine_y = []
        Kine_ylb = []
        Kine_yub = []
        Kine_z = []
        Kine_zlb = []
        Kine_zub = []
        Kine_px = []
        Kine_pxlb = []
        Kine_pxub = []
        Kine_py = []
        Kine_pylb = []
        Kine_pyub = []
        Kine_pz = []
        Kine_pzlb = []
        Kine_pzub = []

        for stepIdx in range(Kine_Nsteps):
            #For CoM state, each phase/step has two knots
            Kine_xtemp = ca.SX.sym('Kine_x'+str(stepIdx+1),2)
            Kine_x.append(Kine_xtemp)
            Kine_xlb.append(np.array([-5,-5]))
            Kine_xub.append(np.array([5,5]))

            Kine_ytemp = ca.SX.sym('Kine_y'+str(stepIdx+1),2)
            Kine_y.append(Kine_ytemp)
            Kine_ylb.append(np.array([-1,-1]))
            Kine_yub.append(np.array([1,1]))

            Kine_ztemp = ca.SX.sym('Kine_z'+str(stepIdx+1),2)
            Kine_z.append(Kine_ztemp)
            Kine_zlb.append(np.array([-3,-3]))
            Kine_zub.append(np.array([3,3]))

            Kine_pxtemp = ca.SX.sym('Kine_px'+str(stepIdx+1))
            Kine_px.append(Kine_pxtemp)
            Kine_pxlb.append(np.array([-5]))
            Kine_pxub.append(np.array([5]))

            Kine_pytemp = ca.SX.sym('Kine_py'+str(stepIdx+1))
            Kine_py.append(Kine_pytemp)
            Kine_pylb.append(np.array([-2]))
            Kine_pyub.append(np.array([2]))

            #   Foot steps are all staying on the ground
            Kine_pztemp = ca.SX.sym('Kine_pz'+str(stepIdx+1))
            Kine_pz.append(Kine_pztemp)
            Kine_pzlb.append(np.array([0]))
            Kine_pzub.append(np.array([0]))

        Kine_x = ca.vertcat(*Kine_x)
        Kine_y = ca.vertcat(*Kine_y)
        Kine_z = ca.vertcat(*Kine_z)
        Kine_px = ca.vertcat(*Kine_px)
        Kine_py = ca.vertcat(*Kine_py)
        Kine_pz = ca.vertcat(*Kine_pz)

        #--------------------------------
        #Update Decision Vars and Bounds
        #--------------------------------
        DecisionVars = ca.vertcat(DecisionVars,Kine_x,Kine_y,Kine_z,Kine_px,Kine_py,Kine_pz) #treat all elements in the list as single variables
        DecisionVarsShape = DecisionVars.shape #get decision variable list shape, for future use

        DecisionVars_lb = np.concatenate((DecisionVars_lb,Kine_xlb,Kine_ylb,Kine_zlb,Kine_pxlb,Kine_pylb,Kine_pzlb),axis=None)
        DecisionVars_ub = np.concatenate((DecisionVars_ub,Kine_xub,Kine_yub,Kine_zub,Kine_pxub,Kine_pyub,Kine_pzub),axis=None)

        #-----------------------------------------------------------------------------------------------------------------------
        #Define Constrains

        #   Initial Condition x-axis (Connect to the First Level)
        g.append(Kine_x[0]-x[-1])
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        #   Initial Condition y-axis (Connect to the First Level)
        g.append(Kine_y[0]-y[-1])
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        #   Initial Condition z-axis (Connect to the First Level)
        g.append(Kine_z[0]-z[-1])
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        #   Terminal Condition
        #   y-axis
        g.append(Kine_y[-1]-y_end)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        #   z-axis
        g.append(Kine_z[-1]-z_end)
        glb.append(np.array([0]))
        gub.append(np.array([0]))

        #   Loop over to have Kinematics Constraint
        for Nph in range(Kine_Nsteps):
            #print('Phase: '+ str(Nph+1))
            Kine_CoM_0 = ca.vertcat(Kine_x[2*Nph],Kine_y[2*Nph],Kine_z[2*Nph])
            #print(Kine_CoM_0)
            Kine_CoM_1 = ca.vertcat(Kine_x[2*Nph+1],Kine_y[2*Nph+1],Kine_z[2*Nph+1])
            #print(Kine_CoM_1)
            if Nph == 0:
                #P_k_current = ca.DM(np.array([px_init,py_init,pz_init]))
                Kine_P_k_current = ca.vertcat(px[-1],py[-1],pz[-1])
                Kine_P_k_next = ca.vertcat(Kine_px[Nph],Kine_py[Nph],Kine_pz[Nph])
            else:
                Kine_P_k_current = ca.vertcat(Kine_px[Nph-1],Kine_py[Nph-1],Kine_pz[Nph-1])
                Kine_P_k_next = ca.vertcat(Kine_px[Nph],Kine_py[Nph],Kine_pz[Nph])
        #print(Kine_P_k_current)
        #print(Kine_P_k_next)

            #Construct Kinematics Constraints
            if Nph%2 == 0: #even number
                #------------------------------------
                #If First Level Swing the Left, then the second level Even Number Phases (the first Phase) Swing the Right -> Left -> Right
                #Left foot in contact for p_current, right foot is going to land as p_next
                #CoM Kinemactis Constraint for the support foot (CoM_0 in *current* contact -> left)
                g.append(ca.if_else(ParaLeftSwingFlag,K_CoM_Left@(Kine_CoM_0-Kine_P_k_current)-ca.DM(k_CoM_Left),np.full((len(k_CoM_Left),),-1)))
                glb.append(np.full((len(k_CoM_Left),),-np.inf))
                gub.append(np.full((len(k_CoM_Left),),0))
                #CoM Kinemactis Constraint for the support foot (CoM_0 in *next* contact -> right)
                g.append(ca.if_else(ParaLeftSwingFlag,K_CoM_Right@(Kine_CoM_0-Kine_P_k_next)-ca.DM(k_CoM_Right),np.full((len(k_CoM_Right),),-1)))
                glb.append(np.full((len(k_CoM_Right),),-np.inf))
                gub.append(np.full((len(k_CoM_Right),),0))
                #CoM Kinemactis Constraint for the swing foot (CoM_1 land but in the *current* contact -> left)
                g.append(ca.if_else(ParaLeftSwingFlag,K_CoM_Left@(Kine_CoM_1-Kine_P_k_current)-ca.DM(k_CoM_Left),np.full((len(k_CoM_Left),),-1)))
                glb.append(np.full((len(k_CoM_Left),),-np.inf))
                gub.append(np.full((len(k_CoM_Left),),0))
                #CoM Kinemactis Constraint for the swing foot (CoM_1 land but in the *next* contact -> right)
                g.append(ca.if_else(ParaLeftSwingFlag,K_CoM_Right@(Kine_CoM_1-Kine_P_k_next)-ca.DM(k_CoM_Right),np.full((len(k_CoM_Right),),-1)))
                glb.append(np.full((len(k_CoM_Right),),-np.inf))
                gub.append(np.full((len(k_CoM_Right),),0))
                #Relative Swing Foot Location (rf in lf)
                g.append(ca.if_else(ParaLeftSwingFlag,Q_rf_in_lf@(Kine_P_k_next-Kine_P_k_current)-ca.DM(q_rf_in_lf),np.full((len(q_rf_in_lf),),-1)))
                glb.append(np.full((len(q_rf_in_lf),),-np.inf))
                gub.append(np.full((len(q_rf_in_lf),),0))
                #-------------------------------------
                #If First Levvel Swing the Right, then the second level Even Number Phases (the first phase) Swing the Left -> Right -> Left
                #Right foot in contact for p_current, left foot is going to land at p_next
                #CoM Kinemactis Constraint for the support foot (CoM_0 in *current* contact -> right)
                g.append(ca.if_else(ParaRightSwingFlag,K_CoM_Right@(Kine_CoM_0-Kine_P_k_current)-ca.DM(k_CoM_Right),np.full((len(k_CoM_Right),),-1)))
                glb.append(np.full((len(k_CoM_Right),),-np.inf))
                gub.append(np.full((len(k_CoM_Right),),0))
                #CoM Kinemactis Constraint for the support foot (CoM_0 in *next* contact -> left)
                g.append(ca.if_else(ParaRightSwingFlag,K_CoM_Left@(Kine_CoM_0-Kine_P_k_next)-ca.DM(k_CoM_Left),np.full((len(k_CoM_Left),),-1)))
                glb.append(np.full((len(k_CoM_Left),),-np.inf))
                gub.append(np.full((len(k_CoM_Left),),0))
                #CoM Kinemactis Constraint for the swing foot (CoM_1 land but in the *current contact* -> right)
                g.append(ca.if_else(ParaRightSwingFlag,K_CoM_Right@(Kine_CoM_1-Kine_P_k_current)-ca.DM(k_CoM_Right),np.full((len(k_CoM_Right),),-1)))
                glb.append(np.full((len(k_CoM_Right),),-np.inf))
                gub.append(np.full((len(k_CoM_Right),),0))
                #CoM Kinemactis Constraint for the swing foot (CoM_1 land but in the *next contact* -> left)
                g.append(ca.if_else(ParaRightSwingFlag,K_CoM_Left@(Kine_CoM_1-Kine_P_k_next)-ca.DM(k_CoM_Left),np.full((len(k_CoM_Left),),-1)))
                glb.append(np.full((len(k_CoM_Left),),-np.inf))
                gub.append(np.full((len(k_CoM_Left),),0))
                #Relative Swing Foot Location (lf in rf)
                g.append(ca.if_else(ParaRightSwingFlag,Q_lf_in_rf@(Kine_P_k_next-Kine_P_k_current)-ca.DM(q_lf_in_rf),np.full((len(q_lf_in_rf),),-1)))
                glb.append(np.full((len(q_lf_in_rf),),-np.inf))
                gub.append(np.full((len(q_lf_in_rf),),0))
            elif Nph%2 ==1: #odd number, right foot in contact for p_current, left foot is going to land at p_next
                #If the first level swings the Left, then the second level for Odd Number Phases (the second phase) swings left -> Right -> Left
                ##Right foot in contact for p_current, left foot is going to land at p_next
                g.append(ca.if_else(ParaLeftSwingFlag,K_CoM_Right@(Kine_CoM_0-Kine_P_k_current)-ca.DM(k_CoM_Right),np.full((len(k_CoM_Right),),-1)))
                glb.append(np.full((len(k_CoM_Right),),-np.inf))
                gub.append(np.full((len(k_CoM_Right),),0))
                #CoM Kinemactis Constraint for the support foot (CoM_0 in *next* contact -> left)
                g.append(ca.if_else(ParaLeftSwingFlag,K_CoM_Left@(Kine_CoM_0-Kine_P_k_next)-ca.DM(k_CoM_Left),np.full((len(k_CoM_Left),),-1)))
                glb.append(np.full((len(k_CoM_Left),),-np.inf))
                gub.append(np.full((len(k_CoM_Left),),0))
                #CoM Kinemactis Constraint for the swing foot (CoM_1 land but in the *current contact* -> right)
                g.append(ca.if_else(ParaLeftSwingFlag,K_CoM_Right@(Kine_CoM_1-Kine_P_k_current)-ca.DM(k_CoM_Right),np.full((len(k_CoM_Right),),-1)))
                glb.append(np.full((len(k_CoM_Right),),-np.inf))
                gub.append(np.full((len(k_CoM_Right),),0))
                #CoM Kinemactis Constraint for the swing foot (CoM_1 land but in the *next contact* -> left)
                g.append(ca.if_else(ParaLeftSwingFlag,K_CoM_Left@(Kine_CoM_1-Kine_P_k_next)-ca.DM(k_CoM_Left),np.full((len(k_CoM_Left),),-1)))
                glb.append(np.full((len(k_CoM_Left),),-np.inf))
                gub.append(np.full((len(k_CoM_Left),),0))
                #Relative Swing Foot Location (lf in rf)
                g.append(ca.if_else(ParaLeftSwingFlag,Q_lf_in_rf@(Kine_P_k_next-Kine_P_k_current)-ca.DM(q_lf_in_rf),np.full((len(q_lf_in_rf),),-1)))
                glb.append(np.full((len(q_lf_in_rf),),-np.inf))
                gub.append(np.full((len(q_lf_in_rf),),0))
                
                #If the first level swings the Right, then the second level for Odd Number Phases (the second phase) swings Right -> Left -> Right
                #Left foot in contact for p_current, right foot is going to land as p_next
                #CoM Kinemactis Constraint for the support foot (CoM_0 in *current* contact -> left)
                g.append(ca.if_else(ParaRightSwingFlag,K_CoM_Left@(Kine_CoM_0-Kine_P_k_current)-ca.DM(k_CoM_Left),np.full((len(k_CoM_Left),),-1)))
                glb.append(np.full((len(k_CoM_Left),),-np.inf))
                gub.append(np.full((len(k_CoM_Left),),0))
                #CoM Kinemactis Constraint for the support foot (CoM_0 in *next* contact -> right)
                g.append(ca.if_else(ParaRightSwingFlag,K_CoM_Right@(Kine_CoM_0-Kine_P_k_next)-ca.DM(k_CoM_Right),np.full((len(k_CoM_Right),),-1)))
                glb.append(np.full((len(k_CoM_Right),),-np.inf))
                gub.append(np.full((len(k_CoM_Right),),0))
                #CoM Kinemactis Constraint for the swing foot (CoM_1 land but in the *current* contact -> left)
                g.append(ca.if_else(ParaRightSwingFlag,K_CoM_Left@(Kine_CoM_1-Kine_P_k_current)-ca.DM(k_CoM_Left),np.full((len(k_CoM_Left),),-1)))
                glb.append(np.full((len(k_CoM_Left),),-np.inf))
                gub.append(np.full((len(k_CoM_Left),),0))
                #CoM Kinemactis Constraint for the swing foot (CoM_1 land but in the *next* contact -> right)
                g.append(ca.if_else(ParaRightSwingFlag,K_CoM_Right@(Kine_CoM_1-Kine_P_k_next)-ca.DM(k_CoM_Right),np.full((len(k_CoM_Right),),-1)))
                glb.append(np.full((len(k_CoM_Right),),-np.inf))
                gub.append(np.full((len(k_CoM_Right),),0))
                #Relative Swing Foot Location (rf in lf)
                g.append(ca.if_else(ParaRightSwingFlag,Q_rf_in_lf@(Kine_P_k_next-Kine_P_k_current)-ca.DM(q_rf_in_lf),np.full((len(q_rf_in_lf),),-1)))
                glb.append(np.full((len(q_rf_in_lf),),-np.inf))
                gub.append(np.full((len(q_rf_in_lf),),0))

    #-----------------------------------------------------------------------------------------------------------------------
    #   reshape all constraints
    g = ca.vertcat(*g)
    glb = np.concatenate(glb)
    gub = np.concatenate(gub)
    #-----------------------------------------------------------------------------------------------------------------------
    #Add Terminal Cost 
    if withSecondLevel == False:
        J = J + 100*(x[-1]-x_end)**2
    elif withSecondLevel == True:
        J = J + 100*(Kine_x[-1]-x_end)**2
    #-----------------------------------------------------------------------------------------------------------------------
    #Generate Initial Guess
    #   Random Initial Guess
    #       Shuffle the Random Seed Generator
    np.random.seed()
    DecisionVars_init = DecisionVars_lb + np.multiply(np.random.rand(DecisionVarsShape[0],DecisionVarsShape[1]).flatten(),(DecisionVars_ub-DecisionVars_lb))#   Fixed Value Initial Guess
    #DecisionVars_init=np.ones((DecisionVarsShape[0],DecisionVarsShape[1]))
    # x_init = np.array([[1.5]*N_K])
    #-----------------------------------------------------------------------------------------------------------------------

    #-----------------------------------------------------------------------------------------------------------------------
    #Build Solver
    prob = {'x': DecisionVars, 'f': J, 'g': g, 'p': paras}
    solver = ca.nlpsol('solver', 'ipopt', prob)

    #-----------------------------------------------------------------------------------------------------------------------
    #Get Variable Index
    #Old Code
    #x_index = (0,Nk_Local*Nphase) #First set of variables start counting from 0
    #y_index = (x_index[1]+1,x_index[1]+Nk_Local*Nphase+1)
    #z_index = (y_index[1]+1,y_index[1]+Nk_Local*Nphase+1)
    #xdot_index = (z_index[1]+1,z_index[1]+Nk_Local*Nphase+1)
    #ydot_index = (xdot_index[1]+1,xdot_index[1]+Nk_Local*Nphase+1)
    #zdot_index = (ydot_index[1]+1,ydot_index[1]+Nk_Local*Nphase+1)
    #Lx_index = (zdot_index[1]+1,zdot_index[1]+Nk_Local*Nphase+1)
    #Ly_index = (Lx_index[1]+1,Lx_index[1]+Nk_Local*Nphase+1)
    #Lz_index = (Ly_index[1]+1,Ly_index[1]+Nk_Local*Nphase+1)
    #Ldotx_index = (Lz_index[1]+1,Lz_index[1]+Nk_Local*Nphase+1)
    #Ldoty_index = (Ldotx_index[1]+1,Ldotx_index[1]+Nk_Local*Nphase+1)
    #Ldotz_index = (Ldoty_index[1]+1,Ldoty_index[1]+Nk_Local*Nphase+1)
    #FL1x_index = (Ldotz_index[1]+1,Ldotz_index[1]+Nk_Local*Nphase+1)
    #FL1y_index = (FL1x_index[1]+1,FL1x_index[1]+Nk_Local*Nphase+1)
    #FL1z_index = (FL1y_index[1]+1,FL1y_index[1]+Nk_Local*Nphase+1)
    #FL2x_index = (FL1z_index[1]+1,FL1z_index[1]+Nk_Local*Nphase+1)
    #FL2y_index = (FL2x_index[1]+1,FL2x_index[1]+Nk_Local*Nphase+1)
    #FL2z_index = (FL2y_index[1]+1,FL2y_index[1]+Nk_Local*Nphase+1)
    #FL3x_index = (FL2z_index[1]+1,FL2z_index[1]+Nk_Local*Nphase+1)
    #FL3y_index = (FL3x_index[1]+1,FL3x_index[1]+Nk_Local*Nphase+1)
    #FL3z_index = (FL3y_index[1]+1,FL3y_index[1]+Nk_Local*Nphase+1)
    #FL4x_index = (FL3z_index[1]+1,FL3z_index[1]+Nk_Local*Nphase+1)
    #FL4y_index = (FL4x_index[1]+1,FL4x_index[1]+Nk_Local*Nphase+1)
    #FL4z_index = (FL4y_index[1]+1,FL4y_index[1]+Nk_Local*Nphase+1)
    #FR1x_index = (FL4z_index[1]+1,FL4z_index[1]+Nk_Local*Nphase+1)
    #FR1y_index = (FR1x_index[1]+1,FR1x_index[1]+Nk_Local*Nphase+1)
    #FR1z_index = (FR1y_index[1]+1,FR1y_index[1]+Nk_Local*Nphase+1)
    #FR2x_index = (FR1z_index[1]+1,FR1z_index[1]+Nk_Local*Nphase+1)
    #FR2y_index = (FR2x_index[1]+1,FR2x_index[1]+Nk_Local*Nphase+1)
    #FR2z_index = (FR2y_index[1]+1,FR2y_index[1]+Nk_Local*Nphase+1)
    #FR3x_index = (FR2z_index[1]+1,FR2z_index[1]+Nk_Local*Nphase+1)
    #FR3y_index = (FR3x_index[1]+1,FR3x_index[1]+Nk_Local*Nphase+1)
    #FR3z_index = (FR3y_index[1]+1,FR3y_index[1]+Nk_Local*Nphase+1)
    #FR4x_index = (FR3z_index[1]+1,FR3z_index[1]+Nk_Local*Nphase+1)
    #FR4y_index = (FR4x_index[1]+1,FR4x_index[1]+Nk_Local*Nphase+1)
    #FR4z_index = (FR4y_index[1]+1,FR4y_index[1]+Nk_Local*Nphase+1)
    #px_index = (FR4z_index[1]+1,FR4z_index[1]+Nstep)
    #py_index = (px_index[1]+1,px_index[1]+Nstep)
    #pz_index = (py_index[1]+1,py_index[1]+Nstep)
    #Ts_index = (pz_index[1]+1,pz_index[1]+Nphase)

    #Get Variable Index - !!!This is the pure Index, when try to get the array using other routines, we need to add "+1" at the last index due to Python indexing conventions
    x_index = (0,N_K-1) #First set of variables start counting from 0, The enumeration of the last knot is N_K-1
    print(DecisionVars[x_index[0]:x_index[1]+1])
    y_index = (x_index[1]+1,x_index[1]+N_K)
    print(DecisionVars[y_index[0]:y_index[1]+1])
    z_index = (y_index[1]+1,y_index[1]+N_K)
    print(DecisionVars[z_index[0]:z_index[1]+1])
    xdot_index = (z_index[1]+1,z_index[1]+N_K)
    print(DecisionVars[xdot_index[0]:xdot_index[1]+1])
    ydot_index = (xdot_index[1]+1,xdot_index[1]+N_K)
    print(DecisionVars[ydot_index[0]:ydot_index[1]+1])
    zdot_index = (ydot_index[1]+1,ydot_index[1]+N_K)
    print(DecisionVars[zdot_index[0]:zdot_index[1]+1])
    Lx_index = (zdot_index[1]+1,zdot_index[1]+N_K)
    print(DecisionVars[Lx_index[0]:Lx_index[1]+1])
    Ly_index = (Lx_index[1]+1,Lx_index[1]+N_K)
    print(DecisionVars[Ly_index[0]:Ly_index[1]+1])
    Lz_index = (Ly_index[1]+1,Ly_index[1]+N_K)
    print(DecisionVars[Lz_index[0]:Lz_index[1]+1])
    Ldotx_index = (Lz_index[1]+1,Lz_index[1]+N_K)
    print(DecisionVars[Ldotx_index[0]:Ldotx_index[1]+1])
    Ldoty_index = (Ldotx_index[1]+1,Ldotx_index[1]+N_K)
    print(DecisionVars[Ldoty_index[0]:Ldoty_index[1]+1])
    Ldotz_index = (Ldoty_index[1]+1,Ldoty_index[1]+N_K)
    print(DecisionVars[Ldotz_index[0]:Ldotz_index[1]+1])
    FL1x_index = (Ldotz_index[1]+1,Ldotz_index[1]+N_K)
    print(DecisionVars[FL1x_index[0]:FL1x_index[1]+1])
    FL1y_index = (FL1x_index[1]+1,FL1x_index[1]+N_K)
    print(DecisionVars[FL1y_index[0]:FL1y_index[1]+1])
    FL1z_index = (FL1y_index[1]+1,FL1y_index[1]+N_K)
    print(DecisionVars[FL1z_index[0]:FL1z_index[1]+1])
    FL2x_index = (FL1z_index[1]+1,FL1z_index[1]+N_K)
    print(DecisionVars[FL2x_index[0]:FL2x_index[1]+1])
    FL2y_index = (FL2x_index[1]+1,FL2x_index[1]+N_K)
    print(DecisionVars[FL2y_index[0]:FL2y_index[1]+1])
    FL2z_index = (FL2y_index[1]+1,FL2y_index[1]+N_K)
    print(DecisionVars[FL2z_index[0]:FL2z_index[1]+1])
    FL3x_index = (FL2z_index[1]+1,FL2z_index[1]+N_K)
    print(DecisionVars[FL3x_index[0]:FL3x_index[1]+1])
    FL3y_index = (FL3x_index[1]+1,FL3x_index[1]+N_K)
    print(DecisionVars[FL3y_index[0]:FL3y_index[1]+1])
    FL3z_index = (FL3y_index[1]+1,FL3y_index[1]+N_K)
    print(DecisionVars[FL3z_index[0]:FL3z_index[1]+1])
    FL4x_index = (FL3z_index[1]+1,FL3z_index[1]+N_K)
    print(DecisionVars[FL4x_index[0]:FL4x_index[1]+1])
    FL4y_index = (FL4x_index[1]+1,FL4x_index[1]+N_K)
    print(DecisionVars[FL4y_index[0]:FL4y_index[1]+1])
    FL4z_index = (FL4y_index[1]+1,FL4y_index[1]+N_K)
    print(DecisionVars[FL4z_index[0]:FL4z_index[1]+1])
    FR1x_index = (FL4z_index[1]+1,FL4z_index[1]+N_K)
    print(DecisionVars[FR1x_index[0]:FR1x_index[1]+1])
    FR1y_index = (FR1x_index[1]+1,FR1x_index[1]+N_K)
    print(DecisionVars[FR1y_index[0]:FR1y_index[1]+1])
    FR1z_index = (FR1y_index[1]+1,FR1y_index[1]+N_K)
    print(DecisionVars[FR1z_index[0]:FR1z_index[1]+1])
    FR2x_index = (FR1z_index[1]+1,FR1z_index[1]+N_K)
    print(DecisionVars[FR2x_index[0]:FR2x_index[1]+1])
    FR2y_index = (FR2x_index[1]+1,FR2x_index[1]+N_K)
    print(DecisionVars[FR2y_index[0]:FR2y_index[1]+1])
    FR2z_index = (FR2y_index[1]+1,FR2y_index[1]+N_K)
    print(DecisionVars[FR2z_index[0]:FR2z_index[1]+1])
    FR3x_index = (FR2z_index[1]+1,FR2z_index[1]+N_K)
    print(DecisionVars[FR3x_index[0]:FR3x_index[1]+1])
    FR3y_index = (FR3x_index[1]+1,FR3x_index[1]+N_K)
    print(DecisionVars[FR3y_index[0]:FR3y_index[1]+1])
    FR3z_index = (FR3y_index[1]+1,FR3y_index[1]+N_K)
    print(DecisionVars[FR3z_index[0]:FR3z_index[1]+1])
    FR4x_index = (FR3z_index[1]+1,FR3z_index[1]+N_K)
    print(DecisionVars[FR4x_index[0]:FR4x_index[1]+1])
    FR4y_index = (FR4x_index[1]+1,FR4x_index[1]+N_K)
    print(DecisionVars[FR4y_index[0]:FR4y_index[1]+1])
    FR4z_index = (FR4y_index[1]+1,FR4y_index[1]+N_K)
    print(DecisionVars[FR4z_index[0]:FR4z_index[1]+1])
    px_index = (FR4z_index[1]+1,FR4z_index[1]+Nstep)
    print(DecisionVars[px_index[0]:px_index[1]+1])
    py_index = (px_index[1]+1,px_index[1]+Nstep)
    print(DecisionVars[py_index[0]:py_index[1]+1])
    pz_index = (py_index[1]+1,py_index[1]+Nstep)
    print(DecisionVars[pz_index[0]:pz_index[1]+1])
    Ts_index = (pz_index[1]+1,pz_index[1]+Nphase)
    print(DecisionVars[Ts_index[0]:Ts_index[1]+1])

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

    if withSecondLevel == True:
        #Get Updated variable index
        Kine_x_index = (Ts_index[1]+1,Ts_index[1]+Kine_Nsteps*2)
        Kine_y_index = (Kine_x_index[1]+1,Kine_x_index[1]+Kine_Nsteps*2)
        Kine_z_index = (Kine_y_index[1]+1,Kine_y_index[1]+Kine_Nsteps*2)
        Kine_px_index = (Kine_z_index[1]+1,Kine_z_index[1]+Kine_Nsteps)
        Kine_py_index = (Kine_px_index[1]+1,Kine_px_index[1]+Kine_Nsteps)
        Kine_pz_index = (Kine_py_index[1]+1,Kine_py_index[1]+Kine_Nsteps)
        #Update var_index dictionary
        var_index.update({"Kine_x":Kine_x_index})
        var_index.update({"Kine_y":Kine_y_index})
        var_index.update({"Kine_z":Kine_z_index})
        var_index.update({"Kine_px":Kine_px_index})
        var_index.update({"Kine_py":Kine_py_index})
        var_index.update({"Kine_pz":Kine_pz_index})

    print(J)

    return solver, DecisionVars_init, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index

def PlotNLPStep(x_opt = None, fig=None, var_index=None, PL_init = None, PR_init = None, LeftSwing = None, RightSwing = None):
    #-----------------------------------------------------------------------------------------------------------------------
    #Plot Result
    if fig==None:
        fig=plt.figure()
    
    ax = Axes3D(fig)
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
        
def CostComputation(Nphase=3,Nk_Local=5,x_opt=None,var_index=None,G = 9.80665,m=95):
    cost_val=0

    #parameter setup
    N_K = Nk_Local*Nphase + 1
    #Time Span Setup
    tau_upper_limit = 1
    tauStepLength = tau_upper_limit/(N_K-1) #Get the interval length, total number of knots - 1

    #Retrieve Compputation results
    Ts_res = x_opt[var_index["Ts"][0]:var_index["Ts"][1]+1]
    Ts_res = np.array(Ts_res)
    print('Ts_res: ',Ts_res)
    x_res = x_opt[var_index["x"][0]:var_index["x"][1]+1]
    x_res = np.array(x_res)
    y_res = x_opt[var_index["y"][0]:var_index["y"][1]+1]
    y_res = np.array(y_res)
    z_res = x_opt[var_index["z"][0]:var_index["z"][1]+1]
    z_res = np.array(z_res)
    Lx_res = x_opt[var_index["Lx"][0]:var_index["Lx"][1]+1]
    Lx_res = np.array(Lx_res)
    Ly_res = x_opt[var_index["Ly"][0]:var_index["Ly"][1]+1]
    Ly_res = np.array(Ly_res)
    Lz_res = x_opt[var_index["Lz"][0]:var_index["Lz"][1]+1]
    Lz_res = np.array(Lz_res)
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

    #Loop over all Phases (Knots)
    for Nph in range(Nphase):
        
        #Decide Number of Knots
        if Nph == Nphase-1:  #The last Knot belongs to the Last Phase
            Nk_ThisPhase = Nk_Local+1
        else:
            Nk_ThisPhase = Nk_Local

        #Decide Time Vector
        if Nph == 0: #first phase
            h = tauStepLength*Nphase*(Ts_res[Nph]-0)
        else: #other phases
            h = tauStepLength*Nphase*(Ts_res[Nph]-Ts_res[Nph-1])

        for Local_k_Count in range(Nk_ThisPhase):
            #Get knot number across the entire time line
            k = Nph*Nk_Local + Local_k_Count
            #print(k)

            #Add Cost Terms
            if k < N_K - 1:
                cost_val = cost_val + h*Lx_res[k]**2 + h*Ly_res[k]**2 + h*Lz_res[k]**2 + h*(FL1x_res[k]/m+FL2x_res[k]/m+FL3x_res[k]/m+FL4x_res[k]/m+FR1x_res[k]/m+FR2x_res[k]/m+FR3x_res[k]/m+FR4x_res[k]/m)**2 + h*(FL1y_res[k]/m+FL2y_res[k]/m+FL3y_res[k]/m+FL4y_res[k]/m+FR1y_res[k]/m+FR2y_res[k]/m+FR3y_res[k]/m+FR4y_res[k]/m)**2 + h*(FL1z_res[k]/m+FL2z_res[k]/m+FL3z_res[k]/m+FL4z_res[k]/m+FR1z_res[k]/m+FR2z_res[k]/m+FR3z_res[k]/m+FR4z_res[k]/m - G)**2

    print("Cost Value: ",cost_val)

    return cost_val



    # res = solver(x0=DecisionVars_init, p = [1,0], lbx=DecisionVars_lb, ubx=DecisionVars_ub, lbg=glb, ubg=gub)
    # x_opt = res['x']
    # x_opt = x_opt.full().flatten()
    # #print('x_opt: ', x_opt)
    # #print(len(np.array(x_opt).flatten()))
    # #-----------------------------------------------------------------------------------------------------------------------
    # #Extract Results
    # #Result Extraction
    # x_index = (0,Nk_Local*Nphase) #First set of variables start counting from 0
    # x_res = x_opt[x_index[0]:x_index[1]+1]
    # x_res = np.array(x_res)
    # print('x_res: ',x_res)
    # y_index = (x_index[1]+1,x_index[1]+Nk_Local*Nphase+1)
    # y_res = x_opt[y_index[0]:y_index[1]+1]
    # y_res = np.array(y_res)
    # print('y_res: ',y_res)
    # z_index = (y_index[1]+1,y_index[1]+Nk_Local*Nphase+1)
    # z_res = x_opt[z_index[0]:z_index[1]+1]
    # z_res = np.array(z_res)
    # print('z_res: ',z_res)
    # xdot_index = (z_index[1]+1,z_index[1]+Nk_Local*Nphase+1)
    # xdot_res = x_opt[xdot_index[0]:xdot_index[1]+1]
    # xdot_res = np.array(xdot_res)
    # print('xdot_res: ',xdot_res)
    # ydot_index = (xdot_index[1]+1,xdot_index[1]+Nk_Local*Nphase+1)
    # ydot_res = x_opt[ydot_index[0]:ydot_index[1]+1]
    # ydot_res = np.array(ydot_res)
    # print('ydot_res: ',ydot_res)
    # zdot_index = (ydot_index[1]+1,ydot_index[1]+Nk_Local*Nphase+1)
    # zdot_res = x_opt[zdot_index[0]:zdot_index[1]+1]
    # zdot_res = np.array(zdot_res)
    # print('zdot_res: ',zdot_res)
    # Lx_index = (zdot_index[1]+1,zdot_index[1]+Nk_Local*Nphase+1)
    # Lx_res = x_opt[Lx_index[0]:Lx_index[1]+1]
    # Lx_res = np.array(Lx_res)
    # print('Lx_res: ',Lx_res)
    # Ly_index = (Lx_index[1]+1,Lx_index[1]+Nk_Local*Nphase+1)
    # Ly_res = x_opt[Ly_index[0]:Ly_index[1]+1]
    # Ly_res = np.array(Ly_res)
    # print('Ly_res: ',Ly_res)
    # Lz_index = (Ly_index[1]+1,Ly_index[1]+Nk_Local*Nphase+1)
    # Lz_res = x_opt[Lz_index[0]:Lz_index[1]+1]
    # Lz_res = np.array(Lz_res)
    # print('Lz_res: ',Lz_res)
    # Ldotx_index = (Lz_index[1]+1,Lz_index[1]+Nk_Local*Nphase+1)
    # Ldotx_res = x_opt[Ldotx_index[0]:Ldotx_index[1]+1]
    # Ldotx_res = np.array(Ldotx_res)
    # print('Ldotx_res: ',Ldotx_res)
    # Ldoty_index = (Ldotx_index[1]+1,Ldotx_index[1]+Nk_Local*Nphase+1)
    # Ldoty_res = x_opt[Ldoty_index[0]:Ldoty_index[1]+1]
    # Ldoty_res = np.array(Ldoty_res)
    # print('Ldoty_res: ',Ldoty_res)
    # Ldotz_index = (Ldoty_index[1]+1,Ldoty_index[1]+Nk_Local*Nphase+1)
    # Ldotz_res = x_opt[Ldotz_index[0]:Ldotz_index[1]+1]
    # Ldotz_res = np.array(Ldotz_res)
    # print('Ldotz_res: ',Ldotz_res)
    # FL1x_index = (Ldotz_index[1]+1,Ldotz_index[1]+Nk_Local*Nphase+1)
    # FL1x_res = x_opt[FL1x_index[0]:FL1x_index[1]+1]
    # FL1x_res = np.array(FL1x_res)
    # print('FL1x_res: ',FL1x_res)
    # FL1y_index = (FL1x_index[1]+1,FL1x_index[1]+Nk_Local*Nphase+1)
    # FL1y_res = x_opt[FL1y_index[0]:FL1y_index[1]+1]
    # FL1y_res = np.array(FL1y_res)
    # print('FL1y_res: ',FL1y_res)
    # FL1z_index = (FL1y_index[1]+1,FL1y_index[1]+Nk_Local*Nphase+1)
    # FL1z_res = x_opt[FL1z_index[0]:FL1z_index[1]+1]
    # FL1z_res = np.array(FL1z_res)
    # print('FL1z_res: ',FL1z_res)
    # FL2x_index = (FL1z_index[1]+1,FL1z_index[1]+Nk_Local*Nphase+1)
    # FL2x_res = x_opt[FL2x_index[0]:FL2x_index[1]+1]
    # FL2x_res = np.array(FL2x_res)
    # print('FL2x_res: ',FL2x_res)
    # FL2y_index = (FL2x_index[1]+1,FL2x_index[1]+Nk_Local*Nphase+1)
    # FL2y_res = x_opt[FL2y_index[0]:FL2y_index[1]+1]
    # FL2y_res = np.array(FL2y_res)
    # print('FL2y_res: ',FL2y_res)
    # FL2z_index = (FL2y_index[1]+1,FL2y_index[1]+Nk_Local*Nphase+1)
    # FL2z_res = x_opt[FL2z_index[0]:FL2z_index[1]+1]
    # FL2z_res = np.array(FL2z_res)
    # print('FL2z_res: ',FL2z_res)
    # FL3x_index = (FL2z_index[1]+1,FL2z_index[1]+Nk_Local*Nphase+1)
    # FL3x_res = x_opt[FL3x_index[0]:FL3x_index[1]+1]
    # FL3x_res = np.array(FL3x_res)
    # print('FL3x_res: ',FL3x_res)
    # FL3y_index = (FL3x_index[1]+1,FL3x_index[1]+Nk_Local*Nphase+1)
    # FL3y_res = x_opt[FL3y_index[0]:FL3y_index[1]+1]
    # FL3y_res = np.array(FL3y_res)
    # print('FL3y_res: ',FL3y_res)
    # FL3z_index = (FL3y_index[1]+1,FL3y_index[1]+Nk_Local*Nphase+1)
    # FL3z_res = x_opt[FL3z_index[0]:FL3z_index[1]+1]
    # FL3z_res = np.array(FL3z_res)
    # print('FL3z_res: ',FL3z_res)
    # FL4x_index = (FL3z_index[1]+1,FL3z_index[1]+Nk_Local*Nphase+1)
    # FL4x_res = x_opt[FL4x_index[0]:FL4x_index[1]+1]
    # FL4x_res = np.array(FL4x_res)
    # print('FL4x_res: ',FL4x_res)
    # FL4y_index = (FL4x_index[1]+1,FL4x_index[1]+Nk_Local*Nphase+1)
    # FL4y_res = x_opt[FL4y_index[0]:FL4y_index[1]+1]
    # FL4y_res = np.array(FL4y_res)
    # print('FL4y_res: ',FL4y_res)
    # FL4z_index = (FL4y_index[1]+1,FL4y_index[1]+Nk_Local*Nphase+1)
    # FL4z_res = x_opt[FL4z_index[0]:FL4z_index[1]+1]
    # FL4z_res = np.array(FL4z_res)
    # print('FL4z_res: ',FL4z_res)
    # FR1x_index = (FL4z_index[1]+1,FL4z_index[1]+Nk_Local*Nphase+1)
    # FR1x_res = x_opt[FR1x_index[0]:FR1x_index[1]+1]
    # FR1x_res = np.array(FR1x_res)
    # print('FR1x_res: ',FR1x_res)
    # FR1y_index = (FR1x_index[1]+1,FR1x_index[1]+Nk_Local*Nphase+1)
    # FR1y_res = x_opt[FR1y_index[0]:FR1y_index[1]+1]
    # FR1y_res = np.array(FR1y_res)
    # print('FR1y_res: ',FR1y_res)
    # FR1z_index = (FR1y_index[1]+1,FR1y_index[1]+Nk_Local*Nphase+1)
    # FR1z_res = x_opt[FR1z_index[0]:FR1z_index[1]+1]
    # FR1z_res = np.array(FR1z_res)
    # print('FR1z_res: ',FR1z_res)
    # FR2x_index = (FR1z_index[1]+1,FR1z_index[1]+Nk_Local*Nphase+1)
    # FR2x_res = x_opt[FR2x_index[0]:FR2x_index[1]+1]
    # FR2x_res = np.array(FR2x_res)
    # print('FR2x_res: ',FR2x_res)
    # FR2y_index = (FR2x_index[1]+1,FR2x_index[1]+Nk_Local*Nphase+1)
    # FR2y_res = x_opt[FR2y_index[0]:FR2y_index[1]+1]
    # FR2y_res = np.array(FR2y_res)
    # print('FR2y_res: ',FR2y_res)
    # FR2z_index = (FR2y_index[1]+1,FR2y_index[1]+Nk_Local*Nphase+1)
    # FR2z_res = x_opt[FR2z_index[0]:FR2z_index[1]+1]
    # FR2z_res = np.array(FR2z_res)
    # print('FR2z_res: ',FR2z_res)
    # FR3x_index = (FR2z_index[1]+1,FR2z_index[1]+Nk_Local*Nphase+1)
    # FR3x_res = x_opt[FR3x_index[0]:FR3x_index[1]+1]
    # FR3x_res = np.array(FR3x_res)
    # print('FR3x_res: ',FR3x_res)
    # FR3y_index = (FR3x_index[1]+1,FR3x_index[1]+Nk_Local*Nphase+1)
    # FR3y_res = x_opt[FR3y_index[0]:FR3y_index[1]+1]
    # FR3y_res = np.array(FR3y_res)
    # print('FR3y_res: ',FR3y_res)
    # FR3z_index = (FR3y_index[1]+1,FR3y_index[1]+Nk_Local*Nphase+1)
    # FR3z_res = x_opt[FR3z_index[0]:FR3z_index[1]+1]
    # FR3z_res = np.array(FR3z_res)
    # print('FR3z_res: ',FR3z_res)
    # FR4x_index = (FR3z_index[1]+1,FR3z_index[1]+Nk_Local*Nphase+1)
    # FR4x_res = x_opt[FR4x_index[0]:FR4x_index[1]+1]
    # FR4x_res = np.array(FR4x_res)
    # print('FR4x_res: ',FR4x_res)
    # FR4y_index = (FR4x_index[1]+1,FR4x_index[1]+Nk_Local*Nphase+1)
    # FR4y_res = x_opt[FR4y_index[0]:FR4y_index[1]+1]
    # FR4y_res = np.array(FR4y_res)
    # print('FR4y_res: ',FR4y_res)
    # FR4z_index = (FR4y_index[1]+1,FR4y_index[1]+Nk_Local*Nphase+1)
    # FR4z_res = x_opt[FR4z_index[0]:FR4z_index[1]+1]
    # FR4z_res = np.array(FR4z_res)
    # print('FR4z_res: ',FR4z_res)
    # px_index = (FR4z_index[1]+1,FR4z_index[1]+Nstep)
    # px_res = x_opt[px_index[0]:px_index[1]+1]
    # px_res = np.array(px_res)
    # print('px_res: ',px_res)
    # py_index = (px_index[1]+1,px_index[1]+Nstep)
    # py_res = x_opt[py_index[0]:py_index[1]+1]
    # py_res = np.array(py_res)
    # print('py_res: ',py_res)
    # pz_index = (py_index[1]+1,py_index[1]+Nstep)
    # pz_res = x_opt[pz_index[0]:pz_index[1]+1]
    # pz_res = np.array(pz_res)
    # print('pz_res: ',pz_res)
    # Ts_index = (pz_index[1]+1,pz_index[1]+Nphase)
    # Ts_res = x_opt[pz_index[0]:Ts_index[1]+1]
    # Ts_res = np.array(Ts_res)
    # print('Ts_res: ',Ts_res)

    # #-----------------------------------------------------------------------------------------------------------------------
    # #Plot Result
    # fig=plt.figure()
    # ax = Axes3D(fig)
    # #ax = fig.add_subplot(111, projection="3d")

    # #ax.plot3D(x_res,y_res,z_res,color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)
    # ax.plot3D(x_res,y_res,z_res,color='green', linestyle='dashed', linewidth=2, markersize=12)
    # ax.set_xlim3d(0, 1)
    # ax.set_ylim3d(-0.5,0.5)
    # ax.set_zlim3d(0,0.8)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # #Initial Footstep Locations
    # ax.scatter(PLx_init, PLy_init, PLz_init, c='r', marker='o', linewidth = 10) 
    # ax.scatter(PRx_init, PRy_init, PRz_init, c='b', marker='o', linewidth = 10) 

    # for PhaseCnt in range(Nphase):
    #     if GaitPattern[PhaseCnt]=='DoubleSupport':
    #         StepCnt = np.max((PhaseCnt-1),0)//2 #Step Count - Start from zero and negelect phase 0
    #         #print('StepCount'+str(StepCnt))
    #         #if StepCnt == 0:#First Step
    #         #        p_next = ca.vertcat(px[StepCnt],py[StepCnt],pz[StepCnt])

    #         if GaitPattern[PhaseCnt-1]=='Swing':
    #             if LeftSwing == 1:
    #                 StepColor = 'r'
    #             if RightSwing == 1:
    #                 StepColor = 'b'
    #         else:
    #             print('No implementation yet')
    #         ax.scatter(px_res[StepCnt], py_res[StepCnt], pz_res[StepCnt], c=StepColor, marker='o', linewidth = 10) 

    # ax.view_init(elev=8.776933438381377, azim=-99.32358055821186)
    # plt.show()

    # plt.plot(FR2z_res[0:-1])
    # plt.show()





    #thisdict = {
    #    "array": [1,2,3]
    #}

    #return thisdict["array"]