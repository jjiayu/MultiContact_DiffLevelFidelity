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
#-----------------------------------------------------------------------------------------------------------------------
#Decide Parameters
#   Gait Pattern, Each action is followed up by a double support phase
#GaitPattern = ['InitialDouble']
GaitPattern = ['InitialDouble','LeftSupport','DoubleSupport'] #,'RightSupport','DoubleSupport','LeftSupport','DoubleSupport'
#GaitPattern = ['InitialDouble','RightSupport','DoubleSupport']
#GaitPattern = ['InitialDouble','LeftSupport','DoubleSupport','RightSupport','DoubleSupport','LeftSupport','DoubleSupport']
#   Number of Phases
Nphase = len(GaitPattern)
#   Number of Steps
Nstep = (Nphase-1)//2 #Integer Division to get number of steps from Number of Phases
#print(Nstep)
#Nstep = 1 #Enumeration of the Steps start from 0, so Number of Steps - 1, use mod function to check it is left or right
#   Number of Phases, each step is associatated with a single support phase and a double support phase, along with a double support phase appended at the beginning
#N_ph = Nstep*2 + 1
#   Number of Knots per Phase - how many intervals do we have for a single phase; 10 intervals need 11 knots/ticks
Nk_Local= 5
#Nk_DoubleLeg = 10
#   Compute Number of total knots/ticks, but the enumeration start from 0 to N_K-1
N_K = Nk_Local*Nphase + 1 #+1 the last knots to finalize the plan
#N_K = Nk_Local + Nstep*(2*Nk_Local) + 1 #Double Support (Initial Phase) + A step (A single Support Phase and a Double Support hase) + 1 (Knot of Time 0)
#N_K = Nk_DoubleLeg + Nstep*(Nk_Local+Nk_Local) + 1 #Double Support (Initial Phase) + A step (A single Support Phase and a Double Support hase) + 1 (Knot of Time 0)
#N_K = Nstep*(Nk_swing+Nk_double) + 1
#   Duration of a Phase
T_SingleLeg = 0.2 #0.8 to 1.2 is a nominal number (See CROC)
T_DoubleLeg = 0.6 #0.2 is typycal number of dynamic case, if we start from null velocity we should change to 
#   Time Discretization
h_SingleLeg = T_SingleLeg/Nk_Local
h_DoubleLeg = T_DoubleLeg/Nk_Local
#   Robot mass
m = 95 #kg
G = 9.80665 #kg/m^2
#   Terrain Model
#       Flat Terrain
TerrainNorm = [0,0,1] 
TerrainTangentX = [1,0,0]
TerrainTangentY = [0,1,0]
miu = 0.3
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
#Set up Initial Condition
#-----------------------------------------------------------------------------------------------------------------------
x_init = 0
y_init = -0.05
z_init = 0.5

xdot_init = 0
ydot_init = 0
zdot_init = 0

Lx_init = 0
Ly_init = 0
Lz_init = 0

Ldotx_init = 0
Ldoty_init = 0
Ldotz_init = 0

PLx_init = 0
PLy_init = 0
PLz_init = 0
PL_init = np.array([PLx_init,PLy_init,PLz_init])

PRx_init = 0
PRy_init = -0.25
PRz_init = 0
PR_init = np.array([PRx_init,PRy_init,PRz_init])

x_end = 0.375
y_end = -0.05
z_end = 0.5

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
#Define Variables and Bounds
#   CoM Position x-axis
x = ca.SX.sym('x',N_K)
x_lb = np.array([[0]*(x.shape[0]*x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
x_ub = np.array([[1]*(x.shape[0]*x.shape[1])])
#   CoM Position y-axis
y = ca.SX.sym('y',N_K)
y_lb = np.array([[-1]*(y.shape[0]*y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
y_ub = np.array([[1]*(y.shape[0]*y.shape[1])])
#   CoM Position z-axis
z = ca.SX.sym('z',N_K)
z_lb = np.array([[0.45]*(z.shape[0]*z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
z_ub = np.array([[0.8]*(z.shape[0]*z.shape[1])])
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
Lx_lb = np.array([[-1]*(Lx.shape[0]*Lx.shape[1])]) #particular way of generating lists in python, [value]*number of elements
Lx_ub = np.array([[1]*(Lx.shape[0]*Lx.shape[1])])
#   Angular Momentum y-axis
Ly = ca.SX.sym('Ly',N_K)
Ly_lb = np.array([[-1]*(Ly.shape[0]*Ly.shape[1])]) #particular way of generating lists in python, [value]*number of elements
Ly_ub = np.array([[1]*(Ly.shape[0]*Ly.shape[1])])
#   Angular Momntum y-axis
Lz = ca.SX.sym('Lz',N_K)
Lz_lb = np.array([[-1]*(Lz.shape[0]*Lz.shape[1])]) #particular way of generating lists in python, [value]*number of elements
Lz_ub = np.array([[1]*(Lz.shape[0]*Lz.shape[1])])
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
FL1x_lb = np.array([[-300]*(FL1x.shape[0]*FL1x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
FL1x_ub = np.array([[300]*(FL1x.shape[0]*FL1x.shape[1])])
#Left Foot Contact Point 1 y-axis
FL1y = ca.SX.sym('FL1y',N_K)
FL1y_lb = np.array([[-300]*(FL1y.shape[0]*FL1y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
FL1y_ub = np.array([[300]*(FL1y.shape[0]*FL1y.shape[1])])
#Left Foot Contact Point 1 z-axis
FL1z = ca.SX.sym('FL1z',N_K)
FL1z_lb = np.array([[-300]*(FL1z.shape[0]*FL1z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
FL1z_ub = np.array([[300]*(FL1z.shape[0]*FL1z.shape[1])])
#Left Foot Contact Point 2 x-axis
FL2x = ca.SX.sym('FL2x',N_K)
FL2x_lb = np.array([[-300]*(FL2x.shape[0]*FL2x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
FL2x_ub = np.array([[300]*(FL2x.shape[0]*FL2x.shape[1])])
#Left Foot Contact Point 2 y-axis
FL2y = ca.SX.sym('FL2y',N_K)
FL2y_lb = np.array([[-300]*(FL2y.shape[0]*FL2y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
FL2y_ub = np.array([[300]*(FL2y.shape[0]*FL2y.shape[1])])
#Left Foot Contact Point 2 z-axis
FL2z = ca.SX.sym('FL2z',N_K)
FL2z_lb = np.array([[-300]*(FL2z.shape[0]*FL2z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
FL2z_ub = np.array([[300]*(FL2z.shape[0]*FL2z.shape[1])])
#Left Foot Contact Point 3 x-axis
FL3x = ca.SX.sym('FL3x',N_K)
FL3x_lb = np.array([[-300]*(FL3x.shape[0]*FL3x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
FL3x_ub = np.array([[300]*(FL3x.shape[0]*FL3x.shape[1])])
#Left Foot Contact Point 3 y-axis
FL3y = ca.SX.sym('FL3y',N_K)
FL3y_lb = np.array([[-300]*(FL3y.shape[0]*FL3y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
FL3y_ub = np.array([[300]*(FL3y.shape[0]*FL3y.shape[1])])
#Left Foot Contact Point 3 z-axis
FL3z = ca.SX.sym('FL3z',N_K)
FL3z_lb = np.array([[-300]*(FL3z.shape[0]*FL3z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
FL3z_ub = np.array([[300]*(FL3z.shape[0]*FL3z.shape[1])])
#Left Foot Contact Point 4 x-axis
FL4x = ca.SX.sym('FL4x',N_K)
FL4x_lb = np.array([[-300]*(FL4x.shape[0]*FL4x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
FL4x_ub = np.array([[300]*(FL4x.shape[0]*FL4x.shape[1])])
#Left Foot Contact Point 4 y-axis
FL4y = ca.SX.sym('FL4y',N_K)
FL4y_lb = np.array([[-300]*(FL4y.shape[0]*FL4y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
FL4y_ub = np.array([[300]*(FL4y.shape[0]*FL4y.shape[1])])
#Left Foot Contact Point 4 z-axis
FL4z = ca.SX.sym('FL4z',N_K)
FL4z_lb = np.array([[-300]*(FL4z.shape[0]*FL4z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
FL4z_ub = np.array([[300]*(FL4z.shape[0]*FL4z.shape[1])])

#Right Contact Force x-axis
#Right Foot Contact Point 1 x-axis
FR1x = ca.SX.sym('FR1x',N_K)
FR1x_lb = np.array([[-300]*(FR1x.shape[0]*FR1x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
FR1x_ub = np.array([[300]*(FR1x.shape[0]*FR1x.shape[1])])
#Right Foot Contact Point 1 y-axis
FR1y = ca.SX.sym('FR1y',N_K)
FR1y_lb = np.array([[-300]*(FR1y.shape[0]*FR1y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
FR1y_ub = np.array([[300]*(FR1y.shape[0]*FR1y.shape[1])])
#Right Foot Contact Point 1 z-axis
FR1z = ca.SX.sym('FR1z',N_K)
FR1z_lb = np.array([[-300]*(FR1z.shape[0]*FR1z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
FR1z_ub = np.array([[300]*(FR1z.shape[0]*FR1z.shape[1])])
#Right Foot Contact Point 2 x-axis
FR2x = ca.SX.sym('FR2x',N_K)
FR2x_lb = np.array([[-300]*(FR2x.shape[0]*FR2x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
FR2x_ub = np.array([[300]*(FR2x.shape[0]*FR2x.shape[1])])
#Right Foot Contact Point 2 y-axis
FR2y = ca.SX.sym('FR2y',N_K)
FR2y_lb = np.array([[-300]*(FR2y.shape[0]*FR2y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
FR2y_ub = np.array([[300]*(FR2y.shape[0]*FR2y.shape[1])])
#Right Foot Contact Point 2 z-axis
FR2z = ca.SX.sym('FR2z',N_K)
FR2z_lb = np.array([[-300]*(FR2z.shape[0]*FR2z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
FR2z_ub = np.array([[300]*(FR2z.shape[0]*FR2z.shape[1])])
#Right Foot Contact Point 3 x-axis
FR3x = ca.SX.sym('FR3x',N_K)
FR3x_lb = np.array([[-300]*(FR3x.shape[0]*FR3x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
FR3x_ub = np.array([[300]*(FR3x.shape[0]*FR3x.shape[1])])
#Right Foot Contact Point 3 y-axis
FR3y = ca.SX.sym('FR3y',N_K)
FR3y_lb = np.array([[-300]*(FR3y.shape[0]*FR3y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
FR3y_ub = np.array([[300]*(FR3y.shape[0]*FR3y.shape[1])])
#Right Foot Contact Point 3 z-axis
FR3z = ca.SX.sym('FR3z',N_K)
FR3z_lb = np.array([[-300]*(FR3z.shape[0]*FR3z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
FR3z_ub = np.array([[300]*(FR3z.shape[0]*FR3z.shape[1])])
#Right Foot Contact Point 4 x-axis
FR4x = ca.SX.sym('FR4x',N_K)
FR4x_lb = np.array([[-300]*(FR4x.shape[0]*FR4x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
FR4x_ub = np.array([[300]*(FR4x.shape[0]*FR4x.shape[1])])
#Right Foot Contact Point 4 y-axis
FR4y = ca.SX.sym('FR4y',N_K)
FR4y_lb = np.array([[-300]*(FR4y.shape[0]*FR4y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
FR4y_ub = np.array([[300]*(FR4y.shape[0]*FR4y.shape[1])])
#Right Foot Contact Point 4 z-axis
FR4z = ca.SX.sym('FR4z',N_K)
FR4z_lb = np.array([[-300]*(FR4z.shape[0]*FR4z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
FR4z_ub = np.array([[300]*(FR4z.shape[0]*FR4z.shape[1])])

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
    pxtemp = ca.SX.sym('px'+str(stepIdx+1))
    px.append(pxtemp)
    px_lb.append(np.array([-0.5]))
    px_ub.append(np.array([1.5]))

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
    Tstemp = ca.SX.sym('Ts'+str(n_phase+1))
    Ts.append(Tstemp)
    Ts_lb.append(np.array([0.15]))
    Ts_ub.append(np.array([1.2]))


#   Collect all Decision Variables
DecisionVars = ca.vertcat(x,y,z,xdot,ydot,zdot,Lx,Ly,Lz,Ldotx,Ldoty,Ldotz,FL1x,FL1y,FL1z,FL2x,FL2y,FL2z,FL3x,FL3y,FL3z,FL4x,FL4y,FL4z,FR1x,FR1y,FR1z,FR2x,FR2y,FR2z,FR3x,FR3y,FR3z,FR4x,FR4y,FR4z,*px,*py,*pz,*Ts)
#DecisionVars = ca.vertcat(x,xdot)
#print(DecisionVars)
#   
DecisionVarsShape = DecisionVars.shape

#   Collect all lower bound and upper bound
DecisionVars_lb = np.concatenate(((x_lb,y_lb,z_lb,xdot_lb,ydot_lb,zdot_lb,Lx_lb,Ly_lb,Lz_lb,Ldotx_lb,Ldoty_lb,Ldotz_lb,FL1x_lb,FL1y_lb,FL1z_lb,FL2x_lb,FL2y_lb,FL2z_lb,FL3x_lb,FL3y_lb,FL3z_lb,FL4x_lb,FL4y_lb,FL4z_lb,FR1x_lb,FR1y_lb,FR1z_lb,FR2x_lb,FR2y_lb,FR2z_lb,FR3x_lb,FR3y_lb,FR3z_lb,FR4x_lb,FR4y_lb,FR4z_lb,px_lb,py_lb,pz_lb,Ts_lb)),axis=None)
DecisionVars_ub = np.concatenate(((x_ub,y_ub,z_ub,xdot_ub,ydot_ub,zdot_ub,Lx_ub,Ly_ub,Lz_ub,Ldotx_ub,Ldoty_ub,Ldotz_ub,FL1x_ub,FL1y_ub,FL1z_ub,FL2x_ub,FL2y_ub,FL2z_ub,FL3x_ub,FL3y_ub,FL3z_ub,FL4x_ub,FL4y_ub,FL4z_ub,FR1x_ub,FR1y_ub,FR1z_ub,FR2x_ub,FR2y_ub,FR2z_ub,FR3x_ub,FR3y_ub,FR3z_ub,FR4x_ub,FR4y_ub,FR4z_ub,px_ub,py_ub,pz_ub,Ts_ub)),axis=None)

#Time Span Setup
tau_upper_limit = 1
tauStepLength = tau_upper_limit/(N_K-1) #Get the interval length, total number of knots - 1

#   Temporary Code for array/matrices multiplication
#x = ca.SX.sym('x',2)
#y = ca.DM(np.array([[1,2]]))
#z = x.T@y
#print(z)

#-----------------------------------------------------------------------------------------------------------------------
#Define Constrains
g = []
glb = []
gub = []
J = (x[-1]-1.5)**2

#Initial and Termianl Conditions
#   Initial CoM x-axis
g.append(x[0]-x_init)
glb.append(np.array([0]))
gub.append(np.array([0]))

#   Initial CoM y-axis
#g.append(y[0]-y_init)
#glb.append(np.array([0]))
#gub.append(np.array([0]))

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

#   Terminal CoM y-axis
#g.append(y[-1]-y_end)
#glb.append(np.array([0]))
#gub.append(np.array([0]))

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
        #print(CoM_k)

        #Phase dependent Constraints and Time Step length
        if GaitPattern[Nph]=='InitialDouble':
            print('Initial Double: Knot',str(k))
            #h = h_DoubleLeg

            #Kinematics Constraint
            #   CoM in the Left foot
            g.append(K_CoM_Left@(CoM_k-ca.DM(PL_init)))
            glb.append(np.full((len(k_CoM_Left),),-np.inf))
            gub.append(k_CoM_Left)
            #   CoM in the Right foot
            g.append(K_CoM_Right@(CoM_k-ca.DM(PR_init)))
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

        elif GaitPattern[Nph]=='LeftSupport':
            
            print('Left Support: Knot',str(k))
            #h = h_SingleLeg

            #   Complementarity Condition (Zero Forces for Right Foot)
            #Zero Forces x-axis (For Right Foot 1)
            g.append(FR1x[k])
            glb.append(np.array([0]))
            gub.append(np.array([0]))
            #Zero Forces y-axis (For Right Foot 1)
            g.append(FR1y[k])
            glb.append(np.array([0]))
            gub.append(np.array([0]))
            #Zero Forces z-axis (For Right Foot 1)
            g.append(FR1z[k])
            glb.append(np.array([0]))
            gub.append(np.array([0]))
            #Zero Forces x-axis (For Right Foot 2)
            g.append(FR2x[k])
            glb.append(np.array([0]))
            gub.append(np.array([0]))
            #Zero Forces y-axis (For Right Foot 2)
            g.append(FR2y[k])
            glb.append(np.array([0]))
            gub.append(np.array([0]))
            #Zero Forces z-axis (For Right Foot 2)
            g.append(FR2z[k])
            glb.append(np.array([0]))
            gub.append(np.array([0]))
            #Zero Forces x-axis (For Right Foot 3)
            g.append(FR3x[k])
            glb.append(np.array([0]))
            gub.append(np.array([0]))
            #Zero Forces y-axis (For Right Foot 3)
            g.append(FR3y[k])
            glb.append(np.array([0]))
            gub.append(np.array([0]))
            #Zero Forces z-axis (For Right Foot 3)
            g.append(FR3z[k])
            glb.append(np.array([0]))
            gub.append(np.array([0]))
            #Zero Forces x-axis (For Right Foot 4)
            g.append(FR4x[k])
            glb.append(np.array([0]))
            gub.append(np.array([0]))
            #Zero Forces y-axis (For Right Foot 4)
            g.append(FR4y[k])
            glb.append(np.array([0]))
            gub.append(np.array([0]))
            #Zero Forces z-axis (For Right Foot 4)
            g.append(FR4z[k])
            glb.append(np.array([0]))
            gub.append(np.array([0]))

            #Kinematic Constraint and Angular Dynamics
            StepCnt = np.max((Nph-1),0)//2 #Step Count - Start from zero and negelect phase 0
            print('StepCount'+str(StepCnt))
            if StepCnt == 0:#First Step
                print('First Step For Left Support')
                #Kinematics Constraint
                #   CoM in the Left foot
                g.append(K_CoM_Left@(CoM_k-ca.DM(PL_init)))
                glb.append(np.full((len(k_CoM_Left),),-np.inf))
                gub.append(k_CoM_Left)
                #Angular Dynamics (Left Support)
                if k<N_K-1:
                    g.append(Ldot_next-Ldot_current-h*(ca.cross((PL_init+np.array([0.11,0.06,0])-CoM_k),FL1_k)+ca.cross((PL_init+np.array([0.11,-0.06,0])-CoM_k),FL2_k)+ca.cross((PL_init+np.array([-0.11,0.06,0])-CoM_k),FL3_k)+ca.cross((PL_init+np.array([-0.11,-0.06,0])-CoM_k),FL4_k)))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))
                #g.append(ca.cross((PL_init-CoM_k),FL_k))
                #glb.append(np.array([-1,-1,-1]))
                #gub.append(np.array([1,1,1]))
 #           else:
 #               print('No implementation yet')


        elif GaitPattern[Nph]=='RightSupport': #Implementation not finished
            print('Right Support: Knot',str(k))
            #h = h_SingleLeg

            #   Complementarity Condition
            #Zero Forces x-axis (For Left Foot 1)
            g.append(FL1x[k])
            glb.append(np.array([0]))
            gub.append(np.array([0]))
            #Zero Forces y-axis (For Left Foot 1)
            g.append(FL1y[k])
            glb.append(np.array([0]))
            gub.append(np.array([0]))
            #Zero Forces z-axis (For Right Foot 1)
            g.append(FL1z[k])
            glb.append(np.array([0]))
            gub.append(np.array([0]))
            #Zero Forces x-axis (For Left Foot 2)
            g.append(FL2x[k])
            glb.append(np.array([0]))
            gub.append(np.array([0]))
            #Zero Forces y-axis (For Left Foot 2)
            g.append(FL2y[k])
            glb.append(np.array([0]))
            gub.append(np.array([0]))
            #Zero Forces z-axis (For Right Foot 2)
            g.append(FL2z[k])
            glb.append(np.array([0]))
            gub.append(np.array([0]))
            #Zero Forces x-axis (For Left Foot 3)
            g.append(FL3x[k])
            glb.append(np.array([0]))
            gub.append(np.array([0]))
            #Zero Forces y-axis (For Left Foot 3)
            g.append(FL3y[k])
            glb.append(np.array([0]))
            gub.append(np.array([0]))
            #Zero Forces z-axis (For Right Foot 3)
            g.append(FL3z[k])
            glb.append(np.array([0]))
            gub.append(np.array([0]))
            #Zero Forces x-axis (For Left Foot 4)
            g.append(FL4x[k])
            glb.append(np.array([0]))
            gub.append(np.array([0]))
            #Zero Forces y-axis (For Left Foot 4)
            g.append(FL4y[k])
            glb.append(np.array([0]))
            gub.append(np.array([0]))
            #Zero Forces z-axis (For Right Foot 4)
            g.append(FL4z[k])
            glb.append(np.array([0]))
            gub.append(np.array([0]))

            #Kinematic Constraint 
            StepCnt = np.max((Nph-1),0)//2 #Step Count - Start from zero and negelect phase 0
            print('StepCount'+str(StepCnt))
            if StepCnt == 0:#First Step
                print('First Step For Right Support')
                #Kinematics Constraint
                #   CoM in the Rgith foot
                g.append(K_CoM_Right@(CoM_k-ca.DM(PR_init)))
                glb.append(np.full((len(k_CoM_Right),),-np.inf))
                gub.append(k_CoM_Right)
                #Angular Dynamics (Right Support)
                if k<N_K-1:
                    g.append(Ldot_next-Ldot_current-h*(ca.cross((PR_init+np.array([0.11,0.06,0])-CoM_k),FR1_k)+ca.cross((PR_init+np.array([0.11,-0.06,0])-CoM_k),FR2_k)+ca.cross((PR_init+np.array([-0.11,0.06,0])-CoM_k),FR3_k)+ca.cross((PR_init+np.array([-0.11,-0.06,0])-CoM_k),FR4_k)))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))
                #g.append(ca.cross((PR_init-CoM_k),FR_k))
                #glb.append(np.array([-1,-1,-1]))
                #gub.append(np.array([1,1,1]))
            else:
                print('No implementation yet')

        elif GaitPattern[Nph]=='DoubleSupport':
            print('Double Support: Knot',str(k))

            #h = h_DoubleLeg

            #Kinematic Constraint and Angular Dynamics
            StepCnt = np.max((Nph-1),0)//2 #Step Count - Start from zero and negelect phase 0
            print('StepCount'+str(StepCnt))
            if StepCnt == 0:#First Step
                print('First Step for Double Support')
                #Kinematics Constraint
                if GaitPattern[Nph-1]=='LeftSupport':
                    #   CoM in the Left foot
                    g.append(K_CoM_Left@(CoM_k-ca.DM(PL_init)))
                    glb.append(np.full((len(k_CoM_Left),),-np.inf))
                    gub.append(k_CoM_Left)
                    #   CoM in the Right foot
                    PR_k = ca.vertcat(px[StepCnt],py[StepCnt],pz[StepCnt])
                    print('Left Support as Previous Phase')
                    print(PR_k)
                    g.append(K_CoM_Right@(CoM_k-PR_k))
                    glb.append(np.full((len(k_CoM_Right),),-np.inf))
                    gub.append(k_CoM_Right)
                    #Angular Dynamics (Double Support)
                    if k<N_K-1:
                        g.append(Ldot_next-Ldot_current-h*(ca.cross((PL_init+np.array([0.11,0.06,0])-CoM_k),FL1_k)+ca.cross((PL_init+np.array([0.11,-0.06,0])-CoM_k),FL2_k)+ca.cross((PL_init+np.array([-0.11,0.06,0])-CoM_k),FL3_k)+ca.cross((PL_init+np.array([-0.11,-0.06,0])-CoM_k),FL4_k)+ca.cross((PR_k+np.array([0.11,0.06,0])-CoM_k),FR1_k)+ca.cross((PR_k+np.array([0.11,-0.06,0])-CoM_k),FR2_k)+ca.cross((PR_k+np.array([-0.11,0.06,0])-CoM_k),FR3_k)+ca.cross((PR_k+np.array([-0.11,-0.06,0])-CoM_k),FR4_k)))
                        glb.append(np.array([0,0,0]))
                        gub.append(np.array([0,0,0]))
                    #g.append(ca.cross((PL_init-CoM_k),FL_k) + ca.cross((PR_k-CoM_k),FR_k))
                    #glb.append(np.array([-1,-1,-1]))
                    #gub.append(np.array([1,1,1]))

                elif GaitPattern[Nph-1]=='RightSupport':
                    #   CoM in the Right foot
                    g.append(K_CoM_Right@(CoM_k-ca.DM(PR_init)))
                    glb.append(np.full((len(k_CoM_Right),),-np.inf))
                    gub.append(k_CoM_Right)
                    #   CoM in the Left foot
                    PL_k = ca.vertcat(px[StepCnt],py[StepCnt],pz[StepCnt])
                    g.append(K_CoM_Left@(CoM_k-PL_k))
                    glb.append(np.full((len(k_CoM_Left),),-np.inf))
                    gub.append(k_CoM_Left)
                    #Angular Dynamics (Double Support)
                    if k<N_K-1:
                        g.append(Ldot_next-Ldot_current-h*(ca.cross((PL_k+np.array([0.11,0.06,0])-CoM_k),FL1_k)+ca.cross((PL_k+np.array([0.11,-0.06,0])-CoM_k),FL2_k)+ca.cross((PL_k+np.array([-0.11,0.06,0])-CoM_k),FL3_k)+ca.cross((PL_k+np.array([-0.11,-0.06,0])-CoM_k),FL4_k)+ca.cross((PR_init+np.array([0.11,0.06,0])-CoM_k),FR1_k)+ca.cross((PR_init+np.array([0.11,-0.06,0])-CoM_k),FR2_k)+ca.cross((PR_init+np.array([-0.11,0.06,0])-CoM_k),FR3_k)+ca.cross((PR_init+np.array([-0.11,-0.06,0])-CoM_k),FR4_k)))
                        glb.append(np.array([0,0,0]))
                        gub.append(np.array([0,0,0]))
                    #g.append(ca.cross((PL_k-CoM_k),FL_k) + ca.cross((PR_init - CoM_k),FR_k))
                    #glb.append(np.array([-1,-1,-1]))
                    #gub.append(np.array([1,1,1]))
            else:
                print('No implementation yet')
            
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

            #Define Cost Function
            #J = J + FLx[k]**2 + FLy[k]**2 + FLz[k]**2 + FRx[k]**2 + FRy[k]**2 + FRz[k]**2

#Relative Footstep Constraints
print('Relative Footstep Constraints')
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

                if GaitPattern[PhaseCnt-1]=='LeftSupport':
                    print('Previous Phase as Left Support')
                    p_current = PL_init
                    
                    g.append(Q_rf_in_lf@(p_next-p_current))
                    glb.append(np.full((len(q_rf_in_lf),),-np.inf))
                    gub.append(q_rf_in_lf)
                elif GaitPattern[PhaseCnt-1]=='RightSupport':
                    p_current = PR_init
                    
                    g.append(Q_lf_in_rf@(p_next-p_current))
                    glb.append(np.full((len(q_lf_in_rf),),-np.inf))
                    gub.append(q_lf_in_rf)
                #print('No Implementation yet')
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
    elif GaitPattern[phase_cnt] == 'LeftSupport':
        g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
        glb.append(np.array([0.5]))
        gub.append(np.array([0.9]))
    elif GaitPattern[phase_cnt] == 'RightSupport':
        g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
        glb.append(np.array([0.5]))
        gub.append(np.array([0.9]))
    elif GaitPattern[phase_cnt] == 'DoubleSupport':
        g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
        glb.append(np.array([0.1]))
        gub.append(np.array([0.25]))

#   reshape all constraints
g = ca.vertcat(*g)
glb = np.concatenate(glb)
gub = np.concatenate(gub)
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
#Define Cost Function
#J = 0
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
#Lower and Upper Bounds of Variables
#x_lb = np.array([[0]*(x.shape[0]*x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
#x_ub = np.array([[5]*(x.shape[0]*x.shape[1])])

#xdot_lb = np.array([[0]*(xdot.shape[0]*xdot.shape[1])]) 
#xdot_ub = np.array([[5]*(xdot.shape[0]*xdot.shape[1])])
#print(DecisionVars_lb)
#-----------------------------------------------------------------------------------------------------------------------

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
prob = {'x': DecisionVars, 'f': J, 'g': g}
solver = ca.nlpsol('solver', 'ipopt', prob)

res = solver(x0=DecisionVars_init, lbx=DecisionVars_lb, ubx=DecisionVars_ub, lbg=glb, ubg=gub)
x_opt = res['x']
x_opt = x_opt.full().flatten()
#print('x_opt: ', x_opt)
#print(len(np.array(x_opt).flatten()))
#-----------------------------------------------------------------------------------------------------------------------
#Extract Results
#Result Extraction
x_index = (0,Nk_Local*Nphase) #First set of variables start counting from 0
x_res = x_opt[x_index[0]:x_index[1]+1]
x_res = np.array(x_res)
print('x_res: ',x_res)
y_index = (x_index[1]+1,x_index[1]+Nk_Local*Nphase+1)
y_res = x_opt[y_index[0]:y_index[1]+1]
y_res = np.array(y_res)
print('y_res: ',y_res)
z_index = (y_index[1]+1,y_index[1]+Nk_Local*Nphase+1)
z_res = x_opt[z_index[0]:z_index[1]+1]
z_res = np.array(z_res)
print('z_res: ',z_res)
xdot_index = (z_index[1]+1,z_index[1]+Nk_Local*Nphase+1)
xdot_res = x_opt[xdot_index[0]:xdot_index[1]+1]
xdot_res = np.array(xdot_res)
print('xdot_res: ',xdot_res)
ydot_index = (xdot_index[1]+1,xdot_index[1]+Nk_Local*Nphase+1)
ydot_res = x_opt[ydot_index[0]:ydot_index[1]+1]
ydot_res = np.array(ydot_res)
print('ydot_res: ',ydot_res)
zdot_index = (ydot_index[1]+1,ydot_index[1]+Nk_Local*Nphase+1)
zdot_res = x_opt[zdot_index[0]:zdot_index[1]+1]
zdot_res = np.array(zdot_res)
print('zdot_res: ',zdot_res)
Lx_index = (zdot_index[1]+1,zdot_index[1]+Nk_Local*Nphase+1)
Lx_res = x_opt[Lx_index[0]:Lx_index[1]+1]
Lx_res = np.array(Lx_res)
print('Lx_res: ',Lx_res)
Ly_index = (Lx_index[1]+1,Lx_index[1]+Nk_Local*Nphase+1)
Ly_res = x_opt[Ly_index[0]:Ly_index[1]+1]
Ly_res = np.array(Ly_res)
print('Ly_res: ',Ly_res)
Lz_index = (Ly_index[1]+1,Ly_index[1]+Nk_Local*Nphase+1)
Lz_res = x_opt[Lz_index[0]:Lz_index[1]+1]
Lz_res = np.array(Lz_res)
print('Lz_res: ',Lz_res)
Ldotx_index = (Lz_index[1]+1,Lz_index[1]+Nk_Local*Nphase+1)
Ldotx_res = x_opt[Ldotx_index[0]:Ldotx_index[1]+1]
Ldotx_res = np.array(Ldotx_res)
print('Ldotx_res: ',Ldotx_res)
Ldoty_index = (Ldotx_index[1]+1,Ldotx_index[1]+Nk_Local*Nphase+1)
Ldoty_res = x_opt[Ldoty_index[0]:Ldoty_index[1]+1]
Ldoty_res = np.array(Ldoty_res)
print('Ldoty_res: ',Ldoty_res)
Ldotz_index = (Ldoty_index[1]+1,Ldoty_index[1]+Nk_Local*Nphase+1)
Ldotz_res = x_opt[Ldotz_index[0]:Ldotz_index[1]+1]
Ldotz_res = np.array(Ldotz_res)
print('Ldotz_res: ',Ldotz_res)
FL1x_index = (Ldotz_index[1]+1,Ldotz_index[1]+Nk_Local*Nphase+1)
FL1x_res = x_opt[FL1x_index[0]:FL1x_index[1]+1]
FL1x_res = np.array(FL1x_res)
print('FL1x_res: ',FL1x_res)
FL1y_index = (FL1x_index[1]+1,FL1x_index[1]+Nk_Local*Nphase+1)
FL1y_res = x_opt[FL1y_index[0]:FL1y_index[1]+1]
FL1y_res = np.array(FL1y_res)
print('FL1y_res: ',FL1y_res)
FL1z_index = (FL1y_index[1]+1,FL1y_index[1]+Nk_Local*Nphase+1)
FL1z_res = x_opt[FL1z_index[0]:FL1z_index[1]+1]
FL1z_res = np.array(FL1z_res)
print('FL1z_res: ',FL1z_res)
FL2x_index = (FL1z_index[1]+1,FL1z_index[1]+Nk_Local*Nphase+1)
FL2x_res = x_opt[FL2x_index[0]:FL2x_index[1]+1]
FL2x_res = np.array(FL2x_res)
print('FL2x_res: ',FL2x_res)
FL2y_index = (FL2x_index[1]+1,FL2x_index[1]+Nk_Local*Nphase+1)
FL2y_res = x_opt[FL2y_index[0]:FL2y_index[1]+1]
FL2y_res = np.array(FL2y_res)
print('FL2y_res: ',FL2y_res)
FL2z_index = (FL2y_index[1]+1,FL2y_index[1]+Nk_Local*Nphase+1)
FL2z_res = x_opt[FL2z_index[0]:FL2z_index[1]+1]
FL2z_res = np.array(FL2z_res)
print('FL2z_res: ',FL2z_res)
FL3x_index = (FL2z_index[1]+1,FL2z_index[1]+Nk_Local*Nphase+1)
FL3x_res = x_opt[FL3x_index[0]:FL3x_index[1]+1]
FL3x_res = np.array(FL3x_res)
print('FL3x_res: ',FL3x_res)
FL3y_index = (FL3x_index[1]+1,FL3x_index[1]+Nk_Local*Nphase+1)
FL3y_res = x_opt[FL3y_index[0]:FL3y_index[1]+1]
FL3y_res = np.array(FL3y_res)
print('FL3y_res: ',FL3y_res)
FL3z_index = (FL3y_index[1]+1,FL3y_index[1]+Nk_Local*Nphase+1)
FL3z_res = x_opt[FL3z_index[0]:FL3z_index[1]+1]
FL3z_res = np.array(FL3z_res)
print('FL3z_res: ',FL3z_res)
FL4x_index = (FL3z_index[1]+1,FL3z_index[1]+Nk_Local*Nphase+1)
FL4x_res = x_opt[FL4x_index[0]:FL4x_index[1]+1]
FL4x_res = np.array(FL4x_res)
print('FL4x_res: ',FL4x_res)
FL4y_index = (FL4x_index[1]+1,FL4x_index[1]+Nk_Local*Nphase+1)
FL4y_res = x_opt[FL4y_index[0]:FL4y_index[1]+1]
FL4y_res = np.array(FL4y_res)
print('FL4y_res: ',FL4y_res)
FL4z_index = (FL4y_index[1]+1,FL4y_index[1]+Nk_Local*Nphase+1)
FL4z_res = x_opt[FL4z_index[0]:FL4z_index[1]+1]
FL4z_res = np.array(FL4z_res)
print('FL4z_res: ',FL4z_res)
FR1x_index = (FL4z_index[1]+1,FL4z_index[1]+Nk_Local*Nphase+1)
FR1x_res = x_opt[FR1x_index[0]:FR1x_index[1]+1]
FR1x_res = np.array(FR1x_res)
print('FR1x_res: ',FR1x_res)
FR1y_index = (FR1x_index[1]+1,FR1x_index[1]+Nk_Local*Nphase+1)
FR1y_res = x_opt[FR1y_index[0]:FR1y_index[1]+1]
FR1y_res = np.array(FR1y_res)
print('FR1y_res: ',FR1y_res)
FR1z_index = (FR1y_index[1]+1,FR1y_index[1]+Nk_Local*Nphase+1)
FR1z_res = x_opt[FR1z_index[0]:FR1z_index[1]+1]
FR1z_res = np.array(FR1z_res)
print('FR1z_res: ',FR1z_res)
FR2x_index = (FR1z_index[1]+1,FR1z_index[1]+Nk_Local*Nphase+1)
FR2x_res = x_opt[FR2x_index[0]:FR2x_index[1]+1]
FR2x_res = np.array(FR2x_res)
print('FR2x_res: ',FR2x_res)
FR2y_index = (FR2x_index[1]+1,FR2x_index[1]+Nk_Local*Nphase+1)
FR2y_res = x_opt[FR2y_index[0]:FR2y_index[1]+1]
FR2y_res = np.array(FR2y_res)
print('FR2y_res: ',FR2y_res)
FR2z_index = (FR2y_index[1]+1,FR2y_index[1]+Nk_Local*Nphase+1)
FR2z_res = x_opt[FR2z_index[0]:FR2z_index[1]+1]
FR2z_res = np.array(FR2z_res)
print('FR2z_res: ',FR2z_res)
FR3x_index = (FR2z_index[1]+1,FR2z_index[1]+Nk_Local*Nphase+1)
FR3x_res = x_opt[FR3x_index[0]:FR3x_index[1]+1]
FR3x_res = np.array(FR3x_res)
print('FR3x_res: ',FR3x_res)
FR3y_index = (FR3x_index[1]+1,FR3x_index[1]+Nk_Local*Nphase+1)
FR3y_res = x_opt[FR3y_index[0]:FR3y_index[1]+1]
FR3y_res = np.array(FR3y_res)
print('FR3y_res: ',FR3y_res)
FR3z_index = (FR3y_index[1]+1,FR3y_index[1]+Nk_Local*Nphase+1)
FR3z_res = x_opt[FR3z_index[0]:FR3z_index[1]+1]
FR3z_res = np.array(FR3z_res)
print('FR3z_res: ',FR3z_res)
FR4x_index = (FR3z_index[1]+1,FR3z_index[1]+Nk_Local*Nphase+1)
FR4x_res = x_opt[FR4x_index[0]:FR4x_index[1]+1]
FR4x_res = np.array(FR4x_res)
print('FR4x_res: ',FR4x_res)
FR4y_index = (FR4x_index[1]+1,FR4x_index[1]+Nk_Local*Nphase+1)
FR4y_res = x_opt[FR4y_index[0]:FR4y_index[1]+1]
FR4y_res = np.array(FR4y_res)
print('FR4y_res: ',FR4y_res)
FR4z_index = (FR4y_index[1]+1,FR4y_index[1]+Nk_Local*Nphase+1)
FR4z_res = x_opt[FR4z_index[0]:FR4z_index[1]+1]
FR4z_res = np.array(FR4z_res)
print('FR4z_res: ',FR4z_res)
px_index = (FR4z_index[1]+1,FR4z_index[1]+Nstep)
px_res = x_opt[px_index[0]:px_index[1]+1]
px_res = np.array(px_res)
print('px_res: ',px_res)
py_index = (px_index[1]+1,px_index[1]+Nstep)
py_res = x_opt[py_index[0]:py_index[1]+1]
py_res = np.array(py_res)
print('py_res: ',py_res)
pz_index = (py_index[1]+1,py_index[1]+Nstep)
pz_res = x_opt[pz_index[0]:pz_index[1]+1]
pz_res = np.array(pz_res)
print('pz_res: ',pz_res)
Ts_index = (pz_index[1]+1,pz_index[1]+Nphase)
Ts_res = x_opt[pz_index[0]:Ts_index[1]+1]
Ts_res = np.array(Ts_res)
print('Ts_res: ',Ts_res)

#-----------------------------------------------------------------------------------------------------------------------
#Plot Result
fig=plt.figure()
ax = Axes3D(fig)
#ax = fig.add_subplot(111, projection="3d")

#ax.plot3D(x_res,y_res,z_res,color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)
ax.plot3D(x_res,y_res,z_res,color='green', linestyle='dashed', linewidth=2, markersize=12)
ax.set_xlim3d(0, 1)
ax.set_ylim3d(-0.5,0.5)
ax.set_zlim3d(0,0.8)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

#Initial Footstep Locations
ax.scatter(PLx_init, PLy_init, PLz_init, c='r', marker='o', linewidth = 10) 
ax.scatter(PRx_init, PRy_init, PRz_init, c='b', marker='o', linewidth = 10) 

for PhaseCnt in range(Nphase):
    if GaitPattern[PhaseCnt]=='DoubleSupport':
        StepCnt = np.max((PhaseCnt-1),0)//2 #Step Count - Start from zero and negelect phase 0
        #print('StepCount'+str(StepCnt))
        #if StepCnt == 0:#First Step
        #        p_next = ca.vertcat(px[StepCnt],py[StepCnt],pz[StepCnt])

        if GaitPattern[PhaseCnt-1]=='LeftSupport':
            #print('Previous Phase as Left Support, land Right Foot')
            StepColor = 'b'
        elif GaitPattern[PhaseCnt-1]=='RightSupport':
            StepColor = 'r'
            #print('Previous Phase as Right Support, land left Foot')
        else:
            print('No implementation yet')
        ax.scatter(px_res[StepCnt], py_res[StepCnt], pz_res[StepCnt], c=StepColor, marker='o', linewidth = 10) 

ax.view_init(elev=8.776933438381377, azim=-99.32358055821186)
plt.show()

plt.plot(FR2z_res[0:-1])
plt.show()

