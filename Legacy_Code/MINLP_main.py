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
miu = 0.6
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
#y_init = 0
z_init = 0.6

xdot_init = 0
ydot_init = 0
zdot_init = 0

PLx_init = 0.1
PLy_init = 0
PLz_init = 0
PL_init = np.array([PLx_init,PLy_init,PLz_init])

PRx_init = -0.1
PRy_init = -0.3
PRz_init = 0
PR_init = np.array([PRx_init,PRy_init,PRz_init])

x_end = 0.3
#y_end = 0
z_end = 0.6

xdot_end = 0
ydot_end = 0
zdot_end = 0

#-----------------------------------------------------------------------------------------------------------------------
#Define Variables and Bounds
#   CoM Position x-axis
x = ca.SX.sym('x',N_K)
x_lb = np.array([[-0]*(x.shape[0]*x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
x_ub = np.array([[1]*(x.shape[0]*x.shape[1])])
#   CoM Position y-axis
y = ca.SX.sym('y',N_K)
y_lb = np.array([[-1]*(y.shape[0]*y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
y_ub = np.array([[1]*(y.shape[0]*y.shape[1])])
#   CoM Position z-axis
z = ca.SX.sym('z',N_K)
z_lb = np.array([[0]*(z.shape[0]*z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
z_ub = np.array([[1]*(z.shape[0]*z.shape[1])])
#   CoM Velocity x-axis
xdot = ca.SX.sym('xdot',N_K)
xdot_lb = np.array([[-2.5]*(xdot.shape[0]*xdot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
xdot_ub = np.array([[2.5]*(xdot.shape[0]*xdot.shape[1])])
#   CoM Velocity y-axis
ydot = ca.SX.sym('ydot',N_K)
ydot_lb = np.array([[-2.5]*(ydot.shape[0]*ydot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
ydot_ub = np.array([[2.5]*(ydot.shape[0]*ydot.shape[1])])
#   CoM Velocity z-axis
zdot = ca.SX.sym('zdot',N_K)
zdot_lb = np.array([[-2.5]*(zdot.shape[0]*zdot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
zdot_ub = np.array([[2.5]*(zdot.shape[0]*zdot.shape[1])])
#   Left Contact Force x-axis
FLx = ca.SX.sym('FLx',N_K)
FLx_lb = np.array([[-1500]*(FLx.shape[0]*FLx.shape[1])]) #particular way of generating lists in python, [value]*number of elements
FLx_ub = np.array([[1500]*(FLx.shape[0]*FLx.shape[1])])
#   Left Contact Force y-axis
FLy = ca.SX.sym('FLy',N_K)
FLy_lb = np.array([[-1500]*(FLy.shape[0]*FLy.shape[1])]) #particular way of generating lists in python, [value]*number of elements
FLy_ub = np.array([[1500]*(FLy.shape[0]*FLy.shape[1])])
#   Left Contact Force z-axis
FLz = ca.SX.sym('FLz',N_K)
FLz_lb = np.array([[-1500]*(FLz.shape[0]*FLz.shape[1])]) #particular way of generating lists in python, [value]*number of elements
FLz_ub = np.array([[1500]*(FLz.shape[0]*FLz.shape[1])])
#   Right Contact Force x-axis
FRx = ca.SX.sym('FRx',N_K)
FRx_lb = np.array([[-1500]*(FRx.shape[0]*FRx.shape[1])]) #particular way of generating lists in python, [value]*number of elements
FRx_ub = np.array([[1500]*(FRx.shape[0]*FRx.shape[1])])
#   Right Contact Force y-axis
FRy = ca.SX.sym('FRy',N_K)
FRy_lb = np.array([[-1500]*(FRy.shape[0]*FRy.shape[1])]) #particular way of generating lists in python, [value]*number of elements
FRy_ub = np.array([[1500]*(FRy.shape[0]*FRy.shape[1])])
#   Right Contact Force z-axis
FRz = ca.SX.sym('FRz',N_K)
FRz_lb = np.array([[-1500]*(FRz.shape[0]*FRz.shape[1])]) #particular way of generating lists in python, [value]*number of elements
FRz_ub = np.array([[1500]*(FRz.shape[0]*FRz.shape[1])])
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
    px_lb.append(np.array([-1]))
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
    Ts_ub.append(np.array([1.1]))


#   Collect all Decision Variables
DecisionVars = ca.vertcat(x,y,z,xdot,ydot,zdot,FLx,FLy,FLz,FRx,FRy,FRz,*px,*py,*pz,*Ts)
#DecisionVars = ca.vertcat(x,xdot)
#print(DecisionVars)
#   
DecisionVarsShape = DecisionVars.shape

#   Collect all lower bound and upper bound
DecisionVars_lb = np.concatenate(((x_lb,y_lb,z_lb,xdot_lb,ydot_lb,zdot_lb,FLx_lb,FLy_lb,FLz_lb,FRx_lb,FRy_lb,FRz_lb,px_lb,py_lb,pz_lb,Ts_lb)),axis=None)
DecisionVars_ub = np.concatenate(((x_ub,y_ub,z_ub,xdot_ub,ydot_ub,zdot_ub,FLx_ub,FLy_ub,FLz_ub,FRx_ub,FRy_ub,FRz_ub,px_ub,py_ub,pz_ub,Ts_ub)),axis=None)

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
J = 0

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

#   Terminal CoM x-axis
g.append(x[-1]-x_end)
glb.append(np.array([0]))
gub.append(np.array([0]))

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

        FL_k = ca.vertcat(FLx[k],FLy[k],FLz[k])
        FR_k = ca.vertcat(FRx[k],FRy[k],FRz[k])
        CoM_k = ca.vertcat(x[k],y[k],z[k])
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
            g.append(ca.cross((PL_init-CoM_k),FL_k)+ca.cross((PR_init-CoM_k),FR_k))
            glb.append(np.array([0,0,0]))
            gub.append(np.array([0,0,0]))

        elif GaitPattern[Nph]=='LeftSupport':
            
            print('Left Support: Knot',str(k))
            #h = h_SingleLeg

            #   Complementarity Condition
            #Zero Forces x-axis (For Right Foot)
            g.append(FRx[k])
            glb.append(np.array([0]))
            gub.append(np.array([0]))

            #Zero Forces y-axis (For Right Foot)
            g.append(FRy[k])
            glb.append(np.array([0]))
            gub.append(np.array([0]))

            #Zero Forces z-axis (For Right Foot)
            g.append(FRz[k])
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
                g.append(ca.cross((PL_init-CoM_k),FL_k))
                glb.append(np.array([0,0,0]))
                gub.append(np.array([0,0,0]))
 #           else:
 #               print('No implementation yet')


        elif GaitPattern[Nph]=='RightSupport': #Implementation not finished
            print('Right Support: Knot',str(k))
            #h = h_SingleLeg

            #   Complementarity Condition
            #Zero Forces x-axis (For Left Foot)
            g.append(FLx[k])
            glb.append(np.array([0]))
            gub.append(np.array([0]))

            #Zero Forces y-axis (For Left Foot)
            g.append(FLy[k])
            glb.append(np.array([0]))
            gub.append(np.array([0]))

            #Zero Forces z-axis (For Right Foot)
            g.append(FLz[k])
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
                g.append(ca.cross((PR_init-CoM_k),FR_k))
                glb.append(np.array([0,0,0]))
                gub.append(np.array([0,0,0]))
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
                    #Angular Dynamics (Left Support)
                    g.append(ca.cross((PL_init-CoM_k),FL_k) + ca.cross((PR_k-CoM_k),FR_k))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))

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
                    #Angular Dynamics (Right Support)
                    g.append(ca.cross((PL_k-CoM_k),FL_k) + ca.cross((PR_init - CoM_k),FR_k))
                    glb.append(np.array([0,0,0]))
                    gub.append(np.array([0,0,0]))
            else:
                print('No implementation yet')

        #Unilateral Forces
        g.append(FL_k.T@TerrainNorm)
        glb.append(np.array([0]))
        gub.append([np.inf])

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
        glb.append(np.array([0.2]))
        gub.append(np.array([0.3]))
    elif GaitPattern[phase_cnt] == 'LeftSupport':
        g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
        glb.append(np.array([0.5]))
        gub.append(np.array([0.8]))
    elif GaitPattern[phase_cnt] == 'RightSupport':
        g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
        glb.append(np.array([0.5]))
        gub.append(np.array([0.8]))
    elif GaitPattern[phase_cnt] == 'DoubleSupport':
        g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
        glb.append(np.array([0.2]))
        gub.append(np.array([0.3]))

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
FLx_index = (zdot_index[1]+1,zdot_index[1]+Nk_Local*Nphase+1)
FLx_res = x_opt[FLx_index[0]:FLx_index[1]+1]
FLx_res = np.array(FLx_res)
print('FLx_res: ',FLx_res)
FLy_index = (FLx_index[1]+1,FLx_index[1]+Nk_Local*Nphase+1)
FLy_res = x_opt[FLy_index[0]:FLy_index[1]+1]
FLy_res = np.array(FLy_res)
print('FLy_res: ',FLy_res)
FLz_index = (FLy_index[1]+1,FLy_index[1]+Nk_Local*Nphase+1)
FLz_res = x_opt[FLz_index[0]:FLz_index[1]+1]
FLz_res = np.array(FLz_res)
print('FLz_res: ',FLz_res)
FRx_index = (FLz_index[1]+1,FLz_index[1]+Nk_Local*Nphase+1)
FRx_res = x_opt[FRx_index[0]:FRx_index[1]+1]
FRx_res = np.array(FRx_res)
print('FRx_res: ',FRx_res)
FRy_index = (FRx_index[1]+1,FRx_index[1]+Nk_Local*Nphase+1)
FRy_res = x_opt[FRy_index[0]:FRy_index[1]+1]
FRy_res = np.array(FRy_res)
print('FRy_res: ',FRy_res)
FRz_index = (FRy_index[1]+1,FRy_index[1]+Nk_Local*Nphase+1)
FRz_res = x_opt[FRz_index[0]:FRz_index[1]+1]
FRz_res = np.array(FRz_res)
print('FRz_res: ',FRz_res)
px_index = (FRz_index[1]+1,FRz_index[1]+Nstep)
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

plt.plot(FLz_res[0:-1])
plt.show()

