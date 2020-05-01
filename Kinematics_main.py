#Fully Kinematics Problem, No check of dynamics, therefore, even no time involved
#Currently Resemble the SL1M formuation
#   two CoM konts for a step. One footstep location knot for a step - Footstep location vector represent the 

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
GaitPattern = ['Left','Right','Left','Right','Left','Right'] #Left support -> Double Support (land Right Foot) -> Right Support
#   Number of Steps
Nstep = 3 #Enumeration of the Steps start from 0, so Number of Steps - 1, use mod function to check it is left or right
#   Define Initial Condition
x_init = 0
#y_init = 0
z_init = 0.6
#   Initial Contact Foot Location, according to the defined gait pattern, it is a left foot
px_init = 0
py_init = 0
pz_init = 0
#   Define Terminal Condition
x_end = 0.7
#y_end = 0
z_end = 0.6
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
#Define Variables and Lower and Upper Bounds
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
px = []
pxlb = []
pxub = []
py = []
pylb = []
pyub = []
pz = []
pzlb = []
pzub = []

for stepIdx in range(Nstep):
    #For CoM state, each phase/step has two knots
    xtemp = ca.SX.sym('x'+str(stepIdx+1),2)
    x.append(xtemp)
    xlb.append(np.array([-5,-5]))
    xub.append(np.array([5,5]))

    ytemp = ca.SX.sym('y'+str(stepIdx+1),2)
    y.append(ytemp)
    ylb.append(np.array([-2,-2]))
    yub.append(np.array([2,2]))

    ztemp = ca.SX.sym('z'+str(stepIdx+1),2)
    z.append(ztemp)
    zlb.append(np.array([-3,-3]))
    zub.append(np.array([3,3]))

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

x = ca.vertcat(*x)
y = ca.vertcat(*y)
z = ca.vertcat(*z)
px = ca.vertcat(*px)
py = ca.vertcat(*py)
pz = ca.vertcat(*pz)

DecisionVars = ca.vertcat(x,y,z,px,py,pz) #treat all elements in the list as single variables
DecisionVarsShape = DecisionVars.shape #get decision variable list shape, for future use

DecisionVars_lb = np.concatenate((xlb,ylb,zlb,pxlb,pylb,pzlb),axis=None)
DecisionVars_ub = np.concatenate((xub,yub,zub,pxub,pyub,pzub),axis=None)

#   Temporary Code for array/matrices multiplication
#x = ca.SX.sym('x',2)
#y = ca.DM(np.array([[1,2]]))
#z = x.T@y
#print(z)
#-----------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------
#Define Constrains
#   Build Containers
g = []
glb = []
gub = []

#   Initial Condition
#g.append(x[0]-x_init)
g.append(x[0]-px_init)
glb.append(np.array([0]))
gub.append(np.array([0]))

#g.append(y[0]-y_init)
#glb.append(np.array([0]))
#gub.append(np.array([0]))

#g.append(y[0]-py_init)
#glb.append(np.array([0]))
#gub.append(np.array([0]))

g.append(z[0]-z_init)
glb.append(np.array([0]))
gub.append(np.array([0]))

#   Terminal Condition
g.append(x[-1]-x_end)
glb.append(np.array([0]))
gub.append(np.array([0]))

#g.append(y[-1]-y_end)
#glb.append(np.array([0]))
#gub.append(np.array([0]))

g.append(z[-1]-z_end)
glb.append(np.array([0]))
gub.append(np.array([0]))

#-----------------------------
#g.append(x[-1]-x_end)
#glb.append(np.array([0]))
#gub.append(np.array([0]))

g.append(px[-1]-x[-1])
glb.append(np.array([0]))
gub.append(np.array([np.inf]))

#g.append(y[-1]-py[-1])
#glb.append(np.array([0]))
#gub.append(np.array([0]))

#g.append(z[-1]-z_end)
#glb.append(np.array([0]))
#gub.append(np.array([0]))

#   Loop over to have Kinematics Constraint
for Nph in range(Nstep):
    print('Phase: '+ str(Nph+1))
    CoM_0 = ca.vertcat(x[2*Nph],y[2*Nph],z[2*Nph])
    print(CoM_0)
    CoM_1 = ca.vertcat(x[2*Nph+1],y[2*Nph+1],z[2*Nph+1])
    print(CoM_1)
    if Nph == 0:
        P_k_current = ca.DM(np.array([px_init,py_init,pz_init]))
        P_k_next = ca.vertcat(px[Nph],py[Nph],pz[Nph])
    else:
        P_k_current = ca.vertcat(px[Nph-1],py[Nph-1],pz[Nph-1])
        P_k_next = ca.vertcat(px[Nph],py[Nph],pz[Nph])
    print(P_k_current)
    print(P_k_next)

    #Put Kinematics Constraint

    if Nph%2 == 0: #even number, left foot in contact for p_current, right foot is going to land as p_next
        print('Left in contact, move right')
        #CoM Kinemactis Constraint for the support foot (CoM_0 in *current* contact -> left)
        g.append(K_CoM_Left@(CoM_0-P_k_current))
        glb.append(np.full((len(k_CoM_Left),),-np.inf))
        gub.append(k_CoM_Left)
        #CoM Kinemactis Constraint for the support foot (CoM_0 in *next* contact -> right)
        g.append(K_CoM_Right@(CoM_0-P_k_next))
        glb.append(np.full((len(k_CoM_Right),),-np.inf))
        gub.append(k_CoM_Right)
        #CoM Kinemactis Constraint for the swing foot (CoM_1 land but in the *current* contact -> left)
        g.append(K_CoM_Left@(CoM_1-P_k_current))
        glb.append(np.full((len(k_CoM_Left),),-np.inf))
        gub.append(k_CoM_Left)
        #CoM Kinemactis Constraint for the swing foot (CoM_1 land but in the *next* contact -> right)
        g.append(K_CoM_Right@(CoM_1-P_k_next))
        glb.append(np.full((len(k_CoM_Right),),-np.inf))
        gub.append(k_CoM_Right)
        #Relative Swing Foot Location (rf in lf)
        g.append(Q_rf_in_lf@(P_k_next-P_k_current))
        glb.append(np.full((len(q_rf_in_lf),),-np.inf))
        gub.append(q_rf_in_lf)

        #test
        #g.append(Q_lf_in_rf@(P_k_next-P_k_current))
        #glb.append(np.full((len(q_lf_in_rf),),-np.inf))
        #gub.append(q_lf_in_rf)

    elif Nph%2 ==1: #odd number, right foot in contact for p_current, left foot is going to land at p_next
        print('Right in contact, move left')
        #CoM Kinemactis Constraint for the support foot (CoM_0 in *current* contact -> right)
        g.append(K_CoM_Right@(CoM_0-P_k_current))
        glb.append(np.full((len(k_CoM_Right),),-np.inf))
        gub.append(k_CoM_Right)
        #CoM Kinemactis Constraint for the support foot (CoM_0 in *next* contact -> left)
        g.append(K_CoM_Left@(CoM_0-P_k_next))
        glb.append(np.full((len(k_CoM_Left),),-np.inf))
        gub.append(k_CoM_Left)
        #CoM Kinemactis Constraint for the swing foot (CoM_1 land but in the *current contact* -> right)
        g.append(K_CoM_Right@(CoM_1-P_k_current))
        glb.append(np.full((len(k_CoM_Right),),-np.inf))
        gub.append(k_CoM_Right)
        #CoM Kinemactis Constraint for the swing foot (CoM_1 land but in the *next contact* -> left)
        g.append(K_CoM_Left@(CoM_1-P_k_next))
        glb.append(np.full((len(k_CoM_Left),),-np.inf))
        gub.append(k_CoM_Left)
        #Relative Swing Foot Location (lf in rf)
        g.append(Q_lf_in_rf@(P_k_next-P_k_current))
        glb.append(np.full((len(q_lf_in_rf),),-np.inf))
        gub.append(q_lf_in_rf)

    #Regulate the CoM of each phase
    g.append(CoM_0[0]-P_k_current[0])
    glb.append(np.array([-0.05]))
    gub.append(np.array([0.05]))

    g.append(P_k_next[0]-CoM_1[0])
    glb.append(np.array([-0.05]))
    gub.append(np.array([0.05]))

#g.append(K_CoM_Left@ca.vertcat(x[0],y[0],z[0]))
#glb.append(np.full((len(k_CoM_Left),),-np.inf))
#gub.append(k_CoM_Left)

#   reshape all constraints
g = ca.vertcat(*g)
glb = np.concatenate(glb)
gub = np.concatenate(gub)

#g = x[0]+x[1]
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
#Define Cost Function
J = 0
#-----y------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
#Generate Initial Guess
#   Random Initial Guess
#       Shuffle the Random Seed Generator
np.random.seed()
DecisionVars_init = DecisionVars_lb + np.multiply(np.random.rand(DecisionVarsShape[0],DecisionVarsShape[1]).flatten(),(DecisionVars_ub-DecisionVars_lb))
#   Fixed Value Initial Guess
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
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
#Result Extraction
x_index = (0,0+Nstep*2-1)
x_res = x_opt[x_index[0]:x_index[1]+1]
x_res = np.array(x_res)
print('x_res: ',x_res)
y_index = (x_index[1]+1,x_index[1]+Nstep*2)
y_res = x_opt[y_index[0]:y_index[1]+1]
y_res = np.array(y_res)
print('y_res: ',y_res)
z_index = (y_index[1]+1,y_index[1]+Nstep*2)
z_res = x_opt[z_index[0]:z_index[1]+1]
z_res = np.array(z_res)
print('z_res: ',z_res)
Px_index = (z_index[1]+1,z_index[1]+Nstep)
Px_res = x_opt[Px_index[0]:Px_index[1]+1]
Px_res = np.array(Px_res)
print('Px_res: ',Px_res)
Py_index = (Px_index[1]+1,Px_index[1]+Nstep)
Py_res = x_opt[Py_index[0]:Py_index[1]+1]
Py_res = np.array(Py_res)
print('Py_res: ',Py_res)
Pz_index = (Py_index[1]+1,Py_index[1]+Nstep)
Pz_res = x_opt[Pz_index[0]:Pz_index[1]+1]
Pz_res = np.array(Pz_res)
print('Pz_res: ',Pz_res)
#-----------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------
#Plot Result
#------------------------------------------------------------------------------------
fig=plt.figure()
ax = Axes3D(fig)
#ax = fig.add_subplot(111, projection="3d")

ax.plot3D(x_res,y_res,z_res,color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)
ax.set_xlim3d(0, 1)
ax.set_ylim3d(-0.5,0.5)
ax.set_zlim3d(0,0.8)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

#Initial Footstep Locations
ax.scatter(px_init, py_init, pz_init, c='r', marker='o', linewidth = 10) 
ax.scatter(px_init, py_init-0.25, pz_init, c='b', marker='o', linewidth = 10) 

for FrameNum in range(Nstep):
    if FrameNum%2 == 0: #even number, left foot in contact for p_current, right foot is going to land as p_next
        StepColor = 'b'
    elif FrameNum%2 == 1: #odd number, right foot in contact for p_current, left foot is going to land as p_next
        StepColor = 'r'
    
    ax.scatter(Px_res[FrameNum], Py_res[FrameNum], Pz_res[FrameNum], c=StepColor, marker='o', linewidth = 10) 

ax.view_init(elev=8.776933438381377, azim=-99.32358055821186)
plt.show()
#x = ca.SX.sym('x'); y = ca.SX.sym('y'); z = ca.SX.sym('z')
#nlp = {'x':ca.vertcat(x,y,z), 'f':x**2+100*z**2, 'g':z+(1-x)**2-y}
#solver = ca.nlpsol('nlp_solver', 'ipopt', nlp)

#res = solver(x0=[2.5,3.0,0.75], lbx=-5, ubx=5, lbg=0, ubg=0)
#x_opt = res['x']
#print('x_opt: ', x_opt)
#plt.plot(x_opt)
#plt.show()

#fig=plt.figure()
#ax = fig.add_subplot(111, projection="3d")
#ax.scatter(x_opt[0], x_opt[4], x_opt[8], c='r', marker='o', linewidth = 5) 
#ax=fig.add_axes([0,0,1,1])
#ax.scatter(, girls_grades, color='r')
#ax.scatter(grades_range, boys_grades, color='b')
#ax.set_xlabel('Grades Range')
#ax.set_ylabel('Grades Scored')
#ax.set_title('scatter plot')
plt.show()
