# Import Important Modules
import numpy as np #Numpy
import casadi as ca #Casadi
import matplotlib.pyplot as plt #Matplotlib

#-----------------------------------------------------------------------------------------------------------------------
#Decide Parameters
#   Gait Pattern, Each action is followed up by a double support phase
GaitPattern = ['Left','Right','Left','Right','Left','Right']
#   Number of Steps
Nstep = 1 #Enumeration of the Steps start from 0, so Number of Steps - 1, use mod function to check it is left or right
#   Number of Knots per Phase - how many intervals do we have for a single phase; 10 intervals need 11 knots/ticks
Nk_swing = 10 #for a single swing phase
Nk_double = 10 #for a double support phase
#   Compute Number of total knots/ticks, but the enumeration start from 0 to N_K-1
N_K = Nstep*(Nk_swing+Nk_double) + 1
#   Duration of a Phase
T_swing = 0.8 #0.8 to 1.2 is a nominal number (See CROC)
T_double = 0.2 #0.2 is typycal number of dynamic case, if we start from null velocity we should change to 
#   Time Discretization
h_swing = T_swing/Nk_swing
h_double = T_double/Nk_double
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
#Define Variables
#   CoM Position
x = ca.SX.sym('x',N_K)
xdot = ca.SX.sym('xdot',N_K)
#   Collect all Decision Variables
DecisionVars = ca.vertcat(x,xdot)
print(DecisionVars)
#   
DecisionVarsShape = DecisionVars.shape

#   Temporary Code for array/matrices multiplication
#x = ca.SX.sym('x',2)
#y = ca.DM(np.array([[1,2]]))
#z = x.T@y
#print(z)

#-----------------------------------------------------------------------------------------------------------------------
#Define Constrains
g = x[0]+x[1]
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
#Define Cost Function
J = 0
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
#Lower and Upper Bounds of Variables
x_lb = np.array([[0]*(x.shape[0]*x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
x_ub = np.array([[5]*(x.shape[0]*x.shape[1])])

xdot_lb = np.array([[0]*(xdot.shape[0]*xdot.shape[1])]) 
xdot_ub = np.array([[5]*(xdot.shape[0]*xdot.shape[1])])
#   Collect all lower bound and upper bound
DecisionVars_lb = np.concatenate(((x_lb,xdot_lb)),axis=None)
DecisionVars_ub = np.concatenate(((x_ub,xdot_ub)),axis=None)
print(DecisionVars_lb)
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
#Generate Initial Guess
#   Random Initial Guess
#       Shuffle the Random Seed Generator
np.random.seed()
x_init = np.random.rand(DecisionVarsShape[0],DecisionVarsShape[1])
#   Fixed Value Initial Guess
# x_init = np.array([[1.5]*N_K])
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
#Build Solver
prob = {'x': DecisionVars, 'f': J, 'g': g}
solver = ca.nlpsol('solver', 'ipopt', prob)

res = solver(x0=x_init, lbx=DecisionVars_lb, ubx=DecisionVars_ub, lbg=3, ubg=3)
x_opt = res['x']
print('x_opt: ', x_opt)
#-----------------------------------------------------------------------------------------------------------------------

#x = ca.SX.sym('x'); y = ca.SX.sym('y'); z = ca.SX.sym('z')
#nlp = {'x':ca.vertcat(x,y,z), 'f':x**2+100*z**2, 'g':z+(1-x)**2-y}
#solver = ca.nlpsol('nlp_solver', 'ipopt', nlp)

#res = solver(x0=[2.5,3.0,0.75], lbx=-5, ubx=5, lbg=0, ubg=0)
#x_opt = res['x']
#print('x_opt: ', x_opt)
plt.plot(x_opt[0:N_K])
plt.show()