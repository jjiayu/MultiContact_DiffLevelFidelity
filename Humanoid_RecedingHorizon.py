#Receding Horizon Planning Framework for Humanoid Robot with Single Fidelity or Multi-fidelity
import numpy as np #Numpy
import casadi as ca #Casadi
import matplotlib.pyplot as plt #Matplotlib
from mpl_toolkits.mplot3d import Axes3D
# Import SL1M modules
from sl1m.constants_and_tools import *
from sl1m.planner import *
from constraints import *
from Humanoid_ProblemDescription import *
from PlotResult import *

#   Set Decimal Printing Precision
np.set_printoptions(precision=4)

solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index = BuildSolver(FirstLevel = "NLP_SingleStep", SecondLevel = "Pure_Kinematics_Check", m = 95)
#solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index = BuildSolver(FirstLevel = "Pure_Kinematics_Check", SecondLevel = None, m = 95)


#Generate Initial Guess
#   Random Initial Guess
#       Shuffle the Random Seed Generator
np.random.seed()
DecisionVarsShape = DecisionVars_lb.shape
DecisionVars_init = DecisionVars_lb + np.multiply(np.random.rand(DecisionVarsShape[0],).flatten(),(DecisionVars_ub-DecisionVars_lb))#   Fixed Value Initial Guess

#Receding Horizon
#Define Problem Parameters
#Swing Left Foot
LeftSwingFlag = 1
RightSwingFlag = 0

x_init = 0
y_init = 0
z_init = 0.6

PLx_init = 0
PLy_init = 0.1
PLz_init = 0

PRx_init = 0
PRy_init = -0.1
PRz_init = 0

x_end = 5
y_end = 0
z_end = 0.6

ParaList = [LeftSwingFlag,RightSwingFlag,x_init,y_init,z_init,PLx_init,PLy_init,PLz_init,PRx_init,PRy_init,PRz_init,x_end,y_end,z_end]

res = solver(x0=DecisionVars_init, p = ParaList, lbx = DecisionVars_lb, ubx = DecisionVars_ub, lbg = glb, ubg = gub)
x_opt = res["x"]
x_opt = x_opt.full().flatten()
x_opt_left = x_opt
res_fig = Plot_Both_Levels(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
#res_fig = PlotNLPStep(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
#res_fig = Plot_Pure_Kinematics_Plan(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)

