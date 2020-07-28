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

solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index = BuildSolver(FirstLevel = "NLP_SingleStep", SecondLevel = "CoM_Dynamics", ConservativeFirstStep = False, m = 95)
#solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index = BuildSolver(FirstLevel = "NLP_SingleStep", SecondLevel = None, ConservativeFirstStep = False, m = 95)
#solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index = BuildSolver(FirstLevel = "NLP_SingleStep", SecondLevel = "Pure_Kinematics_Check", ConservativeFirstStep = True, m = 95)
#solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index = BuildSolver(FirstLevel = "Pure_Kinematics_Check", SecondLevel = None, m = 95)

#backup var_index
var_index_Level1 = var_index["Level1_Var_Index"]

#Generate Initial Guess
#   Random Initial Guess
#       Shuffle the Random Seed Generator
np.random.seed()
DecisionVarsShape = DecisionVars_lb.shape
DecisionVars_init = DecisionVars_lb + np.multiply(np.random.rand(DecisionVarsShape[0],).flatten(),(DecisionVars_ub-DecisionVars_lb))#   Fixed Value Initial Guess

#Receding Horizon
#The First Step
#Define Problem Parameters
#Swing Left Foot
LeftSwingFlag = 1
RightSwingFlag = 0

x_init = 0
y_init = 0
z_init = 0.6

xdot_init = 0
ydot_init = 0
zdot_init = 0

PLx_init = 0
PLy_init = 0.1
PLz_init = 0

PRx_init = 0
PRy_init = -0.1
PRz_init = 0

x_end = 5
y_end = 0
z_end = 0.6

xdot_end = 0
ydot_end = 0
zdot_end = 0

ParaList = [LeftSwingFlag,RightSwingFlag,
            x_init,y_init,z_init,
            xdot_init,ydot_init,zdot_init,
            PLx_init,PLy_init,PLz_init,
            PRx_init,PRy_init,PRz_init,
            x_end,y_end,z_end,
            xdot_end,ydot_end,zdot_end]

res = solver(x0=DecisionVars_init, p = ParaList, lbx = DecisionVars_lb, ubx = DecisionVars_ub, lbg = glb, ubg = gub)
x_opt = res["x"]
x_opt = x_opt.full().flatten()
x_opt_left = x_opt
#res_fig = Plot_Both_Levels(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
#res_fig = PlotNLPStep(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
#res_fig = Plot_Pure_Kinematics_Plan(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
PlotSingleOptimiation_and_PrintResult(x_opt = x_opt, var_index=var_index, PL_init = np.array([PLx_init,PLy_init,PLz_init]), PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag, PrintSecondLevel = False, PlotNLP = True, PlotBothLevel = True)


#The Second Step
#   Swing Right Foot
LeftSwingFlag = 0
RightSwingFlag = 1

#Update Init and Terminal Positions
x_res = x_opt[var_index_Level1["x"][0]:var_index_Level1["x"][1]+1]
x_init = x_res[-1]
y_res = x_opt[var_index_Level1["y"][0]:var_index_Level1["y"][1]+1]
y_init = y_res[-1]
z_res = x_opt[var_index_Level1["z"][0]:var_index_Level1["z"][1]+1]
z_init = z_res[-1]

xdot_res  = x_opt[var_index_Level1["xdot"][0]:var_index_Level1["xdot"][1]+1]
xdot_init = xdot_res[-1]
ydot_res  = x_opt[var_index_Level1["ydot"][0]:var_index_Level1["ydot"][1]+1]
ydot_init = ydot_res[-1]
zdot_res  = x_opt[var_index_Level1["zdot"][0]:var_index_Level1["zdot"][1]+1]
zdot_init = zdot_res[-1]

px_res = x_opt[var_index_Level1["px"][0]:var_index_Level1["px"][1]+1]
PLx_init = px_res[-1]
py_res = x_opt[var_index_Level1["py"][0]:var_index_Level1["py"][1]+1]
PLy_init = py_res[-1]
pz_res = x_opt[var_index_Level1["pz"][0]:var_index_Level1["pz"][1]+1]
PLz_init = pz_res[-1]

ParaList = [LeftSwingFlag,RightSwingFlag,
            x_init,y_init,z_init,
            xdot_init,ydot_init,zdot_init,
            PLx_init,PLy_init,PLz_init,
            PRx_init,PRy_init,PRz_init,
            x_end,y_end,z_end,
            xdot_end,ydot_end,zdot_end]

res = solver(x0=x_opt, p = ParaList, lbx = DecisionVars_lb, ubx = DecisionVars_ub, lbg = glb, ubg = gub)
x_opt = res["x"]
x_opt = x_opt.full().flatten()
x_opt_right = x_opt
#res_fig = Plot_Both_Levels(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
#res_fig = PlotNLPStep(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
#res_fig = Plot_Pure_Kinematics_Plan(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
PlotSingleOptimiation_and_PrintResult(x_opt = x_opt, var_index=var_index, PL_init = np.array([PLx_init,PLy_init,PLz_init]), PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag, PrintSecondLevel = False, PlotNLP = True, PlotBothLevel = True)

#The Third Step
#   Swing Left Foot
LeftSwingFlag = 1
RightSwingFlag = 0

#Update Init and Terminal Positions
x_res = x_opt[var_index_Level1["x"][0]:var_index_Level1["x"][1]+1]
x_init = x_res[-1]
y_res = x_opt[var_index_Level1["y"][0]:var_index_Level1["y"][1]+1]
y_init = y_res[-1]
z_res = x_opt[var_index_Level1["z"][0]:var_index_Level1["z"][1]+1]
z_init = z_res[-1]

xdot_res  = x_opt[var_index_Level1["xdot"][0]:var_index_Level1["xdot"][1]+1]
xdot_init = xdot_res[-1]
ydot_res  = x_opt[var_index_Level1["ydot"][0]:var_index_Level1["ydot"][1]+1]
ydot_init = ydot_res[-1]
zdot_res  = x_opt[var_index_Level1["zdot"][0]:var_index_Level1["zdot"][1]+1]
zdot_init = zdot_res[-1]

px_res = x_opt[var_index_Level1["px"][0]:var_index_Level1["px"][1]+1]
PRx_init = px_res[-1]
py_res = x_opt[var_index_Level1["py"][0]:var_index_Level1["py"][1]+1]
PRy_init = py_res[-1]
pz_res = x_opt[var_index_Level1["pz"][0]:var_index_Level1["pz"][1]+1]
PRz_init = pz_res[-1]

ParaList = [LeftSwingFlag,RightSwingFlag,
            x_init,y_init,z_init,
            xdot_init,ydot_init,zdot_init,
            PLx_init,PLy_init,PLz_init,
            PRx_init,PRy_init,PRz_init,
            x_end,y_end,z_end,
            xdot_end,ydot_end,zdot_end]

res = solver(x0=x_opt_left, p = ParaList, lbx = DecisionVars_lb, ubx = DecisionVars_ub, lbg = glb, ubg = gub)
x_opt = res["x"]
x_opt = x_opt.full().flatten()
x_opt_left = x_opt
#res_fig = Plot_Both_Levels(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
#res_fig = PlotNLPStep(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
#res_fig = Plot_Pure_Kinematics_Plan(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
PlotSingleOptimiation_and_PrintResult(x_opt = x_opt, var_index=var_index, PL_init = np.array([PLx_init,PLy_init,PLz_init]), PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag, PrintSecondLevel = False, PlotNLP = True, PlotBothLevel = True)


#The Forth Step
#   Swing Right Foot
LeftSwingFlag = 0
RightSwingFlag = 1

#Update Init and Terminal Positions
x_res = x_opt[var_index_Level1["x"][0]:var_index_Level1["x"][1]+1]
x_init = x_res[-1]
y_res = x_opt[var_index_Level1["y"][0]:var_index_Level1["y"][1]+1]
y_init = y_res[-1]
z_res = x_opt[var_index_Level1["z"][0]:var_index_Level1["z"][1]+1]
z_init = z_res[-1]

xdot_res  = x_opt[var_index_Level1["xdot"][0]:var_index_Level1["xdot"][1]+1]
xdot_init = xdot_res[-1]
ydot_res  = x_opt[var_index_Level1["ydot"][0]:var_index_Level1["ydot"][1]+1]
ydot_init = ydot_res[-1]
zdot_res  = x_opt[var_index_Level1["zdot"][0]:var_index_Level1["zdot"][1]+1]
zdot_init = zdot_res[-1]

px_res = x_opt[var_index_Level1["px"][0]:var_index_Level1["px"][1]+1]
PLx_init = px_res[-1]
py_res = x_opt[var_index_Level1["py"][0]:var_index_Level1["py"][1]+1]
PLy_init = py_res[-1]
pz_res = x_opt[var_index_Level1["pz"][0]:var_index_Level1["pz"][1]+1]
PLz_init = pz_res[-1]

ParaList = [LeftSwingFlag,RightSwingFlag,
            x_init,y_init,z_init,
            xdot_init,ydot_init,zdot_init,
            PLx_init,PLy_init,PLz_init,
            PRx_init,PRy_init,PRz_init,
            x_end,y_end,z_end,
            xdot_end,ydot_end,zdot_end]

res = solver(x0=x_opt_right, p = ParaList, lbx = DecisionVars_lb, ubx = DecisionVars_ub, lbg = glb, ubg = gub)
x_opt = res["x"]
x_opt = x_opt.full().flatten()
x_opt_right = x_opt
#res_fig = Plot_Both_Levels(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
#res_fig = PlotNLPStep(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
#res_fig = Plot_Pure_Kinematics_Plan(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
PlotSingleOptimiation_and_PrintResult(x_opt = x_opt, var_index=var_index, PL_init = np.array([PLx_init,PLy_init,PLz_init]), PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag, PrintSecondLevel = False, PlotNLP = True, PlotBothLevel = True)


#The Fifth Step
#   Swing Left Foot
LeftSwingFlag = 1
RightSwingFlag = 0

#Update Init and Terminal Positions
x_res = x_opt[var_index_Level1["x"][0]:var_index_Level1["x"][1]+1]
x_init = x_res[-1]
y_res = x_opt[var_index_Level1["y"][0]:var_index_Level1["y"][1]+1]
y_init = y_res[-1]
z_res = x_opt[var_index_Level1["z"][0]:var_index_Level1["z"][1]+1]
z_init = z_res[-1]

xdot_res  = x_opt[var_index_Level1["xdot"][0]:var_index_Level1["xdot"][1]+1]
xdot_init = xdot_res[-1]
ydot_res  = x_opt[var_index_Level1["ydot"][0]:var_index_Level1["ydot"][1]+1]
ydot_init = ydot_res[-1]
zdot_res  = x_opt[var_index_Level1["zdot"][0]:var_index_Level1["zdot"][1]+1]
zdot_init = zdot_res[-1]

px_res = x_opt[var_index_Level1["px"][0]:var_index_Level1["px"][1]+1]
PRx_init = px_res[-1]
py_res = x_opt[var_index_Level1["py"][0]:var_index_Level1["py"][1]+1]
PRy_init = py_res[-1]
pz_res = x_opt[var_index_Level1["pz"][0]:var_index_Level1["pz"][1]+1]
PRz_init = pz_res[-1]

ParaList = [LeftSwingFlag,RightSwingFlag,
            x_init,y_init,z_init,
            xdot_init,ydot_init,zdot_init,
            PLx_init,PLy_init,PLz_init,
            PRx_init,PRy_init,PRz_init,
            x_end,y_end,z_end,
            xdot_end,ydot_end,zdot_end]

res = solver(x0=x_opt_left, p = ParaList, lbx = DecisionVars_lb, ubx = DecisionVars_ub, lbg = glb, ubg = gub)
x_opt = res["x"]
x_opt = x_opt.full().flatten()
x_opt_left= x_opt
#res_fig = Plot_Both_Levels(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
#res_fig = PlotNLPStep(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
#res_fig = Plot_Pure_Kinematics_Plan(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
PlotSingleOptimiation_and_PrintResult(x_opt = x_opt, var_index=var_index, PL_init = np.array([PLx_init,PLy_init,PLz_init]), PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag, PrintSecondLevel = False, PlotNLP = True, PlotBothLevel = True)

#The Sixth Step
#   Swing Right Foot
LeftSwingFlag = 0
RightSwingFlag = 1

#Update Init and Terminal Positions
x_res = x_opt[var_index_Level1["x"][0]:var_index_Level1["x"][1]+1]
x_init = x_res[-1]
y_res = x_opt[var_index_Level1["y"][0]:var_index_Level1["y"][1]+1]
y_init = y_res[-1]
z_res = x_opt[var_index_Level1["z"][0]:var_index_Level1["z"][1]+1]
z_init = z_res[-1]

xdot_res  = x_opt[var_index_Level1["xdot"][0]:var_index_Level1["xdot"][1]+1]
xdot_init = xdot_res[-1]
ydot_res  = x_opt[var_index_Level1["ydot"][0]:var_index_Level1["ydot"][1]+1]
ydot_init = ydot_res[-1]
zdot_res  = x_opt[var_index_Level1["zdot"][0]:var_index_Level1["zdot"][1]+1]
zdot_init = zdot_res[-1]

px_res = x_opt[var_index_Level1["px"][0]:var_index_Level1["px"][1]+1]
PLx_init = px_res[-1]
py_res = x_opt[var_index_Level1["py"][0]:var_index_Level1["py"][1]+1]
PLy_init = py_res[-1]
pz_res = x_opt[var_index_Level1["pz"][0]:var_index_Level1["pz"][1]+1]
PLz_init = pz_res[-1]

ParaList = [LeftSwingFlag,RightSwingFlag,
            x_init,y_init,z_init,
            xdot_init,ydot_init,zdot_init,
            PLx_init,PLy_init,PLz_init,
            PRx_init,PRy_init,PRz_init,
            x_end,y_end,z_end,
            xdot_end,ydot_end,zdot_end]

res = solver(x0=x_opt_right, p = ParaList, lbx = DecisionVars_lb, ubx = DecisionVars_ub, lbg = glb, ubg = gub)
x_opt = res["x"]
x_opt = x_opt.full().flatten()
x_opt_right = x_opt
#res_fig = Plot_Both_Levels(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
#res_fig = PlotNLPStep(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
#res_fig = Plot_Pure_Kinematics_Plan(Rightx_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
PlotSingleOptimiation_and_PrintResult(x_opt = x_opt, var_index=var_index, PL_init = np.array([PLx_init,PLy_init,PLz_init]), PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag, PrintSecondLevel = False, PlotNLP = True, PlotBothLevel = True)

#The Seventh Step
#   Swing Left Foot
LeftSwingFlag = 1
RightSwingFlag = 0

#Update Init and Terminal Positions
x_res = x_opt[var_index_Level1["x"][0]:var_index_Level1["x"][1]+1]
x_init = x_res[-1]
y_res = x_opt[var_index_Level1["y"][0]:var_index_Level1["y"][1]+1]
y_init = y_res[-1]
z_res = x_opt[var_index_Level1["z"][0]:var_index_Level1["z"][1]+1]
z_init = z_res[-1]

xdot_res  = x_opt[var_index_Level1["xdot"][0]:var_index_Level1["xdot"][1]+1]
xdot_init = xdot_res[-1]
ydot_res  = x_opt[var_index_Level1["ydot"][0]:var_index_Level1["ydot"][1]+1]
ydot_init = ydot_res[-1]
zdot_res  = x_opt[var_index_Level1["zdot"][0]:var_index_Level1["zdot"][1]+1]
zdot_init = zdot_res[-1]

px_res = x_opt[var_index_Level1["px"][0]:var_index_Level1["px"][1]+1]
PRx_init = px_res[-1]
py_res = x_opt[var_index_Level1["py"][0]:var_index_Level1["py"][1]+1]
PRy_init = py_res[-1]
pz_res = x_opt[var_index_Level1["pz"][0]:var_index_Level1["pz"][1]+1]
PRz_init = pz_res[-1]

ParaList = [LeftSwingFlag,RightSwingFlag,
            x_init,y_init,z_init,
            xdot_init,ydot_init,zdot_init,
            PLx_init,PLy_init,PLz_init,
            PRx_init,PRy_init,PRz_init,
            x_end,y_end,z_end,
            xdot_end,ydot_end,zdot_end]

res = solver(x0=x_opt_left, p = ParaList, lbx = DecisionVars_lb, ubx = DecisionVars_ub, lbg = glb, ubg = gub)
x_opt = res["x"]
x_opt = x_opt.full().flatten()
x_opt_left = x_opt
#res_fig = Plot_Both_Levels(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
#res_fig = PlotNLPStep(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
#res_fig = Plot_Pure_Kinematics_Plan(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
PlotSingleOptimiation_and_PrintResult(x_opt = x_opt, var_index=var_index, PL_init = np.array([PLx_init,PLy_init,PLz_init]), PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag, PrintSecondLevel = False, PlotNLP = True, PlotBothLevel = True)

#The Eighth Step
#   Swing Right Foot
LeftSwingFlag = 0
RightSwingFlag = 1

#Update Init and Terminal Positions
x_res = x_opt[var_index_Level1["x"][0]:var_index_Level1["x"][1]+1]
x_init = x_res[-1]
y_res = x_opt[var_index_Level1["y"][0]:var_index_Level1["y"][1]+1]
y_init = y_res[-1]
z_res = x_opt[var_index_Level1["z"][0]:var_index_Level1["z"][1]+1]
z_init = z_res[-1]

xdot_res  = x_opt[var_index_Level1["xdot"][0]:var_index_Level1["xdot"][1]+1]
xdot_init = xdot_res[-1]
ydot_res  = x_opt[var_index_Level1["ydot"][0]:var_index_Level1["ydot"][1]+1]
ydot_init = ydot_res[-1]
zdot_res  = x_opt[var_index_Level1["zdot"][0]:var_index_Level1["zdot"][1]+1]
zdot_init = zdot_res[-1]

px_res = x_opt[var_index_Level1["px"][0]:var_index_Level1["px"][1]+1]
PLx_init = px_res[-1]
py_res = x_opt[var_index_Level1["py"][0]:var_index_Level1["py"][1]+1]
PLy_init = py_res[-1]
pz_res = x_opt[var_index_Level1["pz"][0]:var_index_Level1["pz"][1]+1]
PLz_init = pz_res[-1]

ParaList = [LeftSwingFlag,RightSwingFlag,
            x_init,y_init,z_init,
            xdot_init,ydot_init,zdot_init,
            PLx_init,PLy_init,PLz_init,
            PRx_init,PRy_init,PRz_init,
            x_end,y_end,z_end,
            xdot_end,ydot_end,zdot_end]

res = solver(x0=x_opt_right, p = ParaList, lbx = DecisionVars_lb, ubx = DecisionVars_ub, lbg = glb, ubg = gub)
x_opt = res["x"]
x_opt = x_opt.full().flatten()
x_opt_right = x_opt
#res_fig = Plot_Both_Levels(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
#res_fig = PlotNLPStep(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
#res_fig = Plot_Pure_Kinematics_Plan(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
PlotSingleOptimiation_and_PrintResult(x_opt = x_opt, var_index=var_index, PL_init = np.array([PLx_init,PLy_init,PLz_init]), PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag, PrintSecondLevel = False, PlotNLP = True, PlotBothLevel = True)

#The Ninth Step
#   Swing Left Foot
LeftSwingFlag = 1
RightSwingFlag = 0

#Update Init and Terminal Positions
x_res = x_opt[var_index_Level1["x"][0]:var_index_Level1["x"][1]+1]
x_init = x_res[-1]
y_res = x_opt[var_index_Level1["y"][0]:var_index_Level1["y"][1]+1]
y_init = y_res[-1]
z_res = x_opt[var_index_Level1["z"][0]:var_index_Level1["z"][1]+1]
z_init = z_res[-1]

xdot_res  = x_opt[var_index_Level1["xdot"][0]:var_index_Level1["xdot"][1]+1]
xdot_init = xdot_res[-1]
ydot_res  = x_opt[var_index_Level1["ydot"][0]:var_index_Level1["ydot"][1]+1]
ydot_init = ydot_res[-1]
zdot_res  = x_opt[var_index_Level1["zdot"][0]:var_index_Level1["zdot"][1]+1]
zdot_init = zdot_res[-1]

px_res = x_opt[var_index_Level1["px"][0]:var_index_Level1["px"][1]+1]
PRx_init = px_res[-1]
py_res = x_opt[var_index_Level1["py"][0]:var_index_Level1["py"][1]+1]
PRy_init = py_res[-1]
pz_res = x_opt[var_index_Level1["pz"][0]:var_index_Level1["pz"][1]+1]
PRz_init = pz_res[-1]

ParaList = [LeftSwingFlag,RightSwingFlag,
            x_init,y_init,z_init,
            xdot_init,ydot_init,zdot_init,
            PLx_init,PLy_init,PLz_init,
            PRx_init,PRy_init,PRz_init,
            x_end,y_end,z_end,
            xdot_end,ydot_end,zdot_end]

res = solver(x0=x_opt_left, p = ParaList, lbx = DecisionVars_lb, ubx = DecisionVars_ub, lbg = glb, ubg = gub)
x_opt = res["x"]
x_opt = x_opt.full().flatten()
x_opt_left = x_opt
#res_fig = Plot_Both_Levels(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
#res_fig = PlotNLPStep(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
#res_fig = Plot_Pure_Kinematics_Plan(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
PlotSingleOptimiation_and_PrintResult(x_opt = x_opt, var_index=var_index, PL_init = np.array([PLx_init,PLy_init,PLz_init]), PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag, PrintSecondLevel = False, PlotNLP = True, PlotBothLevel = True)

#The Seventh Step
#   Swing Right Foot
LeftSwingFlag = 0
RightSwingFlag = 1

#Update Init and Terminal Positions
x_res = x_opt[var_index_Level1["x"][0]:var_index_Level1["x"][1]+1]
x_init = x_res[-1]
y_res = x_opt[var_index_Level1["y"][0]:var_index_Level1["y"][1]+1]
y_init = y_res[-1]
z_res = x_opt[var_index_Level1["z"][0]:var_index_Level1["z"][1]+1]
z_init = z_res[-1]

xdot_res  = x_opt[var_index_Level1["xdot"][0]:var_index_Level1["xdot"][1]+1]
xdot_init = xdot_res[-1]
ydot_res  = x_opt[var_index_Level1["ydot"][0]:var_index_Level1["ydot"][1]+1]
ydot_init = ydot_res[-1]
zdot_res  = x_opt[var_index_Level1["zdot"][0]:var_index_Level1["zdot"][1]+1]
zdot_init = zdot_res[-1]

px_res = x_opt[var_index_Level1["px"][0]:var_index_Level1["px"][1]+1]
PLx_init = px_res[-1]
py_res = x_opt[var_index_Level1["py"][0]:var_index_Level1["py"][1]+1]
PLy_init = py_res[-1]
pz_res = x_opt[var_index_Level1["pz"][0]:var_index_Level1["pz"][1]+1]
PLz_init = pz_res[-1]

ParaList = [LeftSwingFlag,RightSwingFlag,
            x_init,y_init,z_init,
            xdot_init,ydot_init,zdot_init,
            PLx_init,PLy_init,PLz_init,
            PRx_init,PRy_init,PRz_init,
            x_end,y_end,z_end,
            xdot_end,ydot_end,zdot_end]

res = solver(x0=x_opt_right, p = ParaList, lbx = DecisionVars_lb, ubx = DecisionVars_ub, lbg = glb, ubg = gub)
x_opt = res["x"]
x_opt = x_opt.full().flatten()
x_opt_right = x_opt
#res_fig = Plot_Both_Levels(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
#res_fig = PlotNLPStep(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
#res_fig = Plot_Pure_Kinematics_Plan(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)
PlotSingleOptimiation_and_PrintResult(x_opt = x_opt, var_index=var_index, PL_init = np.array([PLx_init,PLy_init,PLz_init]), PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag, PrintSecondLevel = False, PlotNLP = True, PlotBothLevel = True)
