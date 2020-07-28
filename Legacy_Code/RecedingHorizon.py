import numpy as np #Numpy
import casadi as ca #Casadi

from Humanoid_NLP_Constructor import *

#Build the problem
solver, DecisionVars_init, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index = Humanoid_NLP_MultiFidelity_Constructor(withSecondLevel = False)

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

res = solver(x0=DecisionVars_init, p = ParaList, lbx=DecisionVars_lb, ubx=DecisionVars_ub, lbg=glb, ubg=gub)
x_opt = res['x']
x_opt = x_opt.full().flatten()
x_opt_left = x_opt

CostComputation(Nphase=3,Nk_Local=5,x_opt=x_opt,var_index=var_index,G = 9.80665,m=95)

res_fig = PlotNLPStep(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)

#Second Step
#   Swing Right Foot
LeftSwingFlag = 0
RightSwingFlag = 1

#Update Init and Terminal Positions
x_res = x_opt[var_index["x"][0]:var_index["x"][1]+1]
x_init = x_res[-1]
y_res = x_opt[var_index["y"][0]:var_index["y"][1]+1]
y_init = y_res[-1]
z_res = x_opt[var_index["z"][0]:var_index["z"][1]+1]
z_init = z_res[-1]

px_res = x_opt[var_index["px"][0]:var_index["px"][1]+1]
PLx_init = px_res[-1]
py_res = x_opt[var_index["py"][0]:var_index["py"][1]+1]
PLy_init = py_res[-1]
pz_res = x_opt[var_index["pz"][0]:var_index["pz"][1]+1]
PLz_init = pz_res[-1]

ParaList = [LeftSwingFlag,RightSwingFlag,x_init,y_init,z_init,PLx_init,PLy_init,PLz_init,PRx_init,PRy_init,PRz_init,x_end,y_end,z_end]

res = solver(x0=x_opt, p = ParaList, lbx=DecisionVars_lb, ubx=DecisionVars_ub, lbg=glb, ubg=gub)
x_opt = res['x']
x_opt = x_opt.full().flatten()
x_opt_right = x_opt

x_res = x_opt[var_index["x"][0]:var_index["x"][1]+1]
print('x_result',x_res)
px_res = x_opt[var_index["px"][0]:var_index["px"][1]+1]
print('px_result',px_res)
py_res = x_opt[var_index["py"][0]:var_index["py"][1]+1]
print('py_result',py_res)

CostComputation(Nphase=3,Nk_Local=5,x_opt=x_opt,var_index=var_index,G = 9.80665,m=95)

res_fig = PlotNLPStep(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)

#Third Step 
#Swing Left Foot
#   Swing Right Foot
LeftSwingFlag = 1
RightSwingFlag = 0

#Update Init and Terminal Positions
x_res = x_opt[var_index["x"][0]:var_index["x"][1]+1]
x_init = x_res[-1]
y_res = x_opt[var_index["y"][0]:var_index["y"][1]+1]
y_init = y_res[-1]
z_res = x_opt[var_index["z"][0]:var_index["z"][1]+1]
z_init = z_res[-1]

px_res = x_opt[var_index["px"][0]:var_index["px"][1]+1]
PRx_init = px_res[-1]
py_res = x_opt[var_index["py"][0]:var_index["py"][1]+1]
PRy_init = py_res[-1]
pz_res = x_opt[var_index["pz"][0]:var_index["pz"][1]+1]
PRz_init = pz_res[-1]

ParaList = [LeftSwingFlag,RightSwingFlag,x_init,y_init,z_init,PLx_init,PLy_init,PLz_init,PRx_init,PRy_init,PRz_init,x_end,y_end,z_end]

res = solver(x0=x_opt_left, p = ParaList, lbx=DecisionVars_lb, ubx=DecisionVars_ub, lbg=glb, ubg=gub)
x_opt = res['x']
x_opt = x_opt.full().flatten()
x_opt_left = x_opt

x_res = x_opt[var_index["x"][0]:var_index["x"][1]+1]
print('x_result',x_res)

CostComputation(Nphase=3,Nk_Local=5,x_opt=x_opt,var_index=var_index,G = 9.80665,m=95)

res_fig = PlotNLPStep(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)

#Forth Step 
#   Swing Right Foot
LeftSwingFlag = 0
RightSwingFlag = 1

#Update Init and Terminal Positions
x_res = x_opt[var_index["x"][0]:var_index["x"][1]+1]
x_init = x_res[-1]
y_res = x_opt[var_index["y"][0]:var_index["y"][1]+1]
y_init = y_res[-1]
z_res = x_opt[var_index["z"][0]:var_index["z"][1]+1]
z_init = z_res[-1]

px_res = x_opt[var_index["px"][0]:var_index["px"][1]+1]
PLx_init = px_res[-1]
py_res = x_opt[var_index["py"][0]:var_index["py"][1]+1]
PLy_init = py_res[-1]
pz_res = x_opt[var_index["pz"][0]:var_index["pz"][1]+1]
PLz_init = pz_res[-1]

ParaList = [LeftSwingFlag,RightSwingFlag,x_init,y_init,z_init,PLx_init,PLy_init,PLz_init,PRx_init,PRy_init,PRz_init,x_end,y_end,z_end]

res = solver(x0=x_opt_right, p = ParaList, lbx=DecisionVars_lb, ubx=DecisionVars_ub, lbg=glb, ubg=gub)
x_opt = res['x']
x_opt = x_opt.full().flatten()
x_opt_right = x_opt

x_res = x_opt[var_index["x"][0]:var_index["x"][1]+1]
print('x_result',x_res)

CostComputation(Nphase=3,Nk_Local=5,x_opt=x_opt,var_index=var_index,G = 9.80665,m=95)

res_fig = PlotNLPStep(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)

#Fifth Step 
#   Swing Left Foot
LeftSwingFlag = 1
RightSwingFlag = 0

#Update Init and Terminal Positions
x_res = x_opt[var_index["x"][0]:var_index["x"][1]+1]
x_init = x_res[-1]
y_res = x_opt[var_index["y"][0]:var_index["y"][1]+1]
y_init = y_res[-1]
z_res = x_opt[var_index["z"][0]:var_index["z"][1]+1]
z_init = z_res[-1]

px_res = x_opt[var_index["px"][0]:var_index["px"][1]+1]
PRx_init = px_res[-1]
py_res = x_opt[var_index["py"][0]:var_index["py"][1]+1]
PRy_init = py_res[-1]
pz_res = x_opt[var_index["pz"][0]:var_index["pz"][1]+1]
PRz_init = pz_res[-1]

ParaList = [LeftSwingFlag,RightSwingFlag,x_init,y_init,z_init,PLx_init,PLy_init,PLz_init,PRx_init,PRy_init,PRz_init,x_end,y_end,z_end]

res = solver(x0=x_opt_left, p = ParaList, lbx=DecisionVars_lb, ubx=DecisionVars_ub, lbg=glb, ubg=gub)
x_opt = res['x']
x_opt = x_opt.full().flatten()
x_opt_left = x_opt

x_res = x_opt[var_index["x"][0]:var_index["x"][1]+1]
print('x_result',x_res)

CostComputation(Nphase=3,Nk_Local=5,x_opt=x_opt,var_index=var_index,G = 9.80665,m=95)

res_fig = PlotNLPStep(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)

#Sixth Step 
#   Swing Right Foot
LeftSwingFlag = 0
RightSwingFlag = 1

#Update Init and Terminal Positions
x_res = x_opt[var_index["x"][0]:var_index["x"][1]+1]
x_init = x_res[-1]
y_res = x_opt[var_index["y"][0]:var_index["y"][1]+1]
y_init = y_res[-1]
z_res = x_opt[var_index["z"][0]:var_index["z"][1]+1]
z_init = z_res[-1]

px_res = x_opt[var_index["px"][0]:var_index["px"][1]+1]
PLx_init = px_res[-1]
py_res = x_opt[var_index["py"][0]:var_index["py"][1]+1]
PLy_init = py_res[-1]
pz_res = x_opt[var_index["pz"][0]:var_index["pz"][1]+1]
PLz_init = pz_res[-1]

ParaList = [LeftSwingFlag,RightSwingFlag,x_init,y_init,z_init,PLx_init,PLy_init,PLz_init,PRx_init,PRy_init,PRz_init,x_end,y_end,z_end]

res = solver(x0=x_opt_right, p = ParaList, lbx=DecisionVars_lb, ubx=DecisionVars_ub, lbg=glb, ubg=gub)
x_opt = res['x']
x_opt = x_opt.full().flatten()
x_opt_right = x_opt

x_res = x_opt[var_index["x"][0]:var_index["x"][1]+1]
print('x_result',x_res)

CostComputation(Nphase=3,Nk_Local=5,x_opt=x_opt,var_index=var_index,G = 9.80665,m=95)

res_fig = PlotNLPStep(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)

#Seventh Step 
#   Swing Left Foot
LeftSwingFlag = 1
RightSwingFlag = 0

#Update Init and Terminal Positions
x_res = x_opt[var_index["x"][0]:var_index["x"][1]+1]
x_init = x_res[-1]
y_res = x_opt[var_index["y"][0]:var_index["y"][1]+1]
y_init = y_res[-1]
z_res = x_opt[var_index["z"][0]:var_index["z"][1]+1]
z_init = z_res[-1]

px_res = x_opt[var_index["px"][0]:var_index["px"][1]+1]
PRx_init = px_res[-1]
py_res = x_opt[var_index["py"][0]:var_index["py"][1]+1]
PRy_init = py_res[-1]
pz_res = x_opt[var_index["pz"][0]:var_index["pz"][1]+1]
PRz_init = pz_res[-1]

ParaList = [LeftSwingFlag,RightSwingFlag,x_init,y_init,z_init,PLx_init,PLy_init,PLz_init,PRx_init,PRy_init,PRz_init,x_end,y_end,z_end]

res = solver(x0=x_opt_left, p = ParaList, lbx=DecisionVars_lb, ubx=DecisionVars_ub, lbg=glb, ubg=gub)
x_opt = res['x']
x_opt = x_opt.full().flatten()
x_opt_left = x_opt

x_res = x_opt[var_index["x"][0]:var_index["x"][1]+1]
print('x_result',x_res)

CostComputation(Nphase=3,Nk_Local=5,x_opt=x_opt,var_index=var_index,G = 9.80665,m=95)

res_fig = PlotNLPStep(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)

#Eighth Step 
#   Swing Right Foot
LeftSwingFlag = 0
RightSwingFlag = 1

#Update Init and Terminal Positions
x_res = x_opt[var_index["x"][0]:var_index["x"][1]+1]
x_init = x_res[-1]
y_res = x_opt[var_index["y"][0]:var_index["y"][1]+1]
y_init = y_res[-1]
z_res = x_opt[var_index["z"][0]:var_index["z"][1]+1]
z_init = z_res[-1]

px_res = x_opt[var_index["px"][0]:var_index["px"][1]+1]
PLx_init = px_res[-1]
py_res = x_opt[var_index["py"][0]:var_index["py"][1]+1]
PLy_init = py_res[-1]
pz_res = x_opt[var_index["pz"][0]:var_index["pz"][1]+1]
PLz_init = pz_res[-1]

ParaList = [LeftSwingFlag,RightSwingFlag,x_init,y_init,z_init,PLx_init,PLy_init,PLz_init,PRx_init,PRy_init,PRz_init,x_end,y_end,z_end]

res = solver(x0=x_opt_right, p = ParaList, lbx=DecisionVars_lb, ubx=DecisionVars_ub, lbg=glb, ubg=gub)
x_opt = res['x']
x_opt = x_opt.full().flatten()
x_opt_right = x_opt

x_res = x_opt[var_index["x"][0]:var_index["x"][1]+1]
print('x_result',x_res)

CostComputation(Nphase=3,Nk_Local=5,x_opt=x_opt,var_index=var_index,G = 9.80665,m=95)

res_fig = PlotNLPStep(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)

#Ninth Step 
#   Swing Left Foot
LeftSwingFlag = 1
RightSwingFlag = 0

#Update Init and Terminal Positions
x_res = x_opt[var_index["x"][0]:var_index["x"][1]+1]
x_init = x_res[-1]
y_res = x_opt[var_index["y"][0]:var_index["y"][1]+1]
y_init = y_res[-1]
z_res = x_opt[var_index["z"][0]:var_index["z"][1]+1]
z_init = z_res[-1]

px_res = x_opt[var_index["px"][0]:var_index["px"][1]+1]
PRx_init = px_res[-1]
py_res = x_opt[var_index["py"][0]:var_index["py"][1]+1]
PRy_init = py_res[-1]
pz_res = x_opt[var_index["pz"][0]:var_index["pz"][1]+1]
PRz_init = pz_res[-1]

ParaList = [LeftSwingFlag,RightSwingFlag,x_init,y_init,z_init,PLx_init,PLy_init,PLz_init,PRx_init,PRy_init,PRz_init,x_end,y_end,z_end]

res = solver(x0=x_opt_left, p = ParaList, lbx=DecisionVars_lb, ubx=DecisionVars_ub, lbg=glb, ubg=gub)
x_opt = res['x']
x_opt = x_opt.full().flatten()
x_opt_left = x_opt

x_res = x_opt[var_index["x"][0]:var_index["x"][1]+1]
print('x_result',x_res)

CostComputation(Nphase=3,Nk_Local=5,x_opt=x_opt,var_index=var_index,G = 9.80665,m=95)

res_fig = PlotNLPStep(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)

#Tenth Step 
#   Swing Right Foot
LeftSwingFlag = 0
RightSwingFlag = 1

#Update Init and Terminal Positions
x_res = x_opt[var_index["x"][0]:var_index["x"][1]+1]
x_init = x_res[-1]
y_res = x_opt[var_index["y"][0]:var_index["y"][1]+1]
y_init = y_res[-1]
z_res = x_opt[var_index["z"][0]:var_index["z"][1]+1]
z_init = z_res[-1]

px_res = x_opt[var_index["px"][0]:var_index["px"][1]+1]
PLx_init = px_res[-1]
py_res = x_opt[var_index["py"][0]:var_index["py"][1]+1]
PLy_init = py_res[-1]
pz_res = x_opt[var_index["pz"][0]:var_index["pz"][1]+1]
PLz_init = pz_res[-1]

ParaList = [LeftSwingFlag,RightSwingFlag,x_init,y_init,z_init,PLx_init,PLy_init,PLz_init,PRx_init,PRy_init,PRz_init,x_end,y_end,z_end]

res = solver(x0=x_opt_right, p = ParaList, lbx=DecisionVars_lb, ubx=DecisionVars_ub, lbg=glb, ubg=gub)
x_opt = res['x']
x_opt = x_opt.full().flatten()
x_opt_right = x_opt

x_res = x_opt[var_index["x"][0]:var_index["x"][1]+1]
print('x_result',x_res)

CostComputation(Nphase=3,Nk_Local=5,x_opt=x_opt,var_index=var_index,G = 9.80665,m=95)

res_fig = PlotNLPStep(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)

#Eleventh Step 
#   Swing Left Foot
LeftSwingFlag = 1
RightSwingFlag = 0

#Update Init and Terminal Positions
x_res = x_opt[var_index["x"][0]:var_index["x"][1]+1]
x_init = x_res[-1]
y_res = x_opt[var_index["y"][0]:var_index["y"][1]+1]
y_init = y_res[-1]
z_res = x_opt[var_index["z"][0]:var_index["z"][1]+1]
z_init = z_res[-1]

px_res = x_opt[var_index["px"][0]:var_index["px"][1]+1]
PRx_init = px_res[-1]
py_res = x_opt[var_index["py"][0]:var_index["py"][1]+1]
PRy_init = py_res[-1]
pz_res = x_opt[var_index["pz"][0]:var_index["pz"][1]+1]
PRz_init = pz_res[-1]

ParaList = [LeftSwingFlag,RightSwingFlag,x_init,y_init,z_init,PLx_init,PLy_init,PLz_init,PRx_init,PRy_init,PRz_init,x_end,y_end,z_end]

res = solver(x0=x_opt_left, p = ParaList, lbx=DecisionVars_lb, ubx=DecisionVars_ub, lbg=glb, ubg=gub)
x_opt = res['x']
x_opt = x_opt.full().flatten()

x_res = x_opt[var_index["x"][0]:var_index["x"][1]+1]
print('x_result',x_res)

CostComputation(Nphase=3,Nk_Local=5,x_opt=x_opt,var_index=var_index,G = 9.80665,m=95)

res_fig = PlotNLPStep(x_opt=x_opt,fig=None,var_index=var_index,PL_init=np.array([PLx_init,PLy_init,PLz_init]),PR_init = np.array([PRx_init,PRy_init,PRz_init]), LeftSwing = LeftSwingFlag, RightSwing = RightSwingFlag)

