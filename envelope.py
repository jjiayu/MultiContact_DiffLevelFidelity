import casadi as ca

import numpy as np

x = ca.SX.sym('x')
xlb = np.array([0])
xub = np.array([6])

y = ca.SX.sym('y')
ylb = np.array([0])
ulb = np.array([3])

J = x*y - 2*x

var_lb = 