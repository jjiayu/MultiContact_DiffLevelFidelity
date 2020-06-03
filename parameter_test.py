import casadi as ca
import numpy as np

# define params
x = ca.MX.sym('x',3); 
p = ca.MX.sym('p',3);

# short-circuiting if_else / conditional in your expression graph
g = ca.if_else(p[0], x-1, 0)

# build nlp problem
nlp = {'x':ca.vertcat(x), 'f':x[0]**2, 'g':g, 'p':p }

#  construct solver
S = ca.nlpsol('S', 'ipopt', nlp)

# init conditions, parameter value and constraint bounds
r = S(x0=[2.5,2.5,2.5], p = [1,1,1], lbg=[0,0,0], ubg=[0,0,0] )

# solve
x_opt = r['x']
print('x_opt: ', x_opt)