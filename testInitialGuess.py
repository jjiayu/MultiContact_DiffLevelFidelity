from casadi import *
import numpy as np

# Lotka-Volerra problem from mintoc.de

T = 12. # Time horizon
N = 400 # number of control intervals

# Model parameters
c0 = .4
c1 = .2

# Initial condition for x
x0 = [0.5, 0.7]

# Bounds on x
lbx = [0., 0.]
ubx = [2., 2.]

# Bounds on u
lbu = 0.
ubu = 1.

# Declare model variables
x = SX.sym('x', 2)
u = SX.sym('u')

# Model equations
xdot = vertcat(x[0] - x[0] * x[1] - c0 * x[0] * u,
              -x[1] + x[0] * x[1] - c1 * x[1] * u)

# Objective term
L = (x[0] - 1)**2 + (x[1] - 1)**2 + 1e-4*u**2

# Formulate discrete time dynamics
if False:
   # CVODES from the SUNDIALS suite
   dae = {'x':x, 'p':u, 'ode':xdot, 'quad':L}
   opts = {'tf':T/N}
   F = integrator('F', 'cvodes', dae, opts)
else:
   # Fixed step Runge-Kutta 4 integrator
   M = 4 # RK4 steps per interval
   DT = T/N/M
   f = Function('f', [x, u], [xdot, L])
   X0 = MX.sym('X0', 2)
   U = MX.sym('U')
   X = X0
   Q = 0
   for j in range(M):
       k1, k1_q = f(X, U)
       k2, k2_q = f(X + DT/2 * k1, U)
       k3, k3_q = f(X + DT/2 * k2, U)
       k4, k4_q = f(X + DT * k3, U)
       X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
       Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
   F = Function('F', [X0, U], [X, Q],['x0','p'],['xf','qf'])

# Initial guess for u
u_start = [DM(0.)] * N

# Get a feasible trajectory as an initial guess
xk = DM(x0)
x_start = [xk]
for k in range(N):
    xk = F(x0=xk, p=u_start[k])['xf']
    x_start += [xk]

# Start with an empty NLP
w=[]
w0 = []
lbw = []
ubw = []
discrete = []
J = 0
g=[]
lbg = []
ubg = []

# "Lift" initial conditions
X0 = MX.sym('X0', 2)
w += [X0]
lbw += x0
ubw += x0
w0 += [x_start[0]]
discrete += [False, False]

# Formulate the NLP
Xk = X0
for k in range(N):
    # New NLP variable for the control
    Uk = MX.sym('U_' + str(k))
    w   += [Uk]
    lbw += [lbu]
    ubw += [ubu]
    w0  += [u_start[k]]
    discrete += [True]

    # Integrate till the end of the interval
    Fk = F(x0=Xk, p=Uk)
    Xk_end = Fk['xf']
    J=J+Fk['qf']

    # New NLP variable for state at end of interval
    Xk = MX.sym('X_' + str(k+1), 2)
    w   += [Xk]
    lbw += lbx
    ubw += ubx
    w0  += [x_start[k+1]]
    discrete += [False, False]

    # Add equality constraint
    g   += [Xk_end-Xk]
    lbg += [0, 0]
    ubg += [0, 0]

# Concatenate decision variables and constraint terms
w = vertcat(*w)
g = vertcat(*g)

# Create an NLP solver
nlp_prob = {'f': J, 'x': w, 'g': g}
#nlp_solver = nlpsol('nlp_solver', 'bonmin', nlp_prob, {"discrete": discrete});
#nlp_solver = nlpsol('nlp_solver', 'knitro', nlp_prob, {"discrete": discrete});


opts = {}
# opts["knitro.presolve"] = 0
# opts["knitro.honorbnds"] = 0
# opts["knitro.OutLev"] = 3
# nlp_solver = nlpsol('nlp_solver', 'knitro', nlp_prob,opts); # Solve relaxed problem

# opts["ipopt.bound_push"] = 1e-7
# opts["ipopt.bound_frac"] = 1e-7
nlp_solver = nlpsol('nlp_solver', 'ipopt', nlp_prob,opts); # Solve relaxed problem

# Plot the solution
tgrid = [T/N*k for k in range(N+1)]
import matplotlib.pyplot as plt
plt.figure(1)
plt.clf()
def plot_sol(w_opt):
    w_opt = w_opt.full().flatten()
    x0_opt = w_opt[0::3]
    x1_opt = w_opt[1::3]
    u_opt = w_opt[2::3]
    plt.plot(tgrid, x0_opt, '--')
    plt.plot(tgrid, x1_opt, '-')
    plt.step(tgrid, vertcat(DM.nan(1), u_opt), '-.')
    plt.xlabel('t')
    plt.legend(['x0','x1','u'])
    plt.grid(True)

# Solve the NLP
w_init = np.array(vertcat(*w0))
lbw = np.array(lbw)
ubw = np.array(ubw)

np.random.seed()
initVarsShape = w_init.shape

initvar = lbw + np.multiply(np.random.rand(initVarsShape[0],).flatten(),(ubw - lbw))#   Fixed Value Initial Guess

sol = nlp_solver(x0=initvar, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
#sol = nlp_solver(x0=vertcat(*w0), lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

print("Initial Seed")
print(vertcat(*w0))

print(nlp_solver.stats())

w1_opt = sol['x']
lam_w_opt = sol['lam_x']
lam_g_opt = sol['lam_g']
plot_sol(w1_opt)

plt.show()

initvat = w1_opt

#print Second round

print("Second Round - Random Seed")

sol = nlp_solver(x0=initvat, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

w1_opt = sol['x']
lam_w_opt = sol['lam_x']
lam_g_opt = sol['lam_g']
plot_sol(w1_opt)

plt.show()

print(nlp_solver.stats())

