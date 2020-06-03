import curves  # noqa - necessary to register curves::bezier_curve
import numpy as np
from numpy import array
from hpp_centroidal_dynamics import Equilibrium, EquilibriumAlgorithm, SolverLP
from hpp_bezier_com_traj import (SOLVER_QUADPROG, ConstraintFlag, Constraints, ContactData, ProblemData,
                                 computeCOMTraj, zeroStepCapturability)

#importing tools to plot bezier curves
from curves.plot import (plotBezier)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# testing constructors
eq = Equilibrium("test", 54., 4)
eq = Equilibrium("test", 54., 4, SolverLP.SOLVER_LP_QPOASES)
eq = Equilibrium("test", 54., 4, SolverLP.SOLVER_LP_QPOASES)
eq = Equilibrium("test", 54., 4, SolverLP.SOLVER_LP_QPOASES, False)
eq = Equilibrium("test", 54., 4, SolverLP.SOLVER_LP_QPOASES, False, 1)
eq = Equilibrium("test", 54., 4, SolverLP.SOLVER_LP_QPOASES, True, 1, True)

# whether useWarmStart is enable (True by default)
previous = eq.useWarmStart()
# enable warm start in solver (only for QPOases)
eq.setUseWarmStart(False)
assert (previous != eq.useWarmStart())

# access solver name
assert (eq.getName() == "test")

z = array([0., 0., 1.])
P = array([array([x, y, 0]) for x in [-0.05, 0.05] for y in [-0.1, 0.1]])
N = array([z for _ in range(4)])

# setting contact positions and normals, as well as friction coefficients
eq.setNewContacts(P, N, 0.3, EquilibriumAlgorithm.EQUILIBRIUM_ALGORITHM_PP)
# eq.setNewContacts(P,N,0.3,EquilibriumAlgorithm.EQUILIBRIUM_ALGORITHM_LP)

# setting up optimization problem
c0 = array([0., 0., 1.])
# dc0 = array(np.random.uniform(-1, 1, size=3));
dc0 = array([0.1, 0., 0.])
l0 = array([0., 0., 0.])
T = 1.2
tstep = -1.

a = zeroStepCapturability(eq, c0, dc0, l0, False, T, tstep)

assert (a.success)