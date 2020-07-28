import curves  # noqa - necessary to register curves::bezier_curve
import numpy as np
from numpy import array
from hpp_centroidal_dynamics import Equilibrium, EquilibriumAlgorithm, SolverLP
from hpp_bezier_com_traj import (SOLVER_QUADPROG, ConstraintFlag, Constraints, ContactData, ProblemData,
                                 computeCOMTraj, zeroStepCapturability)

# Import SL1M modules, Mainly for Kinematcis Constraints
from sl1m.constants_and_tools import *
from sl1m.planner import *
from sl1m.stand_alone_scenarios.constraints import *

#Get Kinematics Constraints
#   CoM Kinematics Constraint for Talos
kinematicConstraints = genKinematicConstraints(left_foot_constraints, right_foot_constraints)
K_CoM_Left = kinematicConstraints[0][0]
k_CoM_Left = kinematicConstraints[0][1]
K_CoM_Right = kinematicConstraints[1][0]
k_CoM_Right = kinematicConstraints[1][1]

#Check Kinematics Constraints
#print((K_CoM_Left.dot(array([0,0,1]))-k_CoM_Left).max())


#Define Contact Positions and Normal Vectors (of the contact surface)
z = array([0., 0., 1.])
P = array([array([x, y, 0]) for x in [-0.11, 0.11] for y in [-0.06, 0.06]])
N = array([z for _ in range(4)])

# setting up optimization problem
c0 = array([0., 0., 0.5])
# dc0 = array(np.random.uniform(-1, 1, size=3));
dc0 = array([0.1, 0., 0.])


#Build Problem Description
pD = ProblemData()
pD.constraints_.flag_ = ConstraintFlag.INIT_POS | ConstraintFlag.INIT_VEL | ConstraintFlag.END_VEL


def initContactData(pD, K_CoM, k_CoM, ContactLocs, Norms):

    #Set-up Contact Data
    cData = ContactData(Equilibrium("test", 95., 4))
    cData.contactPhase_.setNewContacts(ContactLocs, Norms, 0.3, EquilibriumAlgorithm.EQUILIBRIUM_ALGORITHM_PP)
    cData.setKinematicConstraints(K_CoM, k_CoM)
    Id = array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    cData.setAngularConstraints(Id, array([0., 0., 1.]))
    pD.addContact(cData)
    #print(cData.Kin_)


initContactData(pD, K_CoM_Left, k_CoM_Left, P, N)
initContactData(pD, K_CoM_Left, k_CoM_Left, P, N)
initContactData(pD, K_CoM_Left, k_CoM_Left, P, N)

#[initContactData(pD) for i in range(3)]

pD.c0_ = c0
pD.dc0_ = dc0
res = computeCOMTraj(pD, array([0.4, 0.4, 0.4]), -1)

print(res.success)