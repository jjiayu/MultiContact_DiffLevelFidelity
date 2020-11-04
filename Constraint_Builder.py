#The File for making Constratins

# Import Important Modules
import numpy as np #Numpy
import casadi as ca #Casadi

def CoM_Kinematics(g = None, glb = None, gub = None, SwingLegIndicator = None, K_polytope = None, k_polytope = None, CoM_k = None, p = None):
    g.append(ca.if_else(SwingLegIndicator,K_polytope@(CoM_k-p)-ca.DM(k_polytope),np.full((len(k_polytope),),-1)))
    glb.append(np.full((len(k_polytope),),-np.inf))
    gub.append(np.full((len(k_polytope),),0))

    return g, glb, gub

def Angular_Momentum_Rate_DoubleSupport(g = None, glb = None, gub = None, SwingLegIndicator = None, Ldot_next = None, Ldot_current = None, h = None, PL = None, PL_TangentX = None, PL_TangentY = None, PR = None, PR_TangentX = None, PR_TangentY = None, CoM_k = None, FL1_k = None, FL2_k = None, FL3_k = None, FL4_k = None, FR1_k = None, FR2_k = None, FR3_k = None, FR4_k = None):
    g.append(ca.if_else(SwingLegIndicator,Ldot_next - Ldot_current - h*(ca.cross((PL+0.11*PL_TangentX+0.06*PL_TangentY-CoM_k),FL1_k) + 
                                                                        ca.cross((PL+0.11*PL_TangentX-0.06*PL_TangentY-CoM_k),FL2_k) + 
                                                                        ca.cross((PL-0.11*PL_TangentX+0.06*PL_TangentY-CoM_k),FL3_k) + 
                                                                        ca.cross((PL-0.11*PL_TangentX-0.06*PL_TangentY-CoM_k),FL4_k) + 
                                                                        ca.cross((PR+0.11*PR_TangentX+0.06*PR_TangentY-CoM_k),FR1_k) + 
                                                                        ca.cross((PR+0.11*PR_TangentX-0.06*PR_TangentY-CoM_k),FR2_k) + 
                                                                        ca.cross((PR-0.11*PR_TangentX+0.06*PR_TangentY-CoM_k),FR3_k) + 
                                                                        ca.cross((PR-0.11*PR_TangentX-0.06*PR_TangentY-CoM_k),FR4_k)),np.array([0,0,0])))
    glb.append(np.array([0,0,0]))
    gub.append(np.array([0,0,0]))

    return g, glb, gub

def Angular_Momentum_Rate_Swing(g = None, glb = None, gub = None, SwingLegIndicator = None, Ldot_next = None, Ldot_current = None, h = None, P = None, P_TangentX = None, P_TangentY = None, CoM_k = None, F1_k = None, F2_k = None, F3_k = None, F4_k = None):
    g.append(ca.if_else(SwingLegIndicator,Ldot_next - Ldot_current - h*(ca.cross((P+0.11*P_TangentX+0.06*P_TangentY-CoM_k),F1_k) + 
                                                                        ca.cross((P+0.11*P_TangentX-0.06*P_TangentY-CoM_k),F2_k) + 
                                                                        ca.cross((P-0.11*P_TangentX+0.06*P_TangentY-CoM_k),F3_k) + 
                                                                        ca.cross((P-0.11*P_TangentX-0.06*P_TangentY-CoM_k),F4_k)),np.array([0,0,0])))
    glb.append(np.array([0,0,0]))
    gub.append(np.array([0,0,0]))

    return g, glb, gub

#Ponton's Convexfication Constraint
def Ponton_Concex_Constraint(g = None, glb = None, gub = None, SwingLegIndicator = None, x_p_bar = None,x_q_bar = None, y_p_bar = None,y_q_bar = None, z_p_bar = None,z_q_bar = None, l = None,f = None):
    l_length = 1.5
    f_length = 400
    a_cvx = np.array([-l[2]/l_length,l[1]/l_length])
    d_cvx = np.array([f[1]/f_length,f[2]/f_length])
    b_cvx = np.array([l[2]/l_length,-l[0]/l_length])
    e_cvx = np.array([f[0]/f_length,f[2]/f_length])
    c_cvx = np.array([-l[1]/l_length,l[0]/l_length])
    f_cvx = np.array([f[0]/f_length,f[1]/f_length])

    x_p = a_cvx + d_cvx
    x_q = a_cvx - d_cvx

    y_p = b_cvx + e_cvx
    y_q = b_cvx - e_cvx

    z_p = c_cvx + f_cvx
    z_q = c_cvx - f_cvx

    g.append(ca.if_else(SwingLegIndicator,x_p_bar-x_p@x_p,np.array([1])))
    glb.append(np.array([0]))
    gub.append(np.array([np.inf]))

    g.append(ca.if_else(SwingLegIndicator,x_q_bar-x_q@x_q,np.array([1])))
    glb.append(np.array([0]))
    gub.append(np.array([np.inf]))

    g.append(ca.if_else(SwingLegIndicator,y_p_bar-y_p@y_p,np.array([1])))
    glb.append(np.array([0]))
    gub.append(np.array([np.inf]))

    g.append(ca.if_else(SwingLegIndicator,y_q_bar-y_q@y_q,np.array([1])))
    glb.append(np.array([0]))
    gub.append(np.array([np.inf]))

    g.append(ca.if_else(SwingLegIndicator,z_p_bar-z_p@z_p,np.array([1])))
    glb.append(np.array([0]))
    gub.append(np.array([np.inf]))

    g.append(ca.if_else(SwingLegIndicator,z_q_bar-z_q@z_q,np.array([1])))
    glb.append(np.array([0]))
    gub.append(np.array([np.inf]))

    return g, glb, gub

#Unilateral Constraints
#Activate a Unilateral Constraint with given Terrain Norm based on the SwingLegIndicator
def Unilateral_Constraints(g = None, glb = None, gub = None, SwingLegIndicator = None, F_k = None, TerrainNorm = None):

    if SwingLegIndicator == None: #For Initial Double Support
        g.append(F_k.T@TerrainNorm)
        glb.append(np.array([0]))
        gub.append([np.inf])
    else:
        #Activating and de-activating depending on the SwingLegIndicator
        g.append(ca.if_else(SwingLegIndicator,F_k.T@TerrainNorm,np.array([1])))
        glb.append(np.array([0]))
        gub.append([np.inf])

    return g, glb, gub

def ZeroForces(g = None, glb = None, gub = None, SwingLegIndicator = None, F_k = None):
    g.append(ca.if_else(SwingLegIndicator,F_k,np.array([0,0,0])))
    glb.append(np.array([0,0,0]))
    gub.append(np.array([0,0,0]))
    
    return g, glb, gub

def FrictionCone(g = None, glb = None, gub = None, SwingLegIndicator = None, F_k = None, TerrainTangentX = None, TerrainTangentY = None, TerrainNorm = None, miu = None):
    
    if SwingLegIndicator == None:
        #For Initial Phase
        #Friction Cone x-axis Set 1
        g.append(F_k.T@TerrainTangentX - miu*F_k.T@TerrainNorm)
        glb.append([-np.inf])
        gub.append(np.array([0]))

        #Friction Cone x-axis Set 2
        g.append(F_k.T@TerrainTangentX + miu*F_k.T@TerrainNorm)
        glb.append(np.array([0]))
        gub.append([np.inf])

        #Friction Cone y-axis Set 1
        g.append(F_k.T@TerrainTangentY - miu*F_k.T@TerrainNorm)
        glb.append([-np.inf])
        gub.append(np.array([0]))   

        #Friction Cone y-axis Set 2
        g.append(F_k.T@TerrainTangentY + miu*F_k.T@TerrainNorm)
        glb.append(np.array([0]))
        gub.append([np.inf])
    
    else:
        #Activate based on the SwingLegIndicator
        #Friction Cone x-axis Set 1
        g.append(ca.if_else(SwingLegIndicator,F_k.T@TerrainTangentX - miu*F_k.T@TerrainNorm,np.array([-1])))
        glb.append([-np.inf])
        gub.append(np.array([0]))

        #Friction Cone x-axis Set 2
        g.append(ca.if_else(SwingLegIndicator,F_k.T@TerrainTangentX + miu*F_k.T@TerrainNorm,np.array([1])))
        glb.append(np.array([0]))
        gub.append([np.inf])

        #Friction Cone y-axis Set 1
        g.append(ca.if_else(SwingLegIndicator,F_k.T@TerrainTangentY - miu*F_k.T@TerrainNorm,np.array([-1])))
        glb.append([-np.inf])
        gub.append(np.array([0]))   

        #Friction Cone y-axis Set 2
        g.append(ca.if_else(SwingLegIndicator,F_k.T@TerrainTangentY + miu*F_k.T@TerrainNorm,np.array([1])))
        glb.append(np.array([0]))
        gub.append([np.inf])

    return g, glb, gub


#Unilateral Constraints
def Unilateral_Constraints_and_Friction_FirstLevel(miu = 0.3,
                                                   g = None, glb = None, gub = None,
                                                   PhaseIndicator = None,
                                                   PhaseNumber = None,
                                                   LeftSwingFirstFlag = None,
                                                   RightSwingFirstFlag = None,
                                                   FL_k = None, FR_k = None,
                                                   PL_init_TangentX = None, PL_init_TangentY = None, PL_init_Norm = None,
                                                   PR_init_TangentX = None, PR_init_TangentY = None, PR_init_Norm = None,
                                                   Pcurrent_TangentX = None, Pcurrent_TangentY = None, Pcurrent_Norm = None,
                                                   Pnext_TangentX = None, Pnext_TangentY = None, Pnext_Norm = None):

    if PhaseIndicator == "InitialDouble":
        #Unilateral Constraints
        #Left Foot
        g.append(FL_k.T@PL_init_Norm)
        glb.append(np.array([0]))
        gub.append([np.inf])

        #Right Foot
        g.append(FR_k.T@PR_init_Norm)
        glb.append(np.array([0]))
        gub.append([np.inf])

    elif PhaseIndicator == "Swing":
        #Unilateral Constraints

        #If swing the left foot first
        #Then there is no point to put unilateral constraints on Left Foot 
        g.append(ca.if_else(LeftSwingFirstFlag,np.array([1]),FL_k.T@PL_init_Norm))
        glb.append(np.array([0]))
        gub.append([np.inf])
        #But we have to constrain zero force constraint on the Left foot
        g.append(ca.if_else(LeftSwingFirstFlag,FL_k,np.array([0,0,0])))
        glb.append(np.array([0,0,0]))
        gub.append(np.array([0,0,0]))
        #And unlaterial force constraint on the Right foot
        g.append(ca.if_else(LeftSwingFirstFlag,FR_k.T@PR_init_Norm,np.array([1])))
        glb.append(np.array([0]))
        gub.append([np.inf])


        #If swing the right foot first
        #Then there is no point to put unilateral constraints Right Foot
        g.append(ca.if_else(RightSwingFirstFlag,np.array([1]),FR_k.T@PR_init_Norm))
        glb.append(np.array([0]))
        gub.append([np.inf])
        #But we have to constrain zero force constraint on the Right foot
        g.append(ca.if_else(RightSwingFirstFlag,FR_k,np.array([0,0,0])))
        glb.append(np.array([0,0,0]))
        gub.append(np.array([0,0,0]))
        #And unlaterial force constraint on the Left foot
        g.append(ca.if_else(RightSwingFirstFlag,FL_k.T@PL_init_Norm,np.array([1])))
        glb.append(np.array([0]))
        gub.append([np.inf])

    elif PhaseIndicator == "DoubleSupport":

        #If swing the left foot first
        #Then the right foot is the stance foot (current/Initial), 
        g.append(ca.if_else(LeftSwingFirstFlag,np.array([1]),FR_k.T@PR_init_Norm))
        glb.append(np.array([0]))
        gub.append([np.inf])
        #the left foot is the newly landing foot
        g.append(ca.if_else(LeftSwingFirstFlag,np.array([1]),FL_k.T@Pnext_Norm))
        glb.append(np.array([0]))
        gub.append([np.inf])

        #If swing the right foot first
        #Then the left foot is the stance foot (current/initial), 
        g.append(ca.if_else(RightSwingFirstFlag,np.array([1]),FL_k.T@PL_init_Norm))
        glb.append(np.array([0]))
        gub.append([np.inf])
        #the left right is the newly landing foot
        g.append(ca.if_else(RightSwingFirstFlag,np.array([1]),FR_k.T@Pnext_Norm))
        glb.append(np.array([0]))
        gub.append([np.inf])


    return g, glb, gub