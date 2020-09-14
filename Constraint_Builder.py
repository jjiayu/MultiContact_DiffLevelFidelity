#The File for making Constratins

# Import Important Modules
import numpy as np #Numpy
import casadi as ca #Casadi

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