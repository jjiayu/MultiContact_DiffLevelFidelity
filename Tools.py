import numpy as np
from scipy.spatial.transform import Rotation as R

def FirstLevelCost(Nk_Local=5,x_opt=None,var_index=None,G = 9.80665,m=95):
    
    cost_val=0
    cost_acc = 0
    cost_momentum = 0
    cost_momentum_rate = 0

    #Number of NLP Phases
    NLPphase=3

    #parameter setup
    N_K = Nk_Local*NLPphase + 1
    #Time Span Setup
    tau_upper_limit = 1
    tauStepLength = tau_upper_limit/(N_K-1) #Get the interval length, total number of knots - 1

    #Retrieve Compputation results

    var_index = var_index["Level1_Var_Index"]

    Ts_res = x_opt[var_index["Ts"][0]:var_index["Ts"][1]+1]
    Ts_res = np.array(Ts_res)
    #print('Ts_res: ',Ts_res)
    x_res = x_opt[var_index["x"][0]:var_index["x"][1]+1]
    x_res = np.array(x_res)
    y_res = x_opt[var_index["y"][0]:var_index["y"][1]+1]
    y_res = np.array(y_res)
    z_res = x_opt[var_index["z"][0]:var_index["z"][1]+1]
    z_res = np.array(z_res)
    Lx_res = x_opt[var_index["Lx"][0]:var_index["Lx"][1]+1]
    Lx_res = np.array(Lx_res)
    Ly_res = x_opt[var_index["Ly"][0]:var_index["Ly"][1]+1]
    Ly_res = np.array(Ly_res)
    Lz_res = x_opt[var_index["Lz"][0]:var_index["Lz"][1]+1]
    Lz_res = np.array(Lz_res)

    Ldotx_res = x_opt[var_index["Ldotx"][0]:var_index["Ldotx"][1]+1]
    Ldotx_res = np.array(Ldotx_res)
    Ldoty_res = x_opt[var_index["Ldoty"][0]:var_index["Ldoty"][1]+1]
    Ldoty_res = np.array(Ldoty_res)
    Ldotz_res = x_opt[var_index["Ldotz"][0]:var_index["Ldotz"][1]+1]
    Ldotz_res = np.array(Ldotz_res)

    FL1x_res = x_opt[var_index["FL1x"][0]:var_index["FL1x"][1]+1]
    FL1x_res = np.array(FL1x_res)
    FL1y_res = x_opt[var_index["FL1y"][0]:var_index["FL1y"][1]+1]
    FL1y_res = np.array(FL1y_res)
    FL1z_res = x_opt[var_index["FL1z"][0]:var_index["FL1z"][1]+1]
    FL1z_res = np.array(FL1z_res)
    FL2x_res = x_opt[var_index["FL2x"][0]:var_index["FL2x"][1]+1]
    FL2x_res = np.array(FL2x_res)
    FL2y_res = x_opt[var_index["FL2y"][0]:var_index["FL2y"][1]+1]
    FL2y_res = np.array(FL2y_res)
    FL2z_res = x_opt[var_index["FL2z"][0]:var_index["FL2z"][1]+1]
    FL2z_res = np.array(FL2z_res)
    FL3x_res = x_opt[var_index["FL3x"][0]:var_index["FL3x"][1]+1]
    FL3x_res = np.array(FL3x_res)
    FL3y_res = x_opt[var_index["FL3y"][0]:var_index["FL3y"][1]+1]
    FL3y_res = np.array(FL3y_res)
    FL3z_res = x_opt[var_index["FL3z"][0]:var_index["FL3z"][1]+1]
    FL3z_res = np.array(FL3z_res)
    FL4x_res = x_opt[var_index["FL4x"][0]:var_index["FL4x"][1]+1]
    FL4x_res = np.array(FL4x_res)
    FL4y_res = x_opt[var_index["FL4y"][0]:var_index["FL4y"][1]+1]
    FL4y_res = np.array(FL4y_res)
    FL4z_res = x_opt[var_index["FL4z"][0]:var_index["FL4z"][1]+1]
    FL4z_res = np.array(FL4z_res)

    FR1x_res = x_opt[var_index["FR1x"][0]:var_index["FR1x"][1]+1]
    FR1x_res = np.array(FR1x_res)
    FR1y_res = x_opt[var_index["FR1y"][0]:var_index["FR1y"][1]+1]
    FR1y_res = np.array(FR1y_res)
    FR1z_res = x_opt[var_index["FR1z"][0]:var_index["FR1z"][1]+1]
    FR1z_res = np.array(FR1z_res)
    FR2x_res = x_opt[var_index["FR2x"][0]:var_index["FR2x"][1]+1]
    FR2x_res = np.array(FR2x_res)
    FR2y_res = x_opt[var_index["FR2y"][0]:var_index["FR2y"][1]+1]
    FR2y_res = np.array(FR2y_res)
    FR2z_res = x_opt[var_index["FR2z"][0]:var_index["FR2z"][1]+1]
    FR2z_res = np.array(FR2z_res)
    FR3x_res = x_opt[var_index["FR3x"][0]:var_index["FR3x"][1]+1]
    FR3x_res = np.array(FR3x_res)
    FR3y_res = x_opt[var_index["FR3y"][0]:var_index["FR3y"][1]+1]
    FR3y_res = np.array(FR3y_res)
    FR3z_res = x_opt[var_index["FR3z"][0]:var_index["FR3z"][1]+1]
    FR3z_res = np.array(FR3z_res)
    FR4x_res = x_opt[var_index["FR4x"][0]:var_index["FR4x"][1]+1]
    FR4x_res = np.array(FR4x_res)
    FR4y_res = x_opt[var_index["FR4y"][0]:var_index["FR4y"][1]+1]
    FR4y_res = np.array(FR4y_res)
    FR4z_res = x_opt[var_index["FR4z"][0]:var_index["FR4z"][1]+1]
    FR4z_res = np.array(FR4z_res)

    #Loop over all Phases (Knots)
    for Nph in range(NLPphase):
        
        #Decide Number of Knots
        if Nph == NLPphase-1:  #The last Knot belongs to the Last Phase
            Nk_ThisPhase = Nk_Local+1
        else:
            Nk_ThisPhase = Nk_Local

        #Decide Time Vector
        if Nph == 0: #first phase
            h = tauStepLength*NLPphase*(Ts_res[Nph]-0)
        else: #other phases
            h = tauStepLength*NLPphase*(Ts_res[Nph]-Ts_res[Nph-1])

        for Local_k_Count in range(Nk_ThisPhase):
            #Get knot number across the entire time line
            k = Nph*Nk_Local + Local_k_Count
            #print(k)

            #Add Cost Terms
            if k < N_K - 1:
                cost_val = cost_val + h*Lx_res[k]**2 + h*Ly_res[k]**2 + h*Lz_res[k]**2 + h*(FL1x_res[k]/m+FL2x_res[k]/m+FL3x_res[k]/m+FL4x_res[k]/m+FR1x_res[k]/m+FR2x_res[k]/m+FR3x_res[k]/m+FR4x_res[k]/m)**2 + h*(FL1y_res[k]/m+FL2y_res[k]/m+FL3y_res[k]/m+FL4y_res[k]/m+FR1y_res[k]/m+FR2y_res[k]/m+FR3y_res[k]/m+FR4y_res[k]/m)**2 + h*(FL1z_res[k]/m+FL2z_res[k]/m+FL3z_res[k]/m+FL4z_res[k]/m+FR1z_res[k]/m+FR2z_res[k]/m+FR3z_res[k]/m+FR4z_res[k]/m - G)**2
                cost_acc = cost_acc + h*(FL1x_res[k]/m+FL2x_res[k]/m+FL3x_res[k]/m+FL4x_res[k]/m+FR1x_res[k]/m+FR2x_res[k]/m+FR3x_res[k]/m+FR4x_res[k]/m)**2 + h*(FL1y_res[k]/m+FL2y_res[k]/m+FL3y_res[k]/m+FL4y_res[k]/m+FR1y_res[k]/m+FR2y_res[k]/m+FR3y_res[k]/m+FR4y_res[k]/m)**2 + h*(FL1z_res[k]/m+FL2z_res[k]/m+FL3z_res[k]/m+FL4z_res[k]/m+FR1z_res[k]/m+FR2z_res[k]/m+FR3z_res[k]/m+FR4z_res[k]/m - G)**2
                cost_momentum = cost_momentum + h*Lx_res[k]**2 + h*Ly_res[k]**2 + h*Lz_res[k]**2
                cost_momentum_rate = cost_momentum_rate + h*Ldotx_res[k]**2 + h*Ldoty_res[k]**2 + h*Ldotz_res[k]**2

    print("Full Cost Value: ",cost_val)
    print("Acceleration: ",cost_acc)
    print("Momentum: ",cost_momentum)
    print("Momentum Rate: ",cost_momentum_rate)

    return cost_val,cost_acc,cost_momentum

def getTerrainTagents_and_Norm(Patch):
    #Input Format
    #p2---------------------p1
    # |                      |
    # |                      |
    # |                      |
    #p3---------------------p4

    p1 = Patch[0]
    p2 = Patch[1]
    p3 = Patch[2]
    p4 = Patch[3]

    #Unrotated Terrain Norm and Tangents
    TerrainTangentX = np.array([1,0,0])
    TerrainTangentY = np.array([0,1,0])
    TerrainNorm = np.array([0,0,1])

    #Case 1 all flat
    if p1[2] == p2[2] and p2[2] == p3[2] and p3[2] == p4[2] and p4[2] == p1[2]:
        print("Flat Terrain, use the default set up of terrain tangent and norm")
    #Case 2, tilt arond Y axis
    elif p1[2] == p4[2] and p2[2] == p3[2] and (not p1[2]-p2[2] == 0) and (not p4[2]-p3[2]==0):
        print("tilt arond Y axis")
        tiltAngle = np.arctan2(p2[2]-p1[2],p1[0]-p2[0])
        r = R.from_euler('y', tiltAngle, degrees=False) 
        TerrainTangentX = r.as_matrix()@TerrainTangentX
        TerrainNorm = r.as_matrix()@TerrainNorm
    #Case 3, tilt around X axis    
    elif p1[2] == p2[2] and p3[2] == p4[2] and (not p2[2]-p3[2] == 0) and (not p1[2]-p4[2]==0):
        tiltAngle = np.arctan2(p1[2]-p4[2],p1[1]-p4[1])
        r = R.from_euler('x', tiltAngle, degrees=False) 
        TerrainTangentY = r.as_matrix()@TerrainTangentY
        TerrainNorm = r.as_matrix()@TerrainNorm
        print("tilt arond X axis")
    else:
        raise Exception("Un-defined Terrain Type")

    return TerrainTangentX, TerrainTangentY, TerrainNorm

#Get Computation Time
def GetComputationTimeFromFile(filename = None):
    
    ProgramTime = []
    TotalTime = []

    with open(filename, 'r') as read_obj:
        for line in read_obj:
            if "Total program time (secs)" in line:
                ProgramTime.append(float(line[44:51]))
                #print(float(line[44:51]))


    with open(filename, 'r') as read_obj:
        for line in read_obj:
            if "Total Program Time: " in line:
                TotalTime.append(float(line[21:27]))
                #print(float(line[21:27]))
                    
    print("Program Time:")
    print(ProgramTime)

    print("Total Computation Time:")
    print(TotalTime)

    return ProgramTime, TotalTime

def GetStatsFromOutputStrings(output_log = None):
    
    ProgramTime = []
    TotalTime = []

    for line in output_log:
        if "Total program time (secs)" in line:
            ProgramTime.append(float(line[37:51]))
            #print(float(line[44:51]))

        if "Total Program Time: " in line:
            TotalTime.append(float(line[21:27]))
            #print(float(line[21:27]))

        if "Accumulated Full Cost is:" in line:
            FullCost = float(line[27:33])

        if "Accumulated Acc Cost is:" in line:
            AccCost = float(line[26:32])

        if "Accumulated Momentum Cost is:" in line:
            MomentCost = float(line[31:37])
    
        if "Total Cost is:" in line:
            TotalCost = float(line[16:])
        
        if "Terminal Cost is:" in line:
            TerminalCost = float(line[19:])

        if "Terminal X position is:" in line:
            Terminal_X_pos = float(line[25:])

        if "Terminal Y position is:" in line:
            Terminal_Y_pos = float(line[25:])

        if "Terminal Z position is:" in line:
            Terminal_Z_pos = float(line[25:])

    print("Program Time:")
    print(ProgramTime)

    print("Total Computation Time:")
    print(TotalTime)

    print("Full Cost: ",FullCost)

    print("Acc Cost: ", AccCost)

    print("Moment Cost: ", MomentCost)

    print("Total Cost: ", TotalCost)

    print("Terminal Cost: ",TerminalCost)

    print("Terminal X position: ", Terminal_X_pos)

    print("Terminal Y position: ", Terminal_Y_pos)

    print("Terminal Z position: ", Terminal_Z_pos)

    return ProgramTime, TotalTime, FullCost, AccCost, MomentCost, TotalCost, TerminalCost, Terminal_X_pos, Terminal_Y_pos, Terminal_Z_pos


def getQuaternion(Patch):
    #Input Format
    #p2---------------------p1
    # |                      |
    # |                      |
    # |                      |
    #p3---------------------p4

    p1 = Patch[0]
    p2 = Patch[1]
    p3 = Patch[2]
    p4 = Patch[3]

    #Unrotated Terrain Norm and Tangents
    #TerrainTangentX = np.array([1,0,0])
    #TerrainTangentY = np.array([0,1,0])
    #TerrainNorm = np.array([0,0,1])

    #Case 1 all flat
    if p1[2] == p2[2] and p2[2] == p3[2] and p3[2] == p4[2] and p4[2] == p1[2]:
        #print("Flat Terrain, use the default set up of terrain tangent and norm")
        r = R.from_euler('x', 0, degrees=False) 
        quat = r.as_quat()
    #Case 2, tilt arond Y axis
    elif p1[2] == p4[2] and p2[2] == p3[2] and (not p1[2]-p2[2] == 0) and (not p4[2]-p3[2]==0):
        #print("tilt arond Y axis")
        tiltAngle = np.arctan2(p2[2]-p1[2],p1[0]-p2[0])
        r = R.from_euler('y', tiltAngle, degrees=False) 
        quat = r.as_quat()
        #TerrainTangentX = r.as_matrix()@TerrainTangentX
        #TerrainNorm = r.as_matrix()@TerrainNorm
    #Case 3, tilt around X axis    
    elif p1[2] == p2[2] and p3[2] == p4[2] and (not p2[2]-p3[2] == 0) and (not p1[2]-p4[2]==0):
        tiltAngle = np.arctan2(p1[2]-p4[2],p1[1]-p4[1])
        r = R.from_euler('x', tiltAngle, degrees=False) 
        quat = r.as_quat()
        #TerrainTangentY = r.as_matrix()@TerrainTangentY
        #TerrainNorm = r.as_matrix()@TerrainNorm
        #print("tilt arond X axis")
    else:
        raise Exception("Un-defined Terrain Type")

    return quat
