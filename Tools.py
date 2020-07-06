import numpy as np

def FirstLevelCost(Nk_Local=5,x_opt=None,var_index=None,G = 9.80665,m=95):
    
    cost_val=0

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

    print("Cost Value: ",cost_val)

    return cost_val