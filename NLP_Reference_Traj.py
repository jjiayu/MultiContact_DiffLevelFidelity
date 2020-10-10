import numpy as np


import matplotlib.pyplot as plt #Matplotlib
from mpl_toolkits.mplot3d import Axes3D

def NLP_ref_traj_constructor():
    #all pieces are with 8 knots and we need to remove them when connecting them
    
    LeftStep_InitDouble_y = np.array([0.0339,0.0287,0.0234,0.0182,0.0131,0.0081,0.0033,-0.0012])
    LeftStep_InitDouble_z = np.array([0.7997,0.7996,0.7995,0.7995,0.7995,0.7996,0.7997,0.7998])
    LeftStep_InitDouble_xdot = np.array([0.3706,0.3706,0.3704,0.3699,0.3691,0.3681,0.3668,0.3653])
    LeftStep_InitDouble_ydot = np.array([-1.2165e-01,-1.2231e-01,-1.2166e-01,-1.1969e-01,-1.1641e-01,-1.1182e-01,-1.0591e-01,-9.8695e-02])
    LeftStep_InitDouble_zdot = np.array([-1.7919e-03,-1.0923e-03,-3.8834e-04,3.2008e-04,1.0329e-03,1.7502e-03,2.4719e-03,3.1980e-03])

    LeftStep_Swing_y = np.array([-0.0012,-0.0083,-0.0122,-0.0137,-0.0137,-0.0123,-0.0094,-0.0043])
    LeftStep_Swing_z = np.array([0.7998,0.8,0.8,0.7999,0.7999,0.8,0.8,0.8])
    LeftStep_Swing_xdot = np.array([0.3653,0.3463,0.3353,0.331,0.332,0.338,0.35,0.3681])
    LeftStep_Swing_ydot = np.array([-9.8695e-02,-5.4221e-02,-2.2043e-02,4.8076e-05,2.0051e-02,4.1089e-02,7.0753e-02,1.0935e-01])
    LeftStep_Swing_zdot = np.array([3.1980e-03,2.0148e-08,-7.0958e-04,-1.2479e-04,2.1462e-04,6.1969e-04,-3.6658e-04,-3.4265e-03])

    LeftStep_DoubleSupportCombined_y = np.array([-0.0043,0.0021,0.0087,0.0153,0.0220,0.0285,0.0347,0.0404])
    LeftStep_DoubleSupportCombined_z = np.array([0.8000,0.7998,0.7997,0.7995,0.7995,0.7995,0.7997,0.7998])
    LeftStep_DoubleSupportCombined_xdot = np.array([0.3681,0.3691,0.3696,0.3699,0.3697,0.3690,0.3677,0.3660])
    LeftStep_DoubleSupportCombined_ydot = np.array([0.1094,0.1145,0.1172,0.1167,0.1140,0.1090,0.1017,0.0921])
    LeftStep_DoubleSupportCombined_zdot = np.array([-0.0034,-0.0025,-0.0016,-0.0007,0.0002,0.0012,0.0022,0.0031])


    RightStep_InitDouble_y = np.array([0.007,0.012,0.017,0.022,0.0269,0.0316,0.0362,0.0404])
    RightStep_InitDouble_z = np.array([0.7997,0.7996,0.7995,0.7995,0.7995,0.7996,0.7997,0.7998])
    RightStep_InitDouble_xdot = np.array([0.3695,0.3699,0.3699,0.3697,0.3692,0.3684,0.3673,0.366])
    RightStep_InitDouble_ydot = np.array([0.1168,0.1172,0.1162,0.114,0.1105,0.1056,0.0995,0.0921])
    RightStep_InitDouble_zdot = np.array([-1.8435e-03,-1.1537e-03,-4.5711e-04,2.4621e-04,9.5628e-04,1.6731e-03,2.3967e-03,3.1270e-03])

    RightStep_Swing_y = np.array([0.0404,0.047,0.0504,0.0514,0.051,0.049,0.0454,0.0397])
    RightStep_Swing_z = np.array([0.7998,0.8,0.8,0.7999,0.7999,0.8,0.8,0.8])
    RightStep_Swing_xdot = np.array([0.366,0.3472,0.3366,0.3328,0.334,0.3408,0.3535,0.3693])
    RightStep_Swing_ydot = np.array([0.0921,0.0472,0.015,-0.0068,-0.0274,-0.0499,-0.0808,-0.1151])
    RightStep_Swing_zdot = np.array([3.1270e-03,5.0790e-09,-6.5774e-04,-2.3350e-05,2.2353e-04,4.5751e-04,-7.6749e-04,-3.3921e-03])

    RightStep_DoubleSupportCombined_y = np.array([0.0397,0.0330,0.0261,0.0191,0.0121,0.0052,-0.0013,-0.0074])
    RightStep_DoubleSupportCombined_z = np.array([0.8000,0.7998,0.7997,0.7996,0.7995,0.7995,0.7997,0.7998])
    RightStep_DoubleSupportCombined_xdot = np.array([0.3693,0.3702,0.3707,0.3706,0.3700,0.3690,0.3675,0.3656])
    RightStep_DoubleSupportCombined_ydot = np.array([-0.1151,-0.1200,-0.1226,-0.1224,-0.1200,-0.1152,-0.1082,-0.0987])
    RightStep_DoubleSupportCombined_zdot = np.array([-0.0034,-0.0025,-0.0016,-0.0006,0.0003,0.0013,0.0022,0.0032])

    #def NLP_Ref_Traj_Constructor():
    Timeseries = [np.linspace(0,0.3,8),np.linspace(0.3,0.8,8)[1:],np.linspace(0.8,0.9+0.3,8)[1:],
                np.linspace(1.2,1.2+0.5,8)[1:],np.linspace(1.2+0.5,1.2+0.5+0.4,8)[1:],
                np.linspace(2.1,2.1+0.5,8)[1:],np.linspace(2.1+0.5,2.1+0.5+0.4,8)[1:],
                np.linspace(1.2+0.9*2,1.2+0.9*2+0.5,8)[1:],np.linspace(1.2+0.9*2+0.5,1.2+0.9*2+0.5+0.4,8)[1:],
                np.linspace(1.2+0.9*3,1.2+0.9*3+0.5,8)[1:],np.linspace(1.2+0.9*3+0.5,1.2+0.9*3+0.5+0.4,8)[1:],
                np.linspace(1.2+0.9*4,1.2+0.9*4+0.5,8)[1:],np.linspace(1.2+0.9*4+0.5,1.2+0.9*4+0.5+0.4,8)[1:],
                np.linspace(1.2+0.9*5,1.2+0.9*5+0.5,8)[1:],np.linspace(1.2+0.9*5+0.5,1.2+0.9*5+0.5+0.4,8)[1:],
                np.linspace(1.2+0.9*6,1.2+0.9*6+0.5,8)[1:],np.linspace(1.2+0.9*6+0.5,1.2+0.9*6+0.5+0.4,8)[1:],
                np.linspace(1.2+0.9*7,1.2+0.9*7+0.5,8)[1:],np.linspace(1.2+0.9*7+0.5,1.2+0.9*7+0.5+0.4,8)[1:],
                np.linspace(1.2+0.9*8,1.2+0.9*8+0.5,8)[1:],np.linspace(1.2+0.9*8+0.5,1.2+0.9*8+0.5+0.4,8)[1:],
                np.linspace(1.2+0.9*9,1.2+0.9*9+0.5,8)[1:],np.linspace(1.2+0.9*9+0.5,1.2+0.9*9+0.5+0.4,8)[1:],
                np.linspace(1.2+0.9*10,1.2+0.9*10+0.5,8)[1:],np.linspace(1.2+0.9*10+0.5,1.2+0.9*10+0.5+0.4,8)[1:],
                np.linspace(1.2+0.9*11,1.2+0.9*11+0.5,8)[1:],np.linspace(1.2+0.9*11+0.5,1.2+0.9*11+0.5+0.4,8)[1:],
                np.linspace(1.2+0.9*12,1.2+0.9*12+0.5,8)[1:],np.linspace(1.2+0.9*12+0.5,1.2+0.9*12+0.5+0.4,8)[1:],
    ]

    LeftStart_y = [LeftStep_InitDouble_y,LeftStep_Swing_y[1:],LeftStep_DoubleSupportCombined_y[1:],
                    RightStep_Swing_y[1:],RightStep_DoubleSupportCombined_y[1:],
                    LeftStep_Swing_y[1:],LeftStep_DoubleSupportCombined_y[1:],
                    RightStep_Swing_y[1:],RightStep_DoubleSupportCombined_y[1:],
                    LeftStep_Swing_y[1:],LeftStep_DoubleSupportCombined_y[1:],
                    RightStep_Swing_y[1:],RightStep_DoubleSupportCombined_y[1:],
                    LeftStep_Swing_y[1:],LeftStep_DoubleSupportCombined_y[1:],
                    RightStep_Swing_y[1:],RightStep_DoubleSupportCombined_y[1:],
                    LeftStep_Swing_y[1:],LeftStep_DoubleSupportCombined_y[1:],
                    RightStep_Swing_y[1:],RightStep_DoubleSupportCombined_y[1:],
                    LeftStep_Swing_y[1:],LeftStep_DoubleSupportCombined_y[1:],
                    RightStep_Swing_y[1:],RightStep_DoubleSupportCombined_y[1:],
                    LeftStep_Swing_y[1:],LeftStep_DoubleSupportCombined_y[1:],
                    RightStep_Swing_y[1:],RightStep_DoubleSupportCombined_y[1:]] 

    LeftStart_z = [LeftStep_InitDouble_z,LeftStep_Swing_z[1:],LeftStep_DoubleSupportCombined_z[1:],
                    RightStep_Swing_z[1:],RightStep_DoubleSupportCombined_z[1:],
                    LeftStep_Swing_z[1:],LeftStep_DoubleSupportCombined_z[1:],
                    RightStep_Swing_z[1:],RightStep_DoubleSupportCombined_z[1:],
                    LeftStep_Swing_z[1:],LeftStep_DoubleSupportCombined_z[1:],
                    RightStep_Swing_z[1:],RightStep_DoubleSupportCombined_z[1:],
                    LeftStep_Swing_z[1:],LeftStep_DoubleSupportCombined_z[1:],
                    RightStep_Swing_z[1:],RightStep_DoubleSupportCombined_z[1:],
                    LeftStep_Swing_z[1:],LeftStep_DoubleSupportCombined_z[1:],
                    RightStep_Swing_z[1:],RightStep_DoubleSupportCombined_z[1:],
                    LeftStep_Swing_z[1:],LeftStep_DoubleSupportCombined_z[1:],
                    RightStep_Swing_z[1:],RightStep_DoubleSupportCombined_z[1:],
                    LeftStep_Swing_z[1:],LeftStep_DoubleSupportCombined_z[1:],
                    RightStep_Swing_z[1:],RightStep_DoubleSupportCombined_z[1:]] 

    LeftStart_xdot = [LeftStep_InitDouble_xdot,LeftStep_Swing_xdot[1:],LeftStep_DoubleSupportCombined_xdot[1:],
                    RightStep_Swing_xdot[1:],RightStep_DoubleSupportCombined_xdot[1:],
                    LeftStep_Swing_xdot[1:],LeftStep_DoubleSupportCombined_xdot[1:],
                    RightStep_Swing_xdot[1:],RightStep_DoubleSupportCombined_xdot[1:],
                    LeftStep_Swing_xdot[1:],LeftStep_DoubleSupportCombined_xdot[1:],
                    RightStep_Swing_xdot[1:],RightStep_DoubleSupportCombined_xdot[1:],
                    LeftStep_Swing_xdot[1:],LeftStep_DoubleSupportCombined_xdot[1:],
                    RightStep_Swing_xdot[1:],RightStep_DoubleSupportCombined_xdot[1:],
                    LeftStep_Swing_xdot[1:],LeftStep_DoubleSupportCombined_xdot[1:],
                    RightStep_Swing_xdot[1:],RightStep_DoubleSupportCombined_xdot[1:],
                    LeftStep_Swing_xdot[1:],LeftStep_DoubleSupportCombined_xdot[1:],
                    RightStep_Swing_xdot[1:],RightStep_DoubleSupportCombined_xdot[1:],
                    LeftStep_Swing_xdot[1:],LeftStep_DoubleSupportCombined_xdot[1:],
                    RightStep_Swing_xdot[1:],RightStep_DoubleSupportCombined_xdot[1:]] 

    LeftStart_ydot = [LeftStep_InitDouble_ydot,LeftStep_Swing_ydot[1:],LeftStep_DoubleSupportCombined_ydot[1:],
                    RightStep_Swing_ydot[1:],RightStep_DoubleSupportCombined_ydot[1:],
                    LeftStep_Swing_ydot[1:],LeftStep_DoubleSupportCombined_ydot[1:],
                    RightStep_Swing_ydot[1:],RightStep_DoubleSupportCombined_ydot[1:],
                    LeftStep_Swing_ydot[1:],LeftStep_DoubleSupportCombined_ydot[1:],
                    RightStep_Swing_ydot[1:],RightStep_DoubleSupportCombined_ydot[1:],
                    LeftStep_Swing_ydot[1:],LeftStep_DoubleSupportCombined_ydot[1:],
                    RightStep_Swing_ydot[1:],RightStep_DoubleSupportCombined_ydot[1:],
                    LeftStep_Swing_ydot[1:],LeftStep_DoubleSupportCombined_ydot[1:],
                    RightStep_Swing_ydot[1:],RightStep_DoubleSupportCombined_ydot[1:],
                    LeftStep_Swing_ydot[1:],LeftStep_DoubleSupportCombined_ydot[1:],
                    RightStep_Swing_ydot[1:],RightStep_DoubleSupportCombined_ydot[1:],
                    LeftStep_Swing_ydot[1:],LeftStep_DoubleSupportCombined_ydot[1:],
                    RightStep_Swing_ydot[1:],RightStep_DoubleSupportCombined_ydot[1:]] 

    LeftStart_zdot = [LeftStep_InitDouble_zdot,LeftStep_Swing_zdot[1:],LeftStep_DoubleSupportCombined_zdot[1:],
                    RightStep_Swing_zdot[1:],RightStep_DoubleSupportCombined_zdot[1:],
                    LeftStep_Swing_zdot[1:],LeftStep_DoubleSupportCombined_zdot[1:],
                    RightStep_Swing_zdot[1:],RightStep_DoubleSupportCombined_zdot[1:],
                    LeftStep_Swing_zdot[1:],LeftStep_DoubleSupportCombined_zdot[1:],
                    RightStep_Swing_zdot[1:],RightStep_DoubleSupportCombined_zdot[1:],
                    LeftStep_Swing_zdot[1:],LeftStep_DoubleSupportCombined_zdot[1:],
                    RightStep_Swing_zdot[1:],RightStep_DoubleSupportCombined_zdot[1:],
                    LeftStep_Swing_zdot[1:],LeftStep_DoubleSupportCombined_zdot[1:],
                    RightStep_Swing_zdot[1:],RightStep_DoubleSupportCombined_zdot[1:],
                    LeftStep_Swing_zdot[1:],LeftStep_DoubleSupportCombined_zdot[1:],
                    RightStep_Swing_zdot[1:],RightStep_DoubleSupportCombined_zdot[1:],
                    LeftStep_Swing_zdot[1:],LeftStep_DoubleSupportCombined_zdot[1:],
                    RightStep_Swing_zdot[1:],RightStep_DoubleSupportCombined_zdot[1:]] 

    #Right Start

    RightStart_y = [RightStep_InitDouble_y,RightStep_Swing_y[1:],RightStep_DoubleSupportCombined_y[1:],
                    LeftStep_Swing_y[1:],LeftStep_DoubleSupportCombined_y[1:],
                    RightStep_Swing_y[1:],RightStep_DoubleSupportCombined_y[1:],
                    LeftStep_Swing_y[1:],LeftStep_DoubleSupportCombined_y[1:],
                    RightStep_Swing_y[1:],RightStep_DoubleSupportCombined_y[1:],
                    LeftStep_Swing_y[1:],LeftStep_DoubleSupportCombined_y[1:],
                    RightStep_Swing_y[1:],RightStep_DoubleSupportCombined_y[1:],
                    LeftStep_Swing_y[1:],LeftStep_DoubleSupportCombined_y[1:],
                    RightStep_Swing_y[1:],RightStep_DoubleSupportCombined_y[1:],
                    LeftStep_Swing_y[1:],LeftStep_DoubleSupportCombined_y[1:],
                    RightStep_Swing_y[1:],RightStep_DoubleSupportCombined_y[1:],
                    LeftStep_Swing_y[1:],LeftStep_DoubleSupportCombined_y[1:],
                    RightStep_Swing_y[1:],RightStep_DoubleSupportCombined_y[1:],
                    LeftStep_Swing_y[1:],LeftStep_DoubleSupportCombined_y[1:]] 

    RightStart_z = [RightStep_InitDouble_z,RightStep_Swing_z[1:],RightStep_DoubleSupportCombined_z[1:],
                    LeftStep_Swing_z[1:],LeftStep_DoubleSupportCombined_z[1:],
                    RightStep_Swing_z[1:],RightStep_DoubleSupportCombined_z[1:],
                    LeftStep_Swing_z[1:],LeftStep_DoubleSupportCombined_z[1:],
                    RightStep_Swing_z[1:],RightStep_DoubleSupportCombined_z[1:],
                    LeftStep_Swing_z[1:],LeftStep_DoubleSupportCombined_z[1:],
                    RightStep_Swing_z[1:],RightStep_DoubleSupportCombined_z[1:],
                    LeftStep_Swing_z[1:],LeftStep_DoubleSupportCombined_z[1:],
                    RightStep_Swing_z[1:],RightStep_DoubleSupportCombined_z[1:],
                    LeftStep_Swing_z[1:],LeftStep_DoubleSupportCombined_z[1:],
                    RightStep_Swing_z[1:],RightStep_DoubleSupportCombined_z[1:],
                    LeftStep_Swing_z[1:],LeftStep_DoubleSupportCombined_z[1:],
                    RightStep_Swing_z[1:],RightStep_DoubleSupportCombined_z[1:],
                    LeftStep_Swing_z[1:],LeftStep_DoubleSupportCombined_z[1:]] 

    RightStart_xdot = [RightStep_InitDouble_xdot,RightStep_Swing_xdot[1:],RightStep_DoubleSupportCombined_xdot[1:],
                    LeftStep_Swing_xdot[1:],LeftStep_DoubleSupportCombined_xdot[1:],
                    RightStep_Swing_xdot[1:],RightStep_DoubleSupportCombined_xdot[1:],
                    LeftStep_Swing_xdot[1:],LeftStep_DoubleSupportCombined_xdot[1:],
                    RightStep_Swing_xdot[1:],RightStep_DoubleSupportCombined_xdot[1:],
                    LeftStep_Swing_xdot[1:],LeftStep_DoubleSupportCombined_xdot[1:],
                    RightStep_Swing_xdot[1:],RightStep_DoubleSupportCombined_xdot[1:],
                    LeftStep_Swing_xdot[1:],LeftStep_DoubleSupportCombined_xdot[1:],
                    RightStep_Swing_xdot[1:],RightStep_DoubleSupportCombined_xdot[1:],
                    LeftStep_Swing_xdot[1:],LeftStep_DoubleSupportCombined_xdot[1:],
                    RightStep_Swing_xdot[1:],RightStep_DoubleSupportCombined_xdot[1:],
                    LeftStep_Swing_xdot[1:],LeftStep_DoubleSupportCombined_xdot[1:],
                    RightStep_Swing_xdot[1:],RightStep_DoubleSupportCombined_xdot[1:],
                    LeftStep_Swing_xdot[1:],LeftStep_DoubleSupportCombined_xdot[1:]] 

    RightStart_ydot = [RightStep_InitDouble_ydot,RightStep_Swing_ydot[1:],RightStep_DoubleSupportCombined_ydot[1:],
                    LeftStep_Swing_ydot[1:],LeftStep_DoubleSupportCombined_ydot[1:],
                    RightStep_Swing_ydot[1:],RightStep_DoubleSupportCombined_ydot[1:],
                    LeftStep_Swing_ydot[1:],LeftStep_DoubleSupportCombined_ydot[1:],
                    RightStep_Swing_ydot[1:],RightStep_DoubleSupportCombined_ydot[1:],
                    LeftStep_Swing_ydot[1:],LeftStep_DoubleSupportCombined_ydot[1:],
                    RightStep_Swing_ydot[1:],RightStep_DoubleSupportCombined_ydot[1:],
                    LeftStep_Swing_ydot[1:],LeftStep_DoubleSupportCombined_ydot[1:],
                    RightStep_Swing_ydot[1:],RightStep_DoubleSupportCombined_ydot[1:],
                    LeftStep_Swing_ydot[1:],LeftStep_DoubleSupportCombined_ydot[1:],
                    RightStep_Swing_ydot[1:],RightStep_DoubleSupportCombined_ydot[1:],
                    LeftStep_Swing_ydot[1:],LeftStep_DoubleSupportCombined_ydot[1:],
                    RightStep_Swing_ydot[1:],RightStep_DoubleSupportCombined_ydot[1:],
                    LeftStep_Swing_ydot[1:],LeftStep_DoubleSupportCombined_ydot[1:]]  

    RightStart_zdot = [RightStep_InitDouble_zdot,RightStep_Swing_zdot[1:],RightStep_DoubleSupportCombined_zdot[1:],
                    LeftStep_Swing_zdot[1:],LeftStep_DoubleSupportCombined_zdot[1:],
                    RightStep_Swing_zdot[1:],RightStep_DoubleSupportCombined_zdot[1:],
                    LeftStep_Swing_zdot[1:],LeftStep_DoubleSupportCombined_zdot[1:],
                    RightStep_Swing_zdot[1:],RightStep_DoubleSupportCombined_zdot[1:],
                    LeftStep_Swing_zdot[1:],LeftStep_DoubleSupportCombined_zdot[1:],
                    RightStep_Swing_zdot[1:],RightStep_DoubleSupportCombined_zdot[1:],
                    LeftStep_Swing_zdot[1:],LeftStep_DoubleSupportCombined_zdot[1:],
                    RightStep_Swing_zdot[1:],RightStep_DoubleSupportCombined_zdot[1:],
                    LeftStep_Swing_zdot[1:],LeftStep_DoubleSupportCombined_zdot[1:],
                    RightStep_Swing_zdot[1:],RightStep_DoubleSupportCombined_zdot[1:],
                    LeftStep_Swing_zdot[1:],LeftStep_DoubleSupportCombined_zdot[1:],
                    RightStep_Swing_zdot[1:],RightStep_DoubleSupportCombined_zdot[1:],
                    LeftStep_Swing_zdot[1:],LeftStep_DoubleSupportCombined_zdot[1:]]  

    Timeseries = np.concatenate(Timeseries)  

    LeftStart_y = np.concatenate(LeftStart_y) 
    LeftStart_z = np.concatenate(LeftStart_z) 
    LeftStart_xdot = np.concatenate(LeftStart_xdot)   
    LeftStart_ydot = np.concatenate(LeftStart_ydot) 
    LeftStart_zdot = np.concatenate(LeftStart_zdot) 

    RightStart_y = np.concatenate(RightStart_y) 
    RightStart_z = np.concatenate(RightStart_z) 
    RightStart_xdot = np.concatenate(RightStart_xdot)   
    RightStart_ydot = np.concatenate(RightStart_ydot) 
    RightStart_zdot = np.concatenate(RightStart_zdot) 

    LeftStartMotion = {"Timeseries":Timeseries,
                       "LeftStart_y":LeftStart_y,
                       "LeftStart_z":LeftStart_z,
                       "LeftStart_xdot":LeftStart_xdot,
                       "LeftStart_ydot":LeftStart_ydot,
                       "LeftStart_zdot":LeftStart_zdot,
                        }

    RightStartMotion = {"Timeseries":Timeseries,
                        "RightStart_y":RightStart_y,
                        "RightStart_z":RightStart_z,
                        "RightStart_xdot":RightStart_xdot,
                        "RightStart_ydot":RightStart_ydot, 
                        "RightStart_zdot":RightStart_zdot,
                        }

    return LeftStartMotion, RightStartMotion

#leftmotion, rightmotion = NLP_ref_traj_constructor()

#plt.plot(leftmotion["Timeseries"],leftmotion["LeftStart_ydot"])
#plt.xlim(0,12.8)
#plt.show()

