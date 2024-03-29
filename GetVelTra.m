%From good result 1000 y regu, 7 lookahead

%Round 8 Left
Ts = [0.3,0.8,0.9];
r8_Phase1TimeSeries = linspace(0,Ts(1),8);
r8_Phase2TimeSeries = linspace(0,Ts(2)-Ts(1),8);
r8_Phase3TimeSeries = linspace(0,Ts(3)-Ts(2),8);

r8_Phase1y = [0.0339,0.0287,0.0234,0.0182,0.0131,0.0081,0.0033,-0.0012];

r8_Phase2y = [-0.0012,-0.0083,-0.0122,-0.0137,-0.0137,-0.0123,-0.0094,-0.0043];

r8_Phase3y = [-0.0043,-0.0027,-0.0012,0.0004,0.0021,0.0037,0.0053,0.007];

r8_Phase1z = [0.7997,0.7996,0.7995,0.7995,0.7995,0.7996,0.7997,0.7998];

r8_Phase2z = [0.7998,0.8,0.8,0.7999,0.7999,0.8,0.8,0.8];

r8_Phase3z = [0.8,0.7999,0.7999,0.7998,0.7998,0.7998,0.7997,0.7997];


r8_Phase1xdot = [0.3706,0.3706,0.3704,0.3699,0.3691,0.3681,0.3668,0.3653];

r8_Phase2xdot = [0.3653,0.3463,0.3353,0.331,0.332,0.338,0.35,0.3681];

r8_Phase3xdot = [0.3681,0.3684,0.3687,0.3689,0.3691,0.3693,0.3694,0.3695];

r8_Phase1ydot = [-1.2165e-01,-1.2231e-01,-1.2166e-01,-1.1969e-01,-1.1641e-01,-1.1182e-01,-1.0591e-01,-9.8695e-02];

r8_Phase2ydot = [-9.8695e-02,-5.4221e-02,-2.2043e-02,4.8076e-05,2.0051e-02,4.1089e-02,7.0753e-02,1.0935e-01];  

r8_Phase3ydot = [1.0935e-01,1.1086e-01,1.1222e-01,1.1343e-01,1.1450e-01,1.1542e-01,1.1619e-01,1.1682e-01];

r8_Phase1zdot = [-1.7919e-03,-1.0923e-03,-3.8834e-04,3.2008e-04,1.0329e-03,1.7502e-03,2.4719e-03,3.1980e-03];  

r8_Phase2zdot = [3.1980e-03,2.0148e-08,-7.0958e-04,-1.2479e-04,2.1462e-04,6.1969e-04,-3.6658e-04,-3.4265e-03];

r8_Phase3zdot = [-3.4265e-03,-3.2025e-03,-2.9777e-03,-2.7522e-03,-2.5261e-03,-2.2993e-03,-2.0717e-03,-1.8435e-03];

%length(Phase1TimeSeries)
%length(Phase2TimeSeries)
%length(Phase3TimeSeries)
%length(Phase1xdot)
%length(Phase2xdot)
%length(Phase3xdot)
%length(Phase1ydot)
%length(Phase2ydot)
%length(Phase3ydot)
%length(Phase1zdot)
%length(Phase2zdot)
%length(Phase3zdot)


%Round 9 Right
Ts = [0.3,0.8,0.9];
r9_Phase1TimeSeries = linspace(0,Ts(1),8);
r9_Phase2TimeSeries = linspace(0,Ts(2)-Ts(1),8);
r9_Phase3TimeSeries = linspace(0,Ts(3)-Ts(2),8);


r9_Phase1y = [0.007,0.012,0.017,0.022,0.0269,0.0316,0.0362,0.0404];

r9_Phase2y = [0.0404,0.047,0.0504,0.0514,0.051,0.049,0.0454,0.0397];

r9_Phase3y = [0.0397,0.038,0.0364,0.0347,0.033,0.0313,0.0295,0.0278];

r9_Phase1z = [0.7997,0.7996,0.7995,0.7995,0.7995,0.7996,0.7997,0.7998];

r9_Phase2z = [0.7998,0.8,0.8,0.7999,0.7999,0.8,0.8,0.8];

r9_Phase3z = [0.8,0.7999,0.7999,0.7998,0.7998,0.7998,0.7997,0.7997];


r9_Phase1xdot = [0.3695,0.3699,0.3699,0.3697,0.3692,0.3684,0.3673,0.366];

r9_Phase2xdot = [0.366,0.3472,0.3366,0.3328,0.334,0.3408,0.3535,0.3693];

r9_Phase3xdot = [0.3693,0.3696,0.3698,0.37,0.3702,0.3704,0.3705,0.3706];

r9_Phase1ydot = [0.1168,0.1172,0.1162,0.114,0.1105,0.1056,0.0995,0.0921];

r9_Phase2ydot = [0.0921,0.0472,0.015,-0.0068,-0.0274,-0.0499,-0.0808,-0.1151];

r9_Phase3ydot = [-0.1151,-0.1165,-0.1178,-0.119,-0.12,-0.1209,-0.1216,-0.1222];

r9_Phase1zdot = [-1.8435e-03,-1.1537e-03,-4.5711e-04,2.4621e-04,9.5628e-04,1.6731e-03,2.3967e-03,3.1270e-03];

r9_Phase2zdot = [3.1270e-03,5.0790e-09,-6.5774e-04,-2.3350e-05,2.2353e-04,4.5751e-04,-7.6749e-04,-3.3921e-03];

r9_Phase3zdot = [-3.3921e-03,-3.1649e-03,-2.9372e-03,-2.7089e-03,-2.4800e-03,-2.2506e-03,-2.0206e-03,-1.7901e-03];

%Round 10 Right
Ts = [0.3,0.8,0.9];
r10_Phase1TimeSeries = linspace(0,Ts(1),8);
r10_Phase2TimeSeries = linspace(0,Ts(2)-Ts(1),8);
r10_Phase3TimeSeries = linspace(0,Ts(3)-Ts(2),8);

r10_Phase1y = [0.0278,0.0226,0.0173,0.0121,0.0069,0.0019,-0.0029,-0.0074];

r10_Phase2y = [-0.0074,-0.0145,-0.0183,-0.0199,-0.0199,-0.0184,-0.0155,-0.0104];

r10_Phase3y = [-0.0104,-0.0089,-0.0073,-0.0057,-0.0041,-0.0024,-0.0008,0.0009];

r10_Phase1z = [0.7997,0.7996,0.7996,0.7995,0.7995,0.7996,0.7997,0.7998];

r10_Phase2z = [0.8,0.8,0.7999,0.7999,0.8,0.8,0.8,0.7999];

r10_Phase3z = [0.7999,0.7999,0.7999,0.7998,0.7998,0.7998,0.7997,0.7997];

r10_Phase1xdot = [0.3706,0.3707,0.3705,0.37,0.3693,0.3683,0.3671,0.3656];

r10_Phase2xdot = [0.3656,0.3466,0.3356,0.3313,0.3324,0.3383,0.3502,0.3683];

r10_Phase3xdot = [0.3683,0.3686,0.3688,0.3691,0.3693,0.3694,0.3696,0.3697];

r10_Phase1ydot = [-0.1222,-0.1228,-0.122,-0.12,-0.1166,-0.112,-0.106,-0.0987];

r10_Phase2ydot = [-0.0987,-0.0542,-0.0219,0.0002,0.0202,0.0412,0.0708,0.1093];

r10_Phase3ydot = [0.1093,0.1108,0.1122,0.1134,0.1145,0.1154,0.1162,0.1168];

r10_Phase1zdot = [-1.7901e-03,-1.0947e-03,-3.9368e-04,3.1298e-04,1.0253e-03,1.7432e-03,2.4668e-03,3.1960e-03];

r10_Phase2zdot = [3.1960e-03,4.6914e-07,-7.1112e-04,-1.2966e-04,2.1564e-04,6.2506e-04,-3.5972e-04,-3.4162e-03];

r10_Phase3zdot = [-3.4162e-03,-3.1936e-03,-2.9701e-03,-2.7459e-03,-2.5210e-03,-2.2953e-03,-2.0688e-03,-1.8416e-03];

%figure()
%hold on
%plot([r8_Phase1TimeSeries,r8_Phase2TimeSeries(2:end),r8_Phase3TimeSeries(2:end)], [r8_Phase1ydot,r8_Phase2ydot(2:end),r8_Phase3ydot(2:end)])
%plot([r10_Phase1TimeSeries,r10_Phase2TimeSeries(2:end),r10_Phase3TimeSeries(2:end)], [r10_Phase1ydot,r10_Phase2ydot(2:end),r10_Phase3ydot(2:end)])


%resample trajectory
resample_time = linspace(0,0.4,8);

%r8_Phase3TimeSeries
%r9_Phase1TimeSeries(2:end) + r8_Phase3TimeSeries(end)

Left_resampled_DS_y = interpn([r8_Phase3TimeSeries, r8_Phase3TimeSeries(end) + r9_Phase1TimeSeries(2:end)],[r8_Phase3y,r9_Phase1y(2:end)],resample_time,'spline');
Left_resampled_DS_z = interpn([r8_Phase3TimeSeries, r8_Phase3TimeSeries(end) + r9_Phase1TimeSeries(2:end)],[r8_Phase3z,r9_Phase1z(2:end)],resample_time,'spline');
Left_resampled_DS_xdot = interpn([r8_Phase3TimeSeries, r8_Phase3TimeSeries(end) + r9_Phase1TimeSeries(2:end)],[r8_Phase3xdot,r9_Phase1xdot(2:end)],resample_time,'spline');
Left_resampled_DS_ydot = interpn([r8_Phase3TimeSeries, r8_Phase3TimeSeries(end) + r9_Phase1TimeSeries(2:end)],[r8_Phase3ydot,r9_Phase1ydot(2:end)],resample_time,'spline');
Left_resampled_DS_zdot = interpn([r8_Phase3TimeSeries, r8_Phase3TimeSeries(end) + r9_Phase1TimeSeries(2:end)],[r8_Phase3zdot,r9_Phase1zdot(2:end)],resample_time,'spline');

figure()
plot([r8_Phase3TimeSeries,r9_Phase1TimeSeries(2:end) + r8_Phase3TimeSeries(end)],[r8_Phase3xdot,r9_Phase1xdot(2:end)])
hold on
plot(resample_time,Left_resampled_DS_xdot)
title("Left xdot")
hold off

figure()
plot([r8_Phase3TimeSeries,r9_Phase1TimeSeries(2:end) + r8_Phase3TimeSeries(end)],[r8_Phase3ydot,r9_Phase1ydot(2:end)])
hold on
plot(resample_time,Left_resampled_DS_ydot)
title("Left ydot")
hold off

figure()
plot([r8_Phase3TimeSeries,r9_Phase1TimeSeries(2:end) + r8_Phase3TimeSeries(end)],[r8_Phase3zdot,r9_Phase1zdot(2:end)])
hold on
plot(resample_time,Left_resampled_DS_zdot)
title("Left zdot")
hold off

%Right Start
%resample trajectory
resample_time = linspace(0,0.4,8);

%r8_Phase3TimeSeries
%r9_Phase1TimeSeries(2:end) + r8_Phase3TimeSeries(end)

Right_resampled_DS_y = interpn([r9_Phase3TimeSeries, r9_Phase3TimeSeries(end) + r10_Phase1TimeSeries(2:end)],[r9_Phase3y,r10_Phase1y(2:end)],resample_time,'spline');
Right_resampled_DS_z = interpn([r9_Phase3TimeSeries, r9_Phase3TimeSeries(end) + r10_Phase1TimeSeries(2:end)],[r9_Phase3z,r10_Phase1z(2:end)],resample_time,'spline');
Right_resampled_DS_xdot = interpn([r9_Phase3TimeSeries, r9_Phase3TimeSeries(end) + r10_Phase1TimeSeries(2:end)],[r9_Phase3xdot,r10_Phase1xdot(2:end)],resample_time,'spline');
Right_resampled_DS_ydot = interpn([r9_Phase3TimeSeries, r9_Phase3TimeSeries(end) + r10_Phase1TimeSeries(2:end)],[r9_Phase3ydot,r10_Phase1ydot(2:end)],resample_time,'spline');
Right_resampled_DS_zdot = interpn([r9_Phase3TimeSeries, r9_Phase3TimeSeries(end) + r10_Phase1TimeSeries(2:end)],[r9_Phase3zdot,r10_Phase1zdot(2:end)],resample_time,'spline');

figure()
plot([r9_Phase3TimeSeries,r9_Phase1TimeSeries(2:end) + r10_Phase3TimeSeries(end)],[r9_Phase3xdot,r10_Phase1xdot(2:end)])
hold on
plot(resample_time,Right_resampled_DS_xdot)
title("Right xdot")
hold off

figure()
plot([r9_Phase3TimeSeries,r9_Phase1TimeSeries(2:end) + r10_Phase3TimeSeries(end)],[r9_Phase3ydot,r10_Phase1ydot(2:end)])
hold on
plot(resample_time,Right_resampled_DS_ydot)
title("Right ydot")
hold off

figure()
plot([r9_Phase3TimeSeries,r9_Phase1TimeSeries(2:end) + r10_Phase3TimeSeries(end)],[r9_Phase3zdot,r10_Phase1zdot(2:end)])
hold on
plot(resample_time,Right_resampled_DS_zdot)
title("Right zdot")
hold off