%% Plot for antfarm time
antnlptime = [0.7407,	1.5428,	3.12,	3.6171,	5.7993,	6.8731,	8.3149,	11.0778,	7.8341];
antPontonFulltime = [0.6486,	0.9847,	1.2641,	1.8526,	1.8785,	2.6573,	2.8169,	3.1823,	3.4647];
antPontonSingletime = [0.5616,	0.574,	0.8284,	1.0173,	1.5086,	1.6112,	1.7212,	1.7685,	1.8821];
pridction_horizon = [1,2,3,4,5,6,7,8,9];

figure()
hold on
plot(pridction_horizon,antnlptime,':o','LineWidth',7,'MarkerSize',25)
plot(pridction_horizon,antPontonFulltime,':o','LineWidth',7,'MarkerSize',25)
plot(pridction_horizon,antPontonSingletime,':o','LineWidth',7,'MarkerSize',25)
ylim([0,14])
ylabel('Time (s)')
xlabel('Num. of Steps in Prediction Horizon')
legend('Baseline ','Candidate 2','Candidate 3','Location','northwest','box','off')
ax = gca;
ax.FontSize = 32;
%grid on
xlim([0.5,9.5])

%% Plot for antfarm Cost
antnlpcost = [0.771829965,0.369179979,0.35190796,0.304114112,0.317607467,0.30113035,0.306439314,0.30165954,0.302053345];
antPontonFullcost = [3.077418669,0.974532072,0.86810627,0.921213394,0.879860659,0.937981567,0.880382775,0.942932729,0.881515991];
antPontonSinglecost = [2.617572813,0.746447191,0.695723339,0.706360031,0.69310504,0.72402795,0.694602988,0.723161826,0.705625616];
pridction_horizon = [1,2,3,4,5,6,7,8,9];

figure()
hold on
plot(pridction_horizon,antnlpcost,':o','LineWidth',7,'MarkerSize',25)
plot(pridction_horizon,antPontonFullcost,':o','LineWidth',7,'MarkerSize',25)
plot(pridction_horizon,antPontonSinglecost,':o','LineWidth',7,'MarkerSize',25)
ylim([0,3.5])
ylabel('Accumulated Cost')
xlabel('Num. of Steps in Prediction Horizon')
%legend('Baseline ','Candidate 2','Candidate 3','Location','northeast','box','off')
ax = gca;
ax.FontSize = 32;
%grid on
xlim([0.5,9.5])

%% Plot for up and down Computation Time
updownnlptime = [1.63,2.93,4.4759,5.6642,6.587,8.3021,8.3982,9.6647,8.9827];
updownPontonFulltime = [0.9176,1.3276,1.2802,1.7153,1.9707,2.3981,2.6293,3.0244,3.2528];
updownPontonSingletime = [0.7566,0.5604,0.7389,1.0375,1.3096,1.2245,1.5653,1.6831,2.0879];
pridction_horizon = [1,2,3,4,5,6,7,8,9];

figure()
hold on
plot(pridction_horizon,updownnlptime,':o','LineWidth',7,'MarkerSize',25)
plot(pridction_horizon,updownPontonFulltime,':o','LineWidth',7,'MarkerSize',25)
plot(pridction_horizon,updownPontonSingletime,':o','LineWidth',7,'MarkerSize',25)
ylim([0,14])
ylabel('Time (s)')
xlabel('Num. of Steps in Prediction Horizon')
%legend('Baseline ','Candidate 2','Candidate 3','Location','northwest','box','off')
ax = gca;
ax.FontSize = 32;
%grid on
xlim([0.5,9.5])

%% Plot for up and down Cost
updownnlpcost = [0.622567928,0.444534344,0.313854235,0.321154294,0.296481242,0.314621493,0.295759448,0.300224301,0.296971236];
updownPontonFullcost = [2.997788222,0.624574455,0.485966016,0.639857353,0.499265786,0.630816027,0.498573526,0.628983197,0.497031614];
updownPontonSinglecost = [2.614036577,0.572713328,0.490612741,0.544352386,0.48733062,0.532323698,0.493423951,0.510225914,0.487403772];
pridction_horizon = [1,2,3,4,5,6,7,8,9];

figure()
hold on
plot(pridction_horizon,updownnlpcost,':o','LineWidth',7,'MarkerSize',25)
plot(pridction_horizon,updownPontonFullcost,':o','LineWidth',7,'MarkerSize',25)
plot(pridction_horizon,updownPontonSinglecost,':o','LineWidth',7,'MarkerSize',25)
ylim([0,3.5])
ylabel('Accumulated Cost')
xlabel('Num. of Steps in Prediction Horizon')
%legend('Baseline ','Candidate 2','Candidate 3','Location','northeast','box','off')
ax = gca;
ax.FontSize = 32;
%grid on
xlim([0.5,9.5])

%% Plot for darpa Computation Time
darpanlptime = [1.84,3.5269,4.796,6.4931,9.4068,13.1713,10.3845,10.0237,11.4756];
darpaPontonFulltime = [0.5693,1.096,1.3442,1.8489,2.6998,3.1719,2.6242,2.8404,3.4463];
darpaPontonSingletime = [0.4573,0.6556,0.7602,1.0106,1.2336,1.2075,1.5881,1.7409,1.8755];
pridction_horizon = [1,2,3,4,5,6,7,8,9];

figure()
hold on
plot(pridction_horizon,darpanlptime,':o','LineWidth',7,'MarkerSize',25)
plot(pridction_horizon,darpaPontonFulltime,':o','LineWidth',7,'MarkerSize',25)
plot(pridction_horizon,darpaPontonSingletime,':o','LineWidth',7,'MarkerSize',25)
ylim([0,14])
ylabel('Time (s)')
xlabel('Num. of Steps in Prediction Horizon')
legend('Baseline ','Candidate 2','Candidate 3','Location','northwest','box','off')
ax = gca;
ax.FontSize = 32;
%grid on
xlim([0.5,9.5])

%% Plot for darpa cost
darpanlpcost = [0.809737908,0.474016988,0.360575192,0.336337099,0.31817316,0.31910797,0.313056725,0.309761673,0.302879631];
darpaPontonFullcost = [2.745943647,0.715072178,0.644810806,0.705407333,0.651732528,0.691746198,0.651963252,0.68180388,0.658276701];
darpaPontonSinglecost = [2.530982838,0.593986384,0.552614468,0.578649329,0.567030201,0.537793624,0.53636745,0.539965184,0.537416107];
pridction_horizon = [1,2,3,4,5,6,7,8,9];

figure()
hold on
plot(pridction_horizon,darpanlpcost,':o','LineWidth',7,'MarkerSize',25)
plot(pridction_horizon,darpaPontonFullcost,':o','LineWidth',7,'MarkerSize',25)
plot(pridction_horizon,darpaPontonSinglecost,':o','LineWidth',7,'MarkerSize',25)
ylim([0,3.5])
ylabel('Accumulated Cost')
xlabel('Num. of Steps in Prediction Horizon')
%legend('Baseline ','Candidate 2','Candidate 3','Location','northeast','box','off')
ax = gca;
ax.FontSize = 32;
%grid on
xlim([0.5,9.5])