%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Clear WorkSpace and Command Window
clear;
close;
clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Model Inputs and Outputs
%time = [1990 1991 1992 1993 1994];

time = [1 2 3 4 5 6];
score = [84.9 84.6 84.4 84.2 84.1 83.9];
 
Y_score = score';
X_time = [ones(6,1) time'];
logfit = fittype('a+b*log(x)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Using Fit Functions to get the Model
[Fit_Obj_Linear, GOF_Linear] = fit(X_time(:,2), Y_score, 'poly1')
[Fit_Obj_Quadratic, GOF_Quadratic] = fit(X_time(:,2), Y_score, 'poly2')
[Fit_Obj_log, GOF_logarithmic] =fit(X_time(:,2), Y_score,logfit);
[Fit_Obj_Power1, GOF_Power1] = fit(X_time(:,2), Y_score, 'power1')
[Fit_Obj_Power2, GOF_Power2] = fit(X_time(:,2), Y_score, 'power2')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot Functions
figure(1);
scatter(time, score, 'b', 'fill');
title('Original Data', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'r');
xlabel('time in months', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'b'); 
ylabel('score in %', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'b');
figure(2);
plot(Fit_Obj_Linear, 'r:');
hold on;
plot(Fit_Obj_Quadratic, 'b-.');
hold on;
plot(Fit_Obj_log, 'k');
hold on;
plot(Fit_Obj_Power1, 'g');
hold on;
plot(Fit_Obj_Power2, 'm');
hold on;
scatter(time, score, 'r', 'fill');
title('All Functions with original data', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'r');
xlabel('time in months', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'b'); 
ylabel('score in %', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'b');
legend('Linear', 'Qadratic', 'log', 'Power1', 'Power2', 'Original', 'Location', 'NorthWest');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Best Function with original data
figure(3)
plot(Fit_Obj_log, 'k');
hold on;
scatter(time, score, 'r', 'fill');
title('Best Function (Log) with original data', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'r');
xlabel('time in months', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'b'); 
ylabel('score in %', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'b');
legend('log', 'Original', 'Location', 'NorthWest');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Predict From 1995 to 1999
% Linear
Y_score_1995_Linear = Fit_Obj_Linear(1995);
Y_score_1996_Linear = Fit_Obj_Linear(1996);
Y_score_1997_Linear = Fit_Obj_Linear(1997);
Y_score_1998_Linear = Fit_Obj_Linear(1998);
Y_score_1999_Linear = Fit_Obj_Linear(1999);
% Quadratic
Y_score_1995_Quadratic = Fit_Obj_Quadratic(1995);
Y_score_1996_Quadratic = Fit_Obj_Quadratic(1996);
Y_score_1997_Quadratic = Fit_Obj_Quadratic(1997);
Y_score_1998_Quadratic = Fit_Obj_Quadratic(1998);
Y_score_1999_Quadratic = Fit_Obj_Quadratic(1999);
%Cubic
Y_score_1995_log = Fit_Obj_log(1995);
Y_score_1996_Cubic = Fit_Obj_log(1996);
Y_score_1997_Cubic = Fit_Obj_log(1997);
Y_score_1998_Cubic = Fit_Obj_log(1998);
Y_score_1999_Cubic = Fit_Obj_log(1999);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Prediction
Y_score_2000_Linear = Fit_Obj_Linear(11)
Y_score_2005_Linear = Fit_Obj_Linear(16)
Y_score_2000_Quadratic = Fit_Obj_Quadratic(11)
Y_score_2005_Quadratic = Fit_Obj_Quadratic(16)
Y_score_2000_Cubic = Fit_Obj_log(11)
Y_score_2005_Cubic = Fit_Obj_log(16)
Y_score_2000_Power1 = Fit_Obj_Power1(11)
Y_score_2005_Power1 = Fit_Obj_Power1(16)
Y_score_2000_Power2 = Fit_Obj_Power2(11)
Y_score_2005_Power2 = Fit_Obj_Power2(16)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

