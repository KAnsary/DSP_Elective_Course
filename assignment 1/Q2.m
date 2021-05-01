%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Clear WorkSpace and Command Window
clear;
close;
clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Model Inputs and Outputs
%Year = [1990 1991 1992 1993 1994];
Year = [1987 1988 1989 1990 1991 1992 1993 1994 1995 1996];
banks = [13.7 13.12 12.71 12.34 11.92 11.46 10.96 10.45 9.94 9.53];
 
Y_banks = banks';
X_Year = [ones(10,1) Year'];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Using Fit Functions to get the Model
[Fit_Obj_Linear, GOF_Linear] = fit(X_Year(:,2), Y_banks, 'poly1')
[Fit_Obj_Quadratic, GOF_Quadratic] = fit(X_Year(:,2), Y_banks, 'poly2')
[Fit_Obj_Cubic, GOF_Cubic] = fit(X_Year(:,2), Y_banks, 'poly3')
[Fit_Obj_Power1, GOF_Power1] = fit(X_Year(:,2), Y_banks, 'power1')
[Fit_Obj_Power2, GOF_Power2] = fit(X_Year(:,2), Y_banks, 'power2')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot Functions
figure(1);
scatter(Year, banks, 'b', 'fill');
title('Original Data', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'r');
xlabel('year', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'b'); 
ylabel('Banks in thousands', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'b');
figure(2);
plot(Fit_Obj_Linear, 'r:');
hold on;
plot(Fit_Obj_Quadratic, 'b-.');
hold on;
plot(Fit_Obj_Cubic, 'k');
hold on;
plot(Fit_Obj_Power1, 'g');
hold on;
plot(Fit_Obj_Power2, 'm');
hold on;
scatter(Year, banks, 'r', 'fill');
title('All Functions with original data', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'r');
xlabel('year', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'b'); 
ylabel('Banks in thousands', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'b');
legend('Linear', 'Qadratic', 'Cubic', 'Power1', 'Power2', 'Original', 'Location', 'NorthWest');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Best Function with original data
figure(3)
plot(Fit_Obj_Linear, 'k');
hold on;
scatter(Year, banks, 'r', 'fill');
title('Best Function (Linear) with original data', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'r');
xlabel('year', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'b'); 
ylabel('Banks in thousands', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'b');
legend('Linear', 'Original', 'Location', 'NorthWest');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Predict From 1995 to 1999
% Linear
Y_banks_1995_Linear = Fit_Obj_Linear(1995);
Y_banks_1996_Linear = Fit_Obj_Linear(1996);
Y_banks_1997_Linear = Fit_Obj_Linear(1997);
Y_banks_1998_Linear = Fit_Obj_Linear(1998);
Y_banks_1999_Linear = Fit_Obj_Linear(1999);
% Quadratic
Y_banks_1995_Quadratic = Fit_Obj_Quadratic(1995);
Y_banks_1996_Quadratic = Fit_Obj_Quadratic(1996);
Y_banks_1997_Quadratic = Fit_Obj_Quadratic(1997);
Y_banks_1998_Quadratic = Fit_Obj_Quadratic(1998);
Y_banks_1999_Quadratic = Fit_Obj_Quadratic(1999);
%Cubic
Y_banks_1995_Cubic = Fit_Obj_Cubic(1995);
Y_banks_1996_Cubic = Fit_Obj_Cubic(1996);
Y_banks_1997_Cubic = Fit_Obj_Cubic(1997);
Y_banks_1998_Cubic = Fit_Obj_Cubic(1998);
Y_banks_1999_Cubic = Fit_Obj_Cubic(1999);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Prediction
Y_banks_2000_Linear = Fit_Obj_Linear(11)
Y_banks_2005_Linear = Fit_Obj_Linear(16)
Y_banks_2000_Quadratic = Fit_Obj_Quadratic(11)
Y_banks_2005_Quadratic = Fit_Obj_Quadratic(16)
Y_banks_2000_Cubic = Fit_Obj_Cubic(11)
Y_banks_2005_Cubic = Fit_Obj_Cubic(16)
Y_banks_2000_Power1 = Fit_Obj_Power1(11)
Y_banks_2005_Power1 = Fit_Obj_Power1(16)
Y_banks_2000_Power2 = Fit_Obj_Power2(11)
Y_banks_2005_Power2 = Fit_Obj_Power2(16)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
