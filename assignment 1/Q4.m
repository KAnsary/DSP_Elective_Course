clc;
Population = [3.93  , 5.31  , 7.24 , 9.64 , 12.86 ,17.07 , 23.19 ,31.44];
Year = [1790  , 1800 , 1810 , 1820 , 1830 , 1840 , 1850 , 1860];
Year = Year - 1780;
 
% Prepare the equation
y = Population';
x = [ones(length(Year),1) Year'];
 

% Analyze the model using "fit" function
% [linear , linear_gof] = fit(x(:,2), y, 'poly1')   
% [Quadratic , Quadratic_gof] = fit(x(:,2), y, 'poly2')   
[exponential , exponential_gof] = fit(x(:,2), y, 'exp1')   
% [power , power_gof] = fit(x(:,2), y, 'power1')    
% [logarithmic ,logarithmic_gof] = fit(log(x(:,2)), y, 'poly1')    
 exponential(90)
scatter(x(:,2),y);
 exponential(10)
hold on
plot(exponential)
xlabel('x(Year)')
ylabel('y(Population)')