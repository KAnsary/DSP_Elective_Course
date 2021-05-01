clc;
Height = [1.03754 , 1.40205 , 1.63806 , 1.77412 , 1.80392 , 1.71522 , 1.50942 , 1.21410 , 0.83173];
Time = [0 , 0.1080 , 0.2150 , 0.3225 , 0.4300 , 0.5375 , 0.6450 , 0.7525 , 0.8600];
 
% Prepare the equation
y = Height';
x = [ones(length(Time),1) Time'];
 

% Analyze the model using "fit" function
% [linear , linear_gof] = fit(x(:,2), y, 'poly1')   
[Quadratic , Quadratic_gof] = fit(x(:,2), y, 'poly2')   
% [cubic , cubic_gof] = fit(x(:,2), y, 'poly3')   
% Making the 0 case a small value to allow power function
% for i = 1:length(Time)
%     if x(i,2)== 0
%         x(i,2)= x(i,2)+0.001
%     end
% end
% [power , power_gof] = fit(x(:,2), y, 'power1')    

scatter(x(:,2),y);
 
hold on
plot(Quadratic)
xlabel('x(time)')
ylabel('y(Height)')