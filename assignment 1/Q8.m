close all;clear;clc;
%%%%%%%%%%%%%%%%%%%Data%%%%%%%%%%%%%%%%%
[num,txt,raw] = xlsread('Question_8_Data_without_Gender.xlsx');
FeaturesData=[ones(30,1) num];
%%%%%%%%%%%%%%%%%%%%Run Fitting%%%%%%%%%%%%%%%%%
Fit_Obj_Linear = fitlm(FeaturesData(1:22,1:4), FeaturesData(1:22,5),'poly1111');
Linear_RSqr=Fit_Obj_Linear.Rsquared.Ordinary;
Fit_Obj_Quad = fitlm(FeaturesData(1:22,1:4), FeaturesData(1:22,5),'poly2222');
Quad_RSqr=Fit_Obj_Quad.Rsquared.Ordinary;
Fit_Obj_Cubic = fitlm(FeaturesData(1:22,1:4), FeaturesData(1:22,5),'poly3333');
cubic_RSqr=Fit_Obj_Cubic.Rsquared.Ordinary;
Fit_Pow = @(b,x)(((b(1).*x(:,1))+b(2).*(x(:,2))+...
    b(3).*(x(:,3))+b(4).*(x(:,4))).^b(5)+b(6));
Fit_Obj_Pow = fitnlm(FeaturesData(1:22,1:4), FeaturesData(1:22,5),Fit_Pow,[1 2 3 4 1 5]);
Pow_RSqr=Fit_Obj_Pow.Rsquared.Ordinary;
Fit_Exp=@(b,x)((exp(b(1)*x(:,1)+b(2)*x(:,2)+b(3)*x(:,3)+b(4)*x(:,4)))+b(5));
Fit_Obj_Exp = fitnlm(FeaturesData(1:22,1:4), FeaturesData(1:22,5)./max(FeaturesData(1:22,5))...
    ,Fit_Exp,[-0.001 -0.002 -0.003 -0.004 -0.005]);
Exp_RSqr=Fit_Obj_Exp.Rsquared.Ordinary;
Fit_Log = @(b,x)(b(1).*log(x(:,1))+b(2).*log(x(:,2))+...
    b(3).*log(x(:,3))+b(4).*log(x(:,4))+b(5));
Fit_Obj_Log = fitnlm(FeaturesData(1:22,1:4), FeaturesData(1:22,5),Fit_Log,[1 1 1 1 1]);
Log_RSqr=Fit_Obj_Log.Rsquared.Ordinary;
Fit_Logistic=@(b,x)(1./(1.+exp(-1.*(b(1).*x(:,1)+b(2).*x(:,2)+...
    b(3).*x(:,3)+b(4).*x(:,4))+b(5))));
Fit_Obj_Logistic = fitnlm(FeaturesData(1:22,1:4), FeaturesData(1:22,5),Fit_Logistic,[1 1 1 1 1]);
Logistic_RSqr=Fit_Obj_Logistic.Rsquared.Ordinary;
%%%%%%%%%%%%%%Plot%%%%%%%%%%%%%%
figure(1);
scatter(FeaturesData(:,2),FeaturesData(:,5),'b');
hold on
title('Practical Problem');
xlabel('last Years Grades'); 
ylabel('Third Year Grade');
scatter(FeaturesData(:,3),FeaturesData(:,5),'r');
hold on
scatter(FeaturesData(:,4),FeaturesData(:,5),'kp');
hold on
legend('Secondary Year Grade','First Year Grade','Second Year Grade');
PredictedThirdGrades=feval(Fit_Obj_Linear,FeaturesData(23:end,1:4));
RSqr=[Linear_RSqr,Quad_RSqr,cubic_RSqr,Pow_RSqr,Exp_RSqr,Log_RSqr,Logistic_RSqr];
[maxRSqr,RSqrIdx]=max(RSqr);%cubic is best fit
if RSqrIdx==1
    fprintf('best model: Linear\n')
elseif RSqrIdx==2
    fprintf('best model: Quadratic\n' )
elseif RSqrIdx==3
    fprintf('best model: Cubic\n' )
elseif RSqrIdx==4
    fprintf('best model: Power\n' )
elseif RSqrIdx==5
    fprintf('best model: Exponential\n' )
elseif RSqrIdx==6
    fprintf('best model: Log\n' )
elseif RSqrIdx==7
    fprintf('best model: Logistic\n' )    
end
TotalError=0;
for i=23:30
Dev(1,i)=(FeaturesData(i,5)-PredictedThirdGrades(i-22,1)).^2;
TotalError=TotalError+Dev(1,i);
end
StandardDev=sqrt(TotalError)./8;
