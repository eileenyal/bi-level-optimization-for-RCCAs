           %0724的paretofront_custom函数代码取的是下半边缘的，有误一，修改 0802留
clear;clc;
load Pareto_Strain25_0724.mat
load Pareto_US1000_0724.mat
obj_all = [];
dec_all = [];
for i =1:5
    [dec,obj] = platemo('algorithm',@NSGAII,'problem',{@optimization, net1,net2},'M',2,'D',6,'N', 50000);
    obj_all = [obj_all;obj];
    dec_all = [dec_all;dec];
end
% 反归一化数据
y(:,2) = -mapminmax('reverse', obj_all(:,2), tps2)';
y(:,1) = -mapminmax('reverse', obj_all(:,1), tps1)';
x1 = mapminmax('reverse',dec_all(:,1:4)',xps2)';
x(:,1:4) = x1;
x2 = mapminmax('reverse',dec_all(:,3:6)',xps1)';
x(:,5:6) = x2(:,3:4);
data_all_reverse = [y,x];
% Find non-dominated solutions on Pareto front
[index, pareto_obj] = paretofront_custom(y);
pareto = data_all_reverse(index, 1:end);

% Plot Pareto front
scatter(pareto(:,1), pareto(:,2));
xlabel('Objective 1');
ylabel('Objective 2');
title('Pareto Front');
grid on;
legend('Non-dominated solutions','Location','northwest');