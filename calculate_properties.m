%% 定义函数
function [kai, delta_x_a, G, F, dg, Hv] = calculate_properties(table_composition)
% 计算已知元素含量table_composition的情况下的性质值
%% 填写已知的元素特征参量
% 表中数据顺序 Mo Nb Ta Hf V W Cr Ti Zr
r = [129;134;134;144;122;130;118;132;145];
ec = [6.82, 7.57, 8.1, 6.44, 5.31, 8.9, 4.85, 3.39, 6.25];
g = [125.6, 37.5, 69.2, 56, 46.7, 160.6, 45.6, 26.2, 35];
x_a = [1.47, 1.41, 1.34, 1.16, 1.53, 1.47, 1.65, 1.38, 1.32];
hv = [1530, 1320, 873, 1760, 628, 3430, 1060, 970, 903];
% 原子半径差delta_r
ra = table_composition * r;
num_1 = size(table_composition,1);
num_2 = size(table_composition,2);
Ri = repmat(r',num_1,1);
R = repmat(ra(:,:),1,num_2);
delta_r_i = table_composition .* (ones(num_1,num_2)-Ri ./ R) .* (ones(num_1,num_2)-Ri ./ R);
delta_r = sqrt(sum(delta_r_i,2));

% 组态熵ΔS delta_S
table0506_0_1 = [];
for i = 1:num_1
    for j = 1:num_2
        if table_composition(i,j) == 0
            table0506_0_1(i,j) = 1;
        else
            table0506_0_1(i,j) = table_composition(i,j);
        end
    end
end
delta_S = -8.314*sum(table_composition .* log(table0506_0_1),2);

% 剪切模量G G
G = table_composition * g';

% 局域模量错配D.G dg
dg = [];
for i = 1:num_1
    dgg = 0;
    for j = 1:num_2
        for k = 1:num_2
            dgg = dgg + abs(g(j)-g(k))*table_composition(i,j)*table_composition(i,k);
        end
    end
    dg = [dg;dgg];
end
dg =  dg*0.5;

% Allen电负性 Δχ_allen
x_a_a = table_composition * x_a';
x_ai = repmat(x_a,num_1,1);
x_aa = repmat(x_a_a(:,:),1,num_2);
delta_x_a_i = table_composition .* (x_aa-x_ai) .* (x_aa-x_ai);
delta_x_a = sqrt(sum(delta_x_a_i,2));

% 平均结合能 Ec
Ec = table_composition * ec';

% Λ参数 kai
kai = 0.0001*delta_S ./ (delta_r .* delta_r);

% 晶格畸变能μ miu
miu = 0.5 * (Ec .* delta_r);

% 派纳力因子 F
F = 2* G .* (ones(num_1,1)-miu);

% 硬度Hv hv
Hv = table_composition * hv';
% kai = kai * 0.1;
% G = G * 0.01;
% F = F * 0.001;
% dg = dg * 0.01;
% Hv = Hv * 0.0001;
end