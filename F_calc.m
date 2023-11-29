clc;clear;

%% 1. 特征参数的计算
[data,datatxt,datacell] = xlsread('data.xlsx','data');
num_data1 = size(data,1);
num_data2 = size(data,2);
datacell = datacell(2:139,2:21);
table = zeros(num_data1,20);
for i = 1:num_data1
    for j = 1:20
        if isnan(datacell{i,j})
            table(i,j) = 0;
        else
            table(i,j) = datacell{i,j};
        end
    end
end
composition = table(:,1:20);
% 表中数据顺序 Mo Nb Ta Hf V W Al Cr Ti Zr Ni Co Sc Fe Mn Re Cu C B Si
r = [129;134;134;144;122;130;125;118;132;145;115;116;144;116;117;128;117;77;88;117];
x_p = [1.8,1.6,1.5,1.3,1.6,1.7,1.5,1.6,1.5,1.4,1.9,1.9,1.3,1.8,1.5,1.9,1.9,2.5,2,1.8];
x_a = [1.47,1.41,1.34,1.16,1.53,1.47,1.613,1.65,1.38,1.32,1.88,1.8,1.19,1.8,1.75,1.6,1.85,2.544,2.051,1.916];
x_mb = [1.94,2.03,1.94,1.73,2.22,1.79,1.64,2,1.86,1.7,1.76,1.72,1.5,1.67,2.04,2.06,1.08,2.37,1.9,1.98];
x_ar = [1.3,1.23,1.33,1.23,1.45,1.4,1.47,1.56,1.32,1.22,1.75,1.7,1.2,1.6,1.6,1.46,1.75,2.5,2.01,1.74];
x_jd = [3.9,4,4.11,3.8,3.6,4.4,3.23,3.72,3.45,3.64,4.4,4.3,3.34,4.06,3.72,4.02,4.48,6.27,4.29,4.47];
vec = [6,5,5,4,5,6,3,6,4,4,10,9,3,8,7,7,11,4,3,4];
ea = [1;1;2;2;2;2;3;1;2;2;2;2;2;2;2;2;1;4;3;4];
ec = [6.82,7.57,8.1,6.44,5.31,8.9,4.1,4.85,3.39,6.25,4.44,4.39,3.9,4.28,2.92,8.03,3.49,7.37,5.77,4.63];
g = [125.6,37.5,69.2,56,46.7,160.6,115.3,45.6,26.2,35,76,82,29.1,81,79.5,181,48.4, 0, 0,79.9];
omega = [4.6,4.3,4.3,3.9,4.3,4.6,4.3,4.5,4.3,4.1,5.2,5,3.5,4.5,4.1,5,4.6,4.7,4.5,4.8];
tm = [2890;2741;3273;2495;2199;3687;933;2130;1945;2127;1726;1768;1541;1808;1518;3453;1356;3820;2573;1683];
rou = [10.22,8.57,16.654,13.31,6.11,19.3,2.698,7.19,4.54,6.506,8.902,8.9,2.989,7.86,7.47,21.01,8.93,3.513,2.46,2.329];
ar = [95.54,92.906,180.948,178.94,50.942,183.85,26.982,51.996,47.867,91.224,58.693,58.933,44.956,55.845,54.938,186.207,63.546,12.011,10.811,28.086];
hv = [1530,1320,873,1760,628,3430,167,1060,970,903,638,1043,840,608,196,1320,874,0,0,0];
[Hij,Hijtxt,Hijcell] = xlsread('data.xlsx','ΔHij');
Hijcell = Hijcell(3:22,3:22);
Hij0506 = zeros(20,20);
for i = 1:20
    for j = 1:20
        if isnan(Hijcell{i,j})
            Hij(i,j) = 0;
        else
           Hij(i,j) = Hijcell{i,j};
        end
    end
end
% 原子半径差delta_r
ra = composition * r;
Ri = repmat(r',num_data1,1);
R = repmat(ra(:,:),1,20);
delta_r_i = composition .* (ones(num_data1,20)-Ri ./ R) .* (ones(num_data1,20)-Ri ./ R);
delta_r = sqrt(sum(delta_r_i,2));

% 局域尺寸错配D.r dr
dr = [];
for i = 1:num_data1
    drr = 0;
    for j = 1:20
        for k = 1:20
            drr = drr + abs(r(j)-r(k))*composition(i,j)*composition(i,k);
        end
    end
    dr = [dr;drr];
end
dr =  dr *0.5;

% 原子尺寸差γ参数 gamma
ra = composition * r;
rmax = [];
rmin = [];
for i = 1:num_data1
    max = 0;
    min = 400;
    for j = 1:20
        if composition(i,j)~=0&&r(j)>max
            max = r(j);
        elseif composition(i,j)~=0&&r(j)<min
            min = r(j);
        end
    end
    rmax = [rmax,max];
    rmin = [rmin,min];
end
gamma = [];
for i = 1:num_data1
    ga = (1-sqrt(((ra(i)+rmin(i))^2-ra(i)^2)/(ra(i)+rmin(i))^2)) / (1-sqrt(((ra(i)+rmax(i))^2-ra(i)^2)/(ra(i)+rmax(i))^2));
    gamma = [gamma;ga];
end

% 混合焓ΔH delta_H
delta_H = [];
for k = 1:num_data1
    delta_H_ij = 0;
    for i = 1:20
        for j = 1:20
            delta_H_ij = delta_H_ij + 4*composition(k,i)*composition(k,j)*Hij(i,j);
        end
    end
    delta_H = [delta_H;delta_H_ij];
end
delta_H = delta_H*0.5;

% 组态熵ΔS delta_S
table0506_0_1 = [];
for i = 1:num_data1
    for j = 1:20
        if composition(i,j) == 0
            table0506_0_1(i,j) = 1;
        else
            table0506_0_1(i,j) = composition(i,j);
        end
    end
end
delta_S = -8.314*sum(composition .* log(table0506_0_1),2);

% 熔化温度Tm
Tm = composition * tm;

% 熔化温度差delta_Tm
Tmi = repmat(tm',num_data1,1);
Tma = repmat(Tm(:,:),1,20);
delta_Tm_i = composition .* (ones(num_data1,20)-Tmi ./ Tma) .* (ones(num_data1,20)-Tmi ./ Tma);
delta_Tm = sqrt(sum(delta_Tm_i,2));

% Ω参数 Omega
Omega = Tm .* (0.001*delta_S) ./ abs(delta_H);

% 剪切模量G G
G = composition * g';

% 剪切模量差δG delta_G
Gi = repmat(g,num_data1,1);
Ga = repmat(G(:,:),1,20);
delta_G_i = composition .* (ones(num_data1,20)-Gi ./ Ga) .* (ones(num_data1,20)-Gi ./ Ga);
delta_G = sqrt(sum(delta_G_i,2));

% 局域模量错配D.G dg
dg = [];
for i = 1:num_data1
    dgg = 0;
    for j = 1:20
        for k = 1:20
            dgg = dgg + abs(g(j)-g(k))*composition(i,j)*composition(i,k);
        end
    end
    dg = [dg;dgg];
end
dg =  dg*0.5;

% 模量错配η eta
gi = repmat(g,num_data1,1);
G_a = repmat(G(:,:),1,20);
gi_g = 2*(gi - G_a) ./ (gi + G_a);
eta_i = composition .* gi_g ./ (ones(num_data1,20) + 0.5*abs(composition .* gi_g));
eta = sum(eta_i,2);

% 价电子浓度VEC
VEC = composition * vec';

% 自由电子浓度e/a e_a
e_a = composition * ea;

% 电负性差Δχ
% Pualing电负性 Δχ_pualing delta_x_p
x_p_a = composition * x_p';
x_pi = repmat(x_p,num_data1,1);
x_pa = repmat(x_p_a(:,:),1,20);
delta_x_p_i = composition .* (x_pa-x_pi) .* (x_pa-x_pi);
delta_x_p = sqrt(sum(delta_x_p_i,2));

% Allen电负性 Δχ_allen
x_a_a = composition * x_a';
x_ai = repmat(x_a,num_data1,1);
x_aa = repmat(x_a_a(:,:),1,20);
delta_x_a_i = composition .* (x_aa-x_ai) .* (x_aa-x_ai);
delta_x_a = sqrt(sum(delta_x_a_i,2));

% MB电负性 Δχ_Martynov&Batsanov
x_mb_a = composition * x_mb';
x_mbi = repmat(x_mb,num_data1,1);
x_mba = repmat(x_mb_a(:,:),1,20);
delta_x_mb_i = composition .* (x_mba-x_mbi) .* (x_mba-x_mbi);
delta_x_mb = sqrt(sum(delta_x_mb_i,2));

% AR电负性 Δχ_Alfed&Rochow 
x_ar_a = composition * x_ar';
x_ari = repmat(x_ar,num_data1,1);
x_ara = repmat(x_ar_a(:,:),1,20);
delta_x_ar_i = composition .* (x_ara-x_ari) .* (x_ara-x_ari);
delta_x_ar = sqrt(sum(delta_x_ar_i,2));

% 绝对电负性Δχ_Absolute
x_jd_a = composition * x_jd';
x_jdi = repmat(x_jd,num_data1,1);
x_jda = repmat(x_jd_a(:,:),1,20);
delta_x_jd_i = composition .* (x_jda-x_jdi) .* (x_jda-x_jdi);
delta_x_jd = sqrt(sum(delta_x_jd_i,2));

% 局域电负性差D.χ dx
% Pualing电负性 D.χ_pualing
dx_p = [];
for i = 1:num_data1
    dx = 0;
    for j = 1:20
        for k = 1:20
            dx = dx + abs(x_p(j)-x_p(k))*composition(i,j)*composition(i,k);
        end
    end
    dx_p = [dx_p;dx];
end
 dx_p =  dx_p*0.5;

%Allen电负性 D.χ_allen 
dx_a = [];
for i = 1:num_data1
    dx = 0;
    for j = 1:20
        for k = 1:20
            dx = dx + abs(x_a(j)-x_a(k))*composition(i,j)*composition(i,k);
        end
    end
    dx_a = [dx_a;dx];
end
dx_a =  dx_a*0.5;

% MB电负性 D.χ_Martynov&Batsanov 
dx_mb = [];
for i = 1:num_data1
    dx = 0;
    for j = 1:20
        for k = 1:20
            dx = dx + abs(x_mb(j)-x_mb(k))*composition(i,j)*composition(i,k);
        end
    end
    dx_mb = [dx_mb;dx];
end
dx_mb =  dx_mb*0.5;

% AR电负性 D.χ_Alfed&Rochow 
dx_ar = [];
for i = 1:num_data1
    dx = 0;
    for j = 1:20
        for k = 1:20
            dx = dx + abs(x_ar(j)-x_ar(k))*composition(i,j)*composition(i,k);
        end
    end
    dx_ar = [dx_ar;dx];
end
dx_ar =  dx_ar*0.5;

% 绝对电负性D.χ_Absolute
dx_jd = [];
for i = 1:num_data1
    dx = 0;
    for j = 1:20
        for k = 1:20
            dx = dx + abs(x_jd(j)-x_jd(k))*composition(i,j)*composition(i,k);
        end
    end
    dx_jd = [dx_jd;dx];
end
dx_jd =  dx_jd*0.5;

% 平均结合能 Ec
Ec = composition * ec';

% Λ参数 kai
kai = 0.0001*delta_S ./ (delta_r .* delta_r);

% 晶格畸变能μ miu
miu = 0.5 * (Ec .* delta_r);

% 能量因子 A
A =G .* delta_r .* ((ones(num_data1,1)+miu) ./ (ones(num_data1,1)-miu));

% 派纳力因子 F
F = 2* G .* (ones(num_data1,1)-miu);

% 功函数六次方ω^6 Omega_6
OMEGA_6 = (composition * omega') .^6;

% 密度density
weight = composition * ar';
rou_i = composition .* repmat(ar,num_data1,1) ./ repmat(rou,num_data1,1);
density = weight ./ sum(rou_i,2);

% 硬度Hv hv
Hv = composition * hv'; 

% 生成数据表
ans =horzcat(Tm, delta_Tm, density, delta_r, delta_x_jd, delta_x_p, delta_x_a, delta_x_mb, delta_x_ar, VEC, delta_H, delta_S, Omega, kai, gamma, e_a, Ec, eta, miu, A, F, OMEGA_6, G, delta_G, dx_jd, dx_p, dx_a, dx_mb, dx_ar, dr, dg, Hv);
% writematrix(ans,'data.xlsx','Sheet','data','Range','X2:BC143')