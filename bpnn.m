clc;clear all;close all;
load data.mat
%% 1.读取数据
rowrank = randperm(size(data, 1));
table_new = data(rowrank,:);
x = table_new(:,7:end);
t = table_new(:,4);
%% 2.十折交叉验证
K = 10; % number of folds
indices = crossvalind('Kfold', size(x, 1), K);
mae1 = zeros(1, K);
mse1 = zeros(1, K);
rmse = zeros(1, K);
r_squared = zeros(1, K);
for i = 1:K
    disp(['fold :', num2str(i)]);
    %% 3.数据归一化
    [x_norm,xps] = mapminmax(x',0,1);
    [t_norm,tps] = mapminmax(t',0,1);
    %% 4.划分训练集和测试集
    test_indices = (indices == i);
    train_indices = ~test_indices;
    x_train = x(train_indices,:);
    t_train = t(train_indices,:);
    x_test = x(test_indices,:);
    t_test = t(test_indices,:);
    xn_train = x_norm(:,train_indices);
    tn_train = t_norm(:,train_indices);
    xn_test = x_norm(:,test_indices);
    tn_test = t_norm(:,test_indices);

    %% 5.求解最佳隐含层
    xnum=size(x,2);
    tnum=size(t,2);
    hiddennum_best = 0;
    MSE0=1e+5;
    transform_func={'tansig','purelin'};
    train_func='trainbr';
    for hiddennum=fix(sqrt(xnum+tnum))+1:fix(sqrt(xnum+tnum))+10
        net=newff(xn_train,tn_train,hiddennum,transform_func,train_func);
        net.trainParam.epochs=3000;
        net.trainParam.lr=0.01;
        net.trainParam.goal=0.000001;
        net=train(net,xn_train,tn_train);
        an0=sim(net,xn_train);
        mse0=mse(tn_train,an0);
        if mse0<MSE0
            MSE0=mse0;
            hiddennum_best=hiddennum;
        end
    end

    %% 6.构建最佳隐含层的BP神经网络
    net=newff(xn_train,tn_train,hiddennum_best,transform_func,train_func);
    net.trainParam.epochs=3000;
    net.trainParam.lr=0.01;
    net.trainParam.goal=0.000001;

    %% 7.网络训练
    net=train(net,xn_train,tn_train);

    %% 8.网络测试
    an=sim(net,xn_test);
    test_simu=mapminmax('reverse',an,tps);
    error=test_simu-t_test';
    % 权值阈值
    W1 = net.iw{1, 1};  %输入层到中间层的权值
    B1 = net.b{1};      %中间各层神经元阈值
    W2 = net.lw{2,1};   %中间层到输出层的权值
    B2 = net.b{2};      %输出层各神经元阈值

    %% 9.输出结果参数
    [~,len]=size(t_test');
    SSE=sum(error.^2);
    MAE=sum(abs(error))/len;
    MSE=error*error'/len;
    RMSE=sqrt(MSE);
    MAPE=mean(abs(error./t_test'));
    r=corrcoef(t_test',test_simu);
    R1=r(1,2);

    rmse(i) = RMSE;
    r_squared(i) = R1^2;
    mape(i) = MAPE;
    mse1(i) = MSE;
    mae1(i) = MAE;

end
% 输出结果
disp(['R2: ', num2str(mean(r_squared))]);
disp(['MAE: ', num2str(mean(mae1))]);
disp(['MSE: ', num2str(mean(mse1))]);
disp(['RMSE: ', num2str(mean(rmse))]);
disp(['MAPE: ', num2str(mean(mape)*100),'%']);
disp(['最后一次R2:',num2str(r_squared(1,10))])
% save("Pareto_US1000_0724.mat",'xps','tps',"t_test",'t_train','x_test','x_train','net','r_squared')