# -*- coding: utf-8 -*-
"""
Created on Tue May 13 11:33:15 2023

@author: Eileen
"""
import sys
import numpy as np
from sklearn.svm import LinearSVR
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor,RandomForestRegressor,GradientBoostingRegressor,BaggingRegressor,AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.svm import LinearSVR,SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score,cross_val_predict,train_test_split,GridSearchCV,LeaveOneOut
from sklearn.feature_selection import RFECV,RFE
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import joblib

data = np.loadtxt('data.txt', delimiter='\t')
data = np.random.permutation(data)
data_input = data[:,6:30] 
data_target = data[:,[4]]

#归一化
data_input_normed=(data_input-data_input.min(axis=0))/(data_input.max(axis=0)-data_input.min(axis=0))
data_target_normed=(data_target-data_target.min(axis=0))/(data_target.max(axis=0)-data_target.min(axis=0))
        
#GradientBoostingRegressor(n_estimators=i, max_depth = j)
results_errors = np.zeros((0,5), dtype=float)
for i in range (95,96,1):
    for j in range (6,7,1):      
        num = 0
        mean_num_errors_01 = np.zeros((0,2), dtype=float)
        mean_num_errors_02 = np.zeros((0,2), dtype=float)
        mean_num_errors_11 = np.zeros((0,2), dtype=float)
        mean_num_errors_12 = np.zeros((0,2), dtype=float)
        for num in range (0,10,1):
            num = num+1                 
            simulator = GradientBoostingRegressor(n_estimators=i, max_depth = j,criterion='friedman_mse')
            predicted_target_normed = cross_val_predict(simulator, data_input_normed, data_target_normed.ravel(), cv=10)
            score = cross_val_score(simulator, data_input_normed, data_target_normed.ravel(), cv=10) 
            predict_target = predicted_target_normed*(data_target.max(axis=0)-data_target.min(axis=0))+data_target.min(axis=0) #预测结果反归一化 
            MAE = np.mean(abs((predict_target.ravel() - data_target.ravel())))
            MSE = np.mean((predict_target.ravel() - data_target.ravel())*(predict_target.ravel() - data_target.ravel()))
            MAPE = np.mean(abs((predict_target.ravel() - data_target.ravel()) / data_target.ravel()))
            RMSE = np.sqrt(MSE)
            error_01 = [[np.mean(score), MAE], ]
            error_02 = [[np.mean(score), MAPE], ]
            error_11 = [[np.mean(score), MSE], ]
            error_12 = [[np.mean(score), RMSE], ]
            mean_num_errors_01 = np.append(mean_num_errors_01, error_01, axis=0)
            mean_num_errors_02 = np.append(mean_num_errors_02, error_02, axis=0)
            mean_num_errors_11 = np.append(mean_num_errors_11, error_11, axis=0)
            mean_num_errors_12 = np.append(mean_num_errors_12, error_12, axis=0)
        mean_MAE = np.mean(abs(mean_num_errors_01[:,1]))
        mean_MAPE = np.mean(abs(mean_num_errors_02[:,1]))
        mean_score = np.mean(abs(mean_num_errors_02[:,0]))
        mean_MSE = np.mean(abs(mean_num_errors_11[:,1]))
        mean_RMSE = np.mean(abs(mean_num_errors_12[:,1]))
        error_2 =  [[i, j,  mean_score, mean_MSE, mean_RMSE],] 
        results_errors = np.append(results_errors, error_2, axis=0) 
print('GBR ',mean_score, ' ', mean_MAE,' ',mean_MSE,' ',mean_RMSE,' ',mean_MAPE*100)

# #MLPRegressor
results_errors = np.zeros((0,5), dtype=float)
for i in range (9,10,1):
    for j in range (6,7,1):
        for k in range (7,8,1):
            num = 0
            mean_num_errors_01 = np.zeros((0,2), dtype=float)
            mean_num_errors_02 = np.zeros((0,2), dtype=float)
            mean_num_errors_11 = np.zeros((0,2), dtype=float)
            mean_num_errors_12 = np.zeros((0,2), dtype=float)
            for num in range (0,10,1):
                num = num+1                 
                simulator =  MLPRegressor(max_iter=100000, hidden_layer_sizes=(i, j, k),activation="tanh"
                                          ,learning_rate_init=0.01)
                predicted_target_normed = cross_val_predict(simulator, data_input_normed, data_target_normed.ravel(), cv=10)
                score = cross_val_score(simulator, data_input_normed, data_target_normed.ravel(), cv=10) 
               
                predict_target = predicted_target_normed*(data_target.max(axis=0)-data_target.min(axis=0))+data_target.min(axis=0) #预测结果反归一化              
                MAE = np.mean(abs((predict_target.ravel() - data_target.ravel())))
                MSE = np.mean((predict_target.ravel() - data_target.ravel())*(predict_target.ravel() - data_target.ravel()))
                MAPE = np.mean(abs((predict_target.ravel() - data_target.ravel()) / data_target.ravel()))
                RMSE = np.sqrt(MSE)
                error_01 = [[np.mean(score), MAE], ]
                error_02 = [[np.mean(score), MAPE], ]
                error_11 = [[np.mean(score), MSE], ]
                error_12 = [[np.mean(score), RMSE], ]
                mean_num_errors_01 = np.append(mean_num_errors_01, error_01, axis=0)
                mean_num_errors_02 = np.append(mean_num_errors_02, error_02, axis=0)
                mean_num_errors_11 = np.append(mean_num_errors_11, error_11, axis=0)
                mean_num_errors_12 = np.append(mean_num_errors_12, error_12, axis=0)
            mean_MAE = np.mean(abs(mean_num_errors_01[:,1]))
            mean_MAPE = np.mean(abs(mean_num_errors_02[:,1]))
            mean_score = np.mean(abs(mean_num_errors_02[:,0]))
            mean_MSE = np.mean(abs(mean_num_errors_11[:,1]))
            mean_RMSE = np.mean(abs(mean_num_errors_12[:,1]))
            error_2 =  [[i, j, k, mean_score, mean_MSE],] 
            results_errors = np.append(results_errors, error_2, axis=0)  
print('MLP ',mean_score, ' ', mean_MAE,' ',mean_MSE,' ',mean_RMSE,' ',mean_MAPE*100)

#GaussianProcessRegressor-GPR
results_errors = np.zeros((0,4), dtype=float)
for i in range (1,9,1):
    for j in range (100,1009,100):      
            num = 0
            mean_num_errors_01 = np.zeros((0,2), dtype=float)
            mean_num_errors_02 = np.zeros((0,2), dtype=float)
            mean_num_errors_11 = np.zeros((0,2), dtype=float)
            mean_num_errors_12 = np.zeros((0,2), dtype=float)
            for num in range (0,10,1):
                num = num+1                 
                simulator = GaussianProcessRegressor(n_restarts_optimizer = i, random_state = j)
                predicted_target_normed = cross_val_predict(simulator, data_input_normed, data_target_normed.ravel(), cv=10)
                score = cross_val_score(simulator, data_input_normed, data_target_normed.ravel(), cv=10) 
                predict_target = predicted_target_normed*(data_target.max(axis=0)-data_target.min(axis=0))+data_target.min(axis=0) #预测结果反归一化              
                MAE = np.mean(abs((predict_target.ravel() - data_target.ravel())))
                MSE = np.mean((predict_target.ravel() - data_target.ravel())*(predict_target.ravel() - data_target.ravel()))
                MAPE = np.mean(abs((predict_target.ravel() - data_target.ravel()) / data_target.ravel()))
                RMSE = np.sqrt(MSE)
                error_01 = [[np.mean(score), MAE], ]
                error_02 = [[np.mean(score), MAPE], ]
                error_11 = [[np.mean(score), MSE], ]
                error_12 = [[np.mean(score), RMSE], ]
                mean_num_errors_01 = np.append(mean_num_errors_01, error_01, axis=0)
                mean_num_errors_02 = np.append(mean_num_errors_02, error_02, axis=0)
                mean_num_errors_11 = np.append(mean_num_errors_11, error_11, axis=0)
                mean_num_errors_12 = np.append(mean_num_errors_12, error_12, axis=0)
            mean_MAE = np.mean(abs(mean_num_errors_01[:,1]))
            mean_MAPE = np.mean(abs(mean_num_errors_02[:,1]))
            mean_score = np.mean(abs(mean_num_errors_02[:,0]))
            mean_MSE = np.mean(abs(mean_num_errors_11[:,1]))
            mean_RMSE = np.mean(abs(mean_num_errors_12[:,1]))
            error_2 =  [[i, j,  mean_score, mean_MSE],] 
            results_errors = np.append(results_errors, error_2, axis=0)  
print('GPR ',mean_score, ' ', mean_MAE,' ',mean_MSE,' ',mean_RMSE,' ',mean_MAPE*100)

# SVR (C = i, epsilon= j)
results_errors = np.zeros((0,4), dtype=float)
for i in [1.0]:  
    for j in [0.06]:    
            num = 0
            mean_num_errors_01 = np.zeros((0,2), dtype=float)
            mean_num_errors_02 = np.zeros((0,2), dtype=float)
            mean_num_errors_11 = np.zeros((0,2), dtype=float)
            mean_num_errors_12 = np.zeros((0,2), dtype=float)
            for num in range (0,10,1):
                num = num+1                 
                simulator = SVR(C = i, epsilon= j)
                predicted_target_normed = cross_val_predict(simulator, data_input_normed, data_target_normed.ravel(), cv=10)
                score = cross_val_score(simulator, data_input_normed, data_target_normed.ravel(), cv=10) 
               
                predict_target = predicted_target_normed*(data_target.max(axis=0)-data_target.min(axis=0))+data_target.min(axis=0) #预测结果反归一化              
                MAE = np.mean(abs((predict_target.ravel() - data_target.ravel())))
                MSE = np.mean((predict_target.ravel() - data_target.ravel())*(predict_target.ravel() - data_target.ravel()))
                MAPE = np.mean(abs((predict_target.ravel() - data_target.ravel()) / data_target.ravel()))
                RMSE = np.sqrt(MSE)
                error_01 = [[np.mean(score), MAE], ]
                error_02 = [[np.mean(score), MAPE], ]
                error_11 = [[np.mean(score), MSE], ]
                error_12 = [[np.mean(score), RMSE], ]
                mean_num_errors_01 = np.append(mean_num_errors_01, error_01, axis=0)
                mean_num_errors_02 = np.append(mean_num_errors_02, error_02, axis=0)
                mean_num_errors_11 = np.append(mean_num_errors_11, error_11, axis=0)
                mean_num_errors_12 = np.append(mean_num_errors_12, error_12, axis=0)
            mean_MAE = np.mean(abs(mean_num_errors_01[:,1]))
            mean_MAPE = np.mean(abs(mean_num_errors_02[:,1]))
            mean_score = np.mean(abs(mean_num_errors_02[:,0]))
            mean_MSE = np.mean(abs(mean_num_errors_11[:,1]))
            mean_RMSE = np.mean(abs(mean_num_errors_12[:,1]))
            error_2 =  [[i, j,  mean_score, mean_MSE],] 
            results_errors = np.append(results_errors, error_2, axis=0)   
print('SVR ',mean_score, ' ', mean_MAE,' ',mean_MSE,' ',mean_RMSE,' ',mean_MAPE*100)

#LinearSVR(C = i, epsilon = j, max_iter=100000)
results_errors = np.zeros((0,4), dtype=float)
for i in [ 0.4]: 
    for j in [ 0.14 ]:     
            num = 0
            mean_num_errors_01 = np.zeros((0,2), dtype=float)
            mean_num_errors_02 = np.zeros((0,2), dtype=float)
            mean_num_errors_11 = np.zeros((0,2), dtype=float)
            mean_num_errors_12 = np.zeros((0,2), dtype=float)
            for num in range (0,10,1):
                num = num+1                 
                simulator = LinearSVR(C = i, epsilon = j, max_iter=100000)
                predicted_target_normed = cross_val_predict(simulator, data_input_normed, data_target_normed.ravel(), cv=10)
                score = cross_val_score(simulator, data_input_normed, data_target_normed.ravel(), cv=10) 
               
                predict_target = predicted_target_normed*(data_target.max(axis=0)-data_target.min(axis=0))+data_target.min(axis=0) #预测结果反归一化              
                MAE = np.mean(abs((predict_target.ravel() - data_target.ravel())))
                MSE = np.mean((predict_target.ravel() - data_target.ravel())*(predict_target.ravel() - data_target.ravel()))
                MAPE = np.mean(abs((predict_target.ravel() - data_target.ravel()) / data_target.ravel()))
                RMSE = np.sqrt(MSE)
                error_01 = [[np.mean(score), MAE], ]
                error_02 = [[np.mean(score), MAPE], ]
                error_11 = [[np.mean(score), MSE], ]
                error_12 = [[np.mean(score), RMSE], ]
                mean_num_errors_01 = np.append(mean_num_errors_01, error_01, axis=0)
                mean_num_errors_02 = np.append(mean_num_errors_02, error_02, axis=0)
                mean_num_errors_11 = np.append(mean_num_errors_11, error_11, axis=0)
                mean_num_errors_12 = np.append(mean_num_errors_12, error_12, axis=0)
            mean_MAE = np.mean(abs(mean_num_errors_01[:,1]))
            mean_MAPE = np.mean(abs(mean_num_errors_02[:,1]))
            mean_score = np.mean(abs(mean_num_errors_02[:,0]))
            mean_MSE = np.mean(abs(mean_num_errors_11[:,1]))
            mean_RMSE = np.mean(abs(mean_num_errors_12[:,1]))
            error_2 =  [[i, j,  mean_score, mean_MSE],] 
            results_errors = np.append(results_errors, error_2, axis=0) 
print('SVR.Lin ',mean_score, ' ', mean_MAE,' ',mean_MSE,' ',mean_RMSE,' ',mean_MAPE*100)

#KNeighborsRegressor (n_neighbors=i, leaf_size = j)
results_errors = np.zeros((0,4), dtype=float)
for i in range (3,4,1):
    for j in range (30,31,10):      
            num = 0
            mean_num_errors_01 = np.zeros((0,2), dtype=float)
            mean_num_errors_02 = np.zeros((0,2), dtype=float)
            mean_num_errors_11 = np.zeros((0,2), dtype=float)
            mean_num_errors_12 = np.zeros((0,2), dtype=float)
            for num in range (0,10,1):
                num = num+1                 
                simulator = KNeighborsRegressor (n_neighbors=i, leaf_size = j)
                predicted_target_normed = cross_val_predict(simulator, data_input_normed, data_target_normed.ravel(), cv=10)
                score = cross_val_score(simulator, data_input_normed, data_target_normed.ravel(), cv=10) 
               
                predict_target = predicted_target_normed*(data_target.max(axis=0)-data_target.min(axis=0))+data_target.min(axis=0) #预测结果反归一化              
                MAE = np.mean(abs((predict_target.ravel() - data_target.ravel())))
                MSE = np.mean((predict_target.ravel() - data_target.ravel())*(predict_target.ravel() - data_target.ravel()))
                MAPE = np.mean(abs((predict_target.ravel() - data_target.ravel()) / data_target.ravel()))
                RMSE = np.sqrt(MSE)
                error_01 = [[np.mean(score), MAE], ]
                error_02 = [[np.mean(score), MAPE], ]
                error_11 = [[np.mean(score), MSE], ]
                error_12 = [[np.mean(score), RMSE], ]
                mean_num_errors_01 = np.append(mean_num_errors_01, error_01, axis=0)
                mean_num_errors_02 = np.append(mean_num_errors_02, error_02, axis=0)
                mean_num_errors_11 = np.append(mean_num_errors_11, error_11, axis=0)
                mean_num_errors_12 = np.append(mean_num_errors_12, error_12, axis=0)
            mean_MAE = np.mean(abs(mean_num_errors_01[:,1]))
            mean_MAPE = np.mean(abs(mean_num_errors_02[:,1]))
            mean_score = np.mean(abs(mean_num_errors_02[:,0]))
            mean_MSE = np.mean(abs(mean_num_errors_11[:,1]))
            mean_RMSE = np.mean(abs(mean_num_errors_12[:,1]))
            error_2 =  [[i, j,  mean_score, mean_MSE],] 
            results_errors = np.append(results_errors, error_2, axis=0) 
print('KNN ',mean_score, ' ', mean_MAE,' ',mean_MSE,' ',mean_RMSE,' ',mean_MAPE*100)

# HistGradientBoostingRegressor (random_state=i,max_depth=j )
results_errors = np.zeros((0,4), dtype=float)
for i in range (10,11,10):
    for j in range (6,7,1):      
            num = 0
            mean_num_errors_01 = np.zeros((0,2), dtype=float)
            mean_num_errors_02 = np.zeros((0,2), dtype=float)
            mean_num_errors_11 = np.zeros((0,2), dtype=float)
            mean_num_errors_12 = np.zeros((0,2), dtype=float)
            for num in range (0,10,1):
                num = num+1                 
                simulator =  HistGradientBoostingRegressor (random_state=i,max_depth=j,max_iter=100 )
                predicted_target_normed = cross_val_predict(simulator, data_input_normed, data_target_normed.ravel(), cv=10)
                score = cross_val_score(simulator, data_input_normed, data_target_normed.ravel(), cv=10) 
               
                predict_target = predicted_target_normed*(data_target.max(axis=0)-data_target.min(axis=0))+data_target.min(axis=0) #预测结果反归一化              
                MAE = np.mean(abs((predict_target.ravel() - data_target.ravel())))
                MSE = np.mean((predict_target.ravel() - data_target.ravel())*(predict_target.ravel() - data_target.ravel()))
                MAPE = np.mean(abs((predict_target.ravel() - data_target.ravel()) / data_target.ravel()))
                RMSE = np.sqrt(MSE)
                error_01 = [[np.mean(score), MAE], ]
                error_02 = [[np.mean(score), MAPE], ]
                error_11 = [[np.mean(score), MSE], ]
                error_12 = [[np.mean(score), RMSE], ]
                mean_num_errors_01 = np.append(mean_num_errors_01, error_01, axis=0)
                mean_num_errors_02 = np.append(mean_num_errors_02, error_02, axis=0)
                mean_num_errors_11 = np.append(mean_num_errors_11, error_11, axis=0)
                mean_num_errors_12 = np.append(mean_num_errors_12, error_12, axis=0)
            mean_MAE = np.mean(abs(mean_num_errors_01[:,1]))
            mean_MAPE = np.mean(abs(mean_num_errors_02[:,1]))
            mean_score = np.mean(abs(mean_num_errors_02[:,0]))
            mean_MSE = np.mean(abs(mean_num_errors_11[:,1]))
            mean_RMSE = np.mean(abs(mean_num_errors_12[:,1]))
            error_2 =  [[i, j,  mean_score, mean_MSE],] 
            results_errors = np.append(results_errors, error_2, axis=0)   
print('HGBR ',mean_score, ' ', mean_MAE,' ',mean_MSE,' ',mean_RMSE,' ',mean_MAPE*100)

#RandomForestRegressor (n_estimators=i,max_depth=j )
results_errors = np.zeros((0,4), dtype=float)
for i in range (140,141,10):
    for j in range (8,9,1):      
            num = 0
            mean_num_errors_01 = np.zeros((0,2), dtype=float)
            mean_num_errors_02 = np.zeros((0,2), dtype=float)
            mean_num_errors_11 = np.zeros((0,2), dtype=float)
            mean_num_errors_12 = np.zeros((0,2), dtype=float)
            for num in range (0,10,1):
                num = num+1                 
                simulator = RandomForestRegressor (n_estimators=i,max_depth=j )
                predicted_target_normed = cross_val_predict(simulator, data_input_normed, data_target_normed.ravel(), cv=10)
                score = cross_val_score(simulator, data_input_normed, data_target_normed.ravel(), cv=10) 
               
                predict_target = predicted_target_normed*(data_target.max(axis=0)-data_target.min(axis=0))+data_target.min(axis=0) #预测结果反归一化              
                MAE = np.mean(abs((predict_target.ravel() - data_target.ravel())))
                MSE = np.mean((predict_target.ravel() - data_target.ravel())*(predict_target.ravel() - data_target.ravel()))
                MAPE = np.mean(abs((predict_target.ravel() - data_target.ravel()) / data_target.ravel()))
                RMSE = np.sqrt(MSE)
                error_01 = [[np.mean(score), MAE], ]
                error_02 = [[np.mean(score), MAPE], ]
                error_11 = [[np.mean(score), MSE], ]
                error_12 = [[np.mean(score), RMSE], ]
                mean_num_errors_01 = np.append(mean_num_errors_01, error_01, axis=0)
                mean_num_errors_02 = np.append(mean_num_errors_02, error_02, axis=0)
                mean_num_errors_11 = np.append(mean_num_errors_11, error_11, axis=0)
                mean_num_errors_12 = np.append(mean_num_errors_12, error_12, axis=0)
            mean_MAE = np.mean(abs(mean_num_errors_01[:,1]))
            mean_MAPE = np.mean(abs(mean_num_errors_02[:,1]))
            mean_score = np.mean(abs(mean_num_errors_02[:,0]))
            mean_MSE = np.mean(abs(mean_num_errors_11[:,1]))
            mean_RMSE = np.mean(abs(mean_num_errors_12[:,1]))
            error_2 =  [[i, j,  mean_score, mean_MSE],] 
            results_errors = np.append(results_errors, error_2, axis=0)
print('RF ',mean_score, ' ', mean_MAE,' ',mean_MSE,' ',mean_RMSE,' ',mean_MAPE*100)

#BaggingRegressor (n_estimators=i, random_state = j)
results_errors = np.zeros((0,4), dtype=float)
for i in range (90,91,1):
    for j in range (30,31,1):      
            num = 0
            mean_num_errors_01 = np.zeros((0,2), dtype=float)
            mean_num_errors_02 = np.zeros((0,2), dtype=float)
            mean_num_errors_11 = np.zeros((0,2), dtype=float)
            mean_num_errors_12 = np.zeros((0,2), dtype=float)
            for num in range (0,10,1):
                num = num+1                 
                simulator = BaggingRegressor(n_estimators=i, random_state = j)
                predicted_target_normed = cross_val_predict(simulator, data_input_normed, data_target_normed.ravel(), cv=10)
                score = cross_val_score(simulator, data_input_normed, data_target_normed.ravel(), cv=10) 
               
                predict_target = predicted_target_normed*(data_target.max(axis=0)-data_target.min(axis=0))+data_target.min(axis=0) #预测结果反归一化              
                MAE = np.mean(abs((predict_target.ravel() - data_target.ravel())))
                MSE = np.mean((predict_target.ravel() - data_target.ravel())*(predict_target.ravel() - data_target.ravel()))
                MAPE = np.mean(abs((predict_target.ravel() - data_target.ravel()) / data_target.ravel()))
                RMSE = np.sqrt(MSE)
                error_01 = [[np.mean(score), MAE], ]
                error_02 = [[np.mean(score), MAPE], ]
                error_11 = [[np.mean(score), MSE], ]
                error_12 = [[np.mean(score), RMSE], ]
                mean_num_errors_01 = np.append(mean_num_errors_01, error_01, axis=0)
                mean_num_errors_02 = np.append(mean_num_errors_02, error_02, axis=0)
                mean_num_errors_11 = np.append(mean_num_errors_11, error_11, axis=0)
                mean_num_errors_12 = np.append(mean_num_errors_12, error_12, axis=0)
            mean_MAE = np.mean(abs(mean_num_errors_01[:,1]))
            mean_MAPE = np.mean(abs(mean_num_errors_02[:,1]))
            mean_score = np.mean(abs(mean_num_errors_02[:,0]))
            mean_MSE = np.mean(abs(mean_num_errors_11[:,1]))
            mean_RMSE = np.mean(abs(mean_num_errors_12[:,1]))
            error_2 =  [[i, j,  mean_score, mean_MSE],] 
            results_errors = np.append(results_errors, error_2, axis=0)  
print('Bagging ',mean_score, ' ', mean_MAE,' ',mean_MSE,' ',mean_RMSE,' ',mean_MAPE*100)

#AdaBoostRegressor (n_estimators=i, random_state = j)
results_errors = np.zeros((0,4), dtype=float)
for i in range (80,81,10):
    for j in [1.6]:      
            num = 0
            mean_num_errors_01 = np.zeros((0,2), dtype=float)
            mean_num_errors_02 = np.zeros((0,2), dtype=float)
            mean_num_errors_11 = np.zeros((0,2), dtype=float)
            mean_num_errors_12 = np.zeros((0,2), dtype=float)
            for num in range (0,10,1):
                num = num+1                 
                simulator =  AdaBoostRegressor (n_estimators=i, learning_rate = j)
                predicted_target_normed = cross_val_predict(simulator, data_input_normed, data_target_normed.ravel(), cv=10)
                score = cross_val_score(simulator, data_input_normed, data_target_normed.ravel(), cv=10) 
               
                predict_target = predicted_target_normed*(data_target.max(axis=0)-data_target.min(axis=0))+data_target.min(axis=0) #预测结果反归一化              
                MAE = np.mean(abs((predict_target.ravel() - data_target.ravel())))
                MSE = np.mean((predict_target.ravel() - data_target.ravel())*(predict_target.ravel() - data_target.ravel()))
                MAPE = np.mean(abs((predict_target.ravel() - data_target.ravel()) / data_target.ravel()))
                RMSE = np.sqrt(MSE)
                error_01 = [[np.mean(score), MAE], ]
                error_02 = [[np.mean(score), MAPE], ]
                error_11 = [[np.mean(score), MSE], ]
                error_12 = [[np.mean(score), RMSE], ]
                mean_num_errors_01 = np.append(mean_num_errors_01, error_01, axis=0)
                mean_num_errors_02 = np.append(mean_num_errors_02, error_02, axis=0)
                mean_num_errors_11 = np.append(mean_num_errors_11, error_11, axis=0)
                mean_num_errors_12 = np.append(mean_num_errors_12, error_12, axis=0)
            mean_MAE = np.mean(abs(mean_num_errors_01[:,1]))
            mean_MAPE = np.mean(abs(mean_num_errors_02[:,1]))
            mean_score = np.mean(abs(mean_num_errors_02[:,0]))
            mean_MSE = np.mean(abs(mean_num_errors_11[:,1]))
            mean_RMSE = np.mean(abs(mean_num_errors_12[:,1]))
            error_2 =  [[i, j,  mean_score, mean_MSE],] 
            results_errors = np.append(results_errors, error_2, axis=0) 
print('AdaBoost ',mean_score, ' ', mean_MAE,' ',mean_MSE,' ',mean_RMSE,' ',mean_MAPE*100)

