'''In this script it is trained and saved the regressor model for the 
   Remained outstaning balance given that the type of balance is 1. 
   It was decided to use random forest regressor since the hypothesis for using Multiple 
   linear regression are not satisfied and XGB model had worst performance.
   ** In this case only financial variables were used.'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
import pickle
import xgboost
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV, cross_validate
from sklearn.ensemble import RandomForestRegressor


begin_time=datetime.datetime.now()

# Definition of the WAPE measure
def wape(y, y_pred):
    wape = np.sum(np.abs(y-y_pred))/np.sum(y)
    return wape
 
df = pd.read_pickle('df_train_models_1.pkl')
print(df.info())

# Select only financial features
df = df.iloc[:, np.r_[0:17,39]]
print(df.info())

columnsToBin = ['MP_R', 'Int', 'HA_P']
# One hot encoding
df = pd.get_dummies(df, columns = columnsToBin, drop_first=False)

df_all = df.copy()
df_all = df_all[df_all.Avg_Remain_pros!=0]
print(f'This is the number of balances less than 10 over the filtration {df_all[df_all.Avg_Remain_pros<10].shape[0]}')
# Train over the observations where the balance is greater or equal than 10
df_all = df_all[df_all.Avg_Remain_pros>=10]
df_all = pd.DataFrame(df_all)

# Define X and y
X = df_all.drop(['Avg_Remain_pros'], axis=1)
y = df_all.Avg_Remain_pros

# split in train and test
x_train, x_test, y_train, y_test = train_test_split(X, 
                                                    y,  
                                                    test_size = 0.3,           
                                                    random_state = 170721) 
x_train = pd.DataFrame(x_train, columns = x_train.columns)
x_test = pd.DataFrame(x_test, columns = x_test.columns)

# Define the Random Forest model 
Balance_Model = RandomForestRegressor(n_estimators=100, 
                                      criterion='squared_error',
                                      max_depth=9,
                                      min_samples_split=2,
                                      min_samples_leaf=2,
                                      min_weight_fraction_leaf=0.0,
                                      max_features='sqrt', 
                                      max_leaf_nodes=None,
                                      min_impurity_decrease=0.05,
                                      bootstrap=True,
                                      oob_score=True, 
                                      random_state=170721,
                                      verbose=0,
                                      warm_start=True, 
                                      n_jobs=6)                                               
# Perform the hyperparameter search 
param_grid = dict({'n_estimators': [400, 450, 500, 550, 600, 650],
                   'max_depth': [10, 11, 12, 13, 14, 15, 16],
                 'min_samples_leaf' : [3, 4, 5, 6],
                 })

GridCVBalance =  GridSearchCV(Balance_Model,
                       param_grid,
                       cv = 5,
                       scoring = 'neg_mean_squared_error',
                       n_jobs = 6,
                       refit = True,
                       verbose = 1
                      )
GridCVBalance.fit(x_train, y_train)

filename = 'Balance_modelCV_red.sav'
pickle.dump(GridCVBalance.best_estimator_, open(filename, 'wb'))
print(f' The best set of parameters for the XGB is {GridCVBalance.best_params_}')

cv_best = cross_val_score(Balance_Model,x_train, y_train,cv=5,scoring = 'neg_mean_squared_error')
print(f'The cross validation RMSE for the best XGB Regressor Chain { np.sqrt(-cv_best.mean())}')

Balance_Model.fit(x_train,y_train)
y_pred = Balance_Model.predict(x_test)
df_aux = pd.DataFrame({})
df_aux['y_pred'] = y_pred

data_frame = pd.DataFrame({})
data_frame['y_test'] = y_test
data_frame['y_pred'] = y_pred

# Predictions only over those with balance type 1
data_filter = data_frame[data_frame.y_test>1500]
test_error_filter = mean_absolute_error(data_filter.y_test, data_filter.y_pred)
# Evaluation metrics
print(f'This is the RMSE over the balances greater than 1500: {np.round(test_error_filter,2)}')
print(f'The percentage over the standard deviation of y_test for balances greater than 1500: {test_error_filter/data_frame.y_test.std()}')
print(f'The MAPE  for balances greater than 1500: {np.round(mean_absolute_percentage_error(data_filter.y_test, data_filter.y_pred)*100,2)}%')
print(f'The R2 for balances greater than 1500: {np.round(r2_score(data_filter.y_test, data_filter.y_pred),2)}')

print(f'This is the range of y_test after filtering {data_filter.y_test.describe()}')
print(f'This is the summary statistics of y_pred after the filtering:  {data_filter.y_pred.describe()}')
print(f'The WAPE over the balances that are greater than 1500:  {np.round(wape(data_filter.y_test, data_filter.y_pred),2)}')

Total_time=datetime.datetime.now()-begin_time
print(f'Total time of execution {Total_time}')