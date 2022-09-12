'''In this script it is trained and saved the regressor model for the 
   Remained outstaning balance given that the type of balance is 0. 
   It was decided to use xgb regressor since the hypothesis for using Multiple 
   linear regression are not satisfied. 
   * Only financial features were used.'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
import pickle
import xgboost
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV, cross_validate
from xgboost import  XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestRegressor

begin_time=datetime.datetime.now()

# Definition of WAPE measure used to evaluate the regressor performance,
# since MAPE is not adecuate in this context.

def wape(y, y_pred):
    wape = np.sum(np.abs(y-y_pred))/np.sum(y)
    return wape

df = pd.read_pickle('df_train_models_1.pkl')

# Select only financial features
df = df.iloc[:, np.r_[0:17,39]]

columnsToBin = ['MP_R', 'Int', 'HA_P']
# One hot encoding
df = pd.get_dummies(df, columns = columnsToBin, drop_first=False)
# In this case, the prediction will be for the amount of balances for class 0 
df_all = df.copy()
df_all = df_all[(df_all.Avg_Remain_pros>0)&(df_all.Avg_Remain_pros<=1500)]
print(f'This is the number of observations of this class {df_all.shape}')

df_all = pd.DataFrame(df_all)


# Define X and y
X = df_all.drop(['Avg_Remain_pros'], axis=1)
y = df_all.Avg_Remain_pros
print(f'This is the mean of the remained balance {y.mean()}')

# split in train and test
x_train, x_test, y_train, y_test = train_test_split(X, 
                                                    y,  
                                                    test_size = 0.3,           
                                                    random_state = 170721) 
x_train = pd.DataFrame(x_train, columns = x_train.columns)
x_test = pd.DataFrame(x_test, columns = x_test.columns)

XGB = XGBRegressor(max_depth=3,
                          learning_rate=0.1,
                          n_estimators=100,
                          verbosity=1,
                          objective='reg:squarederror',
                          booster='gbtree',
                          gamma=0.001,
                          subsample=0.632,
                          colsample_bytree=1,
                          colsample_bylevel=1,
                          colsample_bynode=1,
                          reg_alpha=1,
                          reg_lambda=0,
                          random_state=170721,
                          tree_method='hist',
                          n_jobs=6,
                          )    

param_grid = dict({'n_estimators': [10, 15, 20, 25, 30, 35, 40, 45, 50],
                   'max_depth': [2, 3, 4],
                 'learning_rate' : [0.001, 0.01, 0.1, 0.2, 0.3]
                  })

GridCVRegr = GridSearchCV(XGB,      
                       param_grid,          
                       cv = 5,      
                       scoring = 'neg_mean_squared_error', 
                       n_jobs = 6,          
                       refit = True,       
                       verbose = 1         
                      )

GridCVRegr.fit(x_train, y_train)
print(f'These are the best set of parameters :{GridCVRegr.best_params_}')

# Fit the model to save
XGB.fit(x_train, y_train)
XGB.save_model("SmallMediumBalances_xgb.json")

cv_best = cross_val_score(XGB,x_train, y_train,cv=5,scoring = 'neg_mean_squared_error')
print(f'The cross validation RMSE for the best XGB Regressor Chain { np.round(np.sqrt(-cv_best.mean(),2))}')

y_pred = XGB.predict(x_test)

print(f'This is the WAPE {np.round(wape(y_test, y_pred),2)}')
print(f'This is the WAPE if it is used the mean {np.round(wape(y_test, y_train.mean()*np.ones(len(y_test))),2)}')
Total_time=datetime.datetime.now()-begin_time
print(f'Total time of execution {Total_time}')