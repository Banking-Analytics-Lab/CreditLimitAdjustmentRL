'''In this script the ensemble model is evaluated, this is the two stage model, 
   which consists on a classificator and regressor model, in order to predict the 
   average outstanding balance at the end of the next three future months. 
   RMSE and WAPE measures were calculated.'''

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

#Load the models

filename1 = 'Model_Balance_Type_RF_red_SMOTENC.sav'
Model_Balance_Type_red= pickle.load(open(filename1, 'rb'))

# Amount 
# Class 0
small_medium_balances_Model = xgboost.XGBRegressor()
small_medium_balances_Model.load_model('SmallMediumBalances_xgb.json')
# Class 1
filename2 = 'Balance_modelCV_red.sav'
Balance_class1= pickle.load(open(filename2, 'rb'))



begin_time=datetime.datetime.now()
df = pd.read_pickle('df_train_models_1.pkl')
print(df.info())

# Select only financial features
df = df.iloc[:, np.r_[0:17,39]]
print(df.info())

columnsToBin = ['MP_R', 'Int', 'HA_P']
df = pd.get_dummies(df, columns = columnsToBin, drop_first=False)

df_all = pd.DataFrame(df)

# Define X and y
X = df_all.drop(['Avg_Remain_pros'], axis=1)
y = df_all.Avg_Remain_pros
# counts = df_all.Inact_soon_payer.value_counts()
# print(counts)

# split in train and test
x_train, x_test, y_train, y_test = train_test_split(X, 
                                                    y,  
                                                    test_size = 0.3,           
                                                    random_state = 170721) 
x_train = pd.DataFrame(x_train, columns = x_train.columns)
x_test = pd.DataFrame(x_test, columns = x_test.columns)

# Evaluate the ensemble over the test data set 
# Predict the type of Avg balance
x_test_n = x_test.copy()
x_test_n['Type_balance'] = Model_Balance_Type_red.predict(x_test)
y_pred = (x_test_n.Type_balance==1)*Balance_class1.predict(x_test)+(x_test_n.Type_balance==0)*small_medium_balances_Model.predict(x_test)
test_RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'This is the test RMSE {test_RMSE}')
print(f'This is the percentage over the standard error {test_RMSE/y_test.std()}')

# Define weighted WAPE 
def wape(y, y_pred):
    wape = np.sum(np.abs(y-y_pred))/np.sum(y)
    return wape

print(f'This is the WAPE {wape(y_test, y_pred)}')
