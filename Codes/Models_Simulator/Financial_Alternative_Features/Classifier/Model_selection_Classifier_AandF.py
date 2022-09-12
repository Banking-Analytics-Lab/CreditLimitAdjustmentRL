''''In this script different classifications models were trained doing also hyperparameter
search. Here the goal is to classify the balance type, which was ordered according to the their 
proportions in the portfolio
# Class 2 := Inactives or full payers
# Class 1 := Users with consumption greater than 1500
# Class 0 := Consumption (0, 1500]
In this case alternative and financial features were used.
'''
import pandas as pd
import numpy as np
import pickle 
import random
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from sklearn.pipeline import Pipeline,FeatureUnion

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, plot_confusion_matrix, roc_auc_score, classification_report 
from imblearn.over_sampling import SMOTE, SMOTENC
#  Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Needed to reproductible purpose
random.seed(170721)
np.random.seed(170721)

begin_time = datetime.datetime.now()

data = pd.read_pickle('df_train_models_1.pkl')
# Create the variable with the balances type
# Class 2 := Inactives or full payers
# Class 1 := Users with consumption greater than 1500
# Class 0 := Consumption (0, 1500]

data['Type_balance'] = (data.Avg_Remain_pros==0)*2+(data.Avg_Remain_pros>1500)*1
print(data.Type_balance.value_counts())
print(data.Type_balance.value_counts()/data.shape[0])

# X = data.drop(['Type_balance', 'Delta_Provision_1', 'Avg_Remain_pros'], axis=1)
X = data.drop(['Type_balance', 'Avg_Remain_pros'], axis=1)
y = data['Type_balance']

x_train, x_test, y_train, y_test = train_test_split(X, 
                                                    y,  
                                                    test_size = 0.3,           
                                                    random_state = 170721) 
x_train = pd.DataFrame(x_train, columns = x_train.columns)
x_test = pd.DataFrame(x_test, columns = x_test.columns)
y_train = pd.DataFrame(y_train, columns = ['Type_balance'])

print(x_train.info())
# Since the classes are umbalanced
sm = SMOTENC(random_state=170721, categorical_features=[4, 6, 8, 18, 19, 20, 21, 33, 34])
x_train, y_train = sm.fit_resample(x_train, y_train.values.ravel())

# Categorical variables in the data frame.

columnsToBin = ['MP_R', 'Int', 'HA_P','SEGMENT_RFM', 'MATURITY', 'IS_PRIME', 'CATEGORY',
                'FAV_VERTICAL','FAV_PAYMENT_METHOD']
x_total = pd.concat([x_train,x_test], axis=0, ignore_index=True) # by rows

# One hot encoding
x_total = pd.get_dummies(x_total, columns = columnsToBin, drop_first=False)
x_train_n  = x_total.iloc[np.r_[0:x_train.shape[0]], :]
x_test_n  = x_total.iloc[np.r_[x_train.shape[0]:x_total.shape[0]], :]

'''The following models were trained: decision Trees, XGB and Random Forest also doing hyperparameter search'''
# Decision tree
Model_Tr = DecisionTreeClassifier(random_state=170721)

# XGB 
model_xgb = XGBClassifier(max_depth=3,                
                            learning_rate=0.1,           
                            n_estimators=50,             
                            verbosity=1,                  
                            objective='multi:softmax',
                            num_class =3,  
                            booster='gbtree',            
                            n_jobs=6,                     
                            gamma=0.001,                 
                            subsample=0.632,              
                            colsample_bytree=1,           
                            colsample_bylevel=1,         
                            colsample_bynode=1,                   
                            base_score=0.5,              
                            random_state=170721,
                            use_label_encoder=False,
                            eval_metric='logloss'
                            )
# Random Forest
model_RF =  RandomForestClassifier(n_estimators=100, 
                                      criterion='gini',
                                      min_samples_split=2,
                                      min_samples_leaf=1,
                                      min_weight_fraction_leaf=0.0,
                                      max_features='sqrt', 
                                      max_leaf_nodes=None,
                                      bootstrap=True,
                                      random_state=170721,
)

# Hyper parameters

# Decision Tree
param_grid_Tr = dict({'criterion': ['entropy', 'gini'],
                   'min_samples_leaf': [2, 3, 4, 5, 6],
                 'max_depth' : [2, 3, 4, 5, 6, 7, 8],
                 'min_samples_split' : [2, 3, 4, 5, 6]
                  })

# xgb 
param_grid_xgb = dict({'n_estimators': [50, 100, 150, 200, 250, 300],
                   'max_depth': [3, 4, 5, 6],
                 'learning_rate' : [0.001, 0.01, 0.1]
                  })

# RF
param_grid_RF = dict({'n_estimators': [50, 100, 150, 200, 250, 300],
                   'max_depth': [4, 5, 6, 7, 8],
                 'min_samples_split' : [4, 5, 6]
                  })

# Now definition of the grid searchs

GridCV_Tr = GridSearchCV(estimator=Model_Tr,
        param_grid=param_grid_Tr,
        scoring='f1_weighted',
        cv=5)

GridCV_xgb = GridSearchCV(estimator=model_xgb,
        param_grid=param_grid_xgb,
        scoring='f1_weighted',
        cv=5)

GridCV_RF = GridSearchCV(estimator=model_RF,
        param_grid=param_grid_RF,
        scoring='f1_weighted',
        cv=5)

Grids = [GridCV_Tr, GridCV_xgb, GridCV_RF]

for grid in Grids:
    grid.fit(x_train_n,y_train)

Dict_grid = { 0: 'Decision Trees', 
             1: 'XGB', 2:'Random Forest' 
             }
            
for i, model in enumerate(Grids):
    print('{} Test f1-weighted score: {}'.format(Dict_grid[i], np.round(model.score(x_test_n,y_test),2)))
    print('{} Best Parameters: {}'.format(Dict_grid[i], model.best_params_))         

total_time = datetime.datetime.now()-begin_time
print(f'The total time of execution is {total_time}')