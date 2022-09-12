'''In this script the best model is saved, observing the results using ALL features, we have that this model is the best
to use for balance type classification'''

import pandas as pd
import numpy as np
import pickle 
import random
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, XGBClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, balanced_accuracy_score, confusion_matrix, plot_confusion_matrix, roc_auc_score, classification_report 
from imblearn.over_sampling import SMOTE, SMOTENC

# Needed to reproductible purpose
random.seed(170721)
np.random.seed(170721)

data = pd.read_pickle('df_train_models_1.pkl')
# Create the variable with the balances type
# Class 2 := Inactives or full payers
# Class 1 := Users with consumption greater than 1500
# Class 0 := Consumption (0, 1500]

data['Type_balance'] = (data.Avg_Remain_pros==0)*2+(data.Avg_Remain_pros>1500)*1
print(data.Type_balance.value_counts())
print(data.Type_balance.value_counts()/data.shape[0])

X = data.drop(['Type_balance','Avg_Remain_pros'], axis=1)
# Select only financial features
X = X.iloc[:, np.r_[0:17]]
y = data['Type_balance']

x_train, x_test, y_train, y_test = train_test_split(X, 
                                                    y,  
                                                    test_size = 0.3,           
                                                    random_state = 170721) 
x_train = pd.DataFrame(x_train, columns = x_train.columns)
x_test = pd.DataFrame(x_test, columns = x_test.columns)
y_train = pd.DataFrame(y_train, columns = ['Type_balance'])

# Since the classes are umbalanced
sm = SMOTENC(random_state=170721, categorical_features=[4, 6, 8])
x_train, y_train = sm.fit_resample(x_train, y_train.values.ravel())

model_RF = XGBClassifier(max_depth=6,                    
                            learning_rate= 0.1,           
                            n_estimators=300,            
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
                            eval_metric='mlogloss'
                            )
# Then look for the hyperparameter search

columnsToBin = ['MP_R', 'Int', 'HA_P']
x_total = pd.concat([x_train,x_test], axis=0, ignore_index=True) # by row

# One hot encoding
x_total = pd.get_dummies(x_total, columns = columnsToBin, drop_first=False)
x_train_n  = x_total.iloc[np.r_[0:x_train.shape[0]], :]
x_test_n  = x_total.iloc[np.r_[x_train.shape[0]:x_total.shape[0]], :]

model_RF = model_RF.fit(x_train_n,y_train)
print(model_RF.get_params())
y_pred = model_RF.predict(x_test_n)
print(classification_report(y_test, y_pred, labels=[0, 1, 2]))
y_pred = model_RF.predict_proba(x_test_n)
w_avg_AUC_ovR =roc_auc_score(y_test, y_pred, multi_class="ovr", average="weighted")
w_avg_AUC_ovo =roc_auc_score(y_test, y_pred, multi_class="ovo", average="weighted")
print(f'This is the weighted average AUC one vs Rest for the classifier 1: {w_avg_AUC_ovR}')
print(f'This is the weighted average AUC one vs one for the classifier 1: {w_avg_AUC_ovo}')

# Save the model  
filename = 'Model_Balance_Type_RF_red_SMOTENC.sav'
pickle.dump(model_RF , open(filename, 'wb'))

