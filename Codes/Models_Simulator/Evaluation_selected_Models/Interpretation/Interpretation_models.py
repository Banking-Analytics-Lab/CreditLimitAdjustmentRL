'''In this script the variable importance using entropy criterion and SHAP values were 
   generated to give interpretation to the models.'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
import pickle
import xgboost
from sklearn.metrics import plot_confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import  XGBClassifier
import shap

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

columnsToBin = ['MP_R', 'Int', 'HA_P']
x_total = pd.concat([x_train,x_test], axis=0, ignore_index=True) # by row

# One hot encoding
x_total = pd.get_dummies(x_total, columns = columnsToBin, drop_first=False)
x_train_n  = x_total.iloc[np.r_[0:x_train.shape[0]], :]
x_test_n  = x_total.iloc[np.r_[x_train.shape[0]:x_total.shape[0]], :]


filename1 = 'Model_Balance_Type_RF_red_SMOTENC.sav'
Model_Balance_Type_red= pickle.load(open(filename1, 'rb'))
#
print(f'The Model parameters are: {Model_Balance_Type_red.get_params()}')
# Variable importance based in the impurity decrease within each tree
importances = Model_Balance_Type_red.feature_importances_
indices = np.argsort(importances)[::-1][0:21]

f, ax = plt.subplots(figsize=(3, 8))
plt.title("Variable Importance - XGB Classifier")
sns.set_color_codes("pastel")
sns.barplot(y=[x_train_n.columns[i] for i in indices], 
            x=importances[indices], 
            label="Total", color="m")
ax.set(ylabel="Variable",
       xlabel="Variable Importance (Entropy)")
sns.despine(left=True, bottom=True)
plt.savefig('VarIm_Entropy_BalanceType.png', dpi=1200)
plt.show()

# Variable importance according SHAP values
fig, ax = plt.subplots(1,1)
explainer = shap.Explainer(Model_Balance_Type_red)
fig.legend(loc='lower right')
shap_values = explainer.shap_values(x_test_n, approximate=True)
plt.title('Shap Variable Importance Plot for the Multiclass Model'+'\n'+'Class 2 - Inactives/full payers, Class 1 - Greater Balances, Class 0 - Small/Medium Balances')
shap.summary_plot(shap_values, x_test_n, plot_type="bar")
fig.legend(loc='lower right')
fig.savefig('VarIm_SHAP_BalanceType.png', dpi=1200)

# # shap_values = Model_Balance_Type_red.get_feature_importance(x_test, type="ShapValues")
# # shap_values_transposed = shap_values.transpose(1, 0, 2)
# # shap.summary_plot(list(shap_values_transposed[:,:,:-1]))
# # plt.savefig('V_ISHAP_BalanceType.png', dpi=1200)
# # plt.show()

# # Shap plot that distinguishes over the classes
# shap.summary_plot(shap_values, x_test.values, plot_type="bar", class_names= Type_balances, feature_names = x_test.columns, show=False)
# plt.savefig('SHAP_by_Type_Balances.png')
# plt.show()

# # Also Shap summary for each class
# shap.summary_plot(shap_values[0], x_test.values, feature_names = x_test.columns, show=False)
# plt.savefig('SHAP_by_balance_0.png')
# plt.show()

# shap.summary_plot(shap_values[1], x_test.values, feature_names = x_test.columns, show=False)
# plt.savefig('SHAP_by_balance_1.png')
# plt.show()

# shap.summary_plot(shap_values[2], x_test.values, feature_names = x_test.columns, show=False)
# plt.savefig('SHAP_by_balance_2.png')
# plt.show()

# ### Calculation of the avg AUC, ovR and ovo

# y_pred = Model_Balance_Type_red.predict_proba(x_test)
# w_avg_AUC_ovR =roc_auc_score(y_test, y_pred, multi_class="ovr", average="weighted")
# w_avg_AUC_ovo =roc_auc_score(y_test, y_pred, multi_class="ovo", average="weighted")
# print(f'This is the weighted average AUC one vs Rest for the classifier 1: {w_avg_AUC_ovR}')
# print(f'This is the weighted average AUC one vs Rest for the classifier 1: {w_avg_AUC_ovo}')

# y_pred = Model_Balance_Type_red.predict(x_test)
# print(classification_report(y_test, y_pred, labels=[0, 1, 2]))

# ### In all the data set
# cf_matrix_total = plot_confusion_matrix(Model_Balance_Type_red, x, y, display_labels=Type_balances)#, normalize='true')
# # plt.savefig('confusion_Matrix_Clas1_xgb_all.png')
# plt.show()

# ######################################################
# Interpretation about the regressor models 
########################################################

# Amount 
# Class 0
small_medium_balances_Model = xgboost.XGBRegressor()
small_medium_balances_Model.load_model('SmallMediumBalances_xgb.json')

# Select only financial features
data = data.iloc[:, np.r_[0:17,39]]

columnsToBin = ['MP_R', 'Int', 'HA_P']
# One hot encoding
data = pd.get_dummies(data, columns = columnsToBin, drop_first=False)
# In this case, the prediction will be for the amount of balances for class 0 
data_all = data.copy()
data_all = data_all[(data_all.Avg_Remain_pros>0)&(data_all.Avg_Remain_pros<=1500)]
print(f'This is the number of observations of this class {data_all.shape}')

data_all = pd.DataFrame(data_all)


# Define X and y
X = data_all.drop(['Avg_Remain_pros'], axis=1)
y = data_all.Avg_Remain_pros
print(f'This is the mean of the remained balance {y.mean()}')

# split in train and test
x_train, x_test, y_train, y_test = train_test_split(X, 
                                                    y,  
                                                    test_size = 0.3,           
                                                    random_state = 170721) 
x_train = pd.DataFrame(x_train, columns = x_train.columns)
x_test = pd.DataFrame(x_test, columns = x_test.columns)

# Variable importance based in the impurity decrease within each tree
importances = small_medium_balances_Model.feature_importances_
indices = np.argsort(importances)[::-1][0:15]

f, ax = plt.subplots(figsize=(3, 8))
plt.title("Variable Importance - XGB for Balance Amount Class 0")
sns.set_color_codes("pastel")
sns.barplot(y=[x_train.columns[i] for i in indices], 
            x=importances[indices], 
            label="Total", color="m")
ax.set(ylabel="Variable",
       xlabel="Variable Importance (Entropy)")
sns.despine(left=True, bottom=True)
plt.savefig('VarIm_Entropy_BalanceAmount_Class0.png', dpi=1200)
plt.show()

# Variable importance according SHAP values
fig, ax = plt.subplots(1,1)
explainer = shap.Explainer(small_medium_balances_Model)
shap_values = explainer.shap_values(x_test, approximate=True)
plt.title('Shap Variable Importance Plot for the Regressor Model'+'\n'+'Class 0')
shap.summary_plot(shap_values, x_test_n, plot_type="bar")
fig.savefig('VarIm_SHAP_BalanceAmount_Class0.png', dpi=1200)

# Class 1
filename2 = 'Balance_modelCV_red.sav'
Balance_class1= pickle.load(open(filename2, 'rb'))

data = pd.read_pickle('df_train_models_1.pkl')
print(data.info())

# Select only financial features
data = data.iloc[:, np.r_[0:17,39]]
print(data.info())

columnsToBin = ['MP_R', 'Int', 'HA_P']
# One hot encoding
data = pd.get_dummies(data, columns = columnsToBin, drop_first=False)

data_all = data.copy()
data_all = data_all[data_all.Avg_Remain_pros!=0]
print(f'This is the number of balances less than 10 over the filtration {data_all[data_all.Avg_Remain_pros<10].shape[0]}')
# Train over the observations where the balance is greater or equal than 10
data_all = data_all[data_all.Avg_Remain_pros>=10]
data_all = pd.DataFrame(data_all)

# Define X and y
X = data_all.drop(['Avg_Remain_pros'], axis=1)
y = data_all.Avg_Remain_pros

# split in train and test
x_train, x_test, y_train, y_test = train_test_split(X, 
                                                    y,  
                                                    test_size = 0.3,           
                                                    random_state = 170721) 
x_train = pd.DataFrame(x_train, columns = x_train.columns)
x_test = pd.DataFrame(x_test, columns = x_test.columns)

# Variable importance based in the impurity decrease within each tree
importances = Balance_class1.feature_importances_
indices = np.argsort(importances)[::-1][0:15]

f, ax = plt.subplots(figsize=(3, 8))
plt.title("Variable Importance - XGB for Balance Amount Class 1")
sns.set_color_codes("pastel")
sns.barplot(y=[x_train.columns[i] for i in indices], 
            x=importances[indices], 
            label="Total", color="m")
ax.set(ylabel="Variable",
       xlabel="Variable Importance (Entropy)")
sns.despine(left=True, bottom=True)
plt.savefig('VarIm_Entropy_BalanceAmount_Class1.png', dpi=1200)
plt.show()

# Variable importance according SHAP values
fig, ax = plt.subplots(1,1)
explainer = shap.Explainer(Balance_class1)
shap_values = explainer.shap_values(x_test, approximate=True)
plt.title('Shap Variable Importance Plot for the Regressor Model'+'\n'+'Class 1')
shap.summary_plot(shap_values, x_test_n, plot_type="bar")
fig.savefig('VarIm_SHAP_BalanceAmount_Class1.png', dpi=1200)