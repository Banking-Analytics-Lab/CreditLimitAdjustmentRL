'''This script stores the Data frame with the different predicted trajectories.
   The randomness of the simulator is given by the probability of belonging to each of the 
   balance types.
   Output: Data frame with all the options stored: dataframes_simulator_prospective.pkl, which is a data frames' list that contains the different predictions given the performed action (0 maintain, 1 increase)
   and the balance type.
   '''

import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import datetime
import random
import pickle
import xgboost
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

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
print(f'These are the columns after one hot encoding: {X.info()}')
print(X.columns)
# The previous is to verify the order of the features in the predictions

df2 = pd.read_pickle('df_statesRL_1.pkl')
# When training the RL algorithm the next must not done
df2 = df2.drop(['PD_0','Delta_Provision_1'], axis=1)
print(df2.info())

# In this case the number of data frames is only 4, two actions available and two types of balances (notice balance type 2 is
# not required to be stored.)
df_simulator_C = []

list_options = [(0, 0), (0, 1),
                (1, 0), (1, 1)]
# Name the different data frames

for (i,j) in list_options:
    df_simulator_C.append('df'+str(i)+str(j))

dataframes = {dataframes: pd.DataFrame() for dataframes in df_simulator_C}
# Convert to data frames
for (i,j) in list_options:
    dataframes['df'+str(i)+str(j)] = df2.copy()
    # Add the variables we need for the predictive models
    dataframes['df'+str(i)+str(j)]['HA_P_0'] = 1-i
    dataframes['df'+str(i)+str(j)]['HA_P_1'] = i
    if (i==0):
        dataframes['df'+str(i)+str(j)]['L_P'] = dataframes['df'+str(i)+str(j)]['L_R']
    else:
        # Action increase is set to be equal to 20% of the current limit
        dataframes['df'+str(i)+str(j)]['L_P'] = 1.2*(dataframes['df'+str(i)+str(j)]['L_R'])

# Store the predictions

for (i,j) in list_options:
    # Calculation of the vector of probabilities of the balance type
    # Auxiliar df to do the predictions with the models balance type and amount of the balance
    Aux_df_1 = dataframes['df'+str(i)+str(j)][['TC1', 'TC2', 'TC3', 'L_R', 'N_Months_R', 'EI', 'L_P', 'OB_cday_1',
       'P_pday_1', 'OB_cday_2', 'P_pday_2', 'OB_cday_3', 'P_pday_3', 'BS',
       'MP_R_0', 'MP_R_1', 'MP_R_2', 'Int_0.32', 'Int_0.55', 'Int_0.65',
       'HA_P_0', 'HA_P_1']]
    dataframes['df'+str(i)+str(j)]['Remain_3'] = Aux_df_1.OB_cday_3 - Aux_df_1.P_pday_3 
    dataframes['df'+str(i)+str(j)]['Prob_type_balance_0'] = Model_Balance_Type_red.predict_proba(Aux_df_1)[:, 0]
    dataframes['df'+str(i)+str(j)]['Prob_type_balance_1'] = Model_Balance_Type_red.predict_proba(Aux_df_1)[:, 1]
    dataframes['df'+str(i)+str(j)]['Prob_type_balance_2'] = Model_Balance_Type_red.predict_proba(Aux_df_1)[:, 2]
# According with the type of balance
    if (j==0):
        dataframes['df'+str(i)+str(j)]['Avg_Remain_pros'] = small_medium_balances_Model.predict(Aux_df_1)
    else:
        dataframes['df'+str(i)+str(j)]['Avg_Remain_pros'] = Balance_class1.predict(Aux_df_1)

#  Assumption over the action increase 
#  The Avg Remainded is greater than or equal the remainder given the maintain action 

for (i,j) in list_options:
    if (i==1):
        Aux_df_2 = pd.concat({'Avg_Remain_pros_ac1': dataframes['df'+str(i)+str(j)]['Avg_Remain_pros'],
                              'Avg_Remain_pros_ac0':  dataframes['df'+str(i)+str(j-1)]['Avg_Remain_pros']}, axis=1)
        dataframes['df'+str(i)+str(j)]['Avg_Remain_pros'] = Aux_df_2[['Avg_Remain_pros_ac1', 'Avg_Remain_pros_ac0']].apply(max, axis=1)

with open("dataframes_simulator_prospective.pkl", "wb") as file:
    pickle.dump(dataframes, file)

total_time = datetime.datetime.now()-begin_time
print(f'The total time of execution is {total_time}')