'''
In this script the provision difference is included and also the Default probability 
at time to the decision is made, this is assuming the increase will be of 20% over the 
current limit.
* Here to calculate the default probability the Mexican normative was used and CCF=0.48

Output:
- df_train_models_1.pkl is the data frame to train the models for the simulator creation.
- df_statesRL_1.pkl is the data frame for training the RL algorithm, but with prospective 
  information, this is the way in which it has to be done in the deployment stage.
'''
import pandas as pd
import numpy as np
import pickle


df = pd.read_pickle('df_train_6months_allprospective_category.pkl')

# Data frame to train the models
# Finally average prospective outstanding balance at payment date is included
for i in range(6):
    df['Remain_'+str(i+1)] = df['OB_cday_'+str(i+1)]-df['P_pday_'+str(i+1)] 

df['Avg_Remain_pros'] = df[['Remain_4', 'Remain_5', 'Remain_6']].apply(np.mean, axis=1)
print(df[['Remain_1','Remain_2', 'Remain_3', 'Avg_Remain_pros']].describe())
print(f'This is the number of full payer or inactive customers: {df[df.Avg_Remain_pros==0].shape[0]}')
print(f'Proportion of full payers or inactives looking the average debt in the prospective window {df[df.Avg_Remain_pros==0].shape[0]/df.shape[0]}')

# Because of the big proportion of customers inactives or sooner payers, 
# I trained a two stage model for predict the avg balance at the prospective window 

print(f'The proportion of observations where the balance is greater than 0 and less than 1 is: {df[(df.Avg_Remain_pros>0)&(df.Avg_Remain_pros<1)].shape[0]/df.shape[0]}')
df = pd.DataFrame(df)

# Change the outstanding balance that are less than 1 to zero
df['Avg_Remain_pros']= (~((df.Avg_Remain_pros>0)&(df.Avg_Remain_pros<1)))*df.Avg_Remain_pros
print(f'The new proportion of observations where the balance is greater than 0 and less than 1 is: {round(df[(df.Avg_Remain_pros>0)&(df.Avg_Remain_pros<1)].shape[0]/df.shape[0]*100,2)}')

# The decision is going to be taken for those that are up to date
print(f'Unique value of the operative state at the end of retrospective period {df.EO_3.unique()}')

df = df.drop(['OB_cday_4', 'OB_cday_5', 'OB_cday_6', 'P_pday_4', 'P_pday_5', 'P_pday_6', 'Remain_1',
       'Remain_2', 'Remain_3', 'Remain_4', 'Remain_5', 'Remain_6','MP_R_6months','EO_1','EO_2','EO_3', 'EO_4', 'EO_5', 'EO_6', 'TC4', 'TC5', 'TC6'], axis=1) # I will not include the future info only the target balance remained
# Neither I include the operative state in 1 and 2 because that information is in MP_R    
df = df.reset_index(drop=True)
df.to_pickle('df_train_models_1.pkl')
print(f'These are the final features {df.columns} and the dimension is {df.shape}')


# Generation of the data frame to train the RL algorithms
df_RL = pd.read_pickle('df_train_6months_allprospective_category.pkl')
# The decision has to be made only for those custumers up to date
df_RL =  df_RL[df_RL.EO_6==0].reset_index(drop=True)
# Operative state at the end of the retrospective window
# df['ACT_0'] = 0 
Balance_0, Limit_0 = df_RL['OB_cday_6'], df_RL['L_P']
df_RL['pct_use_0'] = Balance_0/Limit_0
Payment_0 = df_RL['P_pday_6']
df_RL['pct_pay_0'] = Payment_0/Balance_0
df_RL['pct_pay_0'] = df_RL['pct_pay_0'].replace(np.nan, 1)
# Number of no payments in the last 6 months (when available)
df_RL['MP_P'] = df_RL['EO_4']+(df_RL['EO_5']-df_RL['EO_4']) 
HIST_0= df_RL['MP_P']+df_RL['EO_4']+df_RL['EO_5']
df_RL['N_Months_P'] = df_RL['N_Months_R']+3 
ANT_0 = df_RL['N_Months_P']
pct_use_0 = df_RL['pct_use_0']
pct_pay_0 = df_RL['pct_pay_0']
df_RL['expon_0'] = -2.9704 +0.4696*HIST_0-0.0075*ANT_0-1.0217*pct_pay_0+1.1513*pct_use_0
df_RL['PD_0'] = (1/(1+np.exp(-df_RL['expon_0'])))
df_RL = df_RL.drop(['pct_use_0', 'pct_pay_0', 'expon_0'], axis=1)

# Inclusion of provisions
CCF = 0.48
Remainder = Balance_0-Payment_0
df_RL['Provision_0'] = df_RL['PD_0']*(Remainder +CCF*(df_RL['L_R']-Remainder))
df_RL['Provision_1'] = df_RL['PD_0']*(Remainder +CCF*(1.2*df_RL['L_R']-Remainder))
df_RL['Delta_Provision_1'] = df_RL['PD_0']*CCF*0.2*df_RL['L_R']


# Save the dataframe for the RL algorithm
# Eliminate the information that is not needed
df_RL = df_RL.drop(['OB_cday_1','OB_cday_2','OB_cday_3','P_pday_1','P_pday_2','P_pday_3','Provision_0','Provision_1','MP_R_6months','EO_1','EO_2',
                    'EO_3','L_R', 'MP_R', 'TC1', 'TC2', 'TC3', 'N_Months_R'], axis=1)
# Rename the variables to train the RL algorithm
df_RL.rename(columns = {'OB_cday_4':'OB_cday_1', 'OB_cday_5':'OB_cday_2', 'OB_cday_6':'OB_cday_3',
                     'P_pday_4':'P_pday_1', 'P_pday_5':'P_pday_2', 'P_pday_6':'P_pday_3',
                     'L_P':'L_R', 'MP_P':'MP_R','TC4':'TC1','TC5':'TC2','TC6':'TC3','N_Months_P':'N_Months_R'}, inplace = True)
df_RL = df_RL[['TC1', 'TC2', 'TC3', 'L_R', 'N_Months_R', 'EI', 'OB_cday_1', 'P_pday_1','OB_cday_2',
               'P_pday_2', 'OB_cday_3', 'P_pday_3', 'BS', 'PD_0','Delta_Provision_1', 'MP_R', 'Int']]
# df_RL = df_RL.drop(['OB_cday_4', 'OB_cday_5', 'OB_cday_6', 'P_pday_4', 'P_pday_5', 'P_pday_6', 'Remain_1',
#        'Remain_2', 'Remain_3', 'Remain_4', 'Remain_5', 'Remain_6', 'Provision_0', 'Provision_1','MP_R_6months','EO_1','EO_2','EO_3', 'EO_4', 'EO_5', 'EO_6'], axis=1) # I will not include the future info only the target balance remained
# # Only financial features
# df_RL = df_RL.iloc[:, np.r_[0:8, 10:17, 39, 40]] # Do not include HA_P nor L_P
columnsToBin = ['MP_R', 'Int']
df_RL = pd.get_dummies(df_RL, columns = columnsToBin, drop_first=False)
print(f'This is the percentage of customers with credit limit greater than 100000 MXN: {round(df_RL[df_RL.L_R>100000].shape[0]/df_RL.shape[0]*100,3)} \% ')
df_RL = df_RL[df_RL.L_R<=100000]
df_RL = df_RL.reset_index(drop=True)
df_RL.to_pickle('df_statesRL_1.pkl')

