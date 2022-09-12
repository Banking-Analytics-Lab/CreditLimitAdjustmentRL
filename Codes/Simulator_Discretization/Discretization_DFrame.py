'''
In this script the states discretization is done, to train RL tabular methods we discretized:
1. index_0 : Discretization of the monthly average usage percentage in the last three months which was discretized in uniform intervals every 5%.
2. index_1 : Discretization of the monthly average percentage of payment in the last three months which was discretized in uniform intervals every 5%.
3. index_2 : Discretization of the monthly average percentage of consumption in the last three months which was discretized in uniform intervals every 5%.
4. index_3 : Number of no payments in the last three months, this is 0, 1 or 2.
5. index_4 : Discretization of the current credit limit in the last three months, because of the characteristics of the given limits, the partition was not defined
             uniformly, instead for the limits less than or equal than 20.000, interval whose lenght is 1.000 were used. On the other hand for credit limits greater than 
             20.000 and less than or equal to 100.000, intervals of lenght 10.000 were used.
6. index_5 : Discretization of the annual interest rate, here there are three options: 0.32, 0.55 amd 0.65 that were discretized as 0, 1 and 2, respectively.
** During the RL training also the provision change is discretized but this will not depend only on the current customer but also in the previous decisions.

Output: Data frame df_discretized_RL_1.pkl
'''

from unicodedata import decimal
import pandas as pd
import numpy as np
import pickle

df_discretized_RL = pd.read_pickle('df_statesRL_1.pkl')
print(f'This is the shape of the data frame for the portfolio {df_discretized_RL.shape}')

df_discretized_RL['Int'] = (df_discretized_RL['Int_0.32']==1)*0.32+(df_discretized_RL['Int_0.55']==1)*0.55+(df_discretized_RL['Int_0.65']==1)*0.65
df_discretized_RL['MP_R'] = (df_discretized_RL['MP_R_1']==1)*1+(df_discretized_RL['MP_R_2']==1)*2
# Use percentage
df_discretized_RL['Avg_month_Pct_use'] = (df_discretized_RL[['OB_cday_1', 'OB_cday_2', 'OB_cday_3']].apply(sum, axis=1))/(3*df_discretized_RL['L_R'])
df_discretized_RL['index_0'] = (df_discretized_RL['Avg_month_Pct_use']==1)*19+(df_discretized_RL['Avg_month_Pct_use']!=1)*(df_discretized_RL['Avg_month_Pct_use']/0.05).astype(int)

# Payment percentage
for i in range(3):
    df_discretized_RL['Pct_pay_'+str(i+1)] = df_discretized_RL['P_pday_'+str(i+1)]/df_discretized_RL['OB_cday_'+str(i+1)]
    df_discretized_RL['Pct_pay_'+str(i+1)] = df_discretized_RL['Pct_pay_'+str(i+1)].replace(np.nan, 1)

df_discretized_RL['Avg_month_Pct_pay'] = (df_discretized_RL[['Pct_pay_1', 'Pct_pay_2', 'Pct_pay_3']].apply(sum, axis=1))/3
df_discretized_RL['index_1'] = (df_discretized_RL['Avg_month_Pct_pay']==1)*19+(df_discretized_RL['Avg_month_Pct_pay']!=1)*(df_discretized_RL['Avg_month_Pct_pay']/0.05).astype(int)

# Consumption over the current limit
df_discretized_RL['Avg_month_Pct_consumption'] = (df_discretized_RL[['TC1', 'TC2', 'TC3']].apply(sum, axis=1))/(3*df_discretized_RL['L_R'])
# index consumption
df_discretized_RL['index_2'] = (df_discretized_RL['Avg_month_Pct_consumption']>=1)*19+(df_discretized_RL['Avg_month_Pct_consumption']<1)*(df_discretized_RL['Avg_month_Pct_consumption']/0.05).astype(int)

# Number of no payment in the retrospective window
df_discretized_RL['index_3'] = df_discretized_RL['MP_R']

# Index current limit
df_discretized_RL['index_4'] = (df_discretized_RL['L_R']<=20000)*(df_discretized_RL['L_R']/1000).astype(int)+((df_discretized_RL['L_R']>20000)&(df_discretized_RL['L_R']<100000))*(20+((df_discretized_RL['L_R']/10000-2).astype(int)))+27*(df_discretized_RL['L_R']==100000)

# Index interest rate
df_discretized_RL['index_5'] = (df_discretized_RL['Int']==0.55)*1+(df_discretized_RL['Int']==0.65)*2

df_discretized_RL = df_discretized_RL[['OB_cday_1', 'OB_cday_2', 'OB_cday_3', 'P_pday_1', 'P_pday_2', 'P_pday_3', 'L_R', 'TC1', 'TC2', 'TC3', 'Int','MP_R','index_0', 'index_1', 'index_2', 'index_3', 'index_4', 'index_5']]

# Convert the type of column
for i in range(5):
    df_discretized_RL['index_'+str(i)] = df_discretized_RL['index_'+str(i)].astype('int')

# for i in range(3):
#     df_discretized_RL['index_'+str(i)] = (df_discretized_RL['index_'+str(i)]>19)*19+((df_discretized_RL['index_'+str(i)]<=19))*df_discretized_RL['index_'+str(i)]

print(df_discretized_RL.head())
print(df_discretized_RL.index_0.describe())
print(df_discretized_RL.index_1.describe())
print(df_discretized_RL.index_2.describe())
print(df_discretized_RL.index_3.describe())
print(df_discretized_RL.index_4.describe())
print(df_discretized_RL.index_5.describe())

print(f'This is the number of observations in the discretization data frame {df_discretized_RL.shape}')

# Download the discretized states
df_discretized_RL = df_discretized_RL[['index_0', 'index_1', 'index_2', 'index_3', 'index_4', 'index_5']]

df_discretized_RL.to_pickle('df_discretized_RL_1.pkl')