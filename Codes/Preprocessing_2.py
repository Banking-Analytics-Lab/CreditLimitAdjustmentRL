'''
In this script, the information about customers ACTIVE from June until November was concilidated,
some inconsistencies are fixed finally is generated the dataframes

Output:
- CCF_revision.pkl to calculate the average CCF factor of the credit card portfolio
- df_transition.pkl and df_train_secon_2.pkl used to generate the transition probability matrix and 
  used to train models for the RL algorithm's simulator. 
'''

import pandas as pd
import matplotlib as plt
import seaborn as sns
import numpy as np
import datetime

# Upload the df with the historical information 
payment_history =pd.read_csv("df_payment_history_280322", low_memory=False)
payment_history = payment_history.reset_index(drop=True)

# Since our RL algorithm will be trained with the best information until the moment the data was given,
# this is to select the information of customers ACTIVE from June until November

retro_prosp_data= pd.DataFrame(payment_history[(payment_history.E_6=='ACTIVE')&(payment_history.E_7=='ACTIVE')&(payment_history.E_8=='ACTIVE')&(payment_history.E_9=='ACTIVE')&(payment_history.E_10=='ACTIVE')&(payment_history.E_11=='ACTIVE')])
retro_prosp_data.shape

# Replace nan values in outstanding balance and payments by zero values
for j in ['1','2', '3', '4', '5','6','7','8','9', '10','11']:
    retro_prosp_data['OB_'+str(j)] = retro_prosp_data['OB_'+str(j)].replace(np.nan,0)
    retro_prosp_data['P_'+str(j)] = retro_prosp_data['P_'+str(j)].replace(np.nan,0)

# Construction of the data frame with information about the customers described previously
retros_prosp_data_with_global_allmonths = pd.DataFrame([])
# Here EO_R is the same as EO_8
for j in [1,2,3,4,5,6,7,8,9,10,11]:
    # TCi = Total consumption in the month i 2021
    retros_prosp_data_with_global_allmonths['TC'+str(j)] = retro_prosp_data['TC_'+str(j)]
    # OB_pday_i = Outstanding balance in the PAYMENT_DATE of the month ith, after the payment
    retros_prosp_data_with_global_allmonths['OB_pday_'+str(j)] = retro_prosp_data['OB_'+str(j)]-retro_prosp_data['P_'+str(j)]
    # EO_i = ESTADO_OPERATIVO in the ith month
    retros_prosp_data_with_global_allmonths['EO_'+str(j)] = retro_prosp_data['EO_'+str(j)]
    # OB_cday_i = Outstanding balance in the CUT_OFF_DATE of the month ith
    retros_prosp_data_with_global_allmonths['OB_cday_'+str(j)] = retro_prosp_data['OB_'+str(j)]
    # P_pday_i = Payment done with respect to OB_cday_i in the payment date (in month ith) 
    retros_prosp_data_with_global_allmonths['P_pday_'+str(j)] = retro_prosp_data['P_'+str(j)]
    # CIM_i = If the customer falled in no payment in the ith month
    retros_prosp_data_with_global_allmonths['CIM_'+str(j)] = retro_prosp_data['CIM_'+str(j)]
    # PMin_i = Minimum payment in the ith month
    retros_prosp_data_with_global_allmonths['PMin_'+str(j)]= retro_prosp_data['PMin_'+str(j)]
    # VAM_i amount of increase in the prospective window (if there was)
    retros_prosp_data_with_global_allmonths['VAM_'+str(j)]=retro_prosp_data['VAM_'+str(j)]
    
# L_R = Limit in the retrospective window, this is in the month 8
retros_prosp_data_with_global_allmonths['L_R'] = retro_prosp_data['C_8']
# FCC = Date in which the credit card contract was created
retros_prosp_data_with_global_allmonths['FCC'] = retro_prosp_data['FCC_11'] # Date in which the contract was created 
#  HA_P = Indicator variable, (1) if there was increase in the prospective window, 0 in the contrary.
retros_prosp_data_with_global_allmonths['HA_P'] =np.where(((retros_prosp_data_with_global_allmonths['VAM_9']>0)|(retros_prosp_data_with_global_allmonths['VAM_10']>0)|(retros_prosp_data_with_global_allmonths['VAM_11']>0)), 1, 0)
#  EI = Estimated income
retros_prosp_data_with_global_allmonths['EI'] = retro_prosp_data['EI_11'] 
# Annual interest in the Credit card
retros_prosp_data_with_global_allmonths['Int'] = retro_prosp_data['Int_11']
# Bureau score at the moment of the application for the credit card
retros_prosp_data_with_global_allmonths['BS'] = retro_prosp_data['BS_11']
# L_P = Limit during the prospective window
retros_prosp_data_with_global_allmonths['L_P'] = retro_prosp_data['C_11']
# Identification
retros_prosp_data_with_global_allmonths['USER_ID'] = retro_prosp_data['APPLICATION_USER_ID']

# Convert the date of creation of the contract as a date type 
retros_prosp_data_with_global_allmonths['FCC']= pd.to_datetime(retros_prosp_data_with_global_allmonths['FCC'])

# Observe extreme values for limits
data_woe_allmonths = retros_prosp_data_with_global_allmonths[~(retros_prosp_data_with_global_allmonths.L_R>100000)]
print(f'The proportion of customers with LR greater than is 100.000 is: {round(1-data_woe_allmonths.shape[0]/retros_prosp_data_with_global_allmonths.shape[0],3)} this is less than 1%')

####################################################################
# Fixing some inconsistencies given in the data
####################################################################

# Observations where the outstanding balance is positive and the payment is zero
data_woe_allmonths = data_woe_allmonths.reset_index(drop=True) 
n = data_woe_allmonths.shape[0]
z0 = np.zeros((n, 1))
for i in range(n):
    z0[i] = 0
    for j in range(11):
        if ((data_woe_allmonths['OB_cday_'+str(j+1)][i]>0)&(data_woe_allmonths['P_pday_'+str(j+1)][i]==0)):
            z0[i] += 1

print(f'There are {int(z0.sum())} observations where the outstanding balance is positive and the payment is zero, this is {round(z0.sum()/(11*n)*100,2)}% of the data set.')

# Calculate the proportion of inconsistencies, this is the observations where the outstanding balance is positive, the payment is zero and EO==0 and CIM=False

z = np.zeros((n, 1))
for i in range(n):
    z[i] = 0
    for j in range(11):
        if ((data_woe_allmonths['OB_cday_'+str(j+1)][i]>0)&(data_woe_allmonths['P_pday_'+str(j+1)][i]==0)&(data_woe_allmonths['EO_'+str(j+1)][i]==0)&(data_woe_allmonths['CIM_'+str(j+1)][i]==False)):
            z[i] += 1 

print(f'There are {int(z.sum())} inconsistent observations , this is {round(z.sum()/(11*n)*100,2)}% of the data set.')

# Ambiguous information 

z2 = np.zeros((n, 1))
for i in range(n):
    z2[i] = 0
    for j in range(11):
        if ((data_woe_allmonths['OB_cday_'+str(j+1)][i]>0)&(data_woe_allmonths['P_pday_'+str(j+1)][i]==0)&(((data_woe_allmonths['EO_'+str(j+1)][i]==0)&(data_woe_allmonths['CIM_'+str(j+1)][i]==True))|(data_woe_allmonths['EO_'+str(j+1)][i]>0)&(data_woe_allmonths['CIM_'+str(j+1)][i]==False))):
            z2[i] += 1

print(f'Percentage of ambiguous information (CIM and ESTADO_OPERATIVO): {round(z2.sum()/(11*n)*100,2)}%')

# Correct the information using the Minimum payment variable from January until November

retros_prosp_data_with_global_consistent = data_woe_allmonths.copy().reset_index(drop=True)


for i in ['1', '2', '3', '4', '5','6','7', '8', '9', '10', '11']:
# Until November because information of payments in december is not provided
# No pay is equal to zero if the outstanding balance is zero
    retros_prosp_data_with_global_consistent['No_Pay_'+i] = (retros_prosp_data_with_global_consistent['P_pday_'+i]<retros_prosp_data_with_global_consistent['PMin_'+i])*1*(retros_prosp_data_with_global_consistent['OB_cday_'+i]!=0)
    if i=='1':
        retros_prosp_data_with_global_consistent['EO_'+i] = retros_prosp_data_with_global_consistent['No_Pay_'+i]  
    else:
        retros_prosp_data_with_global_consistent['EO_'+i] = (retros_prosp_data_with_global_consistent['No_Pay_'+i]!=0)*(1+retros_prosp_data_with_global_consistent['EO_'+str(int(i)-1)])

# Also we have not to include customers with more than 6 delays since in those cases they were blocked
retros_prosp_data_with_global_consistent = retros_prosp_data_with_global_consistent[(retros_prosp_data_with_global_consistent.EO_7<=6)&(retros_prosp_data_with_global_consistent.EO_8<=6)&(retros_prosp_data_with_global_consistent.EO_9<=6)&(retros_prosp_data_with_global_consistent.EO_9<=6)&(retros_prosp_data_with_global_consistent.EO_10<=6)&(retros_prosp_data_with_global_consistent.EO_11<=6)].reset_index(drop=True)
# This is 0.5% of data set

# Generation of the dataframe CCF_revision.pkl
retros_prosp_data_with_global_consistent.to_pickle('CCF_revision.pkl')

## Verification if now the information is consistent

n = retros_prosp_data_with_global_consistent.shape[0]
z0n = np.zeros((n, 1))
for i in range(n):
    z0n[i] = 0
    for j in range(11):
        if ((retros_prosp_data_with_global_consistent['OB_cday_'+str(j+1)][i]>0)&(retros_prosp_data_with_global_consistent['P_pday_'+str(j+1)][i]==0)):
            z0n[i] += 1

zn = np.zeros((n, 1))
for i in range(n):
    zn[i] = 0
    for j in range(11):
        if ((retros_prosp_data_with_global_consistent['OB_cday_'+str(j+1)][i]>0)&(retros_prosp_data_with_global_consistent['P_pday_'+str(j+1)][i]==0)&(retros_prosp_data_with_global_consistent['EO_'+str(j+1)][i]==0)&(retros_prosp_data_with_global_consistent['CIM_'+str(j+1)][i]==False)):
            zn[i] += 1 

print(f'After fixing the problem, the percentage of total inconsistencies: {round(zn.sum()/(11*n)*100,2)}%')

# Ambiguous Information
z2n = np.zeros((n, 1))
for i in range(n):
    z2n[i] = 0
    for j in range(11):
        if ((retros_prosp_data_with_global_consistent['OB_cday_'+str(j+1)][i]>0)&(retros_prosp_data_with_global_consistent['P_pday_'+str(j+1)][i]==0)&(retros_prosp_data_with_global_consistent['EO_'+str(j+1)][i]==0)):
           z2n[i] += 1
    

print(f'After fixing the problem, the percentage of total contradictory information (CIM and State): {round(z2n.sum()/(11*n)*100,2)}%')

'''
Generating data frame with transitions with regards to payment or 
no payment for customers with credit limit less than or equal to 100.000 
''' 
# Select the No_Pay_i states 

list_t = []
for i in ['1', '2', '3', '4', '5','6','7', '8', '9', '10', '11']:
    list_t.extend(['No_Pay_'+i])
list_t.extend(['HA_P'])

Transitions_data = retros_prosp_data_with_global_consistent[retros_prosp_data_with_global_consistent.L_R<=100000][list_t]
Transitions_data.to_pickle('df_transition.pkl')

'''
Generate data frame with the information of the last six months of 
observation, here we are assuming the action was performed at the end
of the first three months, this is August 2021.
'''
df_RL_all = pd.DataFrame([])
'''
TC_i: is the total consumption for the i month of the first quarter for i=1,2,3 (June, July, August with the provided information-retrospective window)
TC_j: is the total consumption for the j-3 month of the second quarter for j = 4, 5, 6 (September, October and November-Prospective window)
The rest of variables are defined similarly as previously and using  the lines 38-68 of this script.
MP_R: Represents the number of no payment in the first three months of observation.
HA_s: Binary variable, 1 if the customer had an increase in the current limit in the retrospective (s=R) or prospective window (s=P)
N_Months_R: Number of months with RappiCard until August 2021

'''
for j in [1,2,3]:   
        df_RL_all['TC'+str(j)] = retros_prosp_data_with_global_consistent['TC'+str(j+5)]
        df_RL_all['OB_pday_'+str(j)] = retros_prosp_data_with_global_consistent['OB_pday_'+str(j+5)]
        df_RL_all['EO_'+str(j)] = retros_prosp_data_with_global_consistent['EO_'+str(j+5)]
        df_RL_all['OB_cday_'+str(j)] = retros_prosp_data_with_global_consistent['OB_cday_'+str(j+5)]
        df_RL_all['P_pday_'+str(j)] = retros_prosp_data_with_global_consistent['P_pday_'+str(j+5)]         

for j in [4,5,6]: 
        df_RL_all['EO_'+str(j)] = retros_prosp_data_with_global_consistent['EO_'+str(j+5)]
        df_RL_all['OB_pday_'+str(j)] = retros_prosp_data_with_global_consistent['OB_pday_'+str(j+5)]
        df_RL_all['OB_cday_'+str(j)] = retros_prosp_data_with_global_consistent['OB_cday_'+str(j+5)]
        df_RL_all['P_pday_'+str(j)] = retros_prosp_data_with_global_consistent['P_pday_'+str(j+5)]   
        
df_RL_all['L_R'] = retros_prosp_data_with_global_consistent['L_R']
df_RL_all['MP_R'] = retros_prosp_data_with_global_consistent['No_Pay_6']+retros_prosp_data_with_global_consistent['No_Pay_7']+retros_prosp_data_with_global_consistent['No_Pay_8'] 
df_RL_all['MP_P'] = retros_prosp_data_with_global_consistent['No_Pay_9']+retros_prosp_data_with_global_consistent['No_Pay_10']+retros_prosp_data_with_global_consistent['No_Pay_11']
df_RL_all['N_Months_R'] = (2021-retros_prosp_data_with_global_consistent['FCC'].dt.year)*12+8-retros_prosp_data_with_global_consistent['FCC'].dt.month
df_RL_all['HA_P'] =np.where(((retros_prosp_data_with_global_consistent['VAM_9']>0)|(retros_prosp_data_with_global_consistent['VAM_10']>0)|(retros_prosp_data_with_global_consistent['VAM_11']>0)), 1, 0)
# df_RL_all['HA_R'] = np.where((retros_prosp_data_with_global_consistent['VAM_6']>0)|(retros_prosp_data_with_global_consistent['VAM_7']>0)|(retros_prosp_data_with_global_consistent['VAM_8']>0), 1, 0)
df_RL_all['L_P'] = retros_prosp_data_with_global_consistent['L_P']
df_RL_all['EI'] = retros_prosp_data_with_global_consistent['EI']
df_RL_all['USER_ID'] = retros_prosp_data_with_global_consistent['USER_ID']
df_RL_all['Int'] = retros_prosp_data_with_global_consistent['Int']
df_RL_all['BS'] = retros_prosp_data_with_global_consistent['BS']

# Since the decision about the action only has to be done for the customers that 
# are up to date at the end of the retrospective window
df_up = df_RL_all[(df_RL_all.EO_3==0)]
df_up = df_up[['USER_ID','TC1', 'EO_1', 'TC2', 'EO_2', 'TC3', 'EO_3', 'L_R', 'MP_R',
       'N_Months_R', 'Int', 'EI', 'HA_P', 'L_P', 'EO_4', 'EO_5', 'OB_cday_1',
       'P_pday_1', 'OB_cday_2', 'P_pday_2', 'OB_cday_3', 'P_pday_3',
       'OB_cday_6', 'P_pday_6', 'BS']]
'''
Now this financial transactional data is mixed with the generic variables,
also in this stage some information was cleaned
'''
Global_features = pd.read_csv('20220215_DATA_RAPPI_UWO.csv')
Global_features_variables = Global_features[['USER_ID','AGE_IN_RAPPI', 'SEGMENT_RFM','MATURITY','IS_PRIME', 'CATEGORY', 'N_ORDERS', 'N_ORDERS_CC','N_RAPPIFAVOR', 'N_RAPPICASH',
       'N_ANTOJOS', 'PCT_RAPPIFAVOR', 'PCT_RAPPICASH', 'PCT_ANTOJOS',
       'N_MARCAS', 'N_ADDRESSES', 'AVG_TIP', 'RAPPICREDITS_LEVEL', 'FAV_VERTICAL','FAV_PAYMENT_METHOD','NATIONAL_CC', 'INTERNATIONAL_CC','MAX_CC_SCORE', 'WEALTH_INDEX']]

# Data frame with financial and generic variables
df_up_global = pd.merge(df_up, Global_features_variables, on='USER_ID')

# Replace nan values in WEALTH_INDEX by the median of this variable
df_up_global.WEALTH_INDEX = df_up_global.WEALTH_INDEX.replace(np.nan,df_up_global.WEALTH_INDEX.median())
# Replace nan values of MATURITY by the mode which is Amateur
df_up_global.MATURITY = df_up_global.MATURITY.replace(np.nan,'Amateur')
df_up_global.IS_PRIME = (df_up_global.IS_PRIME)*1
# Impose age of 18 for those with less ages
df_up_global.AGE_IN_RAPPI=np.where(df_up_global.AGE_IN_RAPPI<18,18,df_up_global.AGE_IN_RAPPI)
# Since only there is 0 value in variable MARCAS, this variable was eliminated
df_up_global = df_up_global.drop(['N_MARCAS'], axis=1)
# Since BS should not be less than 300
df_up_global.BS=np.where(df_up_global.BS<300,300,df_up_global.BS)

# Get dummies variables for categorical ones
columnsToBin = ['SEGMENT_RFM', 'MATURITY', 'CATEGORY', 'FAV_VERTICAL','FAV_PAYMENT_METHOD']

# Binarize
df_global_all = pd.get_dummies(df_up_global, columns = columnsToBin, drop_first=True)
# Drop USER_ID variable
df_global_all = df_global_all.drop('USER_ID', axis=1)

# Since Rappi has at least credit limits equal to 800, I eliminated extreme small values of outstanding balances
# that are smaller than 10
# Data without outstanding greater than zero less than 10 which is the minimum value they are willing to receive
df_global_all['z'] = df_global_all['L_P']/df_global_all['L_R']
condition =(((df_global_all.OB_cday_6>0)&(df_global_all.OB_cday_6<10))|((df_global_all.OB_cday_1>0)&(df_global_all.OB_cday_1<10))|((df_global_all.OB_cday_2>0)&(df_global_all.OB_cday_2<10))|((df_global_all.OB_cday_3>0)&(df_global_all.OB_cday_3<10))|(df_global_all.z>1.4))
print(f'Percentage of customers that satisfy this condition {round(df_global_all[condition].shape[0]/df_global_all.shape[0]*100, 2)}%')
df_all_c = df_global_all[~condition].reset_index()
df_all_c= df_all_c.drop(['index', 'z'], axis=1)
df_all_c.to_pickle('df_train_secon_2.pkl')

