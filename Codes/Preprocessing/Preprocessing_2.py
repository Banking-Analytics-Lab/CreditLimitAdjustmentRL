'''
In this script, the information about customers ACTIVE from June until November was concilidated,
some inconsistencies are fixed finally is generated the dataframes. Also the operative states over the 6 months (if available)
are stored.

Output:
- df_train_6months_allprospective_category.pkl which will be used to calculate PD_0 (Default probability at the end of the prospective period) for each customer, and generate
  the two different data frames to train the models for the simulator construction and for the RL model training.
- The CCF value of the portfolio, this is the mean of the individuals CCF factors. 
'''
import pandas as pd
import matplotlib as plt
import seaborn as sns
import numpy as np
import datetime

# Upload the df with the historical information 
payment_history =pd.read_csv("df_payment_history_280322.pkl", low_memory=False)
payment_history = payment_history.reset_index(drop=True)

# Since our RL algorithm will be trained with the best information until the moment the data was given,
# this is to select the information of customers ACTIVE from June until November

retro_prosp_data= pd.DataFrame(payment_history[(payment_history.E_6=='ACTIVE')&(payment_history.E_7=='ACTIVE')&(payment_history.E_8=='ACTIVE')&(payment_history.E_9=='ACTIVE')&(payment_history.E_10=='ACTIVE')&(payment_history.E_11=='ACTIVE')])
retro_prosp_data.shape

# Replace nan values in outstanding balance and payments by zero values
for j in ['1','2', '3', '4', '5','6','7','8','9', '10','11']:
    retro_prosp_data['OB_'+str(j)] = retro_prosp_data['OB_'+str(j)].replace(np.nan,0)
    # This is because for our formulation of provisions we need the balance less than or equal to the credit limit
    retro_prosp_data['OB_'+str(j)] = retro_prosp_data[['OB_'+str(j),'C_'+str(j)]].apply(min,axis=1)
    retro_prosp_data['P_'+str(j)] = retro_prosp_data['P_'+str(j)].replace(np.nan,0)
    # Consistency also in the payments
    retro_prosp_data['P_'+str(j)] = retro_prosp_data[['P_'+str(j),'OB_'+str(j)]].apply(min,axis=1)


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

for i in ['1', '2', '3', '4', '5','6','7', '8', '9', '10', '11']:
    print(retros_prosp_data_with_global_consistent['EO_'+i].unique())
### There are also inconsistencies in the original data base 
# # Also we do not have to include customers with more than 6 delays since in those cases they were blocked already we could not count them twice
retros_prosp_data_with_global_consistent = retros_prosp_data_with_global_consistent[(retros_prosp_data_with_global_consistent.EO_7<=6)&(retros_prosp_data_with_global_consistent.EO_8<=6)&(retros_prosp_data_with_global_consistent.EO_9<=6)&(retros_prosp_data_with_global_consistent.EO_9<=6)&(retros_prosp_data_with_global_consistent.EO_10<=6)&(retros_prosp_data_with_global_consistent.EO_11<=6)].reset_index(drop=True)


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

'''Calculation of the portfolio CCF factor'''

CCF_df = retros_prosp_data_with_global_consistent.copy()
# Filter for the customers that defaulted 
Defaulters  = CCF_df[(CCF_df['EO_7']==6)|(CCF_df['EO_8']==6)|(CCF_df['EO_9']==6)|(CCF_df['EO_10']==6)|(CCF_df['EO_11']==6)]
print(f'This is the number of Defaulters: {Defaulters.shape[0]}')
# Creation of the data frames for the calculation of the individual CCF

# Name the data frames
names_dfs = []
for i in range(7, 12):
    names_dfs.append('df_'+str(i))

# Create all the empty data frames 
dictionary_dataframes = {name: pd.DataFrame() for name in names_dfs}

# Fill the data frames 
for i in range(7, 12):
    names_dfs[i-7] = Defaulters[Defaulters['EO_'+str(i)]==6]

# Calculate the individuals CCF for defaulters
for i in range(7, 12):
    data = names_dfs[i-7].copy()
    data['one'], data['zero'] = 1, 0 
    # Since the increses were done from July 
    if ((i==11)|(i==10)):
        data = data[data['L_P']!=data['OB_pday_'+str(i-3)]]
        data['CCF'] = (data['OB_cday_'+str(i)]-data['OB_pday_'+str(i-3)])/(data['L_P']-data['OB_pday_'+str(i-3)])
        # data['CCF'] = (data[['OB_cday_'+str(i), 'L_P']].apply(min, axis=1)-data[['OB_pday_'+str(i-3), 'L_R']].apply(min, axis=1))/(data['L_P']-data[['OB_pday_'+str(i-3), 'L_R']].apply(min, axis=1))
        data['CCF'] = data[['CCF','zero']].apply(max, axis=1)
        # data['CCF'] = data[['CCF','one']].apply(min, axis=1)
    else:
        data = data[data['L_R']!=data['OB_pday_'+str(i-3)]]
        data['CCF'] = (data['OB_cday_'+str(i)]-data['OB_pday_'+str(i-3)])/(data['L_R']-data['OB_pday_'+str(i-3)])
        # data['CCF'] = (data[['OB_cday_'+str(i), 'L_R']].apply(min, axis=1)-data[['OB_pday_'+str(i-3), 'L_R']].apply(min, axis=1))/(data['L_R']-data[['OB_pday_'+str(i-3), 'L_R']].apply(min, axis=1))
        data['CCF'] = data[['CCF','zero']].apply(max, axis=1)
        # data['CCF'] = data[['CCF','one']].apply(min, axis=1)
    names_dfs[i-7] = data
# Concatenate all the CCF values
CCF_total1 = []
for i in range(7, 12):
    CCF_total1.extend(names_dfs[i-7]['CCF'])

CCF1_df = pd.DataFrame(CCF_total1, columns=['CCF'])
#  Notice only 65 defaulters are taken into account for the calculation of the CCF since for some of them the outstanding balance is equal to the credit limit

print(f'This is the summary statistics of the CCF factor for the portfolio \n :{CCF1_df.describe()} \n And the mean of the CCF is {np.round(CCF1_df.mean(),2)}')



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
        df_RL_all['TC'+str(j)] = retros_prosp_data_with_global_consistent['TC'+str(j+5)]    
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
# Generating previous operative states (notice that if there is no information is the same as assume No_Pay_i equal to zero)
df_RL_all['MP_R_6months'] = df_RL_all['MP_R']+retros_prosp_data_with_global_consistent['No_Pay_3']+retros_prosp_data_with_global_consistent['No_Pay_4']+retros_prosp_data_with_global_consistent['No_Pay_5']

# Since the decision about the action only has to be done for the customers that 
# are up to date at the end of the retrospective window
df_up = df_RL_all[(df_RL_all.EO_3==0)]
df_up = df_up[['USER_ID','TC1', 'EO_1', 'TC2', 'EO_2', 'TC3', 'EO_3', 'L_R', 'MP_R',
       'N_Months_R', 'Int', 'EI', 'HA_P', 'L_P','TC4', 'EO_4','TC5','EO_5', 'TC6', 'EO_6', 'OB_cday_1',
       'P_pday_1', 'OB_cday_2', 'P_pday_2', 'OB_cday_3', 'P_pday_3',
       'OB_cday_4', 'P_pday_4', 'OB_cday_5', 'P_pday_5',
       'OB_cday_6', 'P_pday_6', 'BS', 'MP_R_6months']]
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


df_global_categories = df_up_global.copy()
df_global_categories = df_global_categories.drop('USER_ID', axis=1)
df_global_categories = df_global_categories.reset_index(drop=True)
# The following data frame contains information of the customers about the last semester and
# (if aply) and without encoding the categorical variables.

df_global_categories.to_pickle('df_train_6months_allprospective_category.pkl')
