'''
In this script the given financial data 20220211_DATA_FINACIERA_UWO.csv is 
transformed in the way that all the information per customer is stored in one 
row instead of several ones.

Output: Data frame df_payment_history_280322 wich was used for the second 
        preprocessing stage.
'''
import pandas as pd
import matplotlib as plt
import seaborn as sns
import numpy as np
import datetime

Finan = pd.read_csv("20220211_DATA_FINACIERA_UWO.csv", sep=',', low_memory=False)

# Change the type of variables for the dates
dates = ['F_CREACION_CONTRATO', 'START_DATE', 'CUT_OFF_DATE', 'PAYMENT_DATE']
for d in dates:
    Finan[d]=pd.to_datetime(Finan[d])

# Replace by zero all the observation where the total consumption is less than zero. 
Finan['TOTAL_CONSUMO']=Finan['SD_CD_TOTAL_CONSUMO_IN'] + Finan['SD_CD_TOTAL_CONSUMO_OUT']
Finan['TOTAL_CONSUMO'] = (Finan['TOTAL_CONSUMO']>=0)*Finan['TOTAL_CONSUMO']

# Since there are customers with interest rate equal to zero, eliminate those observations
print(f"Number of customers with interest rate equal to zero {Finan[Finan['TAX_RATE']==0].shape[0]}")

# Drop this customers with interest rate equal to zero
index_to_drop = Finan[Finan['TAX_RATE']==0].index
Finan.drop(index_to_drop, inplace=True)

# Examine the number of different contracts
print(f'Number of different RPP_USER_ID:{len(Finan.RPP_USER_ID.unique())}')
print(f'Number of different CONTRACT_NUMBER: {len(Finan.CONTRACT_NUMBER.unique())}')
print(f'Number of different APPLICATION_USER_ID: {len(Finan.APPLICATION_USER_ID.unique())}')

# Number of customers with more than one contract
Cont = pd.DataFrame(Finan.groupby('APPLICATION_USER_ID')['CONTRACT_NUMBER'].nunique())
More_than1_Cont = Cont[Cont.CONTRACT_NUMBER>1]
print(f'Number of customers with more than one contract:{More_than1_Cont.shape[0]}')

App_more_1_contract = list(More_than1_Cont.index)
print(f"ESTADO of the contracts for customers with more than one contract: {Finan[Finan['APPLICATION_USER_ID']==App_more_1_contract[-3]]['ESTADO'].unique()}")
# The customers that had several contracts are because in the past they had a contract and it got cancelled, therefore we are only going to take into account those 
# that were active for that customer

for i in App_more_1_contract:
    index_to_drop = Finan[(Finan['APPLICATION_USER_ID']==i)&((Finan['ESTADO']=='CANCEL')|(Finan['ESTADO']=='ERROR'))].index
    Finan.drop(index_to_drop, inplace=True)

# Review again that the number of contracts coincides with the number of customers
print(f'Number of different RPP_USER_ID:{len(Finan.RPP_USER_ID.unique())}')
print(f'Number of different CONTRACT_NUMBER: {len(Finan.CONTRACT_NUMBER.unique())}')
print(f'Number of different APPLICATION_USER_ID: {len(Finan.APPLICATION_USER_ID.unique())}')

# Create PAYMENT with respect to the SALDO_TOTAL 
Finan['PAYMENT'] = (Finan['SALDO_TOTAL']>=Finan['PAGOS_CD_PD'])*Finan['PAGOS_CD_PD']+(Finan['SALDO_TOTAL']<Finan['PAGOS_CD_PD'])*Finan['SALDO_TOTAL']

# Generation payment historical data
# Eliminate variables that will not be taken into consideration
Finan_payment  = Finan.drop(['CONTRACT_NUMBER','RPP_USER_ID','DESCRIPCION_ESTADO',
       'PAGOS_SD_CD', 'PAGOS_CD_PD', 'SD_CD_TOTAL_CONSUMO_OUT',
       'SD_CD_AVG_CONSUMO_OUT', 'SD_CD_STD_CONSUMO_OUT',
       'SD_CD_TOTAL_CONSUMO_IN', 'SD_CD_AVG_CONSUMO_IN',
       'SD_CD_STD_CONSUMO_IN', 'SD_CD_TOTAL_TRX', 'SD_CD_AVG_CUOTAS',
       'SD_CD_MIN_CUOTAS', 'SD_CD_MAX_CUOTAS', 'CD_PD_TOTAL_CONSUMO_OUT',
       'CD_PD_AVG_CONSUMO_OUT', 'CD_PD_STD_CONSUMO_OUT',
       'CD_PD_TOTAL_CONSUMO_IN', 'CD_PD_AVG_CONSUMO_IN',
       'CD_PD_STD_CONSUMO_IN', 'CD_PD_TOTAL_TRX', 'CD_PD_AVG_CUOTAS',
       'CD_PD_MIN_CUOTAS', 'CD_PD_MAX_CUOTAS','CUT_INDEX'], axis=1)

# Create the list with the name of the data frames for the 12 months
names_dfs = []
id_months = []
for i in range(12):
    names_dfs.append('df_month_'+str(i+1))
    
# Create all the empty data frames 
dictionary_dataframes = {name: pd.DataFrame() for name in names_dfs}

# Create columns for each of the months with information about the interval (START_DATE and CUT_OFF_DATE)
# The data set contains information from January 2021 until December of 2021

for i in range(12):
    if i==11:
        id_month = Finan_payment.PAYMENT_DATE.dt.month==1
    else:
        id_month = Finan_payment.PAYMENT_DATE.dt.month==i+2
    names_dfs[i] = Finan_payment[id_month].drop_duplicates()
    names_dfs[i].columns=['APPLICATION_USER_ID', 'FCC_'+str(i+1), 'C_'+str(i+1), 'SD_'+str(i+1), 'CD_'+str(i+1), 'PD_'+str(i+1), 'E_'+str(i+1), 'EO_'+str(i+1), 'NRI_'+str(i+1),'PMin_'+str(i+1),'OB_'+str(i+1), 'CIM_'+str(i+1), 'BS_'+str(i+1), 'EI_'+str(i+1), 'Int_'+str(i+1),'HA_'+str(i+1), 'VAM_'+str(i+1), 'EP_'+str(i+1), 'TC_'+str(i+1), 'P_'+str(i+1)]

# Create the auxiliar dataframes
names_dfs_context = []

for i in range(6):
    names_dfs_context.append('df_payment_'+str(2*i+1)+'_'+str(2*(i+1)))

# Create all the empty data frames 
dictionary_dataframes_context = {name: pd.DataFrame() for name in names_dfs_context}

# Merge each two months
for i in range(6):
    names_dfs_context[i] = pd.merge(names_dfs[2*i], names_dfs[2*i+1], on='APPLICATION_USER_ID', how='outer').drop_duplicates()

# Merge all the information of the months
df_1234 = pd.merge(names_dfs_context[0], names_dfs_context[1], on='APPLICATION_USER_ID', how='outer').drop_duplicates()
df_5678 = pd.merge(names_dfs_context[2], names_dfs_context[3],on='APPLICATION_USER_ID', how='outer').drop_duplicates()
df_9101112 = pd.merge(names_dfs_context[4], names_dfs_context[5],on='APPLICATION_USER_ID', how='outer').drop_duplicates()
df_1_8=pd.merge(df_1234, df_5678,on='APPLICATION_USER_ID', how='outer').drop_duplicates()
df_payment_history_faster = pd.merge(df_1_8, df_9101112,on='APPLICATION_USER_ID', how='outer').drop_duplicates()
df_history_payment = df_payment_history_faster.drop_duplicates(subset=['APPLICATION_USER_ID'])
df_history_payment.to_csv('df_payment_history_280322', index=False)
