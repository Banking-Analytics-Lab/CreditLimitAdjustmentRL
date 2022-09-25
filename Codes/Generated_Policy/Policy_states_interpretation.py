'''In this script it is recognized in which states the increase action is decided.
    Output: Three histograms according to the following groups:
    1. Monthly average utilization rate, payment rate, consumption rate, and current limit.
    2. Number of no payments in the last three months and Charged interest.
    3. Change in the provisions.'''

import pandas as pd
import numpy as np
import datetime
import pickle
import sparse 
import matplotlib.pyplot as plt

plt.style.use('ggplot')

Q_gamma1 = pd.read_pickle('Q2_values_DQ_sparse_permuted0.1_0.01_170721.pkl')
Q_gamma1 = Q_gamma1.todense()

# Indicator of the states in which the action was to increase
decision_increase = (Q_gamma1[1]>Q_gamma1[0])*1

# Indicator of the states in which the action was to maintain 
decision_maintain = (Q_gamma1[0]>Q_gamma1[1])*1

# States maintain 
states_maintain = np.where(~(decision_maintain == 0))

# States increase
states_increase = np.where(~(decision_increase == 0))

Q_data_frame_increase = pd.DataFrame()
Q_data_frame_increase['Avg_U_R'] = states_increase[0]
Q_data_frame_increase['Avg_P_R'] = states_increase[1]
Q_data_frame_increase['Avg_C_R'] = states_increase[2]
Q_data_frame_increase['MP_R'] = states_increase[3]
Q_data_frame_increase['L_R'] = states_increase[4]
Q_data_frame_increase['Int'] = states_increase[5]
Q_data_frame_increase['Delta_Provision'] = states_increase[6]


### Look the states where the action is maintain 
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
axes = axes.flat
columnas_object = ['Avg_U_R', 'Avg_P_R', 'Avg_C_R', 'L_R']

for i, colum in enumerate(columnas_object):
    Q_data_frame_increase[colum].value_counts().sort_index().plot.barh(ax = axes[i])
    axes[i].set_title(colum, fontsize = 14)#, fontweight = "bold")
    axes[i].tick_params(labelsize = 10)
    axes[i].set_xlabel("")
    # if ((i==0) |(i==2)):
    axes[i].set_ylabel("Bins of the discretization")

fig.tight_layout()
plt.subplots_adjust(top=0.9)
# fig.suptitle('States in which is decided to increase (Double Q-learning)',
            #  fontsize = 20, fontweight = "bold")
plt.savefig('States_increase_DQ_first4.png', bbox_inches='tight')
plt.savefig('States_increase_DQ_first4.eps', format='eps', bbox_inches='tight')

# Second graph 
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
axes = axes.flat
columnas_object = ['MP_R', 'Int']

for i, colum in enumerate(columnas_object):
    Q_data_frame_increase[colum].value_counts().sort_index().plot.barh(ax = axes[i])
    axes[i].set_title(colum, fontsize = 14)#, fontweight = "bold")
    axes[i].tick_params(labelsize = 10)
    axes[i].set_xlabel("")
    # if ((i==0) |(i==2)):
    axes[i].set_ylabel("Bins of the discretization")

fig.tight_layout()
plt.subplots_adjust(top=0.9)
# fig.suptitle('States in which is decided to increase (Double Q-learning)',
            #  fontsize = 20, fontweight = "bold")
plt.savefig('States_increase_DQ_second2.png', bbox_inches='tight')
plt.savefig('States_increase_DQ_second2.eps', format='eps', bbox_inches='tight')

# Final graph
fig, axes = plt.subplots(figsize=(15, 15))
Q_data_frame_increase['Delta_Provision'].value_counts().sort_index().plot.barh(ax = axes)
axes.set_title('Delta_Provision', fontsize = 14)
fig.tight_layout()
plt.subplots_adjust(top=0.9)
# fig.suptitle('States in which is decided to increase (Double Q-learning)',
            #  fontsize = 20, fontweight = "bold")
plt.savefig('States_increase_DQ_last.png', bbox_inches='tight')
plt.savefig('States_increase_DQ_last.eps', format='eps', bbox_inches='tight')
