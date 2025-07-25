import pandas as pd
import numpy as np
import pickle
import xgboost
import sklearn
import matplotlib.pyplot as plt
import random
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# August 4th 2022
# In this script I am going to create the simulator in which 
# 1. The revenue is given as a contingent anuity with constant probability equal to the
#  default probability at the end of the trimester
# 2. The simulations are generated by the probability of being of each class: 0, 1, or 2


random.seed(170721)
np.random.seed(170721)

class simulator:
    def transition_values(i, state,df_simulation):
        # In this option I am not going to generate the trajectories instead the expected revenue
        # No_Pay_4 = random.choices([0,1], [1-PDef_0, PDef_0])[0]
        # No_Pay_5 = random.choices([0,1], [1-PDef_0, PDef_0])[0]      
        # if (No_Pay_4==0):
        #     EO_5 = No_Pay_5
        # elif (No_Pay_4==1)&(No_Pay_5==1):
        #     EO_5 = 2
        # else:
        #     EO_5 = 0 
        
        # print(f'This is the No_Pay_4 {No_Pay_4} and the EO_5 is {EO_5}')

        # For the action maintain
        action  = 0
        prob_type_balance_0_ac0 = df_simulation['df00']['Prob_type_balance_0']
        prob_type_balance_1_ac0 = df_simulation['df00']['Prob_type_balance_1']
        prob_type_balance_2_ac0 = df_simulation['df00']['Prob_type_balance_2']
        # Select the balance type according with that balance type probabilities
        prob_type_balance_ac0 = [prob_type_balance_0_ac0[i], prob_type_balance_1_ac0[i], prob_type_balance_2_ac0[i]]
        # print(f'This is the action: {action}')
        type_balance_ac0 = random.choices([0, 1, 2], prob_type_balance_ac0)[0]
        # print(f'This is the balance type {type_balance_ac0} given action {action}')
        if (type_balance_ac0==2):
            balance_pred_ac0 = 0
        else:    
            balance_pred_ac0 =df_simulation['df0'+str(type_balance_ac0)]['Avg_Remain_pros'][i]
        cupo = df_simulation['df00']['L_P'][i]
        # print(f'The balance is {balance_pred_ac0} and cupo {cupo}')

        # Now for the action increase
        action1=1
        prob_type_balance_0_ac1 = df_simulation['df10']['Prob_type_balance_0']
        prob_type_balance_1_ac1 = df_simulation['df10']['Prob_type_balance_1']
        prob_type_balance_2_ac1 = df_simulation['df10']['Prob_type_balance_2']
        prob_type_balance_ac1 = [prob_type_balance_0_ac1[i], prob_type_balance_1_ac1[i], prob_type_balance_2_ac1[i]]
        type_balance_ac1 = random.choices([0, 1, 2], prob_type_balance_ac1)[0]

        #Assumptions about increase
        # If the limit is increased does not make sense the customer spends less than before if the action is maintain
        # This assumption could not be written as before because the classes are not ordered from the smallest to the bigger
        transf_type_balance_ac0 = (type_balance_ac0==2)*3+(type_balance_ac0==0)*4+(type_balance_ac0==1)*5
        transf_type_balance_ac1 = (type_balance_ac1==2)*3+(type_balance_ac1==0)*4+(type_balance_ac1==1)*5

        if (transf_type_balance_ac1<transf_type_balance_ac0):
            type_balance_ac1 = type_balance_ac0

        # print(f'This is the balance type {type_balance_ac1} given action {action1}')

        if (type_balance_ac1==2):
            balance_pred_ac1 = 0
        else:    
            balance_pred_ac1 =df_simulation['df0'+str(type_balance_ac1)]['Avg_Remain_pros'][i]
        # Remain_3 = df_simulation['df_00']['Remain_3']
        cupo1 = df_simulation['df10']['L_P'][i]
        # print(f'The balance is {balance_pred_ac1} and cupo {cupo1}')
        return balance_pred_ac0, balance_pred_ac1

    def reward(action, balance_pred_ac0, balance_pred_ac1, PDef, Delta_provision_1, a_3, Int_Month):
        if (action==0):
            reward = 0
        elif (balance_pred_ac0==balance_pred_ac1):
            reward = -Delta_provision_1
        else:    
            # First look assuming always the balance is the predicted average of the balances
            # After I could see what happen if first is r*Remain_3*v+ AvgR*
            reward = (1-PDef)*Int_Month*(balance_pred_ac1-balance_pred_ac0)*a_3-Delta_provision_1
        return reward    

            
