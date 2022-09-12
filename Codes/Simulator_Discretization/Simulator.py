''' In this script the class simulator is created which contains two functions:
    1. transition_values which predicts the future outstanding balance given the two action maintain (0)
    or increase (1). 
    Assumptions about the simulator:
   * If the action is increasing the average balance is at least the balance given the action maintain
    2. reward. According with the formulation.'''


import pandas as pd
import numpy as np
import pickle
import xgboost
import sklearn
import matplotlib.pyplot as plt
import random
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor


random.seed(170721)
np.random.seed(170721)

class simulator:
    def transition_values(i, state,df_simulation):
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

        # Now for the action increase
        action1=1
        prob_type_balance_0_ac1 = df_simulation['df10']['Prob_type_balance_0']
        prob_type_balance_1_ac1 = df_simulation['df10']['Prob_type_balance_1']
        prob_type_balance_2_ac1 = df_simulation['df10']['Prob_type_balance_2']
        prob_type_balance_ac1 = [prob_type_balance_0_ac1[i], prob_type_balance_1_ac1[i], prob_type_balance_2_ac1[i]]
        type_balance_ac1 = random.choices([0, 1, 2], prob_type_balance_ac1)[0]

        #Assumptions about increase
        # If the limit is increased does not make sense the customer spends less than before if the action is maintain
        # transformation of the balance class to impose the assumption.
        transf_type_balance_ac0 = (type_balance_ac0==2)*3+(type_balance_ac0==0)*4+(type_balance_ac0==1)*5
        transf_type_balance_ac1 = (type_balance_ac1==2)*3+(type_balance_ac1==0)*4+(type_balance_ac1==1)*5

        if (transf_type_balance_ac1<transf_type_balance_ac0):
            type_balance_ac1 = type_balance_ac0

        if (type_balance_ac1==2):
            balance_pred_ac1 = 0
        else:    
            balance_pred_ac1 =df_simulation['df0'+str(type_balance_ac1)]['Avg_Remain_pros'][i]
        
        cupo1 = df_simulation['df10']['L_P'][i]
        return balance_pred_ac0, balance_pred_ac1

    def reward(action, balance_pred_ac0, balance_pred_ac1, PDef, Delta_provision_1, a_3, Int_Month):
        if (action==0):
            reward = 0
        elif (balance_pred_ac0==balance_pred_ac1):
            # To do the calculation more efficient.
            reward = -Delta_provision_1
        else:    
            # First look assuming always the balance is the predicted average of the balances
            # After I could see what happen if first is r*Remain_3*v+ AvgR*
            reward = (1-PDef)*Int_Month*(balance_pred_ac1-balance_pred_ac0)*a_3-Delta_provision_1
        return reward    

            
