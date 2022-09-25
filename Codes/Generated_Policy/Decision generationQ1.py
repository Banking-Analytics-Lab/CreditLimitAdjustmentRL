''' In this script the decisions for each one of the customers in the portfolio is made,
    since the RL (Double Q-learning) was trained over different permutations of the data frame. 
    The last permutation is taken into consideration in this decision making process.
    Here the policy is greedy with respect to Q1 table. 
    Output: pickle file with the decisions for each one of the customers.
'''
import pandas as pd
import numpy as np
import datetime
import random
import pickle
from sympy import frac
import xgboost
import matplotlib.pyplot as plt
import sparse 

begin_time = datetime.datetime.now()

# Since the RL algorithm was already trained.
episodes = 1

# Upload the Q1 table generated after training
Q_gamma1 = pd.read_pickle('Q1_values_DQ_sparse_permuted0.1_0.01_170721.pkl')
# Convert the Q table to dense 
Q_gamma1 = Q_gamma1.todense()

# Q1-greedy policy
def choose_action(Q, state):
    Q_state = Q[:, state[0], state[1], state[2], state[3], state[4], state[5], state[6]]
    # Conservative behaviour
    if (len(np.where(Q_state==Q_state.max())[0])==2)|(len(np.where(Q_state==Q_state.max())[0])==0):
        action = 0
    else:
        action = np.where(Q_state==Q_state.max())[0][0]
    return action

# Upload the states information
df= pd.read_pickle('df_statesRL_1.pkl')
df['MP_R'] = (df['MP_R_1']==1)*1+(df['MP_R_2']==1)*2
df['Int'] = (df['Int_0.32']==1)*0.32+(df['Int_0.55']==1)*0.55+(df['Int_0.65']==1)*0.65
df['Int_Month'] = df['Int']/12
print(f'This is the shape of the data frame {df.shape}')

data_RL =df[['L_R','N_Months_R','Int','OB_cday_1', 'P_pday_1','OB_cday_2', 'P_pday_2', 'OB_cday_3', 'P_pday_3','TC1','TC2', 'TC3', 'MP_R']]
Int_state = df['Int']
Int_Month_all = df['Int_Month']
Delta_provision_1 = df['Delta_Provision_1']

# Upload the discretization of the first components of the states
df_RL_discr = pd.read_pickle('df_discretized_RL_1.pkl')


# To take the decision we set the data frames as the last permutation 
episode = 3000
df = df.sample(frac=1, random_state=episode).reset_index(drop=True)
data_RL = data_RL.sample(frac=1, random_state=episode).reset_index(drop=True)
df_RL_discr = df_RL_discr.sample(frac=1, random_state=episode).reset_index(drop=True)

a_3 = 3
num_customers = data_RL.shape[0]

# List where the decisions will be stored.
actions_portfolio = []
for episode in range(episodes):
    episode_reward, episode_profit, episode_random, episode_random,episode_allincrease, episode_allmaintain = 0, 0, 0, 0, 0, 0
    state= data_RL.iloc[0, :]
    PDef_0 = df['PD_0'][0]
    Int = Int_state[0]
    Int_Month = Int_Month_all[0]
    DProvision_1 = Delta_provision_1[0] 
    print(f'This is the episode: {episode}')
    Delta_global_provision_Q = 0
    for i in range(num_customers):
        state_disc =  df_RL_discr.iloc[i, :]       
        index_state = [state_disc[0], state_disc[1], state_disc[2], state_disc[3], state_disc[4], state_disc[5], int(Delta_global_provision_Q/6924)]
        print(f'This is the index of the state {index_state}')
        total_reward = 0
        action_gamma1 = choose_action(Q_gamma1, index_state)
        actions_portfolio.append(action_gamma1)
        print(f'This is the customer {i}')
        print(f'The action is {action_gamma1}')
        
        #  Update the Delta provision value 
        if (i<num_customers-1):
            Delta_global_provision_Q +=(action_gamma1==1)*DProvision_1
            DProvision_1 = Delta_provision_1[i+1]

# Save the decisions made in the portfolio. 
with open('decision_portfolio_1_Q1.pkl', 'wb') as f:
    pickle.dump(actions_portfolio, f)

print(f'This is the total number of increase recomendations {sum(actions_portfolio)}')
print(f'This is the total number of customers in the portfolio {len(actions_portfolio)}')

total_time = datetime.datetime.now()-begin_time
print(f'The total time of execution is {total_time}')
