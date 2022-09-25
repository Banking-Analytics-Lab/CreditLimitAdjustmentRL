import pandas as pd
import numpy as np
import datetime
import random
import pickle
import xgboost
import matplotlib.pyplot as plt
import Simulator
from Simulator import simulator
import sys
import sparse
from sparse import DOK
'''
In this script Double Q learning algorithm is trained with different values of alpha: {10**(-6), 10**(-5), 10**(-4), 10**(-3), 10**(-2), 10**(-1)}.
and for epsilon equal to 0.1.
'''

plt.style.use('ggplot')

# The random seed is set
random.seed(170721)
np.random.seed(170721)

# Now it is only required the dictionary with the data frames

with open("dataframes_simulator_prospective.pkl", "rb") as file:
    df_simulation = pickle.load(file)


begin_time = datetime.datetime.now()

gamma = 1 # Because I am affecting the Q value like randomly with the following customer who could be very different from previous
alpha_options = [10**(-6), 10**(-5), 10**(-4), 10**(-3), 10**(-2), 10**(-1)]#[0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
index = int(sys.argv[1])
# print(f'This is the index: {index}')
alpha = alpha_options[index]
epsilon = 0.1
episodes = 500

print(f'This is a Q-learning Agent with the hyparameters: alpha {alpha}, epsilon {epsilon}, and CCF {0.48} and number of episodes is {episodes} with random seed: {170721}')
Q_values = np.zeros(shape=([2, 20, 20, 20, 3, 28, 3, 100]))

# Penultimo es cupos antes habia hasta (100000/800=125)
# Ultimo la suma total de todos los limites si se les aumentara a todos es Deltaprovision1.sum/4000
# The last index is related to the change of provisions

def choose_action(Q, state, epsilon):
        roll = random.uniform(0,1)
        if roll <= epsilon:
            action = random.choice(list(range(0,2)))
        else:
            Q_state = Q[:, state[0], state[1], state[2], state[3], state[4], state[5], state[6]]
            action = np.random.choice(np.where(Q_state==Q_state.max())[0])
            # action = np.random.choice(np.where(Q_state==Q_state.max())[0])
        return action

df= pd.read_pickle('df_statesRL_1.pkl')
df['MP_R'] = (df['MP_R_1']==1)*1+(df['MP_R_2']==1)*2
df['Int'] = (df['Int_0.32']==1)*0.32+(df['Int_0.55']==1)*0.55+(df['Int_0.65']==1)*0.65
df['Int_Month'] = df['Int']/12
print(f'This is the shape of the data frame {df.shape}')

data_RL =df[['L_R','N_Months_R','Int','OB_cday_1', 'P_pday_1','OB_cday_2', 'P_pday_2', 'OB_cday_3', 'P_pday_3','TC1','TC2', 'TC3', 'MP_R']]

# Load the data frame with the discretization made directly 
df_RL_discr = pd.read_pickle('df_discretized_RL_1.pkl')

a_3 = 3
print(f'This is the annuity :{a_3}')
mean_episodes_Q= []
mean_episodes_Random = []
mean_episodes_Allincrease = []
Profit_episodes = []
mean_episodes_allmaintain = []
list_options = [(0, 0), (0, 1),
                (1, 0), (1, 1)]
num_customers = data_RL.shape[0]

print(f'This is the total amount of provision if for all the customers the decision taken was to increase:')
print(df['Delta_Provision_1'].sum()) 
print(f'Therefore the 1% of this amount is equal to:')
print(int(np.round(df['Delta_Provision_1'].sum()*0.01,0)))

for episode in range(episodes):
    # Permutation of the data according to each episode
    df = df.sample(frac=1, random_state=episode).reset_index(drop=True)
    data_RL = data_RL.sample(frac=1, random_state=episode).reset_index(drop=True)
    df_RL_discr = df_RL_discr.sample(frac=1, random_state=episode).reset_index(drop=True)
    Int_state = df['Int']
    Int_Month_all = df['Int_Month']
    Delta_provision_1 = df['Delta_Provision_1']
    for (i, j) in list_options:
        df_simulation['df'+str(i)+str(j)] = df_simulation['df'+str(i)+str(j)].sample(frac=1, random_state=episode).reset_index(drop=True)
    episode_reward, episode_random, episode_allincrease, episode_allmaintain = 0, 0, 0, 0
    state= data_RL.iloc[0, :]
    PDef_0 = df['PD_0'][0]
    Int_Month = Int_Month_all[0]
    DProvision_1 = Delta_provision_1[0] 
    Delta_global_provision_Q = 0
    for i in range(num_customers):
        print(f'This is the episode: {episode} and customer {i}, time {datetime.datetime.now()-begin_time}')
        state_disc =  df_RL_discr.iloc[i, :]       
        index_state = [state_disc[0], state_disc[1], state_disc[2], state_disc[3], state_disc[4], state_disc[5], int(Delta_global_provision_Q/6924)] # The last one is the baseline
        total_reward = 0
        action = choose_action(Q_values, index_state, epsilon)
        action_random = random.choice(list(range(0,2)))
        balance_pred_ac0, balance_pred_ac1 = simulator.transition_values(i, state, df_simulation)
        reward_ac = simulator.reward(action, balance_pred_ac0, balance_pred_ac1, PDef_0, DProvision_1, a_3, Int_Month)
        reward_opac = simulator.reward(1-action, balance_pred_ac0, balance_pred_ac1, PDef_0, DProvision_1, a_3, Int_Month)
        if (action_random==action):
            reward_random = reward_ac
        else: 
            reward_random = reward_opac
        if (action==1):
            reward_increase = reward_ac
        else:
            reward_increase = reward_opac

        if (action==0):
            reward_maintain = 0
        roll_2 = random.uniform(0,1)
        if (i<num_customers-1):
            state = data_RL.iloc[i+1, :]
            Delta_global_provision_Q +=(action==1)*DProvision_1   
            PDef_0 = df['PD_0'][i+1]
            Int_Month = Int_Month_all[i+1]
            DProvision_1 = Delta_provision_1[i+1] 
            nstate_disc =  df_RL_discr.iloc[i+1, :]       
            index_newstate = [nstate_disc[0], nstate_disc[1], nstate_disc[2], nstate_disc[3], nstate_disc[4], nstate_disc[5], int(Delta_global_provision_Q/6924)]
            action_tilde = choose_action(Q_values, index_newstate, epsilon)
            Q_nstate = Q_values[action_tilde, index_newstate[0], index_newstate[1], index_newstate[2], index_newstate[3], index_newstate[4], index_newstate[5], index_newstate[6]]
            Q_state = Q_values[action, index_state[0], index_state[1], index_state[2], index_state[3], index_state[4], index_state[5], index_state[6]]
            Q_values[action, index_state[0], index_state[1], index_state[2], index_state[3], index_state[4], index_state[5], index_state[6]] += alpha*(reward_ac+gamma*Q_nstate.max()-Q_state)
        else:
            Q_state = Q_values[action, index_state[0], index_state[1], index_state[2], index_state[3], index_state[4], index_state[5], index_state[6]]
            Q_values[action, index_state[0], index_state[1], index_state[2], index_state[3], index_state[4], index_state[5], index_state[6]] += alpha*(reward_ac-Q_state)
        episode_reward +=reward_ac
        episode_random += reward_random
        episode_allincrease += reward_increase
        episode_allmaintain += reward_maintain
    mean_episodes_Q.append(episode_reward)
    mean_episodes_Random.append(episode_random)
    mean_episodes_Allincrease.append(episode_allincrease)
    mean_episodes_allmaintain.append(episode_allmaintain)


plt.plot(mean_episodes_Q) 
plt.title('SARSA Algorithm')
plt.ylabel('Reward of the algorithm')
plt.xlabel('episodes')
plt.savefig('mean_episodes_SARSA_permuted'+str(epsilon)+'_'+str(alpha)+str(episodes)+'.png', bbox_inches='tight')
plt.close()

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15, 5))
ax1.plot(mean_episodes_Q, label='SARSA Algorithm')
ax1.set_xlabel('episodes')
ax1.set_ylabel('Reward')
ax1.plot(mean_episodes_Random , color="purple", label='Random')
ax1.legend()
ax2.plot(mean_episodes_Q, label='Q-learning')
ax2.set_xlabel('episodes')
ax2.plot(mean_episodes_Allincrease, color="purple", label='All increase')
ax2.legend()
plt.savefig('rewards_strategies_SARSA_permuted'+str(epsilon)+'_'+str(alpha)+str(episodes)+'.png')
plt.close()

# Smoothen the reward of the algorithm 
mean_episodes_Q_smooth = []
mean_episodes_Random_smooth = []
mean_episodes_Allincrease_smooth = []

for i in range(int(episodes/10)):
    r = np.mean(np.array(mean_episodes_Q)[10*i: 10*(1+i)])
    r1 = np.mean(np.array(mean_episodes_Random)[10*i: 10*(1+i)])
    r2 = np.mean(np.array(mean_episodes_Allincrease)[10*i: 10*(1+i)])
    mean_episodes_Q_smooth.append(r)
    mean_episodes_Random_smooth.append(r1)
    mean_episodes_Allincrease_smooth.append(r2)

plt.plot(mean_episodes_Q_smooth)  
plt.title('SARSA')
plt.ylabel('Smoothen avg reward of 10 episodes')
plt.xlabel('episodes % 10')
plt.savefig('Mean_reward_SARSA_smooth_permuted'+str(epsilon)+'_'+str(alpha)+str(episodes)+'.png')
plt.close()

fig, ax1 = plt.subplots(1,1, figsize=(10, 10))
ax1.plot(mean_episodes_Q_smooth, label='SARSA', linewidth=4) 
ax1.set_title('Reward of the algorithm')
plt.xlabel('episodes % 10')
ax1.plot(mean_episodes_Random_smooth , color="purple", label='Random')
ax1.plot(mean_episodes_Allincrease_smooth , color="blue", label='All_increase')
ax1.plot(np.zeros(int(episodes/10),), color='red', label='All_maintain')
ax1.legend()
plt.savefig('Reward_strategies_smooth_SARSA_permuted'+str(epsilon)+'_'+str(alpha)+str(episodes)+'.png')

# Comparison of the policies without smotheen
fig, ax1 = plt.subplots(1,1, figsize=(10, 10))
ax1.plot(mean_episodes_Q, label='SARSA', linewidth=4) 
ax1.set_title('Reward of the algorithm')
plt.xlabel('episodes')
ax1.plot(mean_episodes_Random, color="purple", label='Random')
ax1.plot(mean_episodes_Allincrease , color="blue", label='All_increase')
ax1.plot(np.zeros(int(episodes),), color='red', label='All_maintain')
ax1.legend()
plt.savefig('Reward_strategies_SARSA_permuted'+str(epsilon)+'_'+str(alpha)+str(episodes)+'.png')

# Save the Q-Table and the images of the raw reward function

with open('rewards_SARSA_permuted'+str(epsilon)+'_'+str(alpha)+str(episodes)+'.pkl', 'wb') as f:
    pickle.dump(mean_episodes_Q, f)

with open('rewards_Random_SARSA_permuted'+str(epsilon)+'_'+str(alpha)+str(episodes)+'.pkl', 'wb') as f:
    pickle.dump(mean_episodes_Random, f)

with open('rewards_Allincrease_SARSA_permuted'+str(epsilon)+'_'+str(alpha)+str(episodes)+'.pkl', 'wb') as f:
    pickle.dump(mean_episodes_Allincrease, f)

Q_values = DOK.from_numpy(Q_values)

with open('Q_values_SARSA_sparse_permuted'+str(epsilon)+'_'+str(alpha)+str(episodes)+'.pkl', 'wb') as f:
    pickle.dump(Q_values, f)

total_time = datetime.datetime.now()-begin_time
print(f'The total time of execution is {total_time}')
