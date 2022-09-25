# Scripts Description

* **DQ_learning_epsilon0.1.py:** In this script the Double Q-learning algorithm is trained with epsilon=10%, and different options for $\alpha$. In addition the random seed is 170721 and the number of steps is 3000. To run this script is needed to have the script *Simulator.py* and the data frames: *dataframes_simulator_prospective.pkl*, *df_discretized_RL_1.pkl* and *df_statesRL_1.pkl*.

Moreover to obtain the rest of results for different epsilon value, the epsilon variable should be changed in line
37.

* **DQ_ep10.sh:** Is the job array that was used to run in parallel for different values of $\alpha$.

*Observation:* For the rest of RL algorithms the requirenments are analogous, therefore in those folders it is only stored the script with the determined RL algorithm.