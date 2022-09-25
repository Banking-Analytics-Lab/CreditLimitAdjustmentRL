# Scripts Description

* **Decision generationQ1.py** and **Decision generationQ2.py**: In these scripts the policy for the current portfolio is found, this is the customers that are recommended to have an increase and those that are not. This is done using the generated Q1 and Q2 tables, respectively. We have that these policies recommend the same.  To run this script is needed to have the script *Simulator.py* and the data frames: *df_discretized_RL_1.pkl* and *df_statesRL_1.pkl*, and the Q1 and Q2 tables generated after running the Double Q-learning algorithm: *Q1_values_DQ_sparse_permuted0.1_0.01_170721.pkl* and *Q2_values_DQ_sparse_permuted0.1_0.01_170721*.

* **Policy_states_interpretation.py:** contains the construction of histograms for those states in which to increase was recommended.
