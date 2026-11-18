# Credit Limit Adjustment using Reinforcement Learning

Code for the paper:

> Sherly Alfonso-Sánchez, Jesús Solano, Alejandro Correa-Bahnsen, Kristina P. Sendova, Cristián Bravo (2024). Optimizing credit limit adjustments under adversarial goals using reinforcement learning. *European Journal of Operational Research*, 315(2), 802–817. <https://doi.org/10.1016/j.ejor.2023.12.025>

Preprint: <https://arxiv.org/abs/2306.15585>.

This repository contains the formulation and results of the research project conducted in 2021-2022 with Rappi Company information; this work was done as the first project by Ph.D. candidate Ms. Sherly Alfonso-Sanchez, under the supervision of Dr. Cristián Bravo Roman and Dr. Kristina Sendova, and with colaboration of Dr. Alejandro Correa Bahnsen and MSc. Jesus Solano. The work has been published at [EJOR](https://www.sciencedirect.com/science/article/pii/S037722172300975X). The code preprocesses the credit card data, trains the two-stage simulator of the outstanding balance (a classifier of the balance type, then a regressor per class), discretizes the states, trains tabular reinforcement learning agents (Double Q-learning, Q-learning and SARSA) against the simulator, and derives and plots the credit limit policy of the Double Q-learning agent.

## Abstract

In this research, we aim to find and automatize an optimal policy to credit limit adjustment, a fundamental problem for banking and fintech companies that offer credit card products, using Reinforcement learning techniques. This problem requires the optimization of the expected profit given the performed actions for the portfolio of credit card customers; with this purpose, we have to balance two different objectives, first, maximize the portfolio's renewal and minimize the portfolio's provisions.

To apply reinforcement learning techniques, we use an offline learning strategy that needs the simulation of the impact of the action based on historical data provided by a Supper-App company in Latin America. Our work gives insights into the impact of alternative data used to make decisions about credit adjustment limits in credit card portfolios, as well as a new objective technique to make these decisions that are primarily based on expert-driven systems instead of data-driven methods.

## Requirements

The Python version and the package versions used for the paper were not recorded. `requirements.txt` lists the packages that the code imports:

```bash
pip install -r requirements.txt
```

Four scripts import `plot_confusion_matrix` from `sklearn.metrics`, which scikit-learn deprecated in version 1.0 and removed in version 1.2. With scikit-learn 1.2 or later these imports fail:
`Codes/Models_Simulator/Financial_Features/Classiffier/Model_selection_Classifier.py`,
`Codes/Models_Simulator/Financial_Features/Classiffier/Best_Classifier.py`,
`Codes/Models_Simulator/Financial_Alternative_Features/Classifier/Model_selection_Classifier_AandF.py` and
`Codes/Models_Simulator/Evaluation_selected_Models/Interpretation/Interpretation_models.py`.

The reinforcement learning runs were submitted as a Slurm job array, `Codes/RL_algorithms/Double_Qlearning/DQ_ep10.sh`, which requests one CPU, 2 GB of memory and 2 h 20 min per array task. The time of a full run was not recorded.

`DQ_ep10.sh` reads two environment variables. Set your allocation in its `#SBATCH --account=<your-allocation>` line before you submit it.

| Variable | Default | Meaning |
|---|---|---|
| `VENV_DIR` | `venv` | Python virtual environment that the job activates |
| `CODE_DIR` | `.` | Folder that holds `DQ_learning_epsilon0.1.py` and `Simulator.py` |

## Data

The data come from Rappi, a Latin American super-app. They describe credit card (RappiCard) holders who were active from June to December 2021 in a Latin American country (paper, Section 5.1). The data are not included in this repository, and neither the repository nor the arXiv version of the paper states the terms of access. `data/README.md` lists the files that the code expects.

Every script reads its inputs from the current working directory and writes its outputs there. Run all scripts from one folder that holds the data, for example `data/`. The commands below do so.

## Training

```bash
cd data

# 1. Preprocessing
python ../Codes/Preprocessing/Preprocessing_1.py   # writes df_payment_history_280322.pkl
python ../Codes/Preprocessing/Preprocessing_2.py   # writes df_train_6months_allprospective_category.pkl
python ../Codes/Preprocessing/Preprocessing_3.py   # writes df_train_models_1.pkl and df_statesRL_1.pkl

# 2. Simulator models with financial features only (the models selected in the paper)
python ../Codes/Models_Simulator/Financial_Features/Classiffier/Best_Classifier.py   # writes Model_Balance_Type_RF_red_SMOTENC.sav
python ../Codes/Models_Simulator/Financial_Features/Regressor/XGB_Balance_class0.py  # writes SmallMediumBalances_xgb.json
python ../Codes/Models_Simulator/Financial_Features/Regressor/RF_Balance_class1.py   # writes Balance_modelCV_red.sav

# 3. Simulator data frames and state discretization
python ../Codes/Simulator_Discretization/DataFrames_simulator_creation.py   # writes dataframes_simulator_prospective.pkl
python ../Codes/Simulator_Discretization/Discretization_DFrame.py           # writes df_discretized_RL_1.pkl

# 4. Double Q-learning; the argument (0 to 5) selects alpha from 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1
python ../Codes/RL_algorithms/Double_Qlearning/DQ_learning_epsilon0.1.py 4

# Q-learning and SARSA use the Simulator.py of the Double Q-learning folder
PYTHONPATH=../Codes/RL_algorithms/Double_Qlearning python ../Codes/RL_algorithms/Q-Learning/Q_learning_epsilon0.1.py 4
PYTHONPATH=../Codes/RL_algorithms/Double_Qlearning python ../Codes/RL_algorithms/SARSA/SARSA_epsilon0.1.py 4
```

The paper selected epsilon = 0.1 and alpha = 10^-2 for Double Q-learning (argument 4). It reports results for Double Q-learning; the Q-learning and SARSA scripts are additional.

To run the six values of alpha as a Slurm job array on a cluster:

```bash
cd data
sbatch --export=ALL,VENV_DIR=../venv,CODE_DIR=../Codes/RL_algorithms/Double_Qlearning ../Codes/RL_algorithms/Double_Qlearning/DQ_ep10.sh
```

For the other values of epsilon in the paper's grid (0.05 and 0.15), change `epsilon` on line 37 of the training script. The script sets `episodes = 500` on line 38, the number that the paper's hyperparameter search used (Appendix A). `Codes/RL_algorithms/Double_Qlearning/Readme.md` mentions 3000 steps. The number of episodes of the final run is not documented.

Each training run writes its Q-tables as `Q1_values_DQ_sparse_permuted<epsilon>_<alpha><episodes>.pkl` and `Q2_values_DQ_sparse_permuted<epsilon>_<alpha><episodes>.pkl`, for example `Q1_values_DQ_sparse_permuted0.1_0.01500.pkl`. The policy scripts below read `Q1_values_DQ_sparse_permuted0.1_0.01_170721.pkl` and `Q2_values_DQ_sparse_permuted0.1_0.01_170721.pkl`. Rename or copy the tables before you run them.

## Evaluation

Run these from the same folder. Each script writes its figures and files to the current working directory.

```bash
cd data

# Classifier selection, with financial features only and with alternative and financial features
python ../Codes/Models_Simulator/Financial_Features/Classiffier/Model_selection_Classifier.py
python ../Codes/Models_Simulator/Financial_Alternative_Features/Classifier/Model_selection_Classifier_AandF.py

# Regressors with alternative and financial features
python ../Codes/Models_Simulator/Financial_Alternative_Features/Regressor/XGB_Balance_class0_AandF.py
python ../Codes/Models_Simulator/Financial_Alternative_Features/Regressor/RF_Balance_class1_AandF.py

# Two-stage simulator: test RMSE and WAPE, and variable importance (entropy and SHAP)
python ../Codes/Models_Simulator/Evaluation_selected_Models/Ensemble_Evaluation/Evaluation.py
python ../Codes/Models_Simulator/Evaluation_selected_Models/Interpretation/Interpretation_models.py

# Policy of the Double Q-learning agent and histograms of the states with an increase
python "../Codes/Generated_Policy/Decision generationQ1.py"   # writes decision_portfolio_1_Q1.pkl
python ../Codes/Generated_Policy/Policy_states_interpretation.py
```

`XGB_Balance_class0_AandF.py` writes `SmallMediumBalances_xgb.json`, the same file that `XGB_Balance_class0.py` writes and the simulator reads. Run it in a copy of the working folder, so that it does not replace the financial-features model.

`Decision generationQ2.py` reads the same Q1 table and writes the same output file as `Decision generationQ1.py`, although its description refers to the Q2 table.

## Pretrained models

Not released. The repository holds no trained simulator models, Q-tables or outputs, and it does not say why.

## Results

Table and figure numbers follow the arXiv preprint (version 2). Every output below is written to the current working directory.

| Table or figure in the paper | Command | Expected result |
|---|---|---|
| Table 1 (previous research), Table 2 (feature descriptions) | Not produced by code | |
| Table 3 (classifier selection) | `Model_selection_Classifier.py`, `Model_selection_Classifier_AandF.py` | Prints the test weighted F1-score and the best parameters of each model. The cross-validation scores and the 95% intervals: Not documented. |
| Table 4 (regressors) | `XGB_Balance_class0.py`, `XGB_Balance_class0_AandF.py` (class 0); `RF_Balance_class1.py`, `RF_Balance_class1_AandF.py` (class 1) | Prints the cross-validated RMSE and the test WAPE. The 95% intervals: Not documented. |
| Two-stage simulator, test RMSE and WAPE (Section 6.1) | `Evaluation.py` | Prints the test RMSE and WAPE. The 95% intervals and the mean-prediction baseline: Not documented. |
| Figure 1 | `Interpretation_models.py` | `VarIm_Entropy_BalanceType.png`, `VarIm_SHAP_BalanceType.png` |
| Figure 2 | `Interpretation_models.py` | `VarIm_Entropy_BalanceAmount_Class0.png`, `VarIm_SHAP_BalanceAmount_Class0.png`, `VarIm_Entropy_BalanceAmount_Class1.png`, `VarIm_SHAP_BalanceAmount_Class1.png` |
| Figure 3a and 3b | `DQ_learning_epsilon0.1.py 4` | Writes `mean_episodes_DQ_permuted<epsilon>_<alpha><episodes>.png` (reward per episode) and `Mean_reward_Q_smooth_permuted<epsilon>_<alpha><episodes>.png` (mean over 10 episodes). Which file gives which panel: Not documented. |
| Figure 3c and 3d (10 random seeds) | Not documented | The training scripts fix the seed at 170721. |
| Figure 4 (comparison of strategies) | Not documented | `DQ_learning_epsilon0.1.py` writes `Reward_strategies_smooth_DQ_permuted<epsilon>_<alpha><episodes>.png` with the Double Q-learning, Random, All increase and All maintain curves. The No arrears, Current policy, BS_P85 and BS_P95 strategies are not in the repository. |
| Share of customers with an increase (Section 6.2) | `Decision generationQ1.py` | Prints the number of increase recommendations and the number of customers. |
| Figures 5 and 6 | `Policy_states_interpretation.py` | `States_increase_DQ_first4`, `States_increase_DQ_second2` and `States_increase_DQ_last`, each as `.png` and `.eps` |
| Figure 7 (deployment) | Not produced by code | |
| Figure 8 (hyperparameter search) | `DQ_ep10.sh`, with `epsilon` changed on line 37 of `DQ_learning_epsilon0.1.py` | Each run writes its own reward plots. The combined figure: Not documented. |

## Citation

```bibtex
@article{alfonsosanchez2024optimizing,
  title   = {Optimizing credit limit adjustments under adversarial goals using reinforcement learning},
  author  = {Alfonso-S{\'a}nchez, Sherly and Solano, Jes{\'u}s and Correa-Bahnsen, Alejandro and Sendova, Kristina P. and Bravo, Cristi{\'a}n},
  journal = {European Journal of Operational Research},
  volume  = {315},
  number  = {2},
  pages   = {802--817},
  year    = {2024},
  doi     = {10.1016/j.ejor.2023.12.025}
}
```

`CITATION.cff` holds the same reference for GitHub's "Cite this repository" button.

## Licence

Apache License 2.0. See [LICENSE](LICENSE).

## Acknowledgements

The paper acknowledges the support of the Natural Sciences and Engineering Research Council of Canada (NSERC) [Discovery Grants RGPIN-2020-07114 and RGPIN-2019-06586] and the Canada Research Chair Program [CRC-2018-00082]. Compute Ontario and the Digital Research Alliance of Canada provided computing support. The data come from Rappi.
