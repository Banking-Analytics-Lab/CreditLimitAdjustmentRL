# Data

The data come from Rappi, a Latin American super-app. They describe credit card (RappiCard) holders
who were active from June to December 2021 in a Latin American country, with financial attributes
of the credit card and generic attributes from the use of the app (paper, Section 5.1). The data are
not released with this repository, and neither the repository nor the arXiv version of the paper
states the terms of access. No synthetic sample is provided.

Every script reads its inputs from the current working directory and writes its outputs there. Put
the two raw files in this folder and run the scripts from it, as the main `README.md` shows. The
`.gitignore` keeps everything in this folder except this file out of git.

## Raw files

| File | Read by | Content, as the code describes it |
|---|---|---|
| `20220211_DATA_FINACIERA_UWO.csv` | `Codes/Preprocessing/Preprocessing_1.py` | Financial (transactional) credit card data, several rows per customer |
| `20220215_DATA_RAPPI_UWO.csv` | `Codes/Preprocessing/Preprocessing_2.py` | Generic variables from the use of the app, joined on `USER_ID` |

## Intermediate files written by the code

| File | Written by | Read by |
|---|---|---|
| `df_payment_history_280322.pkl` (a CSV file despite its extension) | `Preprocessing_1.py` | `Preprocessing_2.py` |
| `df_train_6months_allprospective_category.pkl` | `Preprocessing_2.py` | `Preprocessing_3.py` |
| `df_train_models_1.pkl` | `Preprocessing_3.py` | the scripts in `Codes/Models_Simulator/`, `DataFrames_simulator_creation.py` |
| `df_statesRL_1.pkl` | `Preprocessing_3.py` | `Discretization_DFrame.py`, `DataFrames_simulator_creation.py`, the training scripts, `Decision generationQ1.py`, `Decision generationQ2.py` |
| `Model_Balance_Type_RF_red_SMOTENC.sav` | `Best_Classifier.py` | `DataFrames_simulator_creation.py`, `Evaluation.py`, `Interpretation_models.py` |
| `SmallMediumBalances_xgb.json` | `XGB_Balance_class0.py` (also `XGB_Balance_class0_AandF.py`) | `DataFrames_simulator_creation.py`, `Evaluation.py`, `Interpretation_models.py` |
| `Balance_modelCV_red.sav` | `RF_Balance_class1.py` | `DataFrames_simulator_creation.py`, `Evaluation.py`, `Interpretation_models.py` |
| `dataframes_simulator_prospective.pkl` | `DataFrames_simulator_creation.py` | the training scripts in `Codes/RL_algorithms/` |
| `df_discretized_RL_1.pkl` | `Discretization_DFrame.py` | the training scripts, `Decision generationQ1.py`, `Decision generationQ2.py` |
| `Q1_values_DQ_sparse_permuted0.1_0.01_170721.pkl` | `DQ_learning_epsilon0.1.py`, under another name (see the main `README.md`) | `Decision generationQ1.py`, `Decision generationQ2.py` |
| `Q2_values_DQ_sparse_permuted0.1_0.01_170721.pkl` | `DQ_learning_epsilon0.1.py`, under another name (see the main `README.md`) | `Policy_states_interpretation.py` |
