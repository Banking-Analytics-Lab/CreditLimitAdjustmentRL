# Description of the folder *Models_Simulator*

This folder contains different experiments in which several models were trained to construct the environment
simulator.  For all this codes the data frame *df_train_models_1.pkl* is required.

* **Financial_Alternative_Features:** In this folder the experiments to predict the monthly average outstanding 
balance in the next three months using as predictors alternative and financial information are stored.

* **Financial_Features:** In this folder the experiments to predict the monthly average outstanding 
balance in the next three months using only as predictors financial information are stored.

In both scenarios, two stages model was trained.

* **Evaluation_selected_Models:** In this folder, the evaluation over the complete two stage model is done, 
taking into consideration the best models found in the previous stages, these models are saved as: *Model_Balance_Type_RF_red_SMOTENC.sav*, *SmallMediumBalances_xgb.json* and *Balance_modelCV_red.sav*, where the first one is a classifier model while the last two are regressors ones. Finally, in the *Interpretation* folder the variable importance graphs in terms of entropy and SHAP values are generated.