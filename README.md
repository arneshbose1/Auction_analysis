# Auction_analysis


# Files Description : 

Run the files in the following order

1-Powerplay_Analysis.ipynb : Start running this file. This file analyses the importance of powerplay with respect to the winning and losing team.
2-Distribution_Fitting.ipynb : This file fits the distribution on the auction data from 2020 to 2022.
3-IPL_Auction_Analysis.ipynb : This file gives key insights about the prices offered in the auctions.
4-Categorical_regression.ipynb : This file is to fit model on just categorical variables from "data/auction_nostats.csv" as mentioned in Folder section.
5-Price_Regression.ipynb : This file is used to fit models on entire data including stats from "data/auction_stats.csv".
6-Interpreting_model_results.ipynb : This model loads the saved model and evaluates the results



Other Files:

1. Creating_dataset.ipynb : This file created the ipl auctions dataset. Not required to run as they have already been created and saved under "data/"
2. data_config.json : This file have the paths of all the dataset files in "data/" folder
3. Price_Regression_Smogn.ipynb : This file is failed attempt to try SMOTE on the datset to prevent overfitting
4. utils.py: contains the functions to extract the metrics from the database for all players. Used in "Creating_dataset.ipynb".





# Folder Description:

Folder 1: "data"

Downloaded separately from IPL website
1. auction_nostats.csv : auction data from 2020 to 2022 containg only categorical features
2. auction_raw_data.csv : raw auction data downloaded from IPL website
3. auction_stats.csv : auction data from 2020 to 2022 containg all stats and derived metrics
4. auction_stats_price_yearwse : yearwise stats and price of players in ipl auctions

From RCB database for the years 2019-2021 containing matches from IPL, international matches, etc.
1. match_ball_events.csv : ball-by-ball data of all matches from database of RCB for 3 years
2. match_innings.csv : summary of both the innings of all the matches
3. match_scores_batting.csv : batting scorecard of all matches
4. match_scores_bowling.csv : bowling scorecard of all matches
5. matches.csv : description of all the matches
6. players.csv: description of all the players
7. teams.csv: description of all the teams
8. tournaments.csv: description of all the tournaments


Folder 2: "trained_models" : Contains the best regression model saved.

Folder 3: "Role_Definition" - This folder is for the batting and bowling role determination. This requires pytorch to run.

1. data_config.json : file with the location of various data files
2. dataset.py : creates the batting and bowling role dataset for training
3. model.py : contains the model class
4. train.py : This file trains the model and saves it in the folder "trained_torch_models"
5. eval.py : loads the model from the folder "trained_torch_models" and runs evaluations
6. trained_torch_models : contains the best models for batting and bowling classifier
7. sweep.yaml : contains the set of hyper-parameters for tuning the model oon wandb.ai










