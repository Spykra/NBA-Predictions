# -----------------------------------------------------------
# Data was collected at Kaggle.com from the users: Justinas Cirtautas, Nathan Lauga, Jacob Baruch
#
# This script will be used for pre-processing and cleaning our data
# -----------------------------------------------------------


import pandas as pd

### Data Loading ###

Games = pd.read_csv('Random Datasets/All_games_2004-2021.csv')
Games_Details = pd.read_csv('Random Datasets/All_game_details_since_2004-2021.csv')
Players = pd.read_csv('Random Datasets/All_players_since_2009-2019.csv')
Players_Details = pd.read_csv('Random Datasets/Players_stats_since_1983-2020.csv')
PLayers_of_the_week = pd.read_csv('Random Datasets/Players_of_the_week_since_1980-2020.csv')
Teams = pd.read_csv('Random Datasets/Teams_ID_since_2004-2021.csv')
Teams_Ranking = pd.read_csv('Random Datasets/Teams_ranking-records_since_2004-2021.csv')


### Data Cleaning ###

#Games
Games = Games.drop(columns = ["GAME_ID","GAME_STATUS_TEXT","TEAM_ID_home","TEAM_ID_away"])
Games = Games.dropna(subset=["PTS_home"], how='all')
Games_Details = Games_Details.drop(columns = ["COMMENT", "START_POSITION"])
Games_Details = Games_Details.dropna(subset=["PLUS_MINUS"], how='all')

#Players
Players_Details = Players_Details.drop(columns = ["college","country","draft_year","draft_round","draft_number"])
PLayers_of_the_week = PLayers_of_the_week.drop(columns = ["Conference", "Draft Year","Season short","Pre-draft Team","Real_value","Last Season"])

#Teams
Teams_Ranking = Teams_Ranking.drop(columns = ["LEAGUE_ID","CONFERENCE","RETURNTOPLAY"])
Teams_Ranking = Teams_Ranking.drop_duplicates(["TEAM_ID","HOME_RECORD","W","W_PCT"])    #remove the duplicates
