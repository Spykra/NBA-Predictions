# -----------------------------------------------------------
# This script will be used for regression analysis
#
# We will use 4 algorithms: Linear Regression, Lasso/Ridge Regression,
# Decision Tree - Random Forest, Support Vector Regression
#
# The main goal of the script is not to test the performance of ML algorithms but to see
# how easily we can modify/manipulate the given data for our purposes and try some tuning
# on the hyperparameters of our models. Our evaluation metric will be the root mean squared error.
# -----------------------------------------------------------

import numpy as np
from DataCleaner import *
from numpy import arange
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


###Linear Regression###

#Player details 1983-2020 - Predicting the rebounds given the height of the player
def linear_prediction_based_on_height():
    Height = Players_Details[["player_height"]]
    Rebounds = Players_Details["reb"]

    Height_train, Height_test, Rebounds_train, Rebounds_test = train_test_split(Height, Rebounds, test_size=0.2)
    model = LinearRegression().fit(Height_train, Rebounds_train)
    predictions = model.predict(Height_test)
    RMSE = mean_squared_error(Rebounds_test, predictions, squared=False)
    print("RMSE is:", RMSE)


#Player details 1983-2020 - Predicting the rebounds given the height and the points of the player
def linear_prediction_based_on_height_and_points():
    Height_points = Players_Details[["player_height", "pts"]]
    Rebounds = Players_Details["reb"]

    Height_points_train, Height_points_test, Rebounds_train, Rebounds_test = train_test_split(Height_points, Rebounds, test_size=0.2)
    model = LinearRegression().fit(Height_points_train, Rebounds_train)
    predictions = model.predict(Height_points_test)
    RMSE = mean_squared_error(Rebounds_test, predictions, squared=False)
    print("RMSE is:", RMSE)


#Player details 1983-2020 - Predicting the number of games of a player given multiple stats
def linear_prediction_based_on_multiple_stats():
    Stats = Players_Details[["reb", "pts", "ast","age"]]
    Games_played = Players_Details["gp"]

    Stats_train, Stats_test, Games_played_train, Games_played_test = train_test_split(Stats, Games_played, test_size=0.2)
    model = LinearRegression().fit(Stats_train, Games_played_train)
    predictions = model.predict(Stats_test)
    RMSE = mean_squared_error(Games_played_test, predictions, squared=False)
    print("RMSE is:", RMSE)


###Lasso/Ridge Regression###

#Let's try similar predictions as above but now with Lasso regression
#Player details 1983-2020 - Predicting the rebounds given the height and the points of the player
def lasso_prediction_based_on_height_and_points():
    Height_points = Players_Details[["player_height", "pts"]]
    Rebounds = Players_Details["reb"]

    Height_points_train, Height_points_test, Rebounds_train, Rebounds_test = train_test_split(Height_points, Rebounds, test_size=0.2)
    # model = linear_model.Lasso(alpha=0.1).fit(Height_points_train, Rebounds_train)
    model = linear_model.Ridge(alpha=0.1).fit(Height_points_train, Rebounds_train)
    predictions = model.predict(Height_points_test)
    RMSE = mean_squared_error(Rebounds_test, predictions, squared=False)
    print("RMSE is:", RMSE)


#Let's try some tuning with cross-validation
#Player details 1983-2020 - Predicting the rebounds given the height and the points of the player
def lasso_tuned_prediction_based_on_height_and_points():
    Height_points = Players_Details[["player_height", "pts"]]
    Rebounds = Players_Details["reb"]

    Height_points_train, Height_points_test, Rebounds_train, Rebounds_test = train_test_split(Height_points, Rebounds, test_size=0.2)
    cross_validate = RepeatedKFold(random_state=11)
    # model = LassoCV(alphas=arange(0, 1, 0.01), cv=cross_validate).fit(Height_points_train,Rebounds_train)
    model = RidgeCV(alphas=arange(0, 1, 0.01), cv=cross_validate).fit(Height_points_train, Rebounds_train)
    predictions = model.predict(Height_points_test)
    RMSE = mean_squared_error(Rebounds_test, predictions, squared=False)
    print("RMSE is:", RMSE)


###Decision Tree - Random Forest###

#Player details 1983-2020 - Predicting the net-rating given multiple stats
#Net rating measures the plus-minus statistics for a given player when the player is in the game
#relative to the plus-minus statistics when the player is not in the game.
def decistree_prediction_based_on_multiple_stats():
    Stats = Players_Details[["reb", "pts", "ast","age"]]
    Net_rating = Players_Details["net_rating"]

    Stats_train, Stats_test, Net_rating_train, Net_rating_test = train_test_split(Stats, Net_rating, test_size=0.2)
    model = DecisionTreeRegressor(max_depth = 6).fit(Stats_train, Net_rating_train)
    # model = RandomForestRegressor(max_depth=6).fit(Stats_train, Net_rating_train)
    predictions = model.predict(Stats_test)
    RMSE = mean_squared_error(Net_rating_test, predictions, squared=False)
    print("RMSE is:", RMSE)


#It's time to try something different. We want to estimate the relationship bewtween
#the height of the player and his 'usefulness'. We are going to measure the usefulness
#based on points, assists, rebounds and games played and name this metric "MVP".
#Player details 1983-2020 - Predicting the MVP based on height
def decistree_prediction_of_MVP():
    Stats = Players_Details[["reb", "pts", "ast", "gp"]]
    Height = Players_Details[["player_height"]]

    #This will be our handmade formula
    temp = (Stats["reb"] * 0.3) + (Stats["pts"] * 0.3) + (Stats["ast"] * 0.3) + (Stats["gp"] * 0.1)

    #Let's normalize our new dataframe to values of [0,1]
    MVP = (temp-min(temp))/(max(temp)-min(temp))
    MVP = MVP.to_frame()

    # checking which player had the best season according to our formula
    # MVP.columns = MVP.columns.map(str)
    # MVP = MVP.sort_values('0')    #(SPOILER - player:Russell Westbrook, season: 2016-2017, which indeed was true)


    Height_train, Height_test, MVP_train, MVP_test = train_test_split(Height, MVP, test_size=0.2)
    # model = DecisionTreeRegressor(max_depth = 6).fit(Height_train, MVP_train)
    model = RandomForestRegressor(max_depth = 6).fit(Height_train, MVP_train.values.ravel())
    predictions = model.predict(Height_test)
    RMSE = mean_squared_error(MVP_test, predictions, squared=False)
    print("RMSE is:", RMSE)


###Support Vector Regression###

#For our last regression algorithm, we are going to create a basic pipeline.
#Player details 1983-2020 - Predicting the MVP based on true shooting percentage (measures a
#player's efficiency at shooting the ball)

def svr_prediction_of_MVP_based_on_ts_pct():
    Stats = Players_Details[["reb", "pts", "ast", "gp"]]
    Shooting = Players_Details[["ts_pct"]]

    #This will be our handmade formula
    MVP = (Stats["reb"] * 0.3) + (Stats["pts"] * 0.3) + (Stats["ast"] * 0.3) + (Stats["gp"] * 0.1)
    MVP = MVP.to_frame()

    #Some basic Pipeline
    process = []
    process.append(('Scale', StandardScaler()))     #We could also normalize like in the previous example
    process.append(('SVR', SVR()))

    Shooting_train, Shooting_test, MVP_train, MVP_test = train_test_split(Shooting, MVP, test_size=0.2)
    model = Pipeline(process, verbose = 1).fit(Shooting_train, MVP_train.values.ravel())
    predictions = model.predict(Shooting_test)
    RMSE = mean_squared_error(MVP_test, predictions, squared=False)
    print("RMSE is:", RMSE)

