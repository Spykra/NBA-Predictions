# -----------------------------------------------------------
# This script will be used for classification
#
# We will use 2 algorithms: Logistic Regression, K-Nearest Neighbours
#
# The main goal of the script is not to test the performance of ML algorithms but to
# see how easily we can modify/manipulate the given data for our purposes and try some
# tuning on the hyperparameters of our models. Our evaluation metric will be the accuracy.
# -----------------------------------------------------------

import numpy as np
from DataCleaner import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


###Logistic Regression###

#Games 2004-2021 - Predicting the result (victory or loss) given the points of the home team
def log_prediction_based_on_points():
    Points = Games[["PTS_home"]]
    Results = Games["HOME_TEAM_WINS"]

    Points_train, Points_test, Results_train, Results_test = train_test_split(Points, Results, test_size=0.2)
    model = LogisticRegression().fit(Points_train, Results_train)
    accuracy = model.score(Points_test, Results_test)
    print("Accuracy is:", accuracy)


#Games 2004-2021 - Predicting the result given the ID of both the home team and the visitor team
def log_prediction_based_on_both_IDs():
    ID = Games[["HOME_TEAM_ID", "VISITOR_TEAM_ID"]]
    Results = Games["HOME_TEAM_WINS"]

    ID_train, ID_test, Results_train, Results_test = train_test_split(ID, Results, test_size=0.2)
    model = LogisticRegression().fit(ID_train, Results_train)
    accuracy = model.score(ID_test, Results_test)
    print("Accuracy is:", accuracy)


#Let's try some predictions that would normally need regression algorithms
#Games_details 2004-2021 - Predicting the player's points given the ID of the player
def log_prediction_based_on_player_ID():
    Player_ID = Games_Details[["PLAYER_ID"]]
    Results = Games_Details["PTS"]

    # encoder1 = OneHotEncoder(categories="auto")
    # Results = encoder1.fit_transform(Results)
    encoder = LabelEncoder().fit(Results)
    Results = encoder.transform(Results)

    Player_ID_train, Player_ID_test, Results_train, Results_test = train_test_split(Player_ID, Results, test_size=0.2)
    model = LogisticRegression().fit(Player_ID_train, Results_train)
    accuracy = model.score(Player_ID_test,Results_test)
    print("Accuracy is:", accuracy)


#Instead of transforming our target variable to categorical, for our next two examples we will leave it as it is
#Games_details 2004-2021 - Predicting the player's points given the plus-minus of the player (plus-minus refer to the points of
#the team when the specific player is on the court, e.g. plus-minus=+10 means that when the player is on the court, the team is
#usually playing better)
def log_prediction_based_on_the_plus_minus():
    Plus_minus = Games_Details[["PLUS_MINUS"]]
    Results = Games_Details["PTS"]

    Plus_minus_train, Plus_minus_test, Results_train, Results_test = train_test_split(Plus_minus, Results, test_size=0.2)
    model = LogisticRegression().fit(Plus_minus_train, Results_train)
    accuracy = model.score(Plus_minus_test, Results_test)
    print("Accuracy is:", accuracy)


#Games_details 2004-2021 - Predicting the plus-minus of the player given some of his multiple stats
def log_prediction_based_on_stats():
    Stats = Games_Details[["PTS","REB","AST","STL","BLK"]]
    Results = Games_Details["PLUS_MINUS"]

    Stats_train, Stats_test, Results_train, Results_test = train_test_split(Stats, Results, test_size=0.2)
    model = LogisticRegression().fit(Stats_train, Results_train)
    accuracy = model.score(Stats_test, Results_test)
    print("Accuracy is:", accuracy)


###K-Nearest Neighbours###

#Let's repeat one of the above predictions, but now using a knn classifier
#Games 2004-2021 - Predicting the result (victory or loss) given the points of the home team
def knn_prediction_based_on_points():
    Points = Games[["PTS_home"]]
    Results = Games["HOME_TEAM_WINS"]

    Points_train, Points_test, Results_train, Results_test = train_test_split(Points, Results, test_size=0.2)
    model = KNeighborsClassifier(n_neighbors=1).fit(Points_train, Results_train)
    accuracy = model.score(Points_test, Results_test)
    print("Accuracy is:", accuracy)


#Now, let's try the same prediction with a higher number of neighbors (we can keep experimenting on that
#until we are able to find the optimal number)
def knn_prediction_based_on_points_with_more_neighbors():
    Points = Games[["PTS_home"]]
    Results = Games["HOME_TEAM_WINS"]

    Points_train, Points_test, Results_train, Results_test = train_test_split(Points, Results, test_size=0.2)
    model = KNeighborsClassifier(n_neighbors=1000).fit(Points_train, Results_train)
    accuracy = model.score(Points_test, Results_test)
    print("Accuracy is:", accuracy)

    #We can also take a look at the confusion matrix
    cm = confusion_matrix(Points_test, Results_test)
    print(cm)


#Games 2004-2021 - Predicting the result given the rebounds of the home team
def knn_prediction_based_on_rebounds():
    Rebounds = Games[["REB_home"]]
    Results = Games["HOME_TEAM_WINS"]

    Rebounds_train, Rebounds_test, Results_train, Results_test = train_test_split(Rebounds, Results, test_size=0.2)
    model = KNeighborsClassifier(n_neighbors=100).fit(Rebounds_train, Results_train)
    accuracy = model.score(Rebounds_test, Results_test)
    print("Accuracy is:", accuracy)


#Games 2004-2021 - Predicting the result given the rebounds of the home team
#Unlike the previous experiment, we are not satisfied with the random values of k
#We will try to find the (approximately) optimal k for our case by calculating the mean errors for k values
#in a limited range of values - 1 to 501 with a step of 20
def knn_prediction_based_on_rebounds_with_the_best_k():
    Rebounds = Games[["REB_home"]]
    Results = Games["HOME_TEAM_WINS"]

    mean_errors = []
    Rebounds_train, Rebounds_test, Results_train, Results_test = train_test_split(Rebounds, Results, test_size=0.2)

    #We can also do it with gridsearch instead of a heuristic way (check Classification_2)
    for i in range(1, 501, 20):
        model = KNeighborsClassifier(n_neighbors=i).fit(Rebounds_train, Results_train)
        prediction = model.predict(Rebounds_test)
        mean_errors.append(np.mean(prediction != Results_test))
        # minimum_mean_error = np.amin(mean_errors)
        best_k = (np.argmin(mean_errors) * 20) + 21
    # print(mean_errors)
    # print(minimum_mean_error)
    # print(best_k)

    #Now predict with the best k
    model = KNeighborsClassifier(n_neighbors=best_k).fit(Rebounds_train, Results_train)
    accuracy = model.score(Rebounds_test, Results_test)
    print("Accuracy is:", accuracy)

