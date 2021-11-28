# -----------------------------------------------------------
# This script will be used for classification
#
# We will use 3 algorithms: Support Vector Machines, Decision Tree - Random Forest
#
# The main goal of the script is not to test the performance of ML algorithms but to
# see how easily we can modify/manipulate the given data for our purposes and try some
# tuning on the hyperparameters of our models. Our evaluation metric will be accuracy.
# -----------------------------------------------------------

import numpy as np
from DataCleaner import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


###Support Vector Machines - SVMs###

#Players of the week 1980-2020 - Predicting the position of the player based on height
def svm_prediction_based_on_height():
    Height = PLayers_of_the_week[["Height CM"]]
    Position = PLayers_of_the_week["Position"]

    Height_train, Height_test, Position_train, Position_test = train_test_split(Height, Position, test_size=0.2)
    model = svm.SVC(kernel='rbf').fit(Height_train, Position_train)
    accuracy = model.score(Height_test, Position_test)
    print("Accuracy is:", accuracy)


#Players of the week 1980-2020 - Predicting the position of the player based on height and weight
def svm_prediction_based_on_height_and_weight():
    Height = PLayers_of_the_week[["Height CM","Weight KG"]]
    Position = PLayers_of_the_week["Position"]

    Height_train, Height_test, Position_train, Position_test = train_test_split(Height, Position, test_size=0.2)
    model = svm.SVC(kernel='linear').fit(Height_train, Position_train)
    accuracy = model.score(Height_test, Position_test)
    print("Accuracy is:", accuracy)


#Players of the week 1980-2020 - Predicting the position of the player based on height and weight
#We will test some different parameters for our model
def svm_prediction_based_on_height_and_weight():
    Height_weight = PLayers_of_the_week[["Height CM","Weight KG"]]
    Position = PLayers_of_the_week["Position"]

    Height_weight_train, Height_weight_test, Position_train, Position_test = train_test_split(Height_weight, Position, test_size=0.2)
    model = svm.SVC(kernel='poly',C = 5).fit(Height_weight_train, Position_train)
    accuracy = model.score(Height_weight_test, Position_test)
    print("Accuracy is:", accuracy)


#Let's use a cross-validated gridsearch in order to finetune the parameters of our SVM model
#Players of the week 1980-2020 - Predicting the position of the player based on height and weight
def svm_prediction_based_on_height_and_weight_gridsearch():
    Height_weight = PLayers_of_the_week[["Height CM","Weight KG"]]
    Position = PLayers_of_the_week["Position"]

    Height_weight_train, Height_weight_test, Position_train, Position_test = train_test_split(Height_weight, Position, test_size=0.2)
    parameters = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf','sigmoid','linear']}
    model = GridSearchCV(SVC(), parameters, verbose=3).fit(Height_weight_train, Position_train)

    print("The best parameters are:", model.best_params_)

    accuracy = model.score(Height_weight_test, Position_test)
    print("Accuracy is:", accuracy)


###Decision Tree - Random Forest###

#Our final and most important classification
#Teams ranking-records 2004-2021 - Predicting the NBA ranking (regular season) of
#the year 2020-2021 based on win percentage
def decistree_NBA_ranking_prediction():
    # Preprocessing
    Teams_Ranking['STANDINGSDATE'] = pd.to_datetime(Teams_Ranking['STANDINGSDATE'])
    Date_split = Teams_Ranking.set_index(Teams_Ranking['STANDINGSDATE'])
    Date_split = Date_split.sort_index()

    # Split the data
    train_sample = Date_split['2003-10-01':'2019-09-29']
    test_sample = Date_split['2020-12-10':'2021-05-26']

    train_sample = train_sample.drop(train_sample[train_sample["G"] < 82].index)
    test_sample = test_sample.drop(test_sample[test_sample["G"] < 72].index)

    # Fix the NBA rankings for all the seasons
    train_sample = train_sample.sort_values(by=['SEASON_ID', 'W'], ascending=False)
    test_sample = test_sample.sort_values(by='W', ascending=False)

    # Add a column that show the NBA ladder
    i = 0
    j = 0
    k = 1
    temp_list = []
    for n in train_sample['SEASON_ID']:

        if n == j:
            k = k + 1
        else:
            k = 1

        temp_list.append(k)
        j = n

    #Prepare the dataframes
    ranking_train = pd.DataFrame(temp_list)
    ranking_train = ranking_train.set_index(train_sample['G'].index)
    train_sample['Ranking'] = ranking_train

    test_sample['Ranking'] = range(1, 31)
    ranking_test = test_sample['Ranking']

    Wins_train = train_sample[["W_PCT"]]
    Ladder_train = train_sample["Ranking"]
    Wins_test = test_sample[["W_PCT"]]
    Ladder_test = test_sample["Ranking"]

    #Because of the high number of classes, we are going to compare each class with the rest
    # model = OneVsRestClassifier(DecisionTreeClassifier(max_depth=5, min_samples_leaf=15)).fit(Wins_train, Ladder_train)
    model = OneVsRestClassifier(RandomForestClassifier(n_estimators = 20)).fit(Wins_train, Ladder_train)

    accuracy = model.score(Wins_test, Ladder_test)

    prediction = model.predict(Wins_test)
    test_sample['Prediction'] = prediction
    Result = test_sample[["TEAM", "Ranking", "Prediction"]]

    print(Result)
    print("The accuracy for the predictions of the 2021 NBA Championship based on win percentage is:", accuracy)



