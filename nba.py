import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
import graphviz
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')
import pickle
import os

#Pulling dataset
PATH_TO_DATA = 'nba-enhanced-stats'
NAME_OF_DF = '2016-17_teamBoxScore.csv'
NAME_OF_DF2 = '2017-18_teamBoxScore.csv'
df = pd.read_csv(os.path.join(PATH_TO_DATA, NAME_OF_DF))
df2 = pd.read_csv(os.path.join(PATH_TO_DATA, NAME_OF_DF2))

#Setting x and y variables for training set
feature_cols = ['opptPTS', 'teamDrtg', 'teamPF', 'teamTO', 'teamORB', 'teamFGA']
x = df[feature_cols]
y = df['teamRslt']

#Could have used train_test_split to split training and test data but had multiple datasets so it was not necessary
#Initializing model and fitting it to data
clfgtb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(x, y)

#Setting x and y variable for testing set
new_feature_cols = ['opptPTS', 'teamDrtg', 'teamPF', 'teamTO', 'teamORB', 'teamFGA']
x_new = df2[new_feature_cols]
y_new = df2['teamRslt']

#Model had an accuracy of about 74%
print(clfgtb.score(x_new, y_new))

#Tests
games = ['PHX1 vs SAS2', 'DET1 vs PHI2', 'MIN1 vs BOS2', 'NY1 vs MIA2', 'TOR1 vs MIL2', 'CHI1 vs DAL2', 'UTA1 vs DEN2', 'WSH1 vs MEM2', 'ATL1 vs POR2', 'CHA1 vs LAL2']

g1 = [[101.3, 111.9, 22.3, 15.9, 11.3, 87.1]]
g2 = [[108.4, 106.7, 18.7, 14.4, 10.3, 86.0]]
g3 = [[103.4, 109.8, 18.0, 13.2, 10.3, 84.8]]
g4 = [[102.2, 108.0, 20.8, 15.3, 10.8, 86.0]]
g5 = [[105.9, 105.4, 22.1, 13.9, 9.4, 86.5]]
g6 = [[101.6, 108.7, 19.2, 13.8, 9.4, 88.6]]
g7 = [[107.9, 106.8, 20.1, 14.4, 8.4, 82.4]]
g8 = [[98.9, 106.4, 21.7, 13.7, 9.9, 86.1]]
g9 = [[102.5, 110.9, 19.7, 15.5, 9.5, 84.7]]
g10 = [[106.7, 107.4, 18.1, 13.1, 10.5, 86.4]]

pred1 = clfgtb.predict(g1)
prob1 = clfgtb.predict_proba(g1)
prob1 = np.array2string(prob1, precision=2, separator=' ')
#print("It will be a {} for PHX").format(" ".join(pred1).lower())
print("Probaility in percentage to win arranged with Team 2 first and Team 1 second: {}").format(prob1)