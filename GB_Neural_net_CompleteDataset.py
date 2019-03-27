################################################################################################
##  Script Info: It Identifies FakeNews using Neural network formed by weights from Maxent Classifier
##               and extracted features from feature_engineering.py  
##  Author: Mohammed Habibllah Baig 
##  Date : 11/30/2017
################################################################################################

import pandas as pd
import numpy as np
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split ,StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score , recall_score , precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import re
from scipy.sparse import hstack

df_test= pd.read_csv("./tempdata/removed_articles0.csv")
df_train=pd.read_csv("./tempdata/scored_articles0.csv")

#X_body_text = df_all.body.values
#X_headline_text = df_all.headline.values
y_train = df_train.fakeness.values
y_test = df_test.fakeness.values

X_train=[]
"""To read the news article for training"""
for line in open('./tempdata/generated_feats_HBF_train.txt'):
    feat=[]
    feat=line.rstrip().split(',')
    X_train.append(feat)

X_test=[]

"""To read the news article for testing"""
for line in open('./tempdata/generated_feats_HBF_test.txt'):
    feat=[]
    feat=line.rstrip().split(',')
    X_test.append(feat)
    
clf = GradientBoostingClassifier(n_estimators=250, random_state=14128, verbose=True)
#clf = LogisticRegression(penalty='l1',n_jobs=3)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Gradient Boosting with Neural Net : \n")
print ( "F1 score {:.4}%".format( f1_score(y_test, y_pred, average='macro')*100 ) )
print ( "Accuracy score {:.4}%\n\n".format(accuracy_score(y_test, y_pred)*100) )