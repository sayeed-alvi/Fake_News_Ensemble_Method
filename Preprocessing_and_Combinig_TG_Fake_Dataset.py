######################################################################
##  Script Info: It preprocessses, cleans, lemmatizes the news articles  
##  Author: Mohammed Habibllah Baig 
##  Date : 11/22/2017
######################################################################

import pandas as pd
from collections import Counter
import re
import numpy as np
import nltk
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.ensemble import RandomForestClassifier
from sklearn import feature_extraction
from tqdm import tqdm

_wnl = nltk.WordNetLemmatizer()

STOP_LIST=['guardian','theguardian']

def lemmatization(w):
    return _wnl.lemmatize(w).lower()

#####Uses NLTK tokenization to tokenize input string####

def Tokenization(s):
    return [lemmatization(t) for t in nltk.word_tokenize(s)]

#Function to remove Non Alphanumeric Characters#######

def clean(s):
    return " ".join(re.findall(r'\w+', str(s), flags=re.UNICODE)).lower()


def Clean_stopwords(l):
    # Removes stopwords from a list of tokens
    return [ lemmatization(w) for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS and w not in STOP_LIST and len(lemmatization(w)) > 1]
    
#DataFrameTG1 = pd.read_csv("./tempdata/Clean_TheGuardian_Combined_No_Slash1.csv")
#DataFrameTG2 = pd.read_csv("./tempdata/Clean_TheGuardian_Combined_No_Slash1.csv")
#DataFrameTG = DataFrameTG1.append(DataFrameTG2, ignore_index=True)
DataFrameTG=pd.read_csv("./tempdata/Clean_TheGuardian.csv")

DataFrameTG["fakeness"] = 0
DataFrameTG["author"] = "The Guardian"    
DataFrameFake = pd.read_csv("./tempdata/fake.csv")

print("The columns of the guardians are :",DataFrameFake.columns)

DataFrameTG = DataFrameTG.rename(columns={'bodyText' : 'body','webPublicationDate':'pub_date'})

DataFrameFake = DataFrameFake.rename(columns={'text':'body','title':'headline','uuid':'id','published':'pub_date'})

# Dropping unnecesary columns
DataFrameFake.drop([ u'ord_in_thread', 
         u'language', u'crawled', u'site_url', u'country',
        u'thread_title', u'spam_score', u'replies_count', u'participants_count',
        u'likes', u'comments', u'shares', u'type', u'domain_rank',u'main_img_url'],inplace=True,axis=1)

DataFrameTG.drop([u'Unnamed: 0',  u'apiUrl', u'fields', 
        u'isHosted', u'sectionId', u'sectionName', u'type',
         u'webTitle', u'webUrl', u'pillarId', u'pillarName'],inplace=True,axis=1)
         
print("The columns of the guardians are :",DataFrameTG.columns,"Size",len(DataFrameTG))
DataFrameFake["fakeness"] = 1

# Concta the DataFrames of fakeNews and TheGuardian
DataFrameComplete = DataFrameFake.append(DataFrameTG, ignore_index=True)


for index, row in DataFrameComplete.iterrows():
    clean_headline=clean(row['headline'])
    temp_headline=" ".join(Clean_stopwords(clean_headline.split()))
    DataFrameComplete.set_value(index,'headline',temp_headline)
    #if(temp != row['headline']):
    #    print("different headline")
    #row['headline']=temp
    clean_body=clean(row['body'])
    temp_body=" ".join(Clean_stopwords(clean_body.split()))
    DataFrameComplete.set_value(index,'body',temp_body)
    #if(temp != row['body']):
    #    print("different body")
    #row['body']
DataFrameComplete = DataFrameComplete[DataFrameComplete.body != ""]
DataFrameComplete = DataFrameComplete[DataFrameComplete.headline != ""]

#Dropping the Nan values and info
DataFrameComplete.dropna(inplace=True)
print("The final csv shape is:", DataFrameComplete.shape)

DataFrameComplete.info()

DataFrameComplete.head()
    
DataFrameComplete.to_csv("./tempdata/Complete_DataSet.csv")
