#################################################################
##  Script Info: It cleans the news and write it to the csv 
##  Author: Mohammed Habibllah Baig 
##  Date : 11/22/2017
#################################################################

import zipfile
import os
import json
from pprint import pprint
import pandas as pd
import nltk
import numpy as np
import re
from sklearn import feature_extraction

### Starting the DataFrame ################
DataFrame = pd.read_json("./tempdata/articles/2016-08-01.json" , encoding='utf-8')

count = 0

####################################################################
##### Loop over csv directors to add to the DataFrame ##############
####################################################################

for file in os.listdir("./tempdata/articles/"):
    count+=1
    if count>2:
        path = "./tempdata/articles/" + file
        DataFrame_ = pd.read_json(path , encoding='utf-8')
        DataFrame = pd.concat(objs= [DataFrame,DataFrame_], axis=0,ignore_index=True)        
        
# Getting bodytext from the dataframe
bodies = []
for i in range(DataFrame.shape[0]):
    bodies.append(DataFrame.fields[i]["bodyText"])

# Creating a column with bodytext
DataFrame["bodyText"] = bodies

# headlines from the dataframe
headlines = []
for i in range(DataFrame.shape[0]):
        headlines.append(DataFrame.fields[i]["headline"])
        
# Creating a column with headline
DataFrame["headline"] = headlines

# Getting rid of the embty bodies
DataFrame = DataFrame[DataFrame.bodyText != ""]

# Filtering articles based on topics
DataFrame = DataFrame[(DataFrame.sectionName == 'US news') | (DataFrame.sectionName == 'Business') | (DataFrame.sectionName == 'Politics') | (DataFrame.sectionName == 'World news')]

# Getting to see the dataframe info (data type, non-null value etc.)
print(DataFrame.info())

# Publication data range
print(DataFrame.webPublicationDate.min() ,DataFrame.webPublicationDate.max())

for idx,item in enumerate(DataFrame.bodyText):
    DataFrame.bodyText[idx] = re.sub('[^\x00-\x7F]+', "", item)
    DataFrame.bodyText[idx] = re.sub('(\\n)',"",DataFrame.bodyText[idx])
        
for idx,item in enumerate(DataFrame.headline):
    DataFrame.headline[idx] = re.sub('[^\x00-\x7F]+', "", item)
    DataFrame.headline[idx] = re.sub('(\\n)',"",DataFrame.headline[idx])
    
        
# Saving the cleaned data into csv file
DataFrame.to_csv("./tempdata/Clean_TheGuardian.csv")

Print("Successfully created the csv file..")
