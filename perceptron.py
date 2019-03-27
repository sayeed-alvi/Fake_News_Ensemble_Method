# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 21:18:03 2017

@author: Carson Hanel

Perceptron from scratch!
"""
import sys
import getopt
import os
import math
import pandas as pd
import csv
import numpy as np

class Perceptron:
  class TrainSplit:
    """Represents a set of training/testing data. self.train is a list of Examples, as is self.test. 
    """
    def __init__(self):
      self.train = []
      self.test  = []

  class Example:
    """Represents a document with a label. klass is 'pos' or 'neg' by convention.
       words is a list of strings.
    """
    def __init__(self):
      self.klass = ''
      self.words = []


  def __init__(self):
    """Perceptron initialization"""
    self.numFolds     = 10  #Number of times the testing data is folded.
    self.words        = {}  #Dictionary for words in the bag of words.
    self.vocab_length = 0   #Total length of the calculated bag of words
    self.count_docs   = 0   #Count of total documents; just an iterator
    self.weights      = []  #Calculated feature weights
    
  def classify(self, words):
    """ TODO
      'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    """
    #words = [word for word in words if word not in stopwords.words('english')]
    #No longer necessary, feature vector

    weight = 0.
    for w in set(words):
        w = w.lower()
        if w in self.words:
            weight = weight + float(self.weights[int(self.words[w])])
    #print 'The calculated weight is: ', weight
    if weight > 0:
        return 0
    else:
        return 1 

    return 0
  

  def addExample(self, klass, words, doc, iterations):
    """
     * TODO
     * Train your model on an example document with label klass ('pos' or 'neg') and
     * words, a list of strings.
     * You should store whatever data structures you use for your classifier 
     * in the Perceptron class.
     * Returns nothing
     *
     Note:
         y_i is positive or negative depending on the document class. 1 for pos, -1 for neg.
         x_i is the individual feature
    """
    klass_int = 1
    if(klass == 1):
        klass_int = -1
    occurrence = np.zeros(self.vocab_length)
    
    for i in range(0, iterations):
        sum_weights = 0.
        #Set(words) gets the average of the individual words, not of all words in doc.
        for w in set(words):
            sum_weights += self.weights[int(self.words[w])]
            occurrence[int(self.words[w])] += 1
        #Changing set(words) to words for feature engineering. More features = more weight
        for w in set(words):
            occur       = occurrence[int(self.words[w])]
            if np.sign(sum_weights * occur) != klass_int:
                self.weights[int(self.words[w])] += (klass_int - np.sign(sum_weights * occur)) * occur

    pass
  
  def train(self, split, iterations):
      """
      * TODO 
      * iterates through data examples
      * TODO 
      * use weight averages instead of final iteration weights
      *
      * Personal notes:
      *  The initial for-loop iterates through the examples given as training data.
      *  From what it seems right now, train is a complete function.
      """
      np.random.shuffle(split.train)

      curr_word = 0
      for example in split.train:
          self.count_docs += 1
          for w in example.words:
              #w = w.lower()  Not necessary after feature engineering
              if w not in self.words:
                  self.words[w] = curr_word
                  curr_word += 1
      self.vocab_length = len(self.words.keys())
      print 'The vocab length is: %d' % self.vocab_length + '\n'
      self.weights      = np.zeros(self.vocab_length)
      
      ex_doc = 0

      for example in split.train:
          words = example.words
          self.addExample(example.klass, words, ex_doc, iterations)
          ex_doc += 1
      pass
  
  def trainSplit(self, trainDir):
    split = self.TrainSplit()
    with open(trainDir) as fileName:
          reader = pd.read_csv(fileName)
          for index,row in reader.iterrows():
              content  = row['body']
              polarity = int(row['fakeness'])
              example = self.Example()
              if polarity == 0:
                  example.words = content
                  example.klass = polarity
                  split.train.append(example)
              if polarity == 1:
                  example.words = content
                  example.klass = polarity
                  split.train.append(example)
    return split

  def crossValidationSplits(self, trainDir):
    splits   = []
    content  = str("")
    polarity = 0
    
    for fold in range(0, self.numFolds):
        split = self.TrainSplit()
        count = 0
        with open(trainDir) as fileName:
            reader = pd.read_csv(fileName).fillna(value = "")
            for index,row in reader.iterrows():
                content  = row['body']
                polarity = int(row['fakeness'])
                hashing  = count
                count    = count + 1
                example = self.Example()
                example.words = content.split()
                example.klass = polarity
                if hashing != "hash" and hashing % 10 == fold:
                    split.test.append(example)
                else:
                    split.train.append(example)
        splits.append(split)
    return splits
'''
Notes:
    Reads rows in a CSV file as a dict in order to create test splits.
    Prepares data to be sent to the perceptron to test validation of
    feature engineering and perceptron handling.
'''

def test10Fold(args):
  pt         = Perceptron()
  splitName  = "Word_Data"
  splitCount = 0
  
  iterations = int(args[1])
  splits = pt.crossValidationSplits(args[0])
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = Perceptron()
    accuracy = 0.0
    classifier.train(split,iterations)
  
    for example in split.test:
      words = example.words
      guess = classifier.classify(words)
      if example.klass == guess:
        accuracy += 1.0

    accuracy = accuracy / len(split.test)
    avgAccuracy += accuracy
    print '[INFO]\tFold %d Accuracy: %f' % (fold, accuracy) 
    fold += 1
    '''
    TODO: Beyond here, for every split, create a csv holding weights/words across the corpus
    '''
    fileName = splitName + str(splitCount) + ".csv"
    with open(fileName, 'w') as csvfile:
        data    = ['word', 'weight'] 
        archive = csv.DictWriter(csvfile, fieldnames = data)
        archive.writeheader()
        for w in set(classifier.words):
            temp_word   = w
            temp_weight = classifier.weights[int(classifier.words[w])]
            archive.writerow({'word' : temp_word, 'weight' : temp_weight})
        csvfile.close()
    splitCount += 1
  avgAccuracy = avgAccuracy / fold
  print '[INFO]\tAccuracy: %f' % avgAccuracy
  
def classifyDir(trainDir, testDir,iter):
  classifier = Perceptron()
  trainSplit = classifier.trainSplit(trainDir)
  iterations = int(iter)
  classifier.train(trainSplit,iterations)
  testSplit = classifier.trainSplit(testDir)
  accuracy = 0.0
  for example in testSplit.train:
    words = example.words
    guess = classifier.classify(words)
    if example.klass == guess:
      accuracy += 1.0
  accuracy = accuracy / len(testSplit.train)
  print '[INFO]\tAccuracy: %f' % accuracy

def main():
  (options, args) = getopt.getopt(sys.argv[1:], '')
  
  if len(args) == 3:
    classifyDir(args[0], args[1], args[2])
  elif len(args) == 2:
    test10Fold(args)

if __name__ == "__main__":
    main()
