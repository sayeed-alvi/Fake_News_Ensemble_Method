"""
@author: Carson Hanel
"""
import sys
import getopt
import os
import math
import operator
import numpy as np
import pandas as pd
import csv

class Maxent:
  class TrainSplit:
    """Represents a set of training/testing data. self.train is a list of Examples, as is self.test. 
    """
    def __init__(self):
      self.train = []
      self.test = []

  class Example:
    """Represents a document with a label. klass is 'pos' or 'neg' by convention.
       words is a list of strings.
    """
    def __init__(self):
      self.klass = ''
      self.words = []


  def __init__(self):
    """Maxent initialization"""
    
    self.numFolds     = 10  #Number of times the testing data is folded.
    self.words        = {}  #Dictionary for words in the bag of words.
    self.vocab_length = 0   #Total length of the calculated bag of words
    self.count_docs   = 0   #Count of total documents; just an iterator
    self.bag_of_words = []  #All word occurrences for all documents
    self.bag_of_pos   = []  #All word occurrences for positive 
    self.weights      = []  #Calculated feature weights
    self.accum        = []  #Calculated summation for weight update.
    '''
    * The idea with word_freq is to store at 0 the positive occurences of 
    '''
    self.word_freq    = []

  def classify(self, words):
    """ TODO
      'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    """
    #words = [word for word in words if word not in stopwords.words('english')]

    weight = 0.
    for w in set(words):
        if w in self.words:
            weight = weight + float(self.weights[int(self.words[w])])

    weight = 1 / (1 + np.exp(-weight))
    #print 'The calculated weight is: ', weight
    if weight > .5:
        return int(0)
    else:
        return int(1) 

  def addExample(self, klass, words, doc, eta, lambdaa):
    """
     * TODO
     * Train your model on an example document with label klass ('pos' or 'neg') and
     * words, a list of strings.
     * You should store whatever data structures you use for your classifier 
     * in the Maxent class.
     * Returns nothing
     *
     * Calculate empirical count for the document
     *   - the empirical count is the sum of the observed occurrences of a classifier in a document
    """
    occurrence = np.zeros(self.vocab_length)
    
    
    klass_int = 0
    if(klass == 0):
        klass_int = 1
    for w in set(words):
        self.bag_of_words += 1                  #Counts for all words for empirical probability.
        occurrence[int(self.words[w])] = 1
    """
    TODO:
        Add columns for all words in accum; it's the accumulated probability of all prior occurences of the word.
        Add the calculated new weight to accum[word]
        After being able to successfully calculate weights,
        - score a document by adding up weight of all words over the total words seen. If > 50%, it's a positive match
        
        For every word that showed up in the current 
    """
    change = 99
    parse  = 0
    
    #print 'Parsing document: ', doc
    changes = 0
    while(change > eta):
        changes += 1
        if change == 99:
            change = 0
        sum_weights = 0
        for w in set(words):
            w = w.lower()
            sum_weights += self.weights[int(self.words[w])]
        for w in set(words):
            w = w.lower()
            prev_weight = self.weights[int(self.words[w])]
            x_i_j       = occurrence[int(self.words[w])]
            y_i         = klass_int
            lamb_weight = -1 * (lambdaa * prev_weight)
            document_p  = 1 / (1 + np.exp(-sum_weights))
            
            new_weight  = prev_weight + eta*(lamb_weight + x_i_j*(y_i - document_p))
            self.weights[int(self.words[w])] = new_weight
            
            change     += new_weight - prev_weight
            parse      += 1
            
        change          = change / len(set(words))
    #print 'Total changes: ', changes
    pass
  
  def train(self, split, epsilon, eta, lambdaa):
      """
      * iterates through data examples
      https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words
      ^- bag of words optimization
      """

      np.random.shuffle(split.train)

      curr_word = 0
      for example in split.train:
          self.count_docs += 1
          for w in example.words:
              w = w.lower()
              if w not in self.words:
                  self.words[w] = curr_word
                  curr_word += 1
      self.vocab_length = len(self.words.keys())
      print 'The vocab length is: %d' % self.vocab_length + '\n'
      self.bag_of_words = np.zeros(self.vocab_length)
      self.bag_of_pos   = np.zeros(self.vocab_length)
      self.weights      = np.zeros(self.vocab_length)
      
      ex_doc = 0
      for example in split.train:
          words = example.words
          if(len(words) > 0):
            self.addExample(example.klass, words, ex_doc, eta, lambdaa)
            ex_doc += 1
 
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
  pt         = Maxent()
  splitName  = "Word_Data"
  splitCount = 0
  
  splits = pt.crossValidationSplits(args[0])
  epsilon = float(args[1])
  eta = float(args[2])
  lambdaa = float(args[3])
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = Maxent()
    accuracy = 0.0
    classifier.train(split, epsilon, eta, lambdaa)
  
    for example in split.test:
      words = example.words
      guess = classifier.classify(words)
      if int(example.klass) == int(guess):
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
  
def classifyDir(trainDir, testDir, eps, et, lamb):
  classifier = Maxent()
  trainSplit = classifier.trainSplit(trainDir)
  epsilon = float(eps)
  eta = float(et)
  lambdaa = float(lamb)
  classifier.train(trainSplit, epsilon, eta, lambdaa)
  testSplit = classifier.trainSplit(testDir)
  #testFile = classifier.readFile(testFilePath)
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
  
  if len(args) == 5:
    classifyDir(args[0], args[1], args[2], args[3], args[4])
  elif len(args) == 4:
    test10Fold(args)

if __name__ == "__main__":
    main()
