################################################################################################
##  Script Info: Extarcts features from headline and Body  
##  Authors: Mohammed Habibllah Baig, Carson Hanel
##  Date : 11/22/2017
################################################################################################
import os
import re
import nltk
import numpy as np
from sklearn import feature_extraction
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim import models
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from nltk import word_tokenize, pos_tag, ne_chunk, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer
    
_wnl = nltk.WordNetLemmatizer()

def Lemmatization(w):
    return _wnl.lemmatize(w).lower()


def Tokenization(s):
    return [Lemmatization(t) for t in nltk.word_tokenize(s)]

def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric
    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()
    
def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]


def ExtractFeatures(feat_fn, headline, body):
    feats = feat_fn(headline, body)
    return feats

def Jaccard_Similarity(headline, body):
    """ It calculates the degree of overlap between Headline and Body of news article"""
    X = []
    Tokenized_headline = Tokenization(headline)
    Tokenized_body = Tokenization(body)
    if float(len(set(Tokenized_headline).union(Tokenized_body))) == 0:
        return [0.]
    features = [
        len(set(Tokenized_headline).intersection(Tokenized_body)) / float(len(set(Tokenized_headline).union(Tokenized_body)))]
    X.append(features)
    return features
    
"""It calculates the polarity scores of headline and body using
    Vader(Valence Aware Dictionary and Sentiment Reasoner) sentiment analyzer."""
def sentiment_feature(headline,body):
    sent = SentimentIntensityAnalyzer()
    features = []
    headVader = sent.polarity_scores(headline)
    bodyVader = sent.polarity_scores(body)
    features.append(abs(headVader['pos']-bodyVader['pos']))
    features.append(abs(headVader['neg']-bodyVader['neg']))
    return features
    
def named_entity_feature(headline,body):
    """ Retrieves a list of Named Entities from the Headline and Body.
    Returns a list containing the cosine similarity between the counts of the named entities """
    stemmer = PorterStemmer()
    #Extract the Parts of speech Tags
    def Extract_POS_tags(text):
        return pos_tag(word_tokenize(text))

    def filter_pos(named_tags, tag):
        return " ".join([stemmer.stem(name[0]) for name in named_tags if name[1].startswith(tag)])

    named_cosine = []
    tags = ["NN"]
    
    cosine_simi = []
    head = Extract_POS_tags(headline)
    body = Extract_POS_tags(body[:255])

    for tag in tags:
        head_f = filter_pos(head, tag)
        body_f = filter_pos(body, tag)

        if head_f and body_f:
            vect = TfidfVectorizer(min_df=1)
            tfidf = vect.fit_transform([head_f,body_f])
            cosine = (tfidf * tfidf.T).todense().tolist()
            if len(cosine) == 2:
                cosine_simi.append(cosine[1][0])
            else:
                cosine_simi.append(0)
        else:
            cosine_simi.append(0)
    return cosine_simi

'''It indicates the polarity of headline and body data with sentiment'''
def polarity_features(headline, body):
    _refuting_words = [ 'fake','fraud','hoax','false','deny', 'denies','not','despite','nope','doubt', 
                        'doubts','bogus','debunk','pranks','retract']

    def check_polarity(text):
        tokens = Tokenization(text)
        return sum([t in _refuting_words for t in tokens]) % 2
    X = []
    features = []
    features.append(check_polarity(clean(headline)))
    features.append(check_polarity(clean(body)))
    X.append(features)
    return features

def count_chargrams(features, text_headline, text_body, size):
    def chargrams(input, n):
        output = []
        for i in range(len(input) - n + 1):
            output.append(input[i:i + n])
        return output
    
    headline_words = [' '.join(x) for x in chargrams(" ".join(remove_stopwords(text_headline.split())), size)]
    match_count = 0
    match_early_count = 0
    match_first_count = 0
    for word in headline_words:
        if word in text_body:
            match_count += 1
        if word in text_body[:255]:
            match_early_count += 1
        if word in text_body[:100]:
            match_first_count += 1
    features.append(match_count)
    features.append(match_early_count)
    features.append(match_first_count)
    return features


def count_ngrams(features, text_headline, text_body, size):
    def ngrams(input, n):
        input = input.split(' ')
        output = []
        for i in range(len(input) - n + 1):
            output.append(input[i:i + n])
        return output
    
    headline_words = [' '.join(x) for x in ngrams(text_headline, size)]
    match_count = 0
    match_early_count = 0
    for word in headline_words:
        if word in text_body:
            match_count += 1
        if word in text_body[:255]:
            match_early_count += 1
    features.append(match_count)
    features.append(match_early_count)
    return features


'''The counts of various chargrams and n-grams that are occurring in the headline and body'''
def Misc_features(headline, body):

    def count_matching_tokens(headline, body):
        # Count how many times a token in the title
        # appears in the body text.
        count = 0
        count_early = 0
        for headline_token in clean(headline).split(" "):
            if headline_token in clean(body):
                count += 1
            if headline_token in clean(body)[:255]:
                count_early += 1
        return [count, count_early]

    def count_matching_tokens_stops(headline, body):
        # Count how many times a token in the title
        # appears in the body text. Stopwords in the title
        # are ignored.
        count = 0
        count_early = 0
        for headline_token in remove_stopwords(clean(headline).split(" ")):
            if headline_token in clean(body):
                count += 1
                count_early += 1
        return [count, count_early]

    def count_grams(headline, body):
        # Count how many times an n-gram of the title
        # appears in the entire body, and intro paragraph

        clean_body = clean(body)
        clean_headline = clean(headline)
        features = []
        features = count_chargrams(features, clean_headline, clean_body, 2)
        features = count_chargrams(features, clean_headline, clean_body, 8)
        features = count_chargrams(features, clean_headline, clean_body, 4)
        features = count_chargrams(features, clean_headline, clean_body, 16)
        features = count_ngrams(features, clean_headline, clean_body, 2)
        features = count_ngrams(features, clean_headline, clean_body, 3)
        features = count_ngrams(features, clean_headline, clean_body, 4)
        features = count_ngrams(features, clean_headline, clean_body, 5)
        features = count_ngrams(features, clean_headline, clean_body, 6)
        return features

    X = (count_matching_tokens(headline, body)
             + count_matching_tokens_stops(headline, body)
             + count_grams(headline, body))

    return X


#Pseudo perceptron classifier
#author: Carson Hanel
def score(body, weights, words):
    # Utilizes learned scores from a logistic regression perceptron run on the corpus.
    # The idea is to make the learning quicker by doing learning separately, and simply
    # the learned weights rather than learning and then scoring.
    #
    #TODO:
    #  Utilize the maxent classifier and perceptron both in order to generate weights.
    #  See if accuracy improves if not just classification is included, but the document score.
    feature = []
    body    = body.split()
    weight  = 0.
    
    for w in set(body):
        if w in words:
            weight = weight + float(weights[int(words[w])])

    weight = 1 / (1 + np.exp(-weight))
    feature.append(weight)
    if weight > .5:
        feature.append(0)
    else:
        feature.append(1)
    
    return feature
                 
def unaries(body, words):
    # Parses the current document, and finds the frequencies of unaries in the bag of words.
    #
    #TODO:
    #  Important note: unless the BoW and weights are generated within Modelling.py, this will be very slow.
    #                  For a better design, I'll be moving the building of the BoW and the weight gathering 
    #                  from the CSV into the other file. This way, we're not loading 130k words/weights into
    #                  the RAM for every file to be scored; it'll be able to be passed as a parameter.
    feature = np.zeros(len(words.keys()))
    body    = body.split()
    for word in body:
        if word in words:
            feature[int(words[word])] += 1
    return feature


             
    
