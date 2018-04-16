# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 00:01:49 2018

@author: farhan baig
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("labeledTrainData.tsv", delimiter="\t" ,quoting = 3)
unlabeled_train = pd.read_csv("unlabeledTrainData.tsv", delimiter="\t",quoting = 3)
test_data = pd.read_csv("testData.tsv",delimiter="\t",quoting = 3)

from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords


def review_to_words(review):
  # To remove <br></br> tags we use BeautifulSoup Library  
  review = BeautifulSoup(review).get_text()
  
  # Remove any words which do not start with a-z or A-Z or 0-9 using re(regex library)
  review = re.sub("[^a-zA-Z0-9]"," ",review)
  
  #Substitute 0-9 digits with NUM
  review = re.sub("[0-9]","NUM",review)
  
  # Convert to lower case and tokenize
  words = review.lower().split()
  
  #Remove stop words
  stops = set(stopwords.words("english")) 
  words = [w for w in words if not w in stops]
  
  return( " ".join(words))

import nltk.data
# We use NLTK punkt tokenizer to split a paragraph into sentences which takes care of .?!
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def review_to_sentences( review, tokenizer):
    
      sentences = tokenizer.tokenize(review.strip())
     
      clean_sentences = []
      
      for sentence in sentences:
          if len(sentence) > 0:
              clean_sentences.append(review_to_words(sentence))
      
      return clean_sentences


sentences = []
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)    

for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)
    

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size (word +- 10)                                                                                  
downsampling = 1e-3   # Downsample setting for frequent words

from gensim.models import word2vec

model = word2vec.Word2Vec(sentences,workers=num_workers,size=num_features, min_count = min_word_count,window = context, sample = downsampling)

model.init_sims(replace=True)

model_name = "300features_40minwords_10context"
model.save(model_name)

from gensim.models import Word2Vec
model = Word2Vec.load("300features_40minwords_10context")

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given paragraph
    
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")

    nwords = 0

    # Index2word is a list that contains the names of the words in the model's vocabulary. 
    # Convert it to a set, for speed 
    index2word_set = set(model.wv.index2word)

    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])

    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 

    counter = 0

    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model,num_features)
       counter = counter + 1
    return reviewFeatureVecs


clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(review_to_words(review))

trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )

clean_test_reviews = []
for review in test_data["review"]:
    clean_test_reviews.append( review_to_words(review))

testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier( n_estimators = 100 )


forest = forest.fit(trainDataVecs, train["sentiment"] )

# Test & extract results 
result = forest.predict( testDataVecs )

# Write the test results 
output = pd.DataFrame( data={"id":test_data["id"], "sentiment":result} )
output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )





    

    
          
     
     

  
  
  
  
  
  