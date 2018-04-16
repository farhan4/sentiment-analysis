# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 20:48:24 2018

@author: farhan baig
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv("labeledTrainData.tsv", delimiter="\t")
test_data = pd.read_csv("testData.tsv",delimiter="\t")

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

num_reviews = train_data['review'].size  
clean_train_reviews = []
for i in range( 0, num_reviews):
  clean_train_reviews.append( review_to_words(train_data["review"][i]))

num_reviews_test = test_data["review"].size
clean_test_reviews = []
for i in range( 0, num_reviews_test):
  clean_test_reviews.append(review_to_words(test_data["review"][i])) 
  

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer = "word",ngram_range = (1,3),max_features = 6000)
X = vectorizer.fit_transform(clean_train_reviews)
X = X.toarray()
y = train_data.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

test = vectorizer.transform(clean_test_reviews)
test = test.toarray()


# RandomForest Classifier
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 100) 
model.fit(X_train,y_train)

resultTrainData = model.predict(X_test)
resultTestData = model.predict(test)

from sklearn.metrics import confusion_matrix
cm_train_data = confusion_matrix(y_test, resultTrainData)
#      0      1
#  0  2181   367
#  1  373    2079





  
  
  
  