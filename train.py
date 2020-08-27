#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 19:05:45 2020
@author: bahram@live.com.au
"""

# Import required libraries
from joblib import dump 
from nltk.corpus import stopwords 
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF

# Define preprocessing functions and variables
tokenizer = RegexpTokenizer(r"\w+")
stopWords = set(stopwords.words("english")) 
stemmer = SnowballStemmer("english")

# Reading training dataset and split to paragraphs
f = open("./coding-challenge-ml/train.txt", "r")
dataTest = f.read()
split = dataTest.split("\n")

# Strip off the tag and remove stop words - optional stemming (stemmer.stem (t))
split = list(map(lambda x: x[0: x.find("__")], split))
split = list(map(lambda x: " ".join(stemmer.stem (t) for t in tokenizer.tokenize(x) if not t.lower() in stopWords), split))

# Clustering- Vectorising documents and learning cenroids with K-means algorithm
# Vectorize documents using Tfidf- ignores terms in more than 90% of docs
vectorizer = TfidfVectorizer(stop_words = "english",
                             max_features = 500, use_idf = True,
                             max_df = 0.90, min_df = 0.01)
X = vectorizer.fit_transform(split)

## Model1- K-means
print("Clustering with K-means")
nClusters = 10
model1 = KMeans(n_clusters = nClusters,
                init = "k-means++", max_iter = 200, n_init = 1)
model1.fit(X)

## Model2- Non-negative matrix factorization
print("Topic modeling with Non-negative matrix factorization")

# This model is equivalent to Latent Semantic Indexing algorithm
model2 = NMF(n_components = nClusters, random_state = 1,
          beta_loss = "kullback-leibler", solver = "mu",
          max_iter = 200, alpha = 0.1, l1_ratio = 0.5).fit(X)
model2.fit(X)

# Other vairant of NMF
#model2 = NMF(n_components = 10, random_state = 1, alpha = 0.1, l1_ratio = 0.5)

# Save model on disk for later usage
dump(vectorizer, "vectorizer.joblib")
dump(model1, "model1.joblib")
dump(model2, "model2.joblib")

