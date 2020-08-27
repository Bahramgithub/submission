#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 19:05:45 2020
@author: bahram@live.com.au
"""

# Import required libraries
import numpy as np
import pandas as pd
from joblib import load
from nltk.corpus import stopwords 
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer

# Define preprocessing functions and variables
tokenizer = RegexpTokenizer(r"\w+")
stopWords = set(stopwords.words("english")) 
stemmer = SnowballStemmer("english")
 
# Serve trained models to predict topics of new docs
print("Prediction")

# Load models from disk
model1 = load("model1.joblib")
model2 = load("model2.joblib")
vectorizer = load("vectorizer.joblib")
#nClusters = len(model.cluster_centers_)

# Get terms in vector in indexed list
terms = vectorizer.get_feature_names()

# Get centroids and top terms for them for model 1
centroids = model1.cluster_centers_.argsort()[:, ::-1]

# This stores top terms for each cluster for model 1
topTermsModel1 = {}
for i in range(len(model1.cluster_centers_)):
    topTermsModel1[i] = ""
    topTermsModel1[i] += "/".join(terms [j] for j in centroids[i, : 10])

# Get topics and top terms for them for model 2
topTermsModel2 = {}
for i, topic in enumerate(model2.components_):
    topTermsModel2[i] = ""   
    topTermsModel2[i] += "/".join([terms[i] for i in topic.argsort()[: -11 : -1]])
    
# Reading the files and preprocessing
f = open("./coding-challenge-ml/test.txt", "r")
dataTest = f.read()
split = dataTest.split("\n")
split = list(map(lambda x: x[0: x.find("__")], split))

# First column in output dataframe stores document
dfNewDocuments = pd.DataFrame(split, columns = ["Document"])

# Same pre-processes to remove stop words and stem as we did for training data
split = list(map(lambda x: " ".join(stemmer.stem (t) for t in tokenizer.tokenize(x) if not t.lower() in stopWords), split))

# 2nd column will have cluster index
dfNewDocuments["Model1 tag"] = dfNewDocuments["Document"].map(lambda x: model1.predict(vectorizer.transform([x]))[0])

# 3rd column will have the most representative terms
dfNewDocuments["Model1 cluster"] = dfNewDocuments["Model1 tag"].map(lambda x: topTermsModel1[int(x)]) 

# 4th column will have cluster index
dfNewDocuments["Model2 tag"] = dfNewDocuments["Document"].map(lambda x: np.argmax(model2.transform(vectorizer.transform([x]))))

# 5th column will have the most representative terms
dfNewDocuments["Model2 topic"] = dfNewDocuments["Model2 tag"].map(lambda x: topTermsModel2[int(x)]) 

# Storing new documents with additional tags
dfNewDocuments.to_excel("results.xlsx")