#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 19:05:45 2020
@author: bahram@live.com.au
"""

from flask import Flask#, request 
#import requests 

# Load models
from joblib import load
model1 = load("model1.joblib")
vectorizer = load("vectorizer.joblib")

# Create instance app
app = Flask(__name__)

# Tell flask what url should trigger our function
@app.route("/serve/<uuid>", methods=["POST", "GET"])

# Function to serve the model with uuid
def serve (uuid):

#    req = request.get_json()
#    content = req["content"]
    
    # Get centroids and indexed terms
    centroids = model1.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    
    # This stores the top representative word for each cluster
    topTerms = {}
    nClusters = len(model1.cluster_centers_)
    for i in range(nClusters):
        topTerms[i] = ""
        for ind in centroids[i, : 10]:
            topTerms[i] += "/" + terms[ind]
            
    # Predict the tag for received content
    tag = model1.predict(vectorizer.transform([str (uuid)]))
    message = uuid + " is predicted as class "  + str (tag [0]) + " with top terms as: " + topTerms [int (tag [0])]
    print (message)
    
    return message

if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = 5000, threaded = True)