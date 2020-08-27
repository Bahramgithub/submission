## Problem definition

The problem is to develop a model to automatically determine topic of an unseen article. We have been provided with an unlabelled corpus of articles (from same nature of articles to be classified). 

## High level explanation of approach and code description

I have clustered train set with two clustering and topic modelling approaches and then used the models for prediction. 

The main toolkits I have used are **scikit learn** and **NLTK**. NLTK have been used for tokenising, stop word removal and stemming. And scikit learn has been used for feature extraction and vectorisation.

## Code description

**train.py**	uses NLTK (stopwords, Snowball stemmer); Sklearn (Tfidf Vectorizer, Kmeans, NMF); Joblib (dump)	and Train set to train	K-means model; NMF model;
Tf-iDf Vectorizer	and stores vectorizer and 2 models in joblib format for later use.

**predict.py**	uses Pandas; Numpy; NLTK (stopwords, Snowball stemmer); Joblib (load)	and Test set to run model and generate an	Excel file containing 6 columns (id, Document, Model1 tag, Model1 cluster, Model2 tag, Model2 topic).	The code loads models and vectorizer joblib files and stores results including prediction from two models into the excel file.

**serve.py** uses	Flask; Joblib (load)	and String from browser and predicts tag and top terms.	This is just a simple demo to serve a model and not a production solution. Run the app from terminal (go to directory and execute python serve.py) then call the app from browser on this address. (http://0.0.0.0:5000/serve/something) predicted tag and top terms will be sent back to browser and terminal 

**Note:** The aim in solution was not to achieve the best results by optimising parameters and details but to quickly solution an open problem with a justifiable architecture with outline of approaches for optimising models and processes.

## Pre-processing 

I have used NLTK toolkit for pre-processing. I have done tokenisation, stop word removal, optional digit and punctuations removal and stemming for pre-processing. Pre-processing could also include negation detection or lemmatisation which I have not included in the code.

## Vectorization 

Features been used in my code are Tf-iDf of terms across train set. Terms are indexed and weighted based on distribution on train documents. Terms that are seen in a high number of documents receive less weight and terms consolidated in a small number of documents receive higher weights. The reason is that terms in latter case provide more information in differentiating clusters of documents. In the code I have limited number of features to 500 and have ignored terms that exist in more than 90% of training documents- this is because terms existing in almost all documents are seen as not informative in terms of differentiating topics. Also, very rare words occurring in less that 1% of documents have been ignored as these words could introduce noise to the training algorithm. 

## Unsupervised learning code 

The code **train.py** uses train set and two unsupervised learning algorithms for clustering and topic modelling. All we need for running the code is provide the path to train set. The models will be stored in the same folder as training is completed.

## Clustering using K-means

Unlabelled corpus of articles- train set- is used for unsupervised learning to model clusters in feature space. Train documents have been vectorized using Tf-iDf features after pre-processing. Vectors have been then fed to K-means algorithm. K-means outcome is cluster centroid vectors which then could be used to identify allocate an unseen document to a cluster. We have used scikit learn toolkit for both vectorisation and clustering. K-means is an iterative learning algorithm. Meaning, it iteratively updates clusters and centroids to fit best to train set. Higher number of iterations could result to a more representative model. Higher iterations obviously increase training time. In this code I have set number of iterations as 200. 

**Note:** Number of clusters/topics is itself unknown and could be a parameter to be optimised. However, we could take an intuitive approach to start with a reasonable number of clusters- which suits for the use case- and then examine how well models are performing. Number of clusters been set as 10 in the code. Number of clusters can be optimised using metrics that evaluate how well clustering has been done- such as Silhouette score. In other words, for optimising number of clusters, we can measure this score for different number of clusters to figure out what number of clusters maximises Silhouette score.

## Topic modelling with Non-negative Matrix Factorisation 

In the other approach, I have run topic modelling algorithm using term-document matrix decomposition. The output of this algorithm is a decomposition of term-document matrix and transformer matrix. Transformer then could be used to identify association of unseen document in vector space to each of topics. I have used Non-negative Matrix Factorisation (NMF) from scikit learn for topic modelling. The parameters used in NMF model initiation literally makes it like Latent Semantic Indexing (LSI) approach. Number of iterations is set as 200. Higher iterations improve the model prediction and convergence. 

**Note:** Trained models are then stored on disk for later usage for prediction of unseen documents from test set. I have used joblib library for storage on disk. 

## Prediction code 

Prediction loads trained models from disk to make predictions on test set. This is coded in **predict.py** code. All we need for running the code is setting the path to test set folder and models. I have used both K-means and NMF models for prediction. Results will be written back into an excel file using pandas on local folder. 

At first stage, models will be loaded including Tf-iDf vectorizer, K-means centroid vectors and NMF topic vectors and transformer are loaded. Terms along with their indexes are extracted from vectorizer so that for each centroid and topic vector we could identify what terms are having the most significance. This is how we name topics. For example, one topic is “health/people/research/help/care” based on top 5 most relevant terms to the topic.
For each document in test set, pre-processing including stemming and stop word removal are done. Each document then will be vectorized using vocabulary and inverted-document frequency learnt from train stage. 


