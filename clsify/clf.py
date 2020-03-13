#import liberaries
import os
import joblib

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, jsonify, request



PICKLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pickles", "clf.pkl")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "creditcard.csv")

#function to train model
def train_clf():
    
    #read Data
    dataset=pd.read_csv(DATA_DIR) 
    
    #split into dataset_X and dataset_y
    dataset_X=dataset.drop(["Time","Class"],axis=1)
    dataset_y=dataset.Class
    
    #Split dataset to train and test 
    # X_train, X_test, y_train, y_test \
    #     = train_test_split(dataset_X,dataset_y, test_size=0.2,\
    #     stratify=dataset_y, random_state=123, shuffle=True)
    Xb_train, _Xb_test , yb_train, _yb_test = train_test_split(dataset_X, dataset_y, test_size=0.2, random_state=0)

    #Resampling the dataset using SMOT
    sm = SMOTE(random_state=0)
    X_train, y_train = sm.fit_sample(Xb_train, yb_train)
    
    #Train the model
    clf_RFCls=RandomForestClassifier()
    clf_RFCls.fit(X_train,y_train)
    
    #Save Trained model into Pickle file
    joblib.dump(clf_RFCls, PICKLE_DIR, True)
    
    return "Model Trained and Saved" #model fitted

 
#load model after was trained 
def load_model():
    return joblib.load(PICKLE_DIR)
    