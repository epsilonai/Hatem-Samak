#import libaries
import os
from threading import Thread
import joblib

import pandas as pd
import numpy as np

from flask import Flask, jsonify, request , redirect, url_for
from clsify.clf import train_clf, load_model


app=Flask(__name__)



@app.route("/")
def index():
    return """<h1>"welcome to Prediction Fraud Program"</h>
    <a href="Click Start" </a>"""

#Retrain the model to get another pkl file
@app.route("/train")
def train_new_model():
    # trainer_job = Thread(target=train_clf)
    # trainer_job.start()
    train_save=train_clf()
    return "The {}".format(train_save)

#Read single sample and get prediction and confidence
@app.route("/predict", methods=['POST',"GET"])
def predction():
    #load model
    cur_model=load_model()
    
    #load Data 
    feature = request.get_json()
    lst=feature["data"]
    l=np.reshape(lst,(1,-1))
       
    #Prediction & Calculate Probaility
    pred=cur_model.predict(l)
    if pred[0] ==0:
        clfy="Not Fraudlent"
        prob = (cur_model.predict_proba(l)[:,0][0]>=.002).astype(int)
    else:
        clfy="Fraudlent"
        prob = (cur_model.predict_proba(l)[:,1][0]>=.002).astype(int)
    
    
    return  """<h1> Model is Loaded</h1>
            <h1> </h1>
            <h1> The Transaction is : {}</h1>
            <h1> The Confidence is : {} %</h1>""".format(clfy,int(prob*100))
    
    
    # Onther output    
    #return jsonify({
    #     "prediction is :" : x,
    #     "Confidenc is :": prob[1]
    # })

if __name__ == "__main__":
    app.run(debug=True)