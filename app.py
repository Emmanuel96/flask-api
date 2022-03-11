# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 12:37:22 2020

@author: noopa
"""


import pickle
import pandas as pd
from flask import Flask, request
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in = open("model.pkl",'rb')
classifier=pickle.load(pickle_in)

@app.route('/')
def hello():
    return "You are the best Noopa! Thank you for always being understanding."

@app.route('/predict_with_file', methods=["POST"])
def predict_with_file():
    """Let's predict the class for iris
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: file
        in: formData
        type: file
        required: true
    responses:
        200:
            description: The output values
        
    """
    
    df = pd.read_csv(request.files.get("file"))
    prediction=classifier.predict(df)
    predictions_nominal = [ "M" if x < 0.5 else "B" for x in list(prediction)]
    return " The Predicated Class for the TestFile is"+ str(predictions_nominal)


if __name__=='__main__':
    app.run()
    
    
    
    
    
    
    
    
    
    
    
    