from flask import Flask,request,render_template
import pandas as pd
import numpy as np
import pickle
app=Flask(__name__)
pickle_in=open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)
@app.route('/')
def main():
    return "Wellcome"
@app.route('/predict')
def check():
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    return str(prediction)
if __name__=="__main__":
    app.run()