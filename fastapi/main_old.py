# 1. Library imports
import pandas as pd
import pycaret
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn
import joblib,os

# 2. Create the app object
app = FastAPI()

#. Load trained Pipeline
model = open('../lgbm_clf.pkl', 'rb')
lgbm = joblib.load(model)



# Define predict function
@app.get('/predict/{name}')
def predict(name):
    pred = lgbm.predict(name)
    #y_pred = predict_model(model, data=df)
    return pred

#if __name__ == '__main__':
 #   uvicorn.run(app, host='127.0.0.1', port=8000)