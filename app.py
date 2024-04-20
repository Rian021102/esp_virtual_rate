import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Define Pydantic model for the data received from Streamlit
class InputData(BaseModel):
    feature1: float
    feature2: float
    # Add more features as needed

app = FastAPI()

# Load the trained model
model_path = '/Users/rianrachmanto/pypro/project/ESP_Rate_Prediction/model/virmod.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Function to process data
def process_data(data):
    data = data.drop(['ROW_INDEX','WELL_NAME','DATE','AREA','PUMP_BRAND','EQPM_TYPE'], axis=1)
    return data

# Define endpoint to receive data from Streamlit and make prediction
@app.post("/predict/")
def predict(data: InputData):
    # Prepare the input data for prediction
    processed_data = process_data(pd.DataFrame(data.dict(), index=[0]))  # Convert data to DataFrame and preprocess
    input_data = processed_data.values.reshape(1, -1)  # Reshape for prediction

    # Perform prediction using the loaded model
    prediction = model.predict(input_data)

    # You can return the prediction as JSON
    return {"prediction": prediction[0]}  # Assuming the prediction is a single value

if __name__ == "__main__":
    uvicorn.run("app:app", host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
