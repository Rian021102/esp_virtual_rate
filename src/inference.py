#create inference function by loading the model and using it to predict the rate of the ESP

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

'''create function to load data from excel file'''

def read_data(path):
    df = pd.read_excel(path)
    data = df.drop(['ROW_INDEX','WELL_NAME','DATE','AREA','PUMP_BRAND','EQPM_TYPE'], axis=1)
    print(df.head())
    return df, data

'''function to load the model'''

def load_model(path):
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model

'''function to predict the rate of the ESP'''

def predict_rate(model, data):
    prediction = model.predict(data)
    return prediction

'''function to insert prediction to df'''

def insert_prediction(df, prediction):
    df['Predicted_BFPD'] = prediction
    print(df)
    return df

def main():
    #load data
    path = '/Users/rianrachmanto/pypro/project/ESP_Rate_Prediction/data/clg_datatest_group_b_ori.xlsx'
    df,data=read_data(path)
    #load model
    model_path = '/Users/rianrachmanto/pypro/project/ESP_Rate_Prediction/model/virmod.pkl'
    model = load_model(model_path)
    #predict rate
    prediction = predict_rate(model, data)
    #insert prediction to df
    df = insert_prediction(df, prediction)

if __name__ == '__main__':
    main()


