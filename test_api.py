import requests
import pandas as pd

# Define the URL
url = 'http://0.0.0.0:8080'

# Define the input data

'''load excel file'''
def load_data(file_path):
    data = pd.read_excel(file_path)
    return data

file_path = '/Users/rianrachmanto/pypro/project/ESP_Rate_Prediction/data/clg_datatest_group_b_ori.xlsx'
data = load_data(file_path)

# Convert datetime columns to string
data['DATE'] = data['DATE'].astype(str)  # Assuming 'DATE' is the datetime column

# Convert DataFrame to dictionary
data_dict = data.to_dict(orient='records')

'''call the API'''
response = requests.post(url + '/predict/', json=data_dict)
print(response.json())

# Convert response to DataFrame
response_data = pd.DataFrame(response.json())

# Insert the prediction to the original data
data['PREDICTION'] = response_data['prediction']
print(data.head())
