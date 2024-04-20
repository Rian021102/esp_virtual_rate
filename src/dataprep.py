import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


file = '/Users/rianrachmanto/pypro/project/ESP_Rate_Prediction/data/clg_datatrain_group_b_ori.xlsx'
def read_data(path):
    df = pd.read_excel(file)
    print(df.head())
    print(len(df)) 
    print(df['EQPM_TYPE'].nunique())
    print(df['WELL_NAME'].nunique())
    df.drop(['ROW_INDEX','WELL_NAME','DATE','AREA','PUMP_BRAND','EQPM_TYPE'], axis=1, inplace=True)
    print(df.head())
    X=df.drop(['BFPD'], axis=1)
    y=df['BFPD']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
    return X_train, X_test, y_train, y_test

def eda(X_train,y_train):
    df_eda=pd.concat([X_train, y_train], axis=1)
    print(df_eda.head())
    print(df_eda.describe())
    print(df_eda.info())
    print(df_eda.isnull().sum())
    #select all numerical columns except BFPD
    num_cols = df_eda.select_dtypes(include=[np.number]).columns
    print(num_cols)
    #plot histogram for all numerical columns
    for i in num_cols:
        plt.hist(df_eda[i])
        plt.title(i)
        plt.show()
    
    #plot scatter plot of all numerical columns with BFPD
    for i in num_cols:
        plt.scatter(df_eda[i], df_eda['BFPD'])
        plt.title('BFPD vs '+i)
        plt.show()
    #plot correlation matrix
    corr = df_eda.corr()
    sns.heatmap(corr, annot=True)
    plt.show()
    return df_eda
       