from dataprep import read_data
from datapro import dataprocess
from train_rf import random_forest_rs

def main():
    X_train, X_test, y_train, y_test = read_data('/Users/rianrachmanto/pypro/project/ESP_Rate_Prediction/data/clg_datatrain_group_b_ori.xlsx')
    dp = dataprocess(X_train,X_test)
    X_train = dp.process_train()
    X_test = dp.process_test()
    rf_random = random_forest_rs(X_train,X_test,y_train,y_test)
    print('Random Forest model with Random Search:',rf_random)

if __name__ == '__main__':
    main()

