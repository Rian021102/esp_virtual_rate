#train random forest model with random search
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def random_forest_rs(X_train,X_test,y_train,y_test):
    param_dist = {
        'n_estimators': [100, 300, 500],  # Number of trees in the forest
        'max_features': ['sqrt', 'log2'],  # The number of features to consider when looking for the best split
        'max_depth': [None, 10, 20, 30, 40, 50],  # Maximum number of levels in tree
    }

    # Create a RandomForestRegressor object
    rf = RandomForestRegressor()

    # Setup the RandomizedSearchCV object
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

    # Fit the model
    rf_random.fit(X_train, y_train)

    # Print the best parameters and the best score
    print("Best Parameters:", rf_random.best_params_)
    print("Best Score:", rf_random.best_score_)

    # Predictions
    predictions = rf_random.predict(X_test)

    # Evaluation metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R^2 Score:", r2)
    #save the model with pickle
    with open('/Users/rianrachmanto/pypro/project/ESP_Rate_Prediction/model/virmod.pkl', 'wb') as model_file:
        pickle.dump(rf_random, model_file)
    return rf_random