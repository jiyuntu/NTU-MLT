import pandas as pd
import numpy as np
import csv
from preprocessing import each_day_revenue, make_X, make_cancel, make_adr
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from preprocess import Make_train,Make_new_train
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import pickle as pk
import os

filename = 'adr_rf.pk'

def predict_adr_random_forest():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    #if os.path.exists(filename):
    #    model = pk.load(open(filename, 'rb'))
    #    return model
    '''
    X, y = Make_new_train('train.csv','newtrain.csv')  
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, shuffle = True)
    
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1100, num = 6)]
    # Number of features to consider at every split
    #max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 6)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    #min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    #min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    #bootstrap = [True, False]
    # Create the random grid
    grid = {'n_estimators': n_estimators,'max_depth': max_depth}
    print(grid)
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = GridSearchCV(estimator = rf, param_grid = grid, verbose=500)
    # Fit the random search model
    rf_random.fit(X, y.to_numpy().ravel())
    print(rf_random.best_params_)
    '''
    
    X, y = Make_new_train('train.csv','newtrain.csv')

    for i in range(1):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
        #X_train, y_train = X,y

        regr = RandomForestRegressor(n_estimators=700,max_depth=90,verbose=500)
        #regr = RandomForestRegressor(verbose=500)
        regr.fit(X_train, y_train)

        ypred = regr.predict(X_train)
        print(mean_absolute_error(y_train, ypred))
        print(mean_squared_error(y_train, ypred))

    #pk.dump(regr, open(filename,'wb'))
    #return regr
    

if __name__ == '__main__':
    predict_adr_random_forest()
