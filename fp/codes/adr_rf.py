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
train = pd.read_csv('train.csv')

def predict_adr_random_forest():
    #if os.path.exists(filename):
    #    model = pk.load(open(filename, 'rb'))
    #    return model
    
    feature, date = make_X()
    cancel = make_cancel()
    adr = make_adr()
    stay = pd.DataFrame(train, columns = ['stay_in_weekend_night', 'stay_in_week_night']).values.tolist()

    bufx = []
    bufy = []   
    for i in range(len(feature)):
        if cancel[i][1] == 1 or (stay[i][0] + stay[i][1]) == 0:
            continue
        bufx.append(feature[i])
        bufy.append(adr[i])
    feature = bufx
    adr = bufy
    
    X = []
    y = []
    i = 0
    while i < len(feature):
        X.append(feature[i])
        accu = adr[i]
        j = i + 1
        while j < len(feature) and feature[i] == feature[j]:
            accu += adr[j]
            j += 1
        accu /= (j - i)
        y.append(accu)
        i = j
    X = np.array(X)
    y = np.array(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, shuffle = True)

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
    rf_random.fit(X, y)
    print(rf_random.best_params_)
    
    '''
    X, y = Make_new_train('train.csv','newtrain.csv')

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, y_train = X,y

    #regr = RandomForestRegressor(n_estimators=2000,min_samples_split=5,min_samples_leaf=2,max_features='auto',max_depth=50,bootstrap=True,verbose=500)
    regr = RandomForestRegressor(verbose=500)
    regr.fit(X_train, y_train)

    #ypred = regr.predict(X_test)
    #print(mean_absolute_error(y_test, ypred))
    #print(mean_squared_error(y_test, ypred))
    '''

    #pk.dump(regr, open(filename,'wb'))
    #return regr

if __name__ == '__main__':
    predict_adr_random_forest()
