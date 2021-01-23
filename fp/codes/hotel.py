import pandas as pd
import numpy as np
import csv
from adr2level import predict_level
from adr2level_DS import predict_level_DS
from predictAdr import predict_adr
from preprocessing import make_test_X, make_X
from cancel_mergeNN import predict_cancel_mergeNN
from predictAdrMerge import predict_adr_merge
from xgbPredictAdr import xgb_predict_adr 
from cancel_merge_gbdt import predict_cancel_gbdt
from cancel_merge_dart import predict_cancel_dart
from adr_dart import predict_adr_dart
from testLGB import predict_adr_merge_LGB
from adr_rf import predict_adr_random_forest
import os
from preprocess import Make_test, Make_train

test = pd.read_csv("./train.csv")
test_label = pd.read_csv("./train_label.csv")

def f(x):
    beta = 3
    return 1 / (1 + (x/(1-x))**(-beta))

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    feature, date = make_X()

    cancel = predict_cancel_mergeNN() # whether this booking will be canceled (DNN)
    canc1 = cancel.predict(feature)
#   for i in range(len(canc1)):
#       canc1[i] = canc1[i][0]
#   print('cancel prediction done')

#   cancel = predict_cancel_mergeNN()
#   canc2 = cancel.predict(feature)
    
    X,y = Make_train('train.csv')
    adr = predict_adr_random_forest()
    X_adr = adr.predict(X)
#    print(X_adr)
    print('adr prediction done')

    for i in range(len(X_adr)):
        X_adr[i]=X_adr[i]*f(canc1[i][0])
#        if canc1[i][0] < 0.5 and canc1[i][0] > 0.25:
#            X_adr[i] *= canc1[i][0]
#        elif canc1[i][0] < 0.25:
#            X_adr[i] = 0
#       if canc1[i] == 1 and canc2[i][0] > 0.5:
#           continue
#       elif canc1[i] == 1 and canc2[i][0] <= 0.5:
#           X_adr[i] *= 0.5
#       elif canc1[i] == 0 and canc2[i][0] > 0.5:
#           X_adr[i] *= 0.5
#       else:
#           X_adr[i] = 0
    print('adr tune done')
    

#   buf = []
#   datebuf = []
#   for i in range(len(feature)):
#       buf.append(feature[i])
#       datebuf.append(date[i])
#   feature = buf
#   date = datebuf
    
    
    stay = pd.DataFrame(test, columns = ['stays_in_week_nights', 'stays_in_weekend_nights']).values.tolist()
    buf = []
    for i in range(len(stay)):
        buf.append(stay[i][0] + stay[i][1])
    stay = buf
    
    rev = []
    for i in range(len(date)):
        last = len(rev) - 1
        if last < 0 or rev[last][0] != date[i]:
            rev.append([date[i], stay[i] * X_adr[i]])
        else:
            rev[last][1] += stay[i] * X_adr[i]
    
#   act_rev = [] # actual revenue of each day
#   act_night = [] # vice versa
#   for i in range(len(rev)):
#       act_rev.append(0)
#       act_night.append(0)
#   curr = 0
#   for i in range(len(date)):
#       while date[i] != rev[curr][0]:
#           curr += 1
#       act_rev[curr] = X_adr[i]
#       if canc[i] == 1:
#           act_night[curr] = penalty
#       else:
#           act_night[curr] = 1
#       j = curr + 1
#       while j < len(rev) and j - curr < stay[i]:
#           act_rev[j] += X_adr[i]
#           if canc[i] == 1:
#               act_night[curr] = penalty
#           else:
#               act_night[curr] = 1
#           j += 1
    
    X = [rev[i][1] for i in range(len(rev))]
    level = predict_level_DS() # adr_sum and other feature to ht's level (9 decision stump)

    y = []
    for i in range(len(X)):
        score = 0
        for j in range(len(level)):
            if X[i] > level[j]:
                score += 0.5
            else:
                break
        y.append(score)
        print(score)

    df = pd.DataFrame(test_label)
    df["label-predict"] = y
    df["revenue"] = X
    df = df.set_index('arrival_date')
    filename = 'submit.csv'
    df.to_csv(filename)



























