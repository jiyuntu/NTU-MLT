# reference: y

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from datetime import datetime, date

month2number = {
    "January" : "01", "February" : "02", "March" : "03", "April" : "04", "May" : "05", "June" : "06",
    "July" : "07", "August" : "08", "September" : "09", "October" : "10",
    "November" : "11", "December" : "12"
}
def time_elapsed(y, m, d):
    d1 = datetime.today()
    d0 = datetime(int(y), int(m), int(d))
    delta = d1 - d0
    return delta.days

def Make_train(filetrain):

    train=pd.read_csv(filetrain)
    
    train = train.drop(columns=['ID','country','is_canceled','reservation_status','reservation_status_date'])

    # add new feature
    date = pd.DataFrame(train, columns = ["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"]).values.tolist()
    for i, lst in enumerate(date):
        lst[1] = month2number[lst[1]]
        date[i]=time_elapsed(lst[0], lst[1], lst[2])
    date = np.asmatrix(date).transpose()
    df = pd.DataFrame(data=date, columns=["time_elapsed"])

    train = pd.concat([train,df],axis=1)
    
    stays = pd.DataFrame(train, columns = ["stays_in_weekend_nights","stays_in_week_nights"]).values.tolist()
    for i,lst in enumerate(stays):
        stays[i]=int(lst[0])+int(lst[1])
    stays = np.asmatrix(stays).transpose()
    #print(stays.sum())
    df = pd.DataFrame(data=stays, columns=["stays_in_total"])

    train = pd.concat([train,df],axis=1)

    assigned_room_type_to_value = pd.DataFrame(train, columns=["assigned_room_type"]).values.tolist()
    for i,lst in enumerate(assigned_room_type_to_value):
        assigned_room_type_to_value[i]=int(ord(lst[0])-ord('A'))
    assigned_room_type_to_value = np.asmatrix(assigned_room_type_to_value).transpose()
    df = pd.DataFrame(data=assigned_room_type_to_value, columns=["assigned_room_type_to_value"])

    train = pd.concat([train,df],axis=1)
    
    cat_train=[cat for cat in train.columns if train[cat].dtype=='object']
    num_train=[num for num in train.columns if train[num].dtype=='int64' or train[num].dtype=='float64']

    train= pd.get_dummies(train, columns=cat_train, drop_first= True)
    '''
    y=pd.DataFrame(train, columns = ['adr'])
    X=train.drop(columns=['adr'])
    si=SimpleImputer()
    XX=pd.DataFrame(si.fit_transform(X))
    XX.columns=X.columns
    X=XX
    '''
    y=pd.DataFrame(train, columns = ['adr'])
    X=train.drop(columns=['adr'])
    si=SimpleImputer()
    X=si.fit_transform(X)

    print(X.shape)

    return X, y

def Make_new_train(filetrain, filenewtrain):
    train=pd.read_csv(filetrain)
    train = train.drop(columns=['ID','country','is_canceled','reservation_status','reservation_status_date'])
    newtrain = pd.read_csv(filenewtrain)
    newtrain = newtrain.drop(columns=['ID','country','is_canceled','reservation_status','reservation_status_date'])
    row = newtrain.shape[0]
    newtrain=pd.concat([train,newtrain],axis=0,ignore_index=True)
    
    # add new feature
    date = pd.DataFrame(newtrain, columns = ["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"]).values.tolist()
    for i, lst in enumerate(date):
        lst[1] = month2number[lst[1]]
        date[i]=time_elapsed(lst[0], lst[1], lst[2])
    date = np.asmatrix(date).transpose()

    df = pd.DataFrame(data=date, columns=["time_elapsed"])

    newtrain = pd.concat([newtrain,df],axis=1)
    
    stays = pd.DataFrame(newtrain, columns = ["stays_in_weekend_nights","stays_in_week_nights"]).values.tolist()
    for i,lst in enumerate(stays):
        stays[i]=int(lst[0])+int(lst[1])
    stays = np.asmatrix(stays).transpose()
    df = pd.DataFrame(data=stays, columns=["stays_in_total"])

    newtrain = pd.concat([newtrain,df],axis=1)
    
    assigned_room_type_to_value = pd.DataFrame(newtrain, columns=["assigned_room_type"]).values.tolist()
    for i,lst in enumerate(assigned_room_type_to_value):
        assigned_room_type_to_value[i]=int(ord(lst[0])-ord('A'))
    assigned_room_type_to_value = np.asmatrix(assigned_room_type_to_value).transpose()
    df = pd.DataFrame(data=assigned_room_type_to_value, columns=["assigned_room_type_to_value"])

    newtrain = pd.concat([newtrain,df],axis=1)

    cat_newtrain=[cat for cat in newtrain.columns if newtrain[cat].dtype=='object']
    num_newtrain=[num for num in newtrain.columns if newtrain[num].dtype=='int64' or newtrain[num].dtype=='float64']

    newtrain= pd.get_dummies(newtrain, columns=cat_newtrain, drop_first= True)
    newtrain = newtrain.tail(row)

    y=pd.DataFrame(newtrain, columns = ['adr'])
    X=newtrain.drop(columns=['adr'])
    si=SimpleImputer()
    X=si.fit_transform(X)

    print(X.shape)

    return X, y
   

def Make_test(filetrain, filetest):

    train=pd.read_csv(filetrain)
    train = train.drop(columns=['ID','country','is_canceled','reservation_status','reservation_status_date','adr'])
    #print(train.info())
    test=pd.read_csv(filetest)
    row = test.shape[0]
    test = test.drop(columns=['ID','country'])
    #print(test.info())
    test=pd.concat([train,test],axis=0,ignore_index=True)
    # print(test.info())
    
    # add new feature
    date = pd.DataFrame(test, columns = ["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"]).values.tolist()
    for i, lst in enumerate(date):
        lst[1] = month2number[lst[1]]
        date[i]=time_elapsed(lst[0], lst[1], lst[2])
    date = np.asmatrix(date).transpose()

    df = pd.DataFrame(data=date, columns=["time_elapsed"])

    test = pd.concat([test,df],axis=1)
    
    stays = pd.DataFrame(test, columns = ["stays_in_weekend_nights","stays_in_week_nights"]).values.tolist()
    for i,lst in enumerate(stays):
        stays[i]=int(lst[0])+int(lst[1])
    stays = np.asmatrix(stays).transpose()
    df = pd.DataFrame(data=stays, columns=["stays_in_total"])

    test = pd.concat([test,df],axis=1)
    
    assigned_room_type_to_value = pd.DataFrame(test, columns=["assigned_room_type"]).values.tolist()
    for i,lst in enumerate(assigned_room_type_to_value):
        assigned_room_type_to_value[i]=int(ord(lst[0])-ord('A'))
    assigned_room_type_to_value = np.asmatrix(assigned_room_type_to_value).transpose()
    df = pd.DataFrame(data=assigned_room_type_to_value, columns=["assigned_room_type_to_value"])

    test = pd.concat([test,df],axis=1)

    cat_test=[cat for cat in test.columns if test[cat].dtype=='object']
    num_test=[num for num in test.columns if test[num].dtype=='int64' or test[num].dtype=='float64']

    test= pd.get_dummies(test, columns=cat_test, drop_first= True)
    test = test.tail(row)
    X=test
    si=SimpleImputer()
    X=si.fit_transform(X)

    return X

def Make_train_adrcanc(filetrain):
    train=pd.read_csv(filetrain)
    
    train = train.drop(columns=['ID','country','reservation_status','reservation_status_date'])

    # add new feature
    date = pd.DataFrame(train, columns = ["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"]).values.tolist()
    for i, lst in enumerate(date):
        lst[1] = month2number[lst[1]]
        date[i]=time_elapsed(lst[0], lst[1], lst[2])
    date = np.asmatrix(date).transpose()
    df = pd.DataFrame(data=date, columns=["time_elapsed"])

    train = pd.concat([train,df],axis=1)
    
    stays = pd.DataFrame(train, columns = ["stays_in_weekend_nights","stays_in_week_nights"]).values.tolist()
    for i,lst in enumerate(stays):
        stays[i]=int(lst[0])+int(lst[1])
    stays = np.asmatrix(stays).transpose()
    #print(stays.sum())
    df = pd.DataFrame(data=stays, columns=["stays_in_total"])

    train = pd.concat([train,df],axis=1)

    assigned_room_type_to_value = pd.DataFrame(train, columns=["assigned_room_type"]).values.tolist()
    for i,lst in enumerate(assigned_room_type_to_value):
        assigned_room_type_to_value[i]=int(ord(lst[0])-ord('A'))
    assigned_room_type_to_value = np.asmatrix(assigned_room_type_to_value).transpose()
    df = pd.DataFrame(data=assigned_room_type_to_value, columns=["assigned_room_type_to_value"])

    train = pd.concat([train,df],axis=1)
    
    cat_train=[cat for cat in train.columns if train[cat].dtype=='object']
    num_train=[num for num in train.columns if train[num].dtype=='int64' or train[num].dtype=='float64']

    train= pd.get_dummies(train, columns=cat_train, drop_first= True)
    '''
    y=pd.DataFrame(train, columns = ['adr'])
    X=train.drop(columns=['adr'])
    si=SimpleImputer()
    XX=pd.DataFrame(si.fit_transform(X))
    XX.columns=X.columns
    X=XX
    '''
    y=pd.DataFrame(train, columns = ['is_canceled','adr','stays_in_total']).values.tolist()
    yy = [y[i][0]*y[i][1]*y[i][2] for i in range(len(y))]
    X=train.drop(columns=['adr','is_canceled'])
    si=SimpleImputer()
    X=si.fit_transform(X)

    return X, np.array(yy)


if __name__ == '__main__':
   X,y = Make_train_adrcanc('train.csv') 
