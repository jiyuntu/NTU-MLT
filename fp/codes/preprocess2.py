# reference: https://www.kaggle.com/pythonkumar/houseprice-solution-easiest-way-90-accuracy
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
    train = train[train.adr<1000]
    train = train.reset_index()
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
        #print(i)
        #print(lst)
        stays[i]=int(lst[0])+int(lst[1])
    stays = np.asmatrix(stays).transpose()
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

    y=pd.DataFrame(train, columns = ['adr'])
    X=train.drop(columns=['adr'])
    si=SimpleImputer()
    X=pd.DataFrame(si.fit_transform(X))
    m,n = X.shape
    for i in range(m): 
        print(X.loc[i])
        X=X.append(X.loc[[i]]*(stays[i,0]),ignore_index=True)
        y=y.append(y.loc[[i]]*(stays[i,0]),ignore_index=True)
    print(X)
    print(y)
    return X, y

def Make_test(filetrain, filetest):

    train=pd.read_csv(filetrain)
    train = train.drop(columns=['ID','country','is_canceled','reservation_status','reservation_status_date','adr'])
    #print(train.info())
    test=pd.read_csv(filetest)
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
    test = test.tail(27859)
    X=test
    si=SimpleImputer()
    X=si.fit_transform(X)

    return X

if __name__ == '__main__':
    Make_train('train.csv')
