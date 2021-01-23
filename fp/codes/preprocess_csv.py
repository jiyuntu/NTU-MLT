import pandas as pd

filename = 'train.csv'
newfile = 'newtrain.csv'

def remove_adr_5000():
    df = pd.read_csv(filename)
    df=df[df.adr<5000]
    #df=df.drop([31979])
    df.to_csv(newfile,index=False)
if __name__ == '__main__':
    remove_adr_5000()
