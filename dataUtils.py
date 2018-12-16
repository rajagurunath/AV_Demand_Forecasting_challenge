import pandas as pd


def readData():
    print('Reading data')
    train=pd.read_csv(r'data/train.csv')
    test=pd.read_csv(r'data/test_QoiMO9B.csv')
    center_info=pd.read_csv(r'data/fulfilment_center_info.csv')
    meal_info=pd.read_csv(r'data/meal_info.csv')
    return train,test,center_info,meal_info

def mergeData(train,test,center_info,meal_info):
    print('Merging train ')
    train_merged=pd.merge(train,meal_info,on='meal_id')
    train_merged=pd.merge(train_merged,center_info,on='center_id')
    print(train_merged.head(),train_merged.shape)

    print('Merging test')
    test_merged=pd.merge(test,meal_info,on='meal_id')
    test_merged=pd.merge(test_merged,center_info,on='center_id')
    print(test_merged.head(),test_merged.shape)
    return train_merged,test_merged
