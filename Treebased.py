from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import pandas as pd
from catboost import Pool,CatBoostRegressor
from dataUtils import readData,mergeData
from preprocessors import Categorify

numeric_cols=['base_price','checkout_price','op_area']
cat_colums=['week','emailer_for_promotion','homepage_featured','category','cuisine','city_code','region_code','center_type']

def train_trees(df,y):
    rf=RandomForestRegressor(n_estimators=100)
    xgbr=xgb.XGBRegressor(n_estimators=100)
    print('training started')
    rf.fit(df,y)

    print('Random Forest completed-training')
    xgbr.fit(df,y)
    print('xgboost completed-training')
    return rf,xgbr    


def predict_trees(model_list,df):
    df=pd.DataFrame()
    for model in model_list:
        df[str(model.__class__).split('(')[0]]=model.predict(df)
    return df

def catboost_train(df,y):
    col=df.columns.tolist()
    train_pool=Pool(df,y,cat_features=[col.index(i) for i in cat_colums])
    model=CatBoostRegressor(iterations=1000,l2_leaf_reg=5,
                            one_hot_max_size=100,
                            depth=5,
                            bagging_temperature=10,
                            #use_best_model=True,
                            random_strength=5,
                            )
    model.fit(train_pool)
    return model

def catboost_predict(model,df):
    col=df.columns.tolist()
    test_pool=Pool(df,cat_features=[col.index(i) for i in cat_colums])
    return model.predict(test_pool)

def prepare_submission(test_merged,array):
    sub_df=pd.DataFrame(columns=['id','num_orders'])
    sub_df['id']=test_merged['id'].values
    sub_df['num_orders']=array
    #sub_df['num_orders_tree']=arrayt
    #sub_df['avg']=np.mean(array+arrayt,axis=0)
    sub_df.to_csv('sub_tree.csv',index=False)


if __name__=='__main__':
    train_merged,test_merged=mergeData(*readData())
    catProcess=Categorify(cat_colums,numeric_cols)
    catProcess.apply_train(train_merged)
    catProcess.apply_test(test_merged)
    # print(train_merged.head())
    # catmodel=catboost_train(train_merged[numeric_cols+cat_colums],train_merged['num_orders'].astype('float32'))
    # pred=catboost_predict(catmodel,test_merged[numeric_cols+cat_colums])
    # prepare_submission(test_merged,pred)
    rf,xgbr=train_trees(train_merged[numeric_cols+cat_colums],train_merged['num_orders'].astype('float32'))
    df=predict_trees([rf,xgbr],test_merged[numeric_cols+cat_colums])
    df.to_csv('res.csv')


