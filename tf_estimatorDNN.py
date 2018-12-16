import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataUtils import readData ,mergeData
numeric_cols=['base_price','checkout_price','op_area']
cat_colums=['week','emailer_for_promotion','homepage_featured','category','cuisine','city_code','region_code','center_type']

def makeFeatureColum(train_merged):
    print('Making Feature columns for tensorflow estimators')
    numeric_feat=[tf.feature_column.numeric_column(col) for col in numeric_cols]
    week_embed=tf.feature_column.categorical_column_with_identity('week',200)
    city_embed=tf.feature_column.categorical_column_with_vocabulary_list('city_code',
                                        train_merged['city_code'].unique().tolist())
    region_embed=tf.feature_column.categorical_column_with_vocabulary_list('region_code',
                                train_merged['region_code'].unique().tolist())
    category_embed=tf.feature_column.categorical_column_with_vocabulary_list('category',
                                                train_merged['category'].unique().tolist())
    crossed_feat=tf.feature_column.crossed_column([category_embed,city_embed,region_embed],1000)
    total_embed_columns=[tf.feature_column.embedding_column(col,dim) for col,dim in zip([week_embed,city_embed,region_embed,category_embed],
                                                                                [64,16,2,8])]
    #tf.feature_column.crossed_column(city_embed,region_embed)
    cat_feat=[tf.feature_column.categorical_column_with_vocabulary_list('cuisine',train_merged['cuisine'].unique().tolist()),
        tf.feature_column.categorical_column_with_vocabulary_list('center_type',train_merged['center_type'].unique().tolist()),
        tf.feature_column.categorical_column_with_identity('emailer_for_promotion',2),
        tf.feature_column.categorical_column_with_identity('homepage_featured',2)]
    return total_embed_columns,numeric_feat+cat_feat+[crossed_feat]

def makeFeaturesForTrees():
    numeric_feat=[tf.feature_column.numeric_column(col) for col in numeric_cols]
    week_embed=tf.feature_column.categorical_column_with_identity('week',200)
    city_embed=tf.feature_column.categorical_column_with_vocabulary_list('city_code',
                                        train_merged['city_code'].unique().tolist())
    region_embed=tf.feature_column.categorical_column_with_vocabulary_list('region_code',
                                train_merged['region_code'].unique().tolist())
    category_embed=tf.feature_column.categorical_column_with_vocabulary_list('category',
                                                train_merged['category'].unique().tolist())
    cat_feat=[tf.feature_column.categorical_column_with_vocabulary_list('cuisine',train_merged['cuisine'].unique().tolist()),
        tf.feature_column.categorical_column_with_vocabulary_list('center_type',train_merged['center_type'].unique().tolist()),
        tf.feature_column.categorical_column_with_identity('emailer_for_promotion',2),
        tf.feature_column.categorical_column_with_identity('homepage_featured',2)]
    bucket_colum=[tf.feature_column.bucketized_column(f,boundaries=list(range(int(train_merged[c].min()),int(train_merged[c].max()),10)))for f,c in zip(numeric_feat,numeric_cols)]
    indicator_column=[tf.feature_column.indicator_column(c) for c in [week_embed,city_embed,region_embed,category_embed]+cat_feat]
    return bucket_colum,indicator_column

def buildTreeEstimator(features):
    treeBooster=tf.estimator.BoostedTreesRegressor(features,n_batches_per_layer=128,model_dir='modelstree/')
    return treeBooster

def buildEstimator(dense_feat,lin_feat):
    # estimator=tf.estimator.DNNLinearCombinedRegressor(dnn_hidden_units=[300,500,1],
    #                                                 dnn_feature_columns=dense_feat,
    #                                                 linear_feature_columns=lin_feat,
                                                    

    #                                                 model_dir='models2/')


    estimator=tf.estimator.DNNLinearCombinedRegressor(dnn_hidden_units=[300,500,1],
                                                  dnn_feature_columns=dense_feat,
                                                  linear_feature_columns=lin_feat,
                                                  linear_optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1,
                                                                                l1_regularization_strength=0.001,
                                                                                l2_regularization_strength=0.001),

                                                  dnn_optimizer=lambda: tf.train.AdamOptimizer(learning_rate=tf.train.exponential_decay(
                                                      learning_rate=0.1,global_step=tf.train.get_global_step(),decay_steps=10000,
                                                      decay_rate=0.96)),model_dir='modelsdnn/')
    return estimator

def estInpFunc(train=True):
    if train:
        input_fn=tf.estimator.inputs.pandas_input_fn(train_merged[cat_colums+numeric_cols],train_merged['num_orders'],
                                                                shuffle=True)
    else:
        input_fn=tf.estimator.inputs.pandas_input_fn(test_merged[cat_colums+numeric_cols],shuffle=False)
    return input_fn

def train_estimator(estimator,inp_fn):
    estimator.train(input_fn=inp_fn,steps=1000)
    return estimator

def predict_using_trained_estimator(estimator,inp_fn):
    predicted=[]
    for res in estimator.predict(input_fn=inp_fn):
        predicted.append(res['predictions'])
    return np.array(predicted).ravel()

def prepare_submission(test_merged,array,arrayt):
    sub_df=pd.DataFrame(columns=['id','num_orders'])
    sub_df['id']=test_merged['id'].values
    sub_df['num_orders']=array
    sub_df['num_orders_tree']=arrayt
    sub_df['avg']=np.mean(array+arrayt,axis=0)
    sub_df.to_csv('sub_tf4.csv',index=False)



if __name__=='__main__':
    train,test,center_info,meal_info=readData()
    train_merged,test_merged=mergeData(train,test,center_info,meal_info)
    dense_feat,lin_feat=makeFeatureColum(train_merged)
    bucket_colum,indicator_column=makeFeaturesForTrees()
    estimator=buildEstimator(dense_feat,lin_feat)
    treestimator=buildTreeEstimator(bucket_colum+indicator_column)
    train_inp_fn=estInpFunc()
    test_inp_fn=estInpFunc(train=False)

    #train
    estimator=train_estimator(estimator,train_inp_fn)
    treestimator=train_estimator(treestimator,train_inp_fn)

    #test
    arrayt=predict_using_trained_estimator(treestimator,test_inp_fn)
    array=predict_using_trained_estimator(estimator,test_inp_fn)
    prepare_submission(test_merged,array,arrayt)











