# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 23:37:50 2020

@author: Shashank
"""
import os 
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import dispatcher
import joblib

TRAINING_DATA = "input/train_folds.csv"
FOLD = 0


FOLD_MAPPING = {
        0: [1,2,3,4],
        1: [0,2,3,4],
        2: [0,1,3,4],
        3: [0,1,2,3]
        }

if __name__ == "__main__":
    # read the training data
    df = pd.read_csv(TRAINING_DATA)
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.isin.html
    # isin returns true or false,
    
    # to get the current 4 training data from 5 divided parts from k fold
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    
    # to get singel validation data from 5 parts divided k folded data
    valid_df = df[df.kfold == FOLD]
    

    #get correspondig target values.    
    ytrain = train_df.target.values
    yvalid = valid_df.target.values
    
    # get training data
    train_df = train_df.drop(['id','target','kfold'], axis = 1)
    valid_df = valid_df.drop(['id','target','kfold'], axis = 1)
    #print(train_df.head())
    # to keep the same order of variables.
    valid_df = valid_df[train_df.columns]
    
    label_encoders = []
    
    for c in train_df.columns:
        # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
        # to be used only on trget values
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist())
        train_df.loc[:,c] = lbl.fit_transform(train_df[c].values.tolist())
        valid_df.loc[:,c] = lbl.fit_transform(valid_df[c].values.tolist())
        label_encoders.append((c,lbl))
        #print(lbl)
    
    model = input("which model to use : randomforest or extratress:   ")
    # data is ready to train
    clf = dispatcher.MODELS[model]
    clf.fit(train_df, ytrain)
    preds =clf.predict_proba(valid_df)[:,1]
    print(metrics.roc_auc_score(yvalid, preds))
    joblib.dump(label_encoders,f"models/{model}_label_encoder.pkl")
    joblib.dump(clf,f"models/{model}.pkl")
        
