# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 23:40:26 2020

@author: Shashank
"""
import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("input/train.csv")
    df['kfold'] = -1
    
    
    # for sample
    #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sample.html
    #Return a random sample of items from an axis of object.
    df = df.sample(frac =1).reset_index(drop = True)
    # fraction = 1 so whole data will be returned again.
    
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=12)
    
    for fold,(train_idx,val_idx) in enumerate(kf.split(X=df, y= df.target.values)):
        print(len(train_idx), len(val_idx))
        print(val_idx)
        print(train_idx)
        # Access a group of rows and columns by label(s) or a boolean array.
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html
        df.loc[val_idx,'kfold'] = fold
        
    df.to_csv("input/train_folds.csv", index = False) 