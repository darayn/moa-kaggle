import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import seaborn as sns

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from utils import seed_everything

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')



# Train test size check
def sanity_check(): return train_features.shape, test_features.shape


# rank gauss
def rankGauss(train, test, col):
    transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")
    train[col] = transformer.fit_transform(train[col].values)
    test [col] = transformer.transform    (test [col].values)
    return train, test

def pca(train,test, n_comp, prefix , col):
    data =      pd.concat([pd.DataFrame(train[col]), pd.DataFrame(test[col])])
    data2 =     (PCA(n_components=n_comp, random_state=42).fit_transform(data[col]))
    train2 =    data2[:train.shape[0]];
    test2 =     data2[-test.shape[0]:]

    train2 =    pd.DataFrame(train2, columns=[f'pca_{prefix}-{i}' for i in range(n_comp)])
    test2 =      pd.DataFrame(test2, columns=[f'pca_{prefix}-{i}' for i in range(n_comp)])

    # drop_cols = [f'c-{i}' for i in range(n_comp,len(GENES))]
    train = pd.concat((train, train2), axis=1)
    test = pd.concat((test, test2), axis=1)
    return train, test

# thresholdValue = 0.8
def VarianceThresholdOperation(train, test, thresholdValue):
    var_thresh = VarianceThreshold(thresholdValue)  
    data = train.append(test)
    data_transformed = var_thresh.fit_transform(data.iloc[:, 4:])

    train_features_transformed =   data_transformed[ : train.shape[0]]
    test_features_transformed =    data_transformed[-test.shape[0] : ]
    train =   pd.DataFrame(train[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\
                                  columns=['sig_id','cp_type','cp_time','cp_dose'])
    train =   pd.concat([train, pd.DataFrame(train_features_transformed)], axis=1)


    test =   pd.DataFrame(test[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\
                                columns=['sig_id','cp_type','cp_time','cp_dose']) 
    test =   pd.concat([test, pd.DataFrame(test_features_transformed)], axis=1)
    return train , test

def fe_cluster(train, test, n_clusters_g = 35, n_clusters_c = 5, SEED = 123):
    
    features_g = list(train.columns[4:776])
    features_c = list(train.columns[776:876])
    
    def create_cluster(train, test, features, kind = 'g', n_clusters = n_clusters_g):
        train_ = train[features].copy()
        test_ = test[features].copy()
        data = pd.concat([train_, test_], axis = 0)
        kmeans = KMeans(n_clusters = n_clusters, random_state = SEED).fit(data)
        train[f'clusters_{kind}'] = kmeans.labels_[:train.shape[0]]
        test[f'clusters_{kind}'] = kmeans.labels_[train.shape[0]:]
        train = pd.get_dummies(train, columns = [f'clusters_{kind}'])
        test = pd.get_dummies(test, columns = [f'clusters_{kind}'])
        return train, test
    
    train, test = create_cluster(train, test, features_g, kind = 'g', n_clusters = n_clusters_g)
    train, test = create_cluster(train, test, features_c, kind = 'c', n_clusters = n_clusters_c)
    return train, test

def fe_stats(train, test):
    
    features_g = list(train.columns[4:776])
    features_c = list(train.columns[776:876])
    
    for df in train, test:
        df['g_sum'] = df[features_g].sum(axis = 1)
        df['g_mean'] = df[features_g].mean(axis = 1)
        df['g_std'] = df[features_g].std(axis = 1)
        df['g_kurt'] = df[features_g].kurtosis(axis = 1)
        df['g_skew'] = df[features_g].skew(axis = 1)
        df['c_sum'] = df[features_c].sum(axis = 1)
        df['c_mean'] = df[features_c].mean(axis = 1)
        df['c_std'] = df[features_c].std(axis = 1)
        df['c_kurt'] = df[features_c].kurtosis(axis = 1)
        df['c_skew'] = df[features_c].skew(axis = 1)
        df['gc_sum'] = df[features_g + features_c].sum(axis = 1)
        df['gc_mean'] = df[features_g + features_c].mean(axis = 1)
        df['gc_std'] = df[features_g + features_c].std(axis = 1)
        df['gc_kurt'] = df[features_g + features_c].kurtosis(axis = 1)
        df['gc_skew'] = df[features_g + features_c].skew(axis = 1)
        
    return train, test


def process(train,test):

	GENES = [col for col in train_features.columns if col.startswith('g-')]
	CELLS = [col for col in train_features.columns if col.startswith('c-')]

	# Normalize using rank Guass
	col = GENES + CELLS
	train_features, test_features = rankGauss(train_features, test_features ,col)

	#  Get Pca
	train_features, test_features = pca(train_features, test_features, 600, 'G', GENES)
	train_features, test_features = pca(train_features, test_features, 50, 'C', CELLS)

	# feature selection using variance threshold - 0.8 
	train_features, test_features = VarianceThresholdOperation(train_features, test_features, 0.8)


	# feature engineering
	train_features, test_features = fe_cluster(train_features, test_features)
	train_features, test_features = fe_stats(train_features, test_features)

	return train_features, test_features


def process_score(scored, targets, seed=42, folds=7):
    # LOCATE DRUGS
    vc = scored.drug_id.value_counts()
    vc1 = vc.loc[vc<=18].index.sort_values()
    vc2 = vc.loc[vc>18].index.sort_values()

    # STRATIFY DRUGS 18X OR LESS
    dct1 = {}; dct2 = {}
    skf = MultilabelStratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    tmp = scored.groupby('drug_id')[targets].mean().loc[vc1]
    for fold,(idxT,idxV) in enumerate( skf.split(tmp,tmp[targets])):
        dd = {k:fold for k in tmp.index[idxV].values}
        dct1.update(dd)
    
    # STRATIFY DRUGS MORE THAN 18X
    skf = MultilabelStratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    tmp = scored.loc[scored.drug_id.isin(vc2)].reset_index(drop=True)
    for fold,(idxT,idxV) in enumerate( skf.split(tmp,tmp[targets])):
        dd = {k:fold for k in tmp.sig_id[idxV].values}
        dct2.update(dd)
    
    # ASSIGN FOLDS
    scored['kfold'] = scored.drug_id.map(dct1)
    scored.loc[scored.kfold.isna(),'kfold'] =\
        scored.loc[scored.kfold.isna(),'sig_id'].map(dct2)
    scored.kfold = scored.kfold.astype('int8')
    return scored


def prepare(train, test, scored, targets):
    train, test = process(train, test) 
    train_scored = process_score(scored, targets) 
    
    # merge features with scores
    folds = train.merge(scored, on='sig_id')
    folds = folds[folds['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    test  = test [test ['cp_type']!='ctl_vehicle'].reset_index(drop=True)

    folds = folds.drop('cp_type', axis=1)
    test  = test.drop ('cp_type', axis=1)

    # converting column names to str
    folds.columns = [str(c) for c in folds.columns.values.tolist()]

    # One-hot encoding 
    folds = pd.get_dummies(folds, columns=['cp_time', 'cp_dose'])
    test  = pd.get_dummies(test , columns=['cp_time', 'cp_dose'])

    ## Targets
    target_cols = scored.drop(['sig_id', 'kfold', 'drug_id'], axis=1).columns.values.tolist()

    # features columns
    to_drop = target_cols + ['sig_id', 'kfold', 'drug_id']
    feature_cols = [c for c in folds.columns if c not in to_drop]
    
    return folds, test, feature_cols, target_cols


if __name__ == "__main__":
    import sys
    import joblib
    
    path = sys.argv[1]


    train_features = pd.read_csv(f'{path}/train_features.csv')
    test_features  = pd.read_csv(f'{path}/test_features.csv')
    train_targets_scored = pd.read_csv(f'{path}/train_targets_scored.csv')
    drug = pd.read_csv(f'{path}/train_drug.csv')
    
    targets = train_targets_scored.columns[1:]
    train_targets_scored = train_targets_scored.merge(drug, on='sig_id', how='left') 

    folds, test, feature_cols, target_cols = prepare(train_features, 
                                                     test_features, 
                                                     train_targets_scored, 
                                                     targets)
    
    print(folds.shape)
    print(test.shape)
    print(f'Targets : {len(target_cols)}')
    print(f'Features : {len(feature_cols)}')
    
    folds.to_csv(path/'folds.csv', index=False)
    test .to_csv(path/'test.csv' , index=False)
    columns = {'features': feature_cols, 'targets': target_cols}
    joblib.dump(columns, path/'columns.pkl')

