# Import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFE, RFECV
from xgboost import XGBRegressor
from sklearn.model_selection import KFold

import logging
import sys


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(name)s-%(levelname)s-%(message)s')
handler.setFormatter(formatter)
logger.handlers = [handler]


NUM_FOLDS = 5
MIN_NUM_FEATURES = 10
NUM_FEATURES_TO_SELECT = 15


class RfeFeatureSelector(BaseEstimator, TransformerMixin):
    """Uses Recursive Feature Elimination to select the best features"""

    def __init__(self, estimator=None, 
                 cross_validation:bool=True, 
                 nfold:int=NUM_FOLDS,
                 min_feat_to_select:int=MIN_NUM_FEATURES, 
                 num_feat_to_select:int=NUM_FEATURES_TO_SELECT, 
                 **kwargs):

        self.estimator = estimator
        self.cross_validation = cross_validation
        self.nfold = nfold
        self.min_feat_to_select = min_feat_to_select
        self.num_feat_to_select = num_feat_to_select
        self.kwargs = kwargs
        self.check_model()
        assert self.cross_validation in (True, False), f'cross_validation should be either True or False, but {self.cross_validation} was passed'
        

    def check_model(self):

        # check a model object has been passed while initialising the class     
        check_fit = hasattr(self.estimator, 'fit')
        check_predict_proba = hasattr(self.estimator, 'predict')

        if self.estimator is None:
                self.estimator = XGBRegressor(objective='reg:gamma',
                                              verbosity=0,
                                              random_state=42,
                                              n_estimators=250, 
                                              learning_rate=0.1)

        elif check_fit is False and check_predict_proba is False:
            raise AttributeError('Model must contain both the fit() and predict() methods')

        else:
            pass        
    

    def fit(self, X, y=None):
        
        X = X.copy()
       
        if self.cross_validation:
            self.feat_selector = RFECV(estimator=self.estimator,
                                       min_features_to_select=self.min_feat_to_select,
                                       cv=KFold(n_splits=self.nfold, shuffle=True, random_state=42).split(X),
                                       scoring='neg_mean_gamma_deviance', **self.kwargs
                                      )
        
            logger.info('FITTING RECURSIVE FEATURE ELIMINATION WITH CROSS VALIDATION...')
            self.feat_selector.fit(X, y) 
            logger.info(f'OPTIMAL NUMBER OF FEATURES IS: {self.feat_selector.n_features_}')
            
            plt.figure(figsize=(12.6, 12))
            plt.title('Recursive Feature Elimination with Cross-Validation', fontweight='bold', pad=20)
            plt.xlabel('Number of Features Selected', labelpad=20)
            plt.ylabel('Mean Gamma Deviance', labelpad=20)
            plt.plot(range(1, len(self.feat_selector.grid_scores_) + 1), self.feat_selector.grid_scores_, linewidth=3)           
            plt.show()
                          
        else:                  
            self.feat_selector = RFE(estimator=self.estimator,
                                     n_features_to_select=self.num_feat_to_select,
                                     **self.kwargs
                                    )
                
            logger.info('FITTING RECURSIVE FEATURE ELIMINATION...')
            self.feat_selector.fit(X, y)
        
        self.features_selected = X.columns[self.feat_selector.support_]
        
        logger.info(f'FEATURES SELECTED ARE {self.features_selected}')

        return self

    
    def transform(self, X):
        
        X = X.copy()
        
        logger.info(f'TOTAL NUMBER OF FEATURES: {len(X.columns)}') 
        logger.info(f'NUMBER OF FEATURES SELECTED DURING FITTING: {self.feat_selector.n_features_}')
        
        logger.info('CREATING NEW DATAFRAME WITH SELECTED FEATURES...')
        X = X.loc[:, self.feat_selector.support_]

        return X
