import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

import logging 
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(name)s-%(levelname)s-%(message)s')
handler.setFormatter(formatter)
logger.handlers = [handler]


class RareCatLabelsHandler(BaseEstimator, TransformerMixin):
    """Handles rare labels of a categorical feature by giving all of them the same label"""

    def __init__(self, variable):
        if not isinstance(variable, list):
            self.variable = [variable]
        else:
            self.variable = variable

    def fit(self, X, y=None):
        df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        
        self.rare_labels_dict = {}
        
        logger.info('FITTING RARE CATEGORICAL LABEL HANDLER...')
        for feat in self.variable:
            temp = df.groupby(feat).apply(lambda x: x['SalePrice'].count() / len(df[feat])).to_frame('Percentage')
            rare_labels = temp[temp['Percentage'] < 0.01].index
            if rare_labels.values.tolist():
                self.rare_labels_dict[feat] = rare_labels.values.tolist()
                
        logger.info(f'{len(self.rare_labels_dict)} FEATURES HAVE RARE CATEGORICAL LABELS')
                       
        return self

    def transform(self, X):
        X = X.copy()

        logger.info('HANDLING RARE CATEGORICAL LABELS...')
        for feat in self.variable:
            if feat in self.rare_labels_dict.keys():
                X[feat] = np.where(X[feat].isin(self.rare_labels_dict[feat]), 'Rare', X[feat])

        return X


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    """One hot encode the categorical features"""

    def __init__(self, variable=None):
        if not isinstance(variable, list):
            self.variable = [variable]
        else:
            self.variable = variable

    def fit(self, X, y=None):
        self.onehot_dic_ = {}

        logger.info('FITTING ONE HOT ENCODER...')
        for col in self.variable:
            self.onehot_dic_[col+'fitted'] = OneHotEncoder(sparse=False, dtype=int, handle_unknown='ignore')
            self.onehot_dic_[col+'fitted'].fit(np.array(X[col]).reshape(-1, 1)) # Fit the one hot encoder to the categorical column

        return self

    def transform(self, X):
        X = X.copy()

        logger.info('ONE HOT ENCODING CATEGORICAL VARIABLES...')
        for col in self.variable:
            self.heading_list = []
            for i in list(self.onehot_dic_[col+'fitted'].categories_[0]):
                self.heading_list.append(col+'_'+str(i))
            dummies = self.onehot_dic_[col+'fitted'].transform(np.array(X[col]).reshape(-1, 1)) # Transform the categorical column using the one_hot_encoder fitted to it
            dummies = pd.DataFrame(dummies, columns=self.heading_list)
            X.drop(col, inplace=True, axis=1)
            X = pd.concat([X.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)  # Concatenate the dummies dataframe to the original dataframe

        logger.info(f'SHAPE OF DATAFRAME: {X.shape}')

        return  X
