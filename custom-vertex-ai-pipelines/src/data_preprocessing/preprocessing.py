import pandas as pd
import numpy as np

from src.data_preprocessing.utils import copy_df

import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(name)s-%(levelname)s-%(message)s')
handler.setFormatter(formatter)
logger.handlers = [handler]

def drop_id_col(df:pd.DataFrame):
    """Drops the Id column"""

    logger.info('DROPPING THE ID COLUMN...')
    df = df.drop('Id', axis=1)

    return df


def convert_to_correct_dtype(df:pd.DataFrame):
    """Corrects the dtype of the column MSSubClass"""

    logger.info('CONVERTING DTYPE OF MSSubClass COLUMN TO STRING...')
    df['MSSubClass'] = df['MSSubClass'].astype(str)

    return df


def subtract_yearsold_temporal_feat(df:pd.DataFrame):
    """Subtracts each temporal feature from 'YearSold' feature to get new features"""

    temporal_feat = [feat for feat in df.columns if 'Year' in feat or 'Yr' in feat]

    logger.info('CREATING NEW FEATURES BY SUBTRACTNG TEMPORAL VARIABLES FROM YrSold COLUMN...')
    for feat in temporal_feat:
        df[feat] = df['YrSold'] - df[feat]

    return df


def cosine_transform_cyclic_feat(df:pd.DataFrame):
    """Cosine transforms the categorical feature"""

    logger.info('COSINE TRANSFORMING THE MoSold COLUMN...')
    df['MoSold'] = -(np.cos(0.5236 * df['MoSold']))

    return df


def data_preprocessing_pipeline(df:pd.DataFrame):

    df = df.pipe(copy_df)\
         .pipe(drop_id_col)\
         .pipe(convert_to_correct_dtype)\
         .pipe(subtract_yearsold_temporal_feat)\
         .pipe(cosine_transform_cyclic_feat)

    return df



