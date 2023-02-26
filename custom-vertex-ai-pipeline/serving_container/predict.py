# This module is intended to be used in the serving container
import os
from typing import Dict, List, Tuple
import pickle
import pandas as pd

from src.data_preprocessing.preprocessing import data_preprocessing_pipeline

import logging
import sys

logger = logging.getLogger("App")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(name)s-%(levelname)s-%(message)s')
handler.setFormatter(formatter)
logger.handlers = [handler]


def load_model(path:str):
    """Loads a model artifact"""
    
    with open(path, "rb") as f:
        model = pickle.load(f)
    
    return model


def model_predict(model: Dict, data: pd.DataFrame) -> Tuple[List, str]:
    """
    Use the input model to get predictions on the input data.

    model: A dictionary of objects used to make prediction.
           In its simplest case the dictionary has one item e.g. a scikit.learn Estimator
    data: A generic pandas DataFrame

    Returns: A list where each element is the predictions of the model for a single instance
             of input data
    """

    pipeline = model["pipeline"]
    target = model["target"]

    logger.info(f"PREPROCESSING THE DATA...")
    data_preprocessed = data_preprocessing_pipeline(data)

    logger.info(f"STARTING PREDICT ON DATAFRAME WITH SHAPE: {data_preprocessed.shape} and dtypes: {data_preprocessed.dtypes}")
    model_output = pipeline.predict(data_preprocessed)
    
    return model_output, target
