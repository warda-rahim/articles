import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_gamma_deviance as gamma_deviance
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBRegressor
import shap

from src.feature_engineering.feat_eng import RareCatLabelsHandler, CustomOneHotEncoder
from src.feature_selection.rfe import RfeFeatureSelector

import logging 
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(name)s-%(levelname)s-%(message)s')
handler.setFormatter(formatter)
logger.handlers = [handler]


# WE USE TRAIN.CSV FOR TRAINING THE FINAL MODEL AS TEST.CSV IS MISSING THE TARGET.
# HYPERPARAMETER OPTIMISATION IS PERFORMED USING 'PART OF TRAIN.CSV' USING CROSS-VALIDATION FOLLOWED BY MODEL TRAINING.
# AND MODEL IS EVALUATED USING 'THE REMAINING PART OF TRAIN.CSV'.
# THE FINAL MODEL IS TRAINED USING THE ENTIRE TRAIN.CSV.
# THE FINAL PREDICTIONS ARE MADE ON TEST.CSV.


TARGET = 'SalePrice'
RANDOM_STATE = 42
N_KFOLD_SPLITS = 3
N_TRIALS= 10


class HousePriceModel(object):
    def __init__(self,
                 #train_df:pd.DataFrame=None
                 test_df:pd.DataFrame=None,
                 target:str=TARGET,
                 n_kfold_splits:int=N_KFOLD_SPLITS,
                 n_trials:int=N_TRIALS,
                 random_state:int=RANDOM_STATE             
                ):

       # self.train_df = train_df
        self.test_df = test_df
        self.target = target
        self.n_kfold_splits = n_kfold_splits
        self.n_trials = n_trials
        self.random_state  =  random_state

    @staticmethod
    def kfold_generator(X=None, n_splits:str=3):
        kfold = KFold(n_splits=n_splits)
        return kfold.split(X) 


    def create_final_training_dataset(self, train_df, test_df):
        self.df = pd.concat([train_df, test_df], ignore_index=True)


    def make_pipeline(self, **params):
        self.cat_features = self.X_train.select_dtypes(include=['category', object, bool]).columns.tolist()
        
        return Pipeline([
                ('rare_cat_handler', RareCatLabelsHandler(self.cat_features)),
                ('cat_encoder', CustomOneHotEncoder(self.cat_features)),
                ('feature_selector', RfeFeatureSelector(cross_validation=False)),
                ('xgb', XGBRegressor(**params))
                ])


    def fit_and_evaluate_model(self):
        self.best_params = {'objective': 'reg:gamma',
                            'verbosity': 0,
                            'random_state': self.random_state
                            }

        self.best_params.update(self.study.best_params)
        self.model_pipeline = self.make_pipeline(**self.best_params)
        
        logger.info(f'\nTRAINING THE MODEL...')
        logging.getLogger('src.feature_engineering.feat_eng').disabled = False
        logging.getLogger('src.feature_selection.rfe').disabled = False
        
        self.model_pipeline.fit(self.X_train, self.y_train)

        logger.info(f"\nGETTING THE SHAP VALUES")
        self.X_test_transformed = Pipeline(self.model_pipeline.steps[:-1]).transform(self.test_df.drop(self.target, axis=1))
        self.explainer = shap.TreeExplainer(self.model_pipeline.named_steps["xgb"])
        self.shap_values = self.explainer.shap_values(self.X_test_transformed)
        
        baseline_scores = {"gamma_deviance": gamma_deviance(self.y_train, [self.y_train.mean()]*len(self.y_train)),
                           "mse": mse(self.y_train, [self.y_train.mean()]*len(self.y_train)),
                           "r2": r2_score(self.y_train, [self.y_train.mean()]*len(self.y_train)) 
                           } 

        def evaluate(dataset):
            return {"gamma_deviance": gamma_deviance(dataset[self.target], self.model_pipeline.predict(dataset.drop(self.target, axis=1))),
                    "mse": mse(dataset[self.target], self.model_pipeline.predict(dataset.drop(self.target, axis=1))),
                    "r2": r2_score(dataset[self.target], self.model_pipeline.predict(dataset.drop(self.target, axis=1)))
                    }

        train_scores = evaluate(pd.concat([self.X_train, self.y_train], axis=1))
        test_scores = evaluate(self.test_df)

        self.scores = {"baseline_scores": baseline_scores,
                       "train_scores": train_scores,
                       "test_scores": test_scores}

    
    def fit(self, X, y=None):

        self.X_train = X.copy()
        self.y_train = y.copy()
        
        self.study = optuna.create_study(study_name='house_price_study', direction='maximize', sampler=optuna.samplers.RandomSampler())

        def objective(trial):

            eta = trial.suggest_float('eta', 0.01, 0.1)
            n_estimators = trial.suggest_int('n_estimators', 50, 1000)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            min_child_weight = trial.suggest_int('min_child_weight', 1, 6)
            subsample = trial.suggest_float('subsample', 0.7, 1)
            #colsample_bytree = trial.suggest_float('colsample_bytree', 0.7, 1)

            params = {'objective': 'reg:gamma',
                      'verbosity': 0,
                      'random_state': self.random_state,
                      'eta': eta,
                      'n_estimators': n_estimators,
                      'max_depth': max_depth,
                      'min_child_weight': min_child_weight,
                      'subsample': subsample}
                     # 'colsample_bytree': colsample_bytree,

            pipe = self.make_pipeline(**params)
            prep_pipeline = Pipeline(pipe.steps[:-1])
            estimator = pipe.named_steps['xgb']

            opt_cv_results = []

            logging.getLogger('src.feature_engineering.feat_eng').disabled = True
            logging.getLogger('src.feature_selection.rfe').disabled = True

            for nfold, (train_idx, test_idx) in enumerate(self.kfold_generator(self.X_train, n_splits=2)):
                cv_results_dict = {}

                X_train_transformed = prep_pipeline.fit_transform(self.X_train.iloc[train_idx], self.y_train.iloc[train_idx])
                X_valid_transformed = prep_pipeline.transform(self.X_train.iloc[test_idx])

                results = cross_validate(estimator, 
                                         X_train_transformed, 
                                         self.y_train.iloc[train_idx],
                                         cv=self.kfold_generator(X_train_transformed, n_splits=self.n_kfold_splits),
                                         scoring='neg_mean_gamma_deviance',
                                         return_estimator=True,
                                         return_train_score=True, error_score='raise')

                test_ensemble_pred = np.array([results['estimator'][x].predict(X_valid_transformed) for x in range(len(results['estimator']))]).mean(axis=0)
                test_score = -gamma_deviance(self.y_train.iloc[test_idx], test_ensemble_pred)

                cv_results_dict[f'fold_{nfold}_train_score'] = results['train_score'].mean()
                cv_results_dict[f'fold_{nfold}_valid_score'] = results['test_score'].mean()
                cv_results_dict[f'fold_{nfold}_test_score'] = test_score

                for k, v in cv_results_dict.items(): 
                    trial.set_user_attr(k, v)

                opt_cv_results.append(np.mean(results['test_score']))

            return np.mean(opt_cv_results)

        self.study.optimize(objective, n_trials=self.n_trials)
        self.fit_and_evaluate_model()

        logger.info(f'CREATING THE FINAL TRAINING DATASET...')
        self.create_final_training_dataset(pd.concat([self.X_train, self.y_train], axis=1), self.test_df)

        logger.info(f'\nTRAINING THE FINAL MODEL...')
        logging.getLogger('src.feature_engineering.feat_eng').disabled = False
        logging.getLogger('src.feature_selection.rfe').disabled = False

        self.model_pipeline.fit(self.df.drop(self.target, axis=1), self.df[self.target])
 
        return self


    def predict(self, X):

        # Check if fit has been called
        check_is_fitted(self)

        return self.model_pipeline.predict(X)
