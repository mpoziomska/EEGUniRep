from eegunirep.models import BaseNet
from eegunirep.utils.utils import get_logger

from catboost import CatBoostClassifier, metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold


class FeatureClassification():

    def __init__(self, config, loglevel):
        self.logger = get_logger(name='FEAT_CLS', loglevel=loglevel)
        if config['feature_classification']['model']['name'] == 'catboost':
            self.model = CatBoostClassifier(thread_count=-1,
                                            eval_metric=metrics.AUC(),
                                            iterations=config['feature_classification']['model']['iterations'])
        else:
            raise NotImplementedError(f"Model {config['feature_classification']['model']['name']} is not implemented!")
        
        
        
    def train_eval(self, X_train, y_train, X_test, y_test):

        self.model.fit(X_train, y_train, verbose=False, eval_set=(X_test, y_test))

        return self.model.score(X_test, y_test)

    