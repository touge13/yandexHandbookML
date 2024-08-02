# MostFrequentClassifier

from sklearn.base import ClassifierMixin
import numpy as np
from scipy.stats import mode
    
class MostFrequentClassifier(ClassifierMixin):
    # Predicts the rounded (just in case) median of y_train
    def fit(self, X, y):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
            Training data features
        y : array like, shape = (_samples,)
            Training data targets
        '''
        self.mode_ = mode(y)[0]
        self.is_fitted_ = True
        return self

    def predict(self, X):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
            Data to predict
        '''
        return np.full(shape=X.shape[0], fill_value=self.mode_)