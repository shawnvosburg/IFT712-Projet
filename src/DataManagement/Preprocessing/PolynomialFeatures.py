#!/usr/bin/python
#-*- coding: utf-8 -*-

import pandas as pd
from src.DataManagement.Preprocessing import PreprocessingStrategy
from sklearn.preprocessing import PolynomialFeatures as _PolynomialFeatures

class PolynomialFeatures(PreprocessingStrategy):
    def __init__(self,**kwargs):
        super().__init__()
        self._method = _PolynomialFeatures(**kwargs)
        self.hyperparams = self._method.get_params()
        
    def preprocess(self, data):
        """ Return the transformed data """
        return pd.DataFrame(self._method.fit_transform(data), columns = self._method.get_feature_names(data.columns), index = data.index)

    def jsonify(self):
        out = super().jsonify()
        out.update(**{'hyperparams':self.hyperparams})
        return out
