#!/usr/bin/python
#-*- coding: utf-8 -*-

import pandas as pd
from PreprocessingStrategy import PreprocessingStrategy
from sklearn.preprocessing import StandardScaler as _StandardScaler

class StandardScaler(PreprocessingStrategy):
    def __init__(self,**kwargs):
        super().__init__()
        self._method = _StandardScaler(**kwargs)
        self.hyperparams = self._method.get_params()
        
    def preprocess(self, data):
        """ Return the transformed data """
        return pd.DataFrame(self._method.fit_transform(data), columns = data.columns, index = data.index)

    def jsonify(self):
        out = super().jsonify()
        out.update(**{'hyperparams':self.hyperparams})
        return out
