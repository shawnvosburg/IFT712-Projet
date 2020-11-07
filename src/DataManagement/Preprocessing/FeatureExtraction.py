#!/usr/bin/python
#-*- coding: utf-8 -*-

from src.DataManagement.Preprocessing import PreprocessingStrategy

class FeatureExtraction(PreprocessingStrategy):
    def __init__(self,**kwargs):
        super().__init__()
        self._method = None
        self.hyperparams = {"columns" : 'all'}
        self.hyperparams.update(**kwargs)

    def preprocess(self, data):
        #Check to see if all columns are requested.
        if (self.hyperparams['columns'] == 'all'):
            return data
        else:
            return data[self.hyperparams['columns']]

    def jsonify(self, ):
        out = super().jsonify()
        out.update(**{'hyperparams':self.hyperparams})
        return out

