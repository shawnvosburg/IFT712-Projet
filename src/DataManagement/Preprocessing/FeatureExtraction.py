#!/usr/bin/python
#-*- coding: utf-8 -*-

from src.DataManagement.Preprocessing import PreprocessingStrategy
import re

class FeatureExtraction(PreprocessingStrategy):
    def __init__(self,**kwargs):
        super().__init__()
        self._method = None
        self.hyperparams = {}
        self.hyperparams.update(**kwargs)

    def preprocess(self, data):
        #Check to see if all columns are requested.
        out = data
        if ('columns' in self.hyperparams):
            r = re.compile(self.hyperparams['columns'])
            out = out[list(filter(r.match,out.columns))]
        if ('index' in self.hyperparams):
            r = re.compile(self.hyperparams['index'])
            out = out[list(filter(r.match,out.index))]
        return out

    def jsonify(self, ):
        out = super().jsonify()
        out.update(**{'hyperparams':self.hyperparams})
        return out

