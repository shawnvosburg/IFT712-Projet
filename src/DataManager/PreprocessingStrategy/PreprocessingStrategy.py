#!/usr/bin/python
#-*- coding: utf-8 -*-
import json

class PreprocessingStrategy:
    def __init__(self,**kwargs):
        self._method = None
        self.hyperparams = {}

    def preprocess(self, data):
        """ Virtual Function that will return the pre-prosessed data"""
        return data

    def jsonify(self, ):
        #Return json object of itself
        return {'method':self.__class__.__name__}
