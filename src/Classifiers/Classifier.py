#!/usr/bin/python
#-*- coding: utf-8 -*-

class Classifier:
    """Base Class for all classifiers."""
    def __init__(self):
        self._model = None

    def fit(self, ):
        """Training model"""
        pass

    def predict(self, X):
        """Returns prediction label for X."""
        pass

    def getWeights(self, ):
        """Returns model's weights."""
        pass

