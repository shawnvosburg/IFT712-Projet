#!/usr/bin/python
#-*- coding: utf-8 -*-

from src.Classifiers import Classifier

class SVM(Classifier):
    def __init__(self):
        self._model = None

