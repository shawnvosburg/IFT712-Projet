#!/usr/bin/python
#-*- coding: utf-8 -*-

from Classifier import Classifier

class KernelMethod(Classifier):
    def __init__(self):
        self._model = None

