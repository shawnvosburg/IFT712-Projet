#!/usr/bin/python
#-*- coding: utf-8 -*-

class Statistic:
    """
    Class that calculates a model's performance statistics. 

    Calculation produced will be used to rank models.
    """
    def __init__(self):
        self.pred = None
        self.truth = None

    def calculateAccuracy(self, ):
        """
        Calculate accuracy score from "pred" and "truth".
        Both parameters must be N-sized lists.
        """
        pass

    def getConfusionMatrix(self, ):
        """Returns confusion matrix from predictions and truth labels. Arguments "pred" and "truth" must both be an N-sized list."""
        pass

    def getStatistics(self, ):
        """Returns all statistics in a JSON object"""
        pass

