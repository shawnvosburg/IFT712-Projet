#!/usr/bin/python
#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from io import StringIO
from sklearn.metrics import confusion_matrix as _confusion_matrix
from sklearn.metrics import precision_score as _precision_score
from sklearn.metrics import recall_score as _recall_score

class Statistician:
    """
    Class that calculates a model's performance statistics. 

    Calculation produced will be used to rank models.
    """
    def __init__(self,cmd = []):
        """
        Parameters
        ==========
        None.
        """
        self._preds = []
        self._truths = []
        self._cmd = cmd
        
        #Needs updating when implementing new metric
        self._methodDict = {
            'Accuracy': self._calculateAccuracy,
            'Precision': self._calculatePrecision,
            'Recall': self._calculateRecall,
            'ConfusionMatrix': self._getConfusionMatrix
        }


    @property
    def preds(self):
        return np.concatenate(self._preds)
    @property
    def truths(self):
        return np.concatenate(self._truths)

    def appendLabels(self,predictions,truths):
        """
        Add new entry to statistician

        Parameters
        ==========
        prediction: list -> Predicition labels
        truths: list -> Truth Labels

        Returns
        =======
        void.
        """
        self._preds.append(predictions)
        self._truths.append(truths)

    def _calculateAccuracy(self,):
        """
        Calculate accuracy score from "predictions" and "truth".

        Parameters
        ==========
        None.

        Returns
        =======
        float: Accuracy score in range [0.0, 1.0]
        """
        # 1. Linearize all labels
        return np.mean(self.preds == self.truths)

    def _calculatePrecision(self):
        """
        Calculate precision score from "predictions" and "truth".

        Parameters
        ==========
        None.

        Returns
        =======
        float: Accuracy score in range [0.0, 1.0]
        """
        return _precision_score(self.truths,self.preds,average='micro')
    
    def _calculateRecall(self):
        """
        Calculate recall score from "predictions" and "truth".

        Parameters
        ==========
        None.

        Returns
        =======
        float: Accuracy score in range [0.0, 1.0]
        """
        return _recall_score(self.truths,self.preds, average = 'micro')
        


    def _getConfusionMatrix(self, ):
        """
        Calculates confusion matrix from predictions and truth labels.
        
        Parameters
        ==========
        None.

        Returns
        =======
        numpy 2-d array. 
        """
        csv_buffer = StringIO()
        labels = set(self.truths) | set(self.preds)
        cm = _confusion_matrix(self.truths,self.preds,labels = list(labels))
        pd.DataFrame(cm, columns=labels, index=labels).to_csv(csv_buffer)
        return csv_buffer.getvalue()

    def getStatistics(self,):
        """
        Returns all statistics in a JSON object

        Parameters
        ==========
        None.

        Returns
        =======
        float: Accuracy score in range [0.0, 1.0]
        """
        out = {}

        # Get all wanted metric
        for metric in self._cmd:
            out[metric] = self._methodDict[metric]()

        return out

