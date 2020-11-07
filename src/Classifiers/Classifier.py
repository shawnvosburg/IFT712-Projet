#!/usr/bin/python
#-*- coding: utf-8 -*-
import pandas as pd
import sys

def getClassifier(classifier:str, **hyperparams ):
    """ 
    Searches environement for definition of classifier

    Parameters
    ==========
    classifier:str -> Name of the classifier class.
    hyperparams: dict -> Arguments to classifier

    Returns
    =======
    Instance of chosen classifier
    """
    return getattr(sys.modules[__name__],classifier)(**hyperparams)

class Classifier:
    """Base Class for all classifiers."""
    def __init__(self):
        self._model = None

    def fit(self, data:pd.DataFrame):
        """Training model"""
        pass

    def predict(self, X:pd.DataFrame):
        """Returns prediction label for X."""
        pass


if __name__ == '__main__':
    from src.Classifiers import *
    from src.DataManagement.Manager import DataManager

    #Create DataManager object
    dm = DataManager()
    dm.CreateCommand(method='StandardScaler')
    dm.CreateCommand(method='PCA',n_components=None)
    dm.importAndPreprocess(label_name = 'species')
    dm.split_data()

    #Create classifier
    clf = getClassifier(classifier='KernelMethod', alpha = 0.001, kernel = 'rbf')

    #Fit and predict
    clf.fit(dm.df_Train, dm.labels_Train)
    predictions = clf.predict(dm.df_Test)

    #Calculate accuracy score
    #In project, this will be handled by the Statistician package
    accuracyList = (predictions == dm.labels_Test.values)
    accuracy = sum(accuracyList) / len(accuracyList)
    print('Accuracy: {:.3f}%'.format(accuracy))
