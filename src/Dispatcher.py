"""
IFT712 Term Project

Shawn Vosburg
Ismail Idrissi

This file contains the logic to:
    1) Perform cross validation (train & validation sets only)
    2) Train fully on training set and predict on test set
"""

from src.DataManagement.Manager import DataManager
import src.Classifiers as classification
import src.Statistics as statistics
import numpy as np
import uuid
import pandas as pd

def runTestSet(DataManagementParams:dict, ClassificationParams:dict, StatisticianParams:list, verbose = False):
    """
    Trains on complete training set, predicts on test set and returns predictions with statistics. 

    Parameters
    ==========
    DataManagementParams: dict -> Parameters to send to the DataManagement module.
    ClassificationParams: dict -> Parameters to send to the Classifiers module.
    StatisticianParams: list -> Parameters to send to the Statistician module.

    Returns
    =======
    Results of test set with statistics. All in dictionary format.
    """
    cmd = {
        'DataManagementParams':DataManagementParams,
        'ClassificationParams':ClassificationParams,
        'StatisticianParams':StatisticianParams
    }

    # 1. Prepare data
    dm = DataManager(**DataManagementParams)
    dm.importAndPreprocess(label_name = 'species',verbose=verbose)
    dm.split_data(test_ratio=0.1)

    # 2. Create Statistician
    stats = statistics.Statistician(StatisticianParams)

    # 3. Train
    if(verbose): print('Fitting model...',end='')
    clf = classification.getClassifier(**ClassificationParams)
    clf.fit(dm.df_Train,dm.labels_Train)
    if(verbose): print('Done!')

    # 4. Prediction
    if(verbose): print('Prediciting Test Set...',end='')
    predictions = clf.predict(dm.df_Test.values)
    if(verbose): print('Done!')

    # 5. Statistics
    stats.appendLabels(predictions, dm.labels_Test.values)
    return {'predictions':predictions,'truths':dm.labels_Test.values, 'metrics':stats.getStatistics()}



def run(DataManagementParams:dict, ClassificationParams:dict, StatisticianParams:list, verbose = False):
    """
    Launches a machine learning classification evaluation using cross validation

    Parameters
    ==========
    DataManagementParams: dict -> Parameters to send to the DataManagement module.
    ClassificationParams: dict -> Parameters to send to the Classifiers module.
    StatisticianParams: list -> Parameters to send to the Statistician module.

    Returns
    =======
    Results with hyperparameters in dictionary format
    """
    cmd = {
        'DataManagementParams':DataManagementParams,
        'ClassificationParams':ClassificationParams,
        'StatisticianParams':StatisticianParams
    }

    # 1. Prepare data
    dm = DataManager(**DataManagementParams)
    dm.importAndPreprocess(label_name = 'species',verbose=verbose)

    # 2. Create Statistician
    stats = statistics.Statistician(StatisticianParams)

    # 3. Perform K-fold
    if(verbose): print('Performing K-fold....',end='')
    for train_data, val_data, train_labels, val_labels in dm.k_fold(k=10):
        
        # 4. Create Classifier
        clf = classification.getClassifier(**ClassificationParams)

        # 5. Fit classifier with training data and labels
        clf.fit(train_data, train_labels)

        # 6. Get predictions
        predictions = clf.predict(val_data)

        # 7. Add labels to statistician
        stats.appendLabels(predictions, val_labels.values)

    if(verbose): print('Done!')
    # 8. Calculate average statistics
    statisticsJson = stats.getStatistics()
    
    stats_name = str(uuid.uuid1())
    return {stats_name:{'pipeline':cmd,'results':statisticsJson}}
        

if __name__ == '__main__':
    cmd = {
        "DataManagementParams": {
                "seed": 160743167,
                "cmds": [
                    {
                        "method": "StandardScaler",
                        "hyperparams": {}
                    },
                    {
                        "method": "PCA",
                        "hyperparams": {
                            "n_components": 100
                        }
                    }
                ]   
        },
        #CHOOSE ONLY 1.  Note that we index at the end of the list, so no list is actually sent as a param.
        # The reason we did this is to show the different possibilities. 
        'ClassificationParams': [
            {
            'classifier': 'SVM',
            'C': 12,                                    # Regularization parameter.
            'kernel': 'poly',                           # {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
            'degree' : 3,                               # Degree of the polynomial kernel function (‘poly’)
            # gamma : {‘scale’, ‘auto’} or float        # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid
            },
            {
            'classifier': 'NeuralNetwork',
            'activation': 'relu',                   # activation {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
            'solver': 'adam',                           # solver {‘lbfgs’, ‘sgd’, ‘adam’}
            'alpha': 0.001,                             # regularization parameter
            'learning_rate': 'invscaling',              # learning_rate{‘constant’, ‘invscaling’, ‘adaptive’}
            'max_iter': 1000
            },
            {
            'classifier': 'LogisticRegression',
            'solver': 'liblinear',                      # solver {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}
            'random_state': 0,                          # Control randomness
            'penalty': 'l2',                            # penalty {‘l1’, ‘l2’, ‘elasticnet’, ‘none’}
            'tol': 1e-3,                                # Tolerance for stopping criteria
            'C': 2.5,                                   # regularization parameter
            },
            {
            'classifier':'Perceptron',
            'loss': 'perceptron',
            'penalty' : 'l2',
            'alpha': 0.01,                              # Regularization parameter
            'learning_rate': 'invscaling',              # learning_rate {‘constant’,‘optimal’, ‘invscaling’, ‘adaptive’}
            'eta0': 1,                                  # Constant by which the updates are multiplied
            },
            {
            'classifier': 'KernelMethod',
            'alpha': 0.0001,
            'kernel': 'rbf',
            'gamma': 0.001                              # gamma defines how much influence a single training example has
            },
            {
            'classifier': 'GenerativeModel'
            }
        ][5], #IMPORTANT! INDEXING HERE, DO NOT ACTUALLY FEED IN A LIST
        'StatisticianParams':[
            'Accuracy','Precision','Recall'#,'ConfusionMatrix'
        ]
    }

    print(run(**cmd, verbose=True))