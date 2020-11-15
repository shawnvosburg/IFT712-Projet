from src.Dispatcher import run
import numpy as np
import itertools
import random
import functools
import pathlib
import os
import json

SAVEPATH = str(pathlib.Path(__file__).parent.absolute()) + '/./models/results/'
RESULTS_FILENAME =  'results.json'

# Define a classifier gridsearch generator
class ClassifierGridSearch():
    def __init__(self,classifier,**hyperparams):
        self.classifier = classifier
        self.hyperparams = hyperparams

    def gridsearchGenerator(self):
        keys = self.hyperparams.keys()
        vals = self.hyperparams.values()
        for validTuple in itertools.product(*vals):
            yield {'classifier':self.classifier, **dict(zip(keys,validTuple))}

    def __len__(self):
        vals = self.hyperparams.values()
        return functools.reduce(lambda count,x: count*len(x),vals, 1)

# Define a DataManager gridsearch generator
class DataManagerGridSearch():
    def __init__(self,seed,featureAugmenters,scalers,featureExtracters):
        self.seed = seed
        self.scalers = scalers
        self.featureAugmenters = featureAugmenters
        self.featureExtracters = featureExtracters

    def gridsearch(self):
        cmds = []
        if(len(self.featureAugmenters) > 0): cmds.append(self.featureAugmenters)
        if(len(self.scalers) > 0): cmds.append(self.scalers)
        if(len(self.featureExtracters) > 0): cmds.append(self.featureExtracters)
        return list(map(lambda x: {'seed':self.seed, 'cmds':list(x)}, list(itertools.product(*cmds))))

def saveDict(obj,savepath = SAVEPATH + RESULTS_FILENAME):
    with open(savepath, 'w') as f:
        json.dump(obj,f)


if __name__ == '__main__':
    # 0. Load results json file
    if(os.path.isfile(SAVEPATH + RESULTS_FILENAME)):
            with open(SAVEPATH + RESULTS_FILENAME) as f:
                resultsJson = json.load(f)
    else:
        resultsJson = {}

    dgs = DataManagerGridSearch(seed = 16082604, 
                                featureAugmenters = [], #No data augmentation because it takes too much place on disk
                                scalers = [
                                    {   'method':'Normalize',
                                        'hyperparams':{}
                                    },
                                    {   'method':'StandardScaler',
                                        'hyperparams':{}
                                    }
                                ],
                                featureExtracters = [
                                    {   'method':'PCA',
                                        'hyperparams':{'n_components':100}
                                    },
                                    {   'method':'FeatureExtraction',
                                        'hyperparams':{'columns':r'^\w*\d*[02468]$'}
                                    }
                                ]
                                ) 

    #Must create a gridsearch generator for every classifier
    cgsKernelMethod = ClassifierGridSearch( classifier='KernelMethod', 
                                            alpha= np.logspace(-9, np.log10(2), 20), 
                                            kernel = ['rbf','linear','poly'], 
                                            gamma =  np.logspace(-9,np.log10(2), 20)
                                            )
    cgsGenerativeModel = ClassifierGridSearch( classifier='GenerativeModel')
    cgsLogisticRegression = ClassifierGridSearch(   classifier = 'LogisticRegression',
                                                    solver= ['liblinear'],                      # solver {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}
                                                    random_state = [0],                          # Control randomness
                                                    penalty = ['l2'],                            # penalty {‘l1’, ‘l2’, ‘elasticnet’, ‘none’}
                                                    tol = np.logspace(-4,np.log10(2), num=20), # Tolerance for stopping criteria
                                                    C = np.logspace(-4,4,num=20),              # regularization parameter
                                                )
    cgsNeuralNetwork = ClassifierGridSearch(    classifier = 'NeuralNetwork',
                                                hidden_layer_sizes=[(100), (200), (300)],
                                                activation = ['relu','tanh','logistic'],     # activation {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
                                                solver = ['adam'],                           # solver {‘lbfgs’, ‘sgd’, ‘adam’}
                                                alpha = np.logspace(-9,np.log10(2),num=20), # regularization parameter
                                                learning_rate = ['invscaling'],              # learning_rate{‘constant’, ‘invscaling’, ‘adaptive’}
                                                max_iter= [1000]
                                                )
    cgsPerception = ClassifierGridSearch(   classifier = 'Perceptron',
                                            loss = ['perceptron'],
                                            penalty = ['l2'],
                                            alpha =  np.logspace(-9,np.log10(2),num=20),   # Regularization parameter
                                            learning_rate = ['invscaling'],              # learning_rate {‘constant’,‘optimal’, ‘invscaling’, ‘adaptive’}
                                            eta0 = [1],                                  # Constant by which the updates are multiplied
                                                )
    cgsSVM = ClassifierGridSearch(  classifier = 'SVM',
                                    C = np.logspace(-4,4,num=20),             # Regularization parameter.
                                    kernel = ['rbf','linear','poly','sigmoid'], # {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
                                    degree = [2],                               # Degree of the polynomial kernel function (‘poly’)
                                    gamma =  np.logspace(-9,np.log10(2), 20),      # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid
                                                )

    # Create Generatore that generates hyperparameter values
    cgs = itertools.chain(*map(lambda x: x.gridsearchGenerator(), [cgsKernelMethod, cgsGenerativeModel, cgsLogisticRegression, cgsNeuralNetwork, cgsPerception, cgsSVM]))
    gen = itertools.product(cgs,dgs.gridsearch())
    
    #Advance generator to right place 
    for _ in range(len(resultsJson)):
        next(gen)

    #For each hyperparam combination that hasnt been run, run model and update results
    for i,(c,d) in enumerate(gen, start = len(resultsJson)):
        #Checkpoint
        if(i%100 == 0): 
            print('Combination %d'%i)
            saveDict(resultsJson)
        resultsJson.update(run(DataManagementParams = d, ClassificationParams = c, StatisticianParams= ['Accuracy','Precision','Recall'], verbose = False))
    
    #Final save of values
    saveDict(resultsJson)


