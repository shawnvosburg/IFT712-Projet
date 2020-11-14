from src.Dispatcher import run
import numpy as np
import itertools
import random
import functools

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

class DataManagerGridSearch():
    def __init__(self,seed,featureAugmenters,scalers,featureExtracters):
        self.seed = seed
        self.scalers = scalers
        self.featureAugmenters = featureAugmenters
        self.featureExtracters = featureExtracters

    def gridsearchGenerator(self):
        random.seed(self.seed)
        while True:
            cmds = []
            if(len(self.featureAugmenters) > 0): cmds.append(random.choice(self.featureAugmenters))
            if(len(self.scalers) > 0): cmds.append(random.choice(self.scalers))
            if(len(self.featureExtracters) > 0): cmds.append(random.choice(self.featureExtracters))
            yield { 'seed':self.seed, 
                    'cmds':cmds
            }




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


cgs = ClassifierGridSearch( classifier='KernelMethod', 
                            alpha= np.logspace(-9, np.log10(2), 20), 
                            kernel = ['rbf','linear','polynomial'], 
                            gamma =  np.logspace(-9,np.log10(2), 20)
                            )

for c,d in zip(cgs.gridsearchGenerator(),dgs.gridsearchGenerator()):
    print(run(DataManagementParams = d, ClassificationParams = c, StatisticianParams= ['Accuracy','Precision','Recall'], verbose = False))

