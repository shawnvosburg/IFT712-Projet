#!/usr/bin/python
#-*- coding: utf-8 -*-
import pandas as pd
import os
import pathlib
import matplotlib.image as mpimage
import PreprocessingStrategy as pps
import json
import uuid
import numpy as np
from sklearn.model_selection import KFold,train_test_split

DATA_PATH = str(pathlib.Path(__file__).parent.absolute()) + '\\..\\..\\data\\'
RAWDATA_PATH = DATA_PATH + '\\raw\\train.csv'
PROCESSEDDATA_FOLDER = DATA_PATH + '\\processed\\'
PROCESSED_FILENAME = 'processed.json'

class DataManager:
    def __init__(self, seed = 0, cmds = []):
        self._df = None
        self._labels = None
        self.label_name = ""
        self._seed = seed
        self.cmds = cmds

        self._train_indexes = None 
        self._test_indexes = None
    
    # Handeling of common sub-indexing
    @property
    def train_indexes(self):
        if(self._train_indexes is None): return self._df.index
        else:                            return self._train_indexes
    @train_indexes.setter
    def train_indexes(self,val):
        self._train_indexes = val
    @property
    def test_indexes(self):
        if(self._test_indexes is None):  return []
        else:                            return self._test_indexes
    @test_indexes.setter
    def train_indexes(self,val):
        self._test_indexes = val

    # Common datasubsets 
    @property
    def df_Train(self):
        return self._df.loc[self.train_indexes]
    @property
    def df_Test(self):
        return self._df.loc[self.test_indexes]
    @property
    def labels_Train(self):
        return self._labels.loc[self.train_indexes]
    @property
    def labels_Test(self):
        return self._labels.loc[self.test_indexes]

    def importAndPreprocess(self,label_name,filepath = RAWDATA_PATH, savepath = PROCESSEDDATA_FOLDER):
        """ Checks if pre-processed data already saved. Loads pre-processed data if so. Else, generates preprocessed data. """
        #1. Check to see if data has already been preprocessed
         # Load already saved preprocessed data
        if(os.path.isfile(savepath + PROCESSED_FILENAME)):
            with open(savepath + PROCESSED_FILENAME) as f:
                preprocessedJson = json.load(f)
            for key in preprocessedJson:
                if(preprocessedJson[key] == self.cmds):
                    preprocessedPath = savepath + key
                    print('Loading saved preprocessed data from %s'%(preprocessedPath))
                    return self.importData(label_name = label_name,filepath = preprocessedPath)
        #Else, load raw data, preprocess and save
        self.importData(label_name=label_name,filepath=filepath)
        self.preprocess()
        self.saveData(savepath=savepath)

    def importData(self, label_name, filepath = RAWDATA_PATH):
        """Import data from file"""
        #Obtain features from train.csv
        # 1. Obtain csv of features
        self._df = pd.read_csv(filepath, index_col = 'id')

        # 2. Seperate labels
        self.label_name = label_name
        self._labels = self._df[self.label_name]
        del self._df[label_name]
        
    def preprocess(self):
        """Preprocess data. Save Preprocessed data to file."""

        # Convert cmd into a list of Preprocessing Strategies
        print('Commencing preprocessing...')
        for i,cmd in enumerate(self.cmds):
            print('\tMethod #%d:'%i,cmd)
            strategy = getattr(pps,cmd['method'])(**cmd['hyperparams'])
            self._df = strategy.preprocess(self._df)
        print('Done!')

    def saveData(self, savepath = PROCESSEDDATA_FOLDER):
        # Load already saved file
        if(not os.path.isfile(savepath + PROCESSED_FILENAME)):
            preprocessedJson = {}
        else:
            with open(savepath + PROCESSED_FILENAME) as f:
                preprocessedJson = json.load(f)
        
        # Add commands with uuid that referenced saved data
        uuidfilename = str(uuid.uuid1()) + '.csv'
        preprocessedJson[uuidfilename] = self.cmds
        
        # Dump to harddrive
        print('Saving preprocessed data to %s'%(savepath + uuidfilename))
        with open(savepath + PROCESSED_FILENAME,'w') as f:
            json.dump(preprocessedJson, f,indent=4)
        df_tobesaved = self._df.copy()
        df_tobesaved[self.label_name] = self._labels
        df_tobesaved.to_csv(savepath + uuidfilename)

    def CreateCommand(self,method = "",**kwargs):
        """ Function that converts arguments to JSON that will be interpreted by sub-classes. """
        hyperparams = {**kwargs}
        out = {'method' : method, 'hyperparams':hyperparams}
        self.cmds.append(out)

    def split_data(self,test_ratio = 0.1):
        #Seperates dataset into train and test datasets
        self._train_indexes, self._test_indexes = train_test_split(self._df.index, random_state = self._seed)

    def k_fold(self, k):
        """Returns generator that produces training, validation"""
        #Splits data
        if(self._train_indexes is None):    self.split_data(test_ratio=0.1)

        #Creates KFold generator
        kf = KFold(n_splits=k,shuffle=True,random_state=self._seed)
        
        #Create generator that spits out data and labels for train and validation datasets
        for train_idxes, val_idxes in kf.split(self.df_Train.to_numpy(), self.labels_Train.to_numpy()):
            train_data = self.df_Train.iloc[train_idxes]
            val_data = self.df_Train.iloc[val_idxes]
            train_labels = self.labels_Train.iloc[train_idxes]
            val_labels = self.labels_Train.iloc[val_idxes]
            yield  train_data, val_data, train_labels, val_labels

    def setSeed(self, seed):
        """Set the seed for the randomizer."""
        self._seed = seed


if __name__ == '__main__':
    METHOD = 0 
    if(METHOD == 0):
        #Testing IncludeImages, Normalize, PCA
        dm = DataManager()
        dm.CreateCommand(method='PolynomialFeatures',degree = 2)
        dm.CreateCommand(method='StandardScaler')
        dm.CreateCommand(method='PCA',n_components=None)

        #Testing Importing and Preprocessing Data
        dm.importAndPreprocess(label_name = 'species')

        #Spliting data into [Train & Validation] & [Test] datasets
        dm.setSeed(0) #Important in order to always have same test set
        dm.split_data(test_ratio=0.1)

        #Getting K-fold datasets
        for i, (X_train, X_val, Y_train, Y_val) in enumerate(dm.k_fold(k=10)):
            print('K =', i)
            print('\tAmount of Train observations:', len(X_train))
            print('\tAmount of Validation observations:', len(X_val))
    else:
        #Another valid method
        cmds = [
            {   'method':'IncludeImages',
                'hyperparams':{}
            },
            {   'method':'StandardScaler',
                'hyperparams':{}
            },
            {   'method':'PCA',
                'hyperparams':{
                    'n_components':None
                }
            }
        ]
        dm2 = DataManager(cmds=cmds)
        dm2.importAndPreprocess(label_name = 'species')
        dm2.setSeed(0)
        dm2.split_data(test_ratio=0.1)
        for i, (X_train, X_val, Y_train, Y_val) in enumerate(dm2.k_fold(k=10)):
            print('K =', i)
            print('\tAmount of Train observations:', len(X_train))
            print('\tAmount of Validation observations:', len(X_val))