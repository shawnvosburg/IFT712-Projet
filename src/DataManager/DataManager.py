#!/usr/bin/python
#-*- coding: utf-8 -*-
import pandas as pd
import os
import pathlib
import matplotlib.image as mpimage
import PreprocessingStrategy as pps
import json
import uuid

DATA_PATH = str(pathlib.Path(__file__).parent.absolute()) + '\\..\\..\\data\\'
RAWDATA_PATH = DATA_PATH + '\\raw\\train.csv'
PROCESSEDDATA_FOLDER = DATA_PATH + '\\processed\\'
PROCESSED_FILENAME = 'processed.json'

class DataManager:
    def __init__(self, seed = None):
        self._df = None
        self._labels = None
        self.label_name = ""
        self._seed = seed
        self.cmds = []

    def importAndPreprocess(self,label_name,filepath = RAWDATA_PATH, savepath = PROCESSEDDATA_FOLDER):
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

    def k_fold(self, k):
        """Returns generator that produces training, validation and test sets"""
        pass

    def getDataSubset(self, indexes):
        """Returns a subset of the data that corresponds to the indexes passed as a parameter."""
        pass

    def setSeed(self, seed):
        """Set the seed for the randomizer."""
        self._seed = seed


if __name__ == '__main__':
    #Testing IncludeImages, Normalize, PCA
    dm = DataManager()
    dm.CreateCommand(method='IncludeImages')
    dm.CreateCommand(method='StandardScaler')
    dm.CreateCommand(method='PCA',n_components=None)

    #Testing Importing and Preprocessing Data
    dm.importAndPreprocess(label_name = 'species')