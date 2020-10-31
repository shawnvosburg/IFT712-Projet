#!/usr/bin/python
#-*- coding: utf-8 -*-
import pandas as pd
import os
import matplotlib.image as mpimage

DATA_PATH = '../../data/'
RESIZED_IMG = ()

class DataManager:
    def __init__(self, seed = None):
        self._df = None
        self._labels = None
        self._seed = seed

    def importData(self, filepath = DATA_PATH):
        """Import data from file"""
        # Importing Raw Data
        raw_filepath = filepath + 'raw/'
        
        #Idea: Obtain features from CSV and concatenate linearized images.
        self._df = pd.read_csv(raw_filepath + 'train.csv', index_col = 'id')
        images = []
        raw_image_path = raw_filepath + 'images/'
        for img_name in os.listdir(raw_image_path):
            img = mpimage.imread(raw_image_path + img_name)
            images.append( (img_name,img) )
        
    def preprocess(self, savepath):
        """Preprocess data. Save Preprocessed data to file."""
        pass

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
    #Test and show pre-processed data
    dm = DataManager()

    dm.importData()
