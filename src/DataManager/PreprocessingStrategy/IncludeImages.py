#!/usr/bin/python
#-*- coding: utf-8 -*-

from PreprocessingStrategy import PreprocessingStrategy
import pathlib
import os
import pandas as pd
from PIL import Image
import numpy as np

RESIZE_SHAPE = (64,64)
DATA_PATH = str(pathlib.Path(__file__).parent.absolute()) + '\\..\\..\\..\\data\\'

class IncludeImages(PreprocessingStrategy):
    def __init__(self,**kwargs):
        super().__init__()
        self._method = None
        self.hyperparams = {"Images_Path" : 'raw\\images\\'}
        self.hyperparams.update(**kwargs)

    def preprocess(self, data):
        IMAGES_PATH = DATA_PATH + self.hyperparams['Images_Path']

        #Prepare Image DataFrame
        columns = ['Pixel_%d'%i for i in range(np.prod(RESIZE_SHAPE))]
        img_df = pd.DataFrame(columns = columns)
        
        #Import image and insert in DataFrame
        for filename in os.listdir(IMAGES_PATH):
            img = Image.open(IMAGES_PATH + filename)
            img = np.array(img.resize(RESIZE_SHAPE)).flatten()
            img_df.loc[int(filename.strip('.jpg'))] = img
            
        return data.join(img_df)

    def jsonify(self, ):
        out = super().jsonify()
        out.update(**{'hyperparams':self.hyperparams})
        return out

