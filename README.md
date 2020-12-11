IFT712-Projet
==============================

End of semester project of IFT712 for USherbrooke.  
Program written by: **Shawn Vosburg and Ismail Idrissi**

Problem sypnosis
----------------
In the context of the **IFT712 - Techniques d'apprentissage** course offered at the University of Sherbrooke,
students in groups of 1 or 2 are asked to test the performance of 6 different machine learning algorithms on 
a Kaggle dataset. The choice of algorithms is left up to the students. The sklearn python library is to be used for the algorithm implementations.  

The goal of the project is to pratice proper machine learning coding conventions such as: cross-validation,
hyperparameter optimization (grid-search). These techniques are standard in the Machine Learning community to
select the best algorithm with hyperparameters for a specific classification task.

Setup
---------------------------

### Unpacking the Leaf Dataset  

A leaf classification dataset is packed with this git repository.
A description of the dataset can be found on Kaggle at https://www.kaggle.com/c/leaf-classification/data    


#### WINDOWS:
To unpack dataset, navigate to the root project and run:

```bash
make extract-leaf
```

#### Linux/UNIX:
To unpack dataset, either modify the root makefile to use **python3** or run from root folder:
```bash
python3 ./dataset/leaf-classification.py
```

### Installing dependencies 

To run the code, certain python libraries must be installed. To do so, we recommend creating a virtual
python environment. A simple guide is available here: https://docs.python.org/3/library/venv.html

Once inside the virtual environment, the necessary packages can be installed with:
```
pip install -r requirements.txt
```

Validating proper setup
-----------------------
TODO: write tests in different folder using the unittest module.

To verify that the setup is complete, navigate to the project root folder and type in a terminal window:
```bash
python -m src.Dispatcher
```
The setup is correct if no errors appear. 

This test runs the following pipeline:
1. Preprocessing 
    1. A StandardScaler preprocessor that maps attributes to have ```mean = 0``` and ```variance = 1```.
    2. A PCA preprocessor that calculates and keeps the 100 first principal components.
2. Training model
    1. The classifier chosen is KernelMethod (in scikit-learn terminology, this is the KernelRidge algorithm). A RBF kernel is chosen with parameters ```alpha = 0.0001```
    and ```gamma = 0.001```.
3. Calculating performance statistics
    1. The accuracy, precision and recall metrics are calculated once the model is trained.

Notebooks
----------
Two notebooks are available under notebooks/. 
1. **DataExploration.ipynb**. We explore the different distributions of the data to try to hypothesize what type of classifier would work well with the data. 
2. **ResultsAnalysis.ipynb**. We analyse the results from our gridsearch performed on Google's Compute Engine. 

Brief description of modules
----------------------------
Our projet is sub-divided in **4 major modules**.
### 1. DataManagement module

The DataManagement module is responsible of loading and preprocessing the dataset. It also saves the preprocessed data to avoid 
unnecessary reprocessing. A submodule called **Manager** contains the class **DataManager** that serves as the interface with the other parts of the program. 
Other responsibilities of the module is to seperate the dataset into train, validation and test sets during cross-validation.

The DataManager class can be instantiated with parameter **cmds**, which describes the preprocessing steps that the dataset must go through.
Here is an example of a DataManager object:
```python
cmds = [
            {   'method':'IncludeImages',
                'hyperparams':{}
            },
            {   'method':'StandardScaler',
                'hyperparams':{}
            },
            {   'method':'PCA',
                'hyperparams':{
                    'n_components':100
                }
            }
        ]
dm = DataManager(seed=123456, cmds=cmds)
```
In the above example, the DataManager is setup to preprocess the data by:
1. Including the flatten images.
2. StandardScaling all attributes (i.e. ```mean = 0``` and ```variance = 1```)
3. Keeping only the 100 first principal components.

### 2. Classifiers module

This module is responsible for implementing all supported learning algorithms. Given our problem domain, the classifier module is essentially
a wrapper module for the chosen scikit-learn learning algorithms. All classifiers have a fit(X,Y) and predict(X) methods.
Classifiers should also be able to handle class labels that are strings.

### 3. Statistics module

This module is reponsible for calculating the performance of a model using pre-defined metrics. Its main class, **Statistician**,
handles all the metrics calculations.  
Here is an example use:
```python
stats = Statistician(['Accuracy','Precision','Recall'])
stats.appendLabels(predictions = [0,0,1,1,1,2,2,2], truths = [1,0,1,1,1,1,2,0])
stats.getStatistics()

# OUT[1]: {'Accuracy': 0.625, 'Precision': 0.611111111111111, 'Recall': 0.7000000000000001}
```

### 4. Dispatcher module

This module is responsible for establishing the pipeline of the program. Its function **run()** calls
the DataManager to preprocess the data. It then performs KFold cross-validation with the Classifiers module. Finally, it uses a
Statistician to mesure performance. 

This module is the interface of our algorithm evaluatation program. Users should only import the Dispatcher module to evaluate models.

Acknowledgement
----------------
<p><small>Project file structure based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
