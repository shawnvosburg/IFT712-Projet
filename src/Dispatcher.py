from src.DataManagement.Manager import DataManager
import src.Classifiers as classification
import src.Statistics as statistics
import pathlib
import json
import uuid
import os

SAVEPATH = str(pathlib.Path(__file__).parent.absolute()) + '/../models/results/'
RESULTS_FILENAME =  'results.json'

def run(DataManagementParams:dict, ClassificationParams:list, StatisticianParams:list, savepath = SAVEPATH):
    """
    Launches a machine learning classification evaluation

    Parameters
    ==========
    DataManagementParams: dict -> Parameters to send to the DataManagement module.
    ClassificationParams: dict -> Parameters to send to the Classifiers module.
    StatisticianParams: list -> Parameters to send to the Statistician module.

    Returns
    =======
    void.
    """
    # 0. Check to see if work was already done.
    resultsJson = {}
    if(os.path.isfile(savepath + RESULTS_FILENAME)):
            with open(savepath + RESULTS_FILENAME) as f:
                resultsJson = json.load(f)
            for key in resultsJson:
                if(resultsJson[key] == cmd):
                    statisticsPath = savepath + key
                    print('Loading saved statistics data from %s'%(statisticsPath))
                    with open(statisticsPath) as f:
                        statisticsJson = json.load(f)
                    return statisticsJson

    # 1. Prepare data
    dm = DataManager(**DataManagementParams)
    dm.importAndPreprocess(label_name = 'species')

    # 2. Create Statistician
    stats = statistics.Statistician(StatisticianParams)

    # 3. Perform K-fold
    print('Performing K-fold....',end='')
    for train_data, val_data, train_labels, val_labels in dm.k_fold(k=10):
        
        # 4. Create Classifier
        clf = classification.getClassifier(**ClassificationParams[0])

        # 5. Fit classifier with training data and labels
        clf.fit(train_data, train_labels)

        # 6. Get predictions
        predictions = clf.predict(val_data)

        # 7. Add labels to statistician
        stats.appendLabels(predictions, val_labels.values)

    print('Done!')
    # 8. Calculate average statistics
    statisticsJson = stats.getStatistics()

    # 9. Save results
    if(savepath is not None):
        stats_filename = str(uuid.uuid1()) + '.json'

        #Update results.json
        resultsJson.update({stats_filename:cmd})
        with open(savepath + RESULTS_FILENAME, 'w') as f:
            json.dump(resultsJson,f,indent=4)
        
        #save results
        print('Saving statistics in file',savepath + stats_filename)
        with open(savepath + stats_filename,'w') as f:
            json.dump(statisticsJson,f,indent=4)
    

    return statisticsJson
        

if __name__ == '__main__':
    cmd = {
        'DataManagementParams':{
            'seed': 0,
            'cmds': [
                {   'method':'StandardScaler',
                    'hyperparams':{}
                },
                {   'method':'PCA',
                    'hyperparams':{
                        'n_components':100
                    }
                }
            ]   
        },
        'ClassificationParams': [
            {
            'classifier': 'SVM',
            'C': 12,                                    # Regularization parameter.
            'kernel': 'poly',                           # {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
            'degree' : 3,                               # Degree of the polynomial kernel function (‘poly’)
            # gamma : {‘scale’, ‘auto’} or float        # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid
            },
            {
            'classifier': 'LogisticRegression',
            'solver': 'liblinear',                      # solver {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}
            'random_state': 0,                          # Control randomness
            'penalty': 'l2',                            # penalty {‘l1’, ‘l2’, ‘elasticnet’, ‘none’}
            'tol': 1e-3,                                # Tolerance for stopping criteria
            'C': 2.5,                                   # Regularization parameter
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
            }
        ],
        'StatisticianParams':[
            'Accuracy','Precision','Recall'#,'ConfusionMatrix'
        ]
    }

    print(run(**cmd))