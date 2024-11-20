"""
Copyright (c) 2024 Federico Ceriani, University of Sheffield.


A set of utilities for analyzing Auditory Brainstem Response (ABR) data.
This module provides functions for loading, preprocessing, analyzing and visualizing ABR data.

Author: Federico Ceriani
Email: f.ceriani at sheffield.ac.uk
License: APACHE 2.0
"""

from pylab import *
import os
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from scipy import signal, stats
import datetime 
#Global variables
frequencies = [100,3000,6000,12000,18000,24000,30000,36000,42000]
lowestInt = 15 # dB

freqDict = {'100':'Click','3000':'3 kHz','6000':'6 kHz','12000':'12 kHz','18000':'18 kHz','24000':'24 kHz','30000':'30 kHz','36000':'36 kHz','42000':'42 kHz'}
intensityDict = {'0':0,'5':1,'10':2,'15':3,'20':4,'25':5,'30':6,'35':7,'40':8,'45':9,#
        '50':10,'55':11,'60':12,'65':13,'70':14,'75':15,'80':16,'85':17,'90':18,'95':19}   

fs = 195000.0/2.0 # Acquisition sampling rate



def extractABR(filename,removeDuplicates = True,saveConverted=False):
    def extractABR(filename, removeDuplicates=True, saveConverted=False):
        """
        Extract ABR (Auditory Brainstem Response) data from CSV files (output of Biosig).
        Parameters:
        filename (str): The path to the CSV file containing ABR data.
        removeDuplicates (bool): If True, removes duplicated data from the output DataFrame. Default is True.
        saveConverted (bool): If True, saves the converted file as an ordered CSV file in the same folder. Default is False.
        Returns:
        pd.DataFrame: A DataFrame containing the extracted ABR data, indexed by frequency and intensity.
        Notes:
        - The function reads the CSV file line by line, extracting relevant data based on specific markers.
        - It handles cases where the data might be duplicated and removes such duplicates if `removeDuplicates` is set to True. This is useful when the same data is recorded multiple times.
        - If `saveConverted` is set to True, the function saves the extracted data as a new CSV file with 'converted' appended to the original filename.
        - The function assumes that the CSV file follows a specific format as output by Biosig.
        """

    f = open(filename,'r')
    l = f.readlines()
    out=[]
    header1=[]
    header2 = []
    nclicks = 0 # 70 dB 10 Clicks are recorded three times, we keep the one that is part of a stack
    for i,line in enumerate(l):

        if line.startswith('[Trace_'):
            nextL = l[i+1]
            s = nextL.split(',')
            try: 
                indicator = float(s[1])
                nextindex=2
            except:
                indicator = float(s[2])
                nextindex=3
            if s[0].endswith('Cal')==False and s[1].endswith('Cal')==False and indicator!=42001:

                frequency = indicator#float(s[1]) 
                
                intensity = float(s[nextindex])

                if frequency == 100 and intensity == 70:
                    nclicks = nclicks + 1
                if nclicks<3 or  not ((frequency==100) and (intensity == 70)):
                    header1.append(frequency)
                    header2.append(intensity)
                    nextL = l[i+2]
                    j=0
                    column =[]
                    while nextL.startswith('[Trace_')==False:

                        if nextL.startswith('TraceData_'):
                            s0 = nextL.split('=')[1]
                            s = s0.split(',')[:]
                            
                            for el in s:
                                try:
                                    column.append(float(el))
                                except ValueError:
                                    print("weird stuff goin on in "+filename)
                        j=j+1
                        try:
                            nextL=l[i+2+j]
                        except:
                            break

                    if column==[]:
                        print(frequency)
                    out.append(column)

        else:
            pass
    if saveConverted:
        table = np.vstack((header1,header2,np.array(out).T))
        np.savetxt(filename+'converted.csv',table,delimiter=',')

    pdOut = pd.DataFrame(out,index=[header1,header2]) 
    if removeDuplicates: # remove duplicated data
        t2 = pdOut.reset_index()
        pdOut['levels']=(t2['level_0'].astype(str)+'_'+t2['level_1'].astype(str)).values
        pdOut.drop_duplicates(keep='last',subset='levels',inplace=True)
        pdOut.drop('levels',inplace=True,axis=1)
    return pdOut


def makeFigure(h1,h2,out,title,thresholds = None):
    def makeFigure(h1, h2, out, title, thresholds=None):
        """
        Creates a figure displaying ABR (Auditory Brainstem Response) trace data.
        Parameters
        ----------
        h1 : array-like
            Frequencies in Hz for each trace
        h2 : array-like
            Intensities in dB for each trace
        out : array-like
            2D array containing the ABR trace data, where each row represents a trace
        title : str
            Title for the entire figure
        thresholds : dict, optional
            Dictionary mapping frequencies to threshold intensities. If provided, traces above
            threshold will be plotted in red, below threshold in black
        Returns
        -------
        matplotlib.figure.Figure
            Figure object containing the plotted ABR traces arranged in a grid where:
            - Columns represent different frequencies (increasing left to right)
            - Rows represent different intensities (decreasing top to bottom)
            - Each cell contains a single ABR trace
            - For click stimulus, frequency is labeled as 'Click' instead of '100 Hz'
        Notes
        -----
        - The figure layout auto-adjusts based on unique frequencies and intensities
        - If only one frequency or intensity is provided, grid dimension is set to minimum of 2
        - All subplot axes are created without tick marks
        - Figure size is fixed at 15.8 x 16.35 inches
        """
    frequency = list(set(h1))#[100,3000,6000, 12000,18000,24000,30000,36000,42000 ]
    frequency.sort()
    intensity = list(set(h2))#arange(0,100,5)
    intensity.sort()
    nint = len(intensity)
    nfreq=len(frequency)
    freqmap=dict(zip(frequency,np.arange(len(frequency))))
    imap = dict(zip(intensity,np.arange(len(intensity))))
    if nint==1:
        nint=2
    if nfreq==1:
        nfreq=2
    fig,axs=plt.subplots(nint,nfreq,sharex=False, sharey=False,subplot_kw={'xticks': [], 'yticks': []},figsize=np.array([ 15.8 ,  16.35]))
    for i in range(len(h1)):
        column = freqmap[int(h1[i])]
        row = imap[int(h2[i])]
        #plotn = i+row*len(frequency)
        linecol = 'k'
        if thresholds is not None:
            if h2[i]>=thresholds[h1[i]]:
                linecol = 'r'
            else:
                linecol = 'k'

        axs[nint-row-1,column].plot(np.array(out)[i,:],c=linecol)
        #axs[nint-row-1,column].set_ylim((array(out).min(),array(out).max()))
        if nint-row-1==0:
            tit1 = int(h1[i])
            tit = str(tit1)+' Hz'
            if tit1 == 100:
                tit='Click'
            axs[nint-row-1,column].set_title(tit)
        if column==0:
            axs[nint-row-1,column].set_ylabel(str(int(h2[i]))+' dB')
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def find_peaks(ys,xs,fss, distance=0.5e-3, prominence=50, wlen=None,
               invert=False, detrend=True):

    '''
Utility function for ABR wave 1 extraction
    '''
    y = -ys if invert else ys
    if detrend:
        y = signal.detrend(ys)
    x = xs
    fs = fss
    prominence = np.percentile(y, prominence)
    i_distance = round(fs*distance)
    if wlen is not None:
        wlen = round(fs*wlen)
    kwargs = {'distance': i_distance, 'prominence': prominence, 'wlen': wlen}
    indices, metrics = signal.find_peaks(y, **kwargs)

    metrics.pop('left_bases')
    metrics.pop('right_bases')
    metrics['x'] = x[indices]
    metrics['y'] = y[indices]
    metrics['index'] = indices
    metrics = pd.DataFrame(metrics)
    return metrics

def guess_peaks(metrics, latency):
    '''
    Initial guess in ABR wave 1 determination using find_peaks.
    '''
    p_score_norm = metrics['prominences'] / metrics['prominences'].sum()
    guess = {}
    for i in sorted(latency.keys()):
        l = latency[i]
        l_score = metrics['x'].apply(l.pdf)
        l_score_norm = l_score / l_score.sum()
        score = 5 * l_score_norm + p_score_norm
        m = score.idxmax()
        if np.isfinite(m):
            guess[i] = metrics.loc[m]
            metrics = metrics.loc[m+1:]
        else:
            guess[i] = {'x': l.mean(), 'y': 0}

    return pd.DataFrame(guess).T



from sklearn.model_selection import train_test_split

def loadFiles(datafolder ='../data'):
    """
    Load and concatenate Excel files containing ABR experiment data from specified directory.

    This function reads experiment data and threshold files for both repaired and 6N datasets,
    concatenates them, and returns the combined data along with the current version information.

    Parameters
    ----------
    datafolder : str, optional
        Path to the directory containing the data files (default is '../data')

    Returns
    -------
    tuple
        A tuple containing three elements:
        - data : pandas.DataFrame
            Combined experiment data from both repaired and 6N datasets
        - thresholds : pandas.DataFrame
            Combined threshold data from both repaired and 6N datasets
        - dataVersion : str
            Current version of the dataset as specified in Data-version.txt

    Files Required
    -------------
    - Data-version.txt
    - Repaired - MachineLearningABR_ExperimentList.xlsx
    - Repaired - Thresholds.xlsx
    - 6N - MachineLearningABR_ExperimentList.xlsx
    - 6N - Thresholds.xlsx

    Notes
    -----
    All files must be present in the specified datafolder directory
    """
    
    try:
        with open(os.path.join(datafolder,'Data-version.txt')) as f:
            lines = f.readlines()
            #print(lines)
            dataVersion = lines[0]
    except FileNotFoundError:
        dataVersion = 'None'

    print('The dataset version is: ' + str(dataVersion))


    dataRep = pd.read_excel(os.path.join(datafolder,'Repaired - MachineLearningABR_ExperimentList.xlsx'))
    thresholdsRep = pd.read_excel(os.path.join(datafolder,'Repaired - Thresholds.xlsx'))#
    data6N = pd.read_excel(os.path.join(datafolder,'6N - MachineLearningABR_ExperimentList.xlsx'))
    thresholds6N = pd.read_excel(os.path.join(datafolder,'6N - Thresholds.xlsx'))#


    data = pd.concat([dataRep,data6N],ignore_index=True)
    thresholds = pd.concat([thresholdsRep,thresholds6N],ignore_index=True)   # Substitute with a concatenation of 6N and repaired when necessary

    return (data,thresholds,dataVersion)
    
def createClassificationDataset(datafolder ='../data',test_size = 0.2,random_state = 42,  returnValidation = False,val_size=0.2, oversample=False,ages = [1],frequencies = [100,3000,6000,12000,18000,24000,30000,36000,42000],xlimit=None,lowestInt =15, highestInt = 95,verbose=1,returnMouseIDs=False):
    """
    Return datasets for classification of ABR data.
    This function loads auditory brainstem response (ABR) data from the specified data folder,
    processes the data for the given ages and frequencies, and returns datasets suitable
    for classification tasks (e.g., predicting strain from ABR data).
    Parameters
    ----------
    datafolder : str, optional
        The path to the data folder. Default is '../data'.
    test_size : float, optional
        The proportion of the dataset to include in the test split. Default is 0.2.
    random_state : int, optional
        Controls the shuffling applied to the data before applying the split. Default is 42.
    returnValidation : bool, optional
        If True, also return a validation set split from the training data. Default is False.
    val_size : float, optional
        The proportion of the training data to include in the validation split (only if returnValidation is True). Default is 0.2.
    oversample : bool, optional
        If True, perform oversampling on the training data using Borderline SMOTE to balance classes. Default is False.
    ages : list of int, optional
        The ages (in months) of the data to include. Can include [1, 3, 6, 9, 12]. Default is [1].
    frequencies : list of int, optional
        The list of frequencies (in Hz) to include in the dataset. Default is [100, 3000, 6000, 12000, 18000, 24000, 30000, 36000, 42000]. 100 is the Click stimulus.
    xlimit : int or None, optional
        If not None, limit the number of time points in the ABR data to xlimit. Default is None.
    lowestInt : int, optional
        The lowest intensity (in dB) to include. Default is 15.
    highestInt : int, optional
        The highest intensity (in dB) to include. Default is 95.
    verbose : int, optional
        Verbosity level. If greater than 0, print warnings when data for certain ages cannot be found. Default is 1.
    returnMouseIDs : bool, optional
        If True, also return the mouse IDs corresponding to the data splits. Default is False.
    Returns
    -------
    If returnValidation is True:
        tuple
            (X_train, X_test, X_val, y_train, y_test, y_val, dataVersion)
    If returnMouseIDs is True:
        tuple
            (X_train, X_test, y_train, y_test, mouseIDs_train, mouseIDs_test, dataVersion)
    Else:
        tuple
            (X_train, X_test, y_train, y_test, dataVersion)
    """
    
    from collections import Counter
    data,_,dataVersion = loadFiles(datafolder)

    X = []
    y = []
    pairs = []

    for fr in frequencies:
        for i in range(lowestInt,highestInt+5,5):
            pairs.append([fr,i])

    for j,el in data.iterrows():
        if 1 in ages:
            filename = el['Folder 1'].split('./')[1] # DAta at 1 month
            filename = os.path.join(datafolder,filename)
            #mouseID = str(el['ID'])#.split('-')[0]
            t = extractABR(filename)#[arange(1200)]
            if xlimit is not None:
                t = t.iloc[:,:xlimit]
            X.append(t.loc[[(p[0],p[1]) for p in pairs],:].values.ravel())
            y.append(el['Strain'])
        if 3 in ages:

            try:
                filename = el['Folder 2'].split('./')[1] # DAta at 1 month
                filename = os.path.join(datafolder,filename)
                t = extractABR(filename)#[arange(1200)]
                if xlimit is not None:
                    t = t.iloc[:,:xlimit]
                X.append(t.loc[[(p[0],p[1]) for p in pairs],:].values.ravel())
                y.append(el['Strain'])
            except:
                if verbose>0:
                    print('Can''t find 3 month old data')
                else:
                    pass
        if 6 in ages:

            try:
                filename = el['Folder 3'].split('./')[1] # DAta at 1 month
                filename = os.path.join(datafolder,filename)
                t = extractABR(filename)#[arange(1200)]
                if xlimit is not None:
                    t = t.iloc[:,:xlimit]
                X.append(t.loc[[(p[0],p[1]) for p in pairs],:].values.ravel())
                y.append(el['Strain'])
            except:
                if verbose>0:
                    print('Can''t find 6 month old data')
                else:
                    pass

        if 9 in ages:

            try:
                filename = el['Folder 4'].split('./')[1] # DAta at 1 month
                filename = os.path.join(datafolder,filename)
                t = extractABR(filename)#[arange(1200)]
                if xlimit is not None:
                    t = t.iloc[:,:xlimit]
                X.append(t.loc[[(p[0],p[1]) for p in pairs],:].values.ravel())
                y.append(el['Strain'])
            except:
                if verbose>0:
                    print('Can''t find 9 month old data')
                else:
                    pass

        if 12 in ages:

            try:
                filename = el['Folder 5'].split('./')[1] # DAta at 1 month
                filename = os.path.join(datafolder,filename)
                t = extractABR(filename)#[arange(1200)]
                if t.shape[1]!=1953:
                    t = t.dropna(axis=1) #I added this for a weird file with lots of missing values when loaded
                if xlimit is not None:
                    t = t.iloc[:,:xlimit]

                X.append(t.loc[[(p[0],p[1]) for p in pairs],:].values.ravel())
                y.append(el['Strain'])
            except:
                if verbose>0:
                    print('Can''t find 12 month old data')
                else:
                    pass
    
    X = np.array(X)
    y = np.array(y)
    mouseIDs = data['ID'].values

    if oversample:
        print("WARNING!")
        print("Oversampling dataset using BORDERLINE SMOTE")
        from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
        from imblearn.under_sampling import RandomUnderSampler
        #oversample = SMOTE(sampling_strategy=0.5)
        oversample = BorderlineSMOTE(sampling_strategy=1,random_state=random_state)
        #oversample = SVMSMOTE(sampling_strategy=0.5)
        under = RandomUnderSampler(sampling_strategy=1.0,random_state=random_state)
        X,y = oversample.fit_resample(X,y)
        X,y = under.fit_resample(X,y)
        print('Classes')
        print(Counter(y))
    
    print(Counter(y))
    X_train,  X_test,y_train,y_test,mouseIDs_train,mouseIDs_test = train_test_split(X,y,mouseIDs,test_size=test_size,shuffle=True,random_state=random_state,stratify=y)


    if returnValidation:
        X_train,  X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=val_size,shuffle=True,random_state=random_state)
        return (X_train,  X_test,X_val,y_train,y_test,y_val,dataVersion)
    elif returnMouseIDs:
        return (X_train,  X_test,y_train,y_test,mouseIDs_train,mouseIDs_test,dataVersion)


    else:
        return (X_train,  X_test,y_train,y_test,dataVersion)


from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import socket

def initWandb(project,name = str(datetime.datetime.now()), group=None,config = {},dataVersion=None, train_size = 0, test_size = 0):

    import wandb
    wandb.login()
    run = wandb.init(project=project,name=name,group=group,config=config)
    wandb.config.model = group
    wandb.config.data_version = dataVersion

    wandb.config.train_size = train_size
    wandb.config.test_size = test_size
    wandb.config.architecture = socket.gethostname()

    return run

def fitRegModel(model,X_train,y_train,X_test=None,y_test = None,makePlot=True,calculateScores=True,cv=10,saveToWandb=False,modelName = '',config_dict = {}, dataVersion=None, plotOutliers = False,n_jobs=-1,random_state=42,n_splits=5,n_repeats=5):
    """
    Fits and evaluates a regression model with various metrics and visualization options.
    Parameters:
    ----------
    model : estimator object
        The regression model to fit (must implement fit and predict methods)
    X_train : array-like of shape (n_samples, n_features)
        Training data
    y_train : array-like of shape (n_samples,)
        Target values for training
    X_test : array-like of shape (n_samples, n_features), optional
        Test data (default=None)
    y_test : array-like of shape (n_samples,), optional
        Target values for test data (default=None)
    makePlot : bool, optional
        If True, generates scatter plot of predictions vs actual values (default=True)
    calculateScores : bool, optional
        If True, performs cross-validation and calculates performance metrics (default=True)
    cv : int, optional
        Number of folds for cross-validation (default=10)
    saveToWandb : bool, optional
        If True, logs results to Weights & Biases (default=False)
    modelName : str, optional
        Name of the model for logging purposes (default='')
    config_dict : dict, optional
        Configuration dictionary for W&B logging (default={})
    dataVersion : str, optional
        Version of data being used (default=None)
    plotOutliers : bool, optional
        If True, generates outlier plots in W&B (default=False)
    n_jobs : int, optional
        Number of CPU cores to use for parallel processing (default=-1, use all cores)
    random_state : int, optional
        Random seed for reproducibility (default=42)
    n_splits : int, optional
        Number of splits for RepeatedKFold cross-validation (default=5)
    n_repeats : int, optional
        Number of times to repeat cross-validation (default=5)
    Returns:
    -------
    tuple or dict
        If calculateScores=True:
            Returns dictionary containing cross-validation results with keys:
            'RMSE', 'MAE', 'MAPE', 'R2', 'MaxError'
        If calculateScores=False:
            Returns tuple (scores, mse, test_mse) where:
            - scores: cross-validation scores (empty list if calculateScores=False)
            - mse: mean squared error on training data
            - test_mse: mean squared error on test data (None if X_test not provided)
    Notes:
    -----
    Requires sklearn metrics and model selection modules.
    If saveToWandb=True, requires wandb to be installed and configured.
    """

    from sklearn.model_selection import RepeatedKFold, cross_validate
    from sklearn.metrics import make_scorer, mean_squared_error,mean_absolute_error,r2_score,root_mean_squared_error,max_error,mean_absolute_percentage_error

    if saveToWandb:
        run = initWandb(project='Threshold prediction',name=str(datetime.datetime.now()),group=modelName,config=config_dict, dataVersion=dataVersion,
        train_size = X_train.shape[0], test_size= X_test.shape[0])

    print('Fitting '+modelName+' model')

    model.fit(X_train,y_train)
    results = model.predict(X_train)
    
    mse = sqrt(mean_squared_error(y_train,results))

    if X_test is not None:
        test_results = model.predict(X_test)
        test_mse = np.sqrt(mean_squared_error(y_test,test_results))
    else:
        test_mse=None
    
    if makePlot:
        fig = figure()
        plot(y_train,results,'o')
        
        if X_test is not None:
            plot(y_test,test_results,'og')
            
        plot([15,120],[15,120],'-r')
        #xlim(0,7)
        #ylim(0,7)
        xlabel('Real threshold (dB)')
        ylabel('Estimated threshold (dB')
    else:
        fig = None
    
    if calculateScores:
        print('Cross validating')
        scorers = {'RMSE':make_scorer(root_mean_squared_error),
                    'mean_absolute_error':make_scorer(mean_absolute_error),
                    'mean_absolute_percentage_error':make_scorer(mean_absolute_percentage_error),
                    'r2':make_scorer(r2_score),
                    'max_error':make_scorer(max_error)
        }

        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        c1 = cross_validate(model,X_train,y_train,scoring=scorers,n_jobs=n_jobs,cv=cv)

        scores = -c1['test_RMSE']**2
        print('MSE on train '+ str(mse))
        print('CV RMSE : ' + str( np.sqrt(-scores).mean()))
        print('CV RMSE STD : ' + str( np.sqrt(-scores).std()))
        print('CV R2 : ' + str(c1['test_r2'].mean()))
        print('CV R2 STD : ' + str(c1['test_r2'].std()))

        res = {
            'RMSE':c1['test_RMSE'],
            'MAE':c1['test_mean_absolute_error'],
            'MAPE':c1['test_mean_absolute_percentage_error'],
            'R2':c1['test_r2'],
            'MaxError':c1['test_max_error']
        }
        return res

    else:
        scores = []

    if saveToWandb:
        wandb.log({'Train MSE': mse,
        'CV neg MSE scores': scores,
        'CV MSE':np.sqrt(-scores).mean(),
        'CV std':np.sqrt(-scores).std(),
        'Test MSE':test_mse,
        'plot':fig
         })
        #wandb.sklearn.plot_regressor(model, X_train, X_test, y_train, y_test,  model_name=modelName)
        if plotOutliers:
            wandb.sklearn.plot_outlier_candidates(model, X_train, y_train)
            wandb.sklearn.plot_residuals(model, X_train, y_train)

        run.finish()
    
    return scores, mse, test_mse


def fitClassificationModel(model,X_train,y_train,X_test=None,y_test = None,makePlot=True,crossValidation=False,saveToWandb=False,modelName = '',config_dict = {}, dataVersion=None,random_state=42,calculatePValue=False,njobs= -1,n_splits=5,n_repeats=5,encode_labels=False):
    """
    Fits and evaluates a classification model with various options for cross-validation and performance visualization.
    Parameters:
    ----------
    model : sklearn estimator object
        The classification model to fit
    X_train : array-like of shape (n_samples, n_features)
        Training data
    y_train : array-like of shape (n_samples,)
        Target values for training
    X_test : array-like of shape (n_samples, n_features), optional
        Test data
    y_test : array-like of shape (n_samples,), optional
        Target values for test
    makePlot : bool, default=True
        Whether to generate confusion matrices and classification reports
    crossValidation : bool, default=False
        Whether to perform cross-validation
    saveToWandb : bool, default=False
        Whether to log results to Weights & Biases
    modelName : str, default=''
        Name of the model for logging purposes
    config_dict : dict, default={}
        Configuration dictionary for Weights & Biases
    dataVersion : str, optional
        Version of the data being used
    random_state : int, default=42
        Random state for reproducibility
    calculatePValue : bool, default=False
        Whether to calculate permutation test p-value
    njobs : int, default=-1
        Number of jobs for parallel processing (-1 means using all processors)
    n_splits : int, default=5
        Number of folds for cross-validation
    n_repeats : int, default=5
        Number of times to repeat cross-validation
    encode_labels : bool, default=False
        Whether to encode labels as integers (0,1) instead of strings
    Returns:
    -------
    dict or None
        If crossValidation=True, returns a dictionary containing various performance metrics:
        - accuracy: Balanced accuracy scores
        - test_f1_scorer_avg: Weighted average F1 scores
        - test_f1_scorer_6N: F1 scores for class '6N'
        - test_f1_scorer_Rep: F1 scores for class 'Repaired'
        - roc_auc_score: ROC AUC scores
        - test_precision_scorer_avg: Weighted average precision scores
        - test_precision_scorer_6N: Precision scores for class '6N'
        - test_precision_scorer_Rep: Precision scores for class 'Repaired'
        - test_recall_scorer_avg: Weighted average recall scores
        - test_recall_scorer_6N: Recall scores for class '6N'
        - test_recall_scorer_Rep: Recall scores for class 'Repaired'
        - p_value: Permutation test p-value (if calculatePValue=True)
        If crossValidation=False, returns None
    Notes:
    -----
    - Weights & Biases integration is currently not fully implemented
    - The function assumes binary classification with classes '6N' and 'Repaired'
    - Cross-validation uses RepeatedStratifiedKFold for robust performance estimation
    """


    if saveToWandb:
        run = initWandb(project='6N vs repaired',name=str(datetime.datetime.now()),group=modelName,config=config_dict, dataVersion=dataVersion,
        train_size = X_train.shape[0], test_size= X_test.shape[0])

    from sklearn.model_selection import cross_val_score, cross_validate,RepeatedStratifiedKFold,permutation_test_score
    from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix,classification_report,make_scorer,\
                                f1_score, accuracy_score, roc_auc_score,balanced_accuracy_score , precision_score,recall_score
    from sklearn.feature_selection import f_classif,mutual_info_classif

    if crossValidation == False:
        model.fit(X_train,y_train)




        if makePlot:
            print('CLASSIFICATION REPORT ON TRAIN')
            print(classification_report(y_train, model.predict(X_train)))

            if X_test is not None:
                print('CLASSIFICATION REPORT ON TEST')
                print(classification_report(y_test, model.predict(X_test)))

            print('Confusion matrix on train')
            cm = confusion_matrix(y_train,model.predict(X_train),normalize=None,labels=['6N','Repaired'])
            ConfusionMatrixDisplay(cm).plot()
            show()

            print('Confusion matrix on test')
            cm = confusion_matrix(y_test,model.predict(X_test),normalize=None,labels=['6N','Repaired'])
            fig = ConfusionMatrixDisplay(cm).plot()
            show()


        if saveToWandb:
            wandb.sklearn.plot_classifier(model, X_train, X_test, y_train, y_test, model.predict(X_test), 
                                        model.predict_proba(X_test), labels = ['6N','Repaired'], model_name=modelName, feature_names=None)
            wandb.summary['Class report test'] = classification_report(y_test, model.predict(X_test), output_dict=True)
        
            #wandb.sklearn.plot_regressor(model, X_train, X_test, y_train, y_test,  model_name=modelName)
        
            run.finish()

  
    if crossValidation == True:
        print('Cross validating...')
        if encode_labels:
            labelsdf = {
                '6N':0,
                'Repaired':1
            }
        else:
            labelsdf = {
                '6N':'6N',
                'Repaired':'Repaired'
            }

        scorer = {'accuracy':make_scorer(balanced_accuracy_score),
        'f1_scorer':make_scorer(f1_score,pos_label=labelsdf['6N']),
        'f1_scorer_r':make_scorer(f1_score,pos_label=labelsdf['Repaired']),
        'f1_scorer_avg':make_scorer(f1_score,average='weighted'),
        'auc': make_scorer(roc_auc_score,average='weighted',needs_proba=True),
        'precision':make_scorer(precision_score,pos_label=labelsdf['6N']),
        'recall':make_scorer(recall_score,pos_label=labelsdf['6N']),
        'precision_avg':make_scorer(precision_score,average='weighted'),
        'recall_avg':make_scorer(recall_score,average='weighted'),
        'precision_r':make_scorer(precision_score,pos_label=labelsdf['Repaired']),
        'recall_r':make_scorer(recall_score,pos_label=labelsdf['Repaired']),
        }
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        cv2 = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1, random_state=random_state) # for p value calculation we use n=1 for speed
        #c1 = pd.DataFrame()
        #c2 = pd.DataFrame()
        #accuracies = []
        #f1_6Ns = []
        #f1_Repaireds = []
        #rocs = []
        # for el in cv.split(X_train,y_train):
        #     X_train_val = X_train[el[0],:]
        #     y_train_val = y_train[el[0]]
        #     X_val = X_train[el[1],:]
        #     y_val = y_train[el[1]]

        #     model.fit(X_train_val,y_train_val)
        #     y_pred = model.predict(X_val)
            
        #     accuracies.append(accuracy_score(y_val,y_pred))
        #     f1_6Ns.append(f1_score(y_val,y_pred, pos_label='6N'))
        #     f1_Repaireds.append(f1_score(y_val,y_pred, pos_label='Repaired'))

        #     rocs.append(roc_auc_score(y_val,model.predict_proba(X_val)[:,1]))
        
        c1 = cross_validate(model,X_train,y_train,cv=cv,scoring=scorer,n_jobs=njobs)
        #c2 = cross_validate(model,X_train,y_train,cv=cv,scoring='roc_auc',n_jobs=-1) 
        # c1['test_f1_scorer'] = f1_6Ns
        # c1['test_f1_scorer_r'] = f1_Repaireds
        # c1['test_accuracy'] = accuracies
        # c2['test_score'] = rocs   # we use c1 and c2 so the code is back compatible 

        if calculatePValue == True :
            #TODO figure out how to include feature selection with the p value
            pts = permutation_test_score(model, X_train, y_train, groups=None, cv=cv2, n_permutations=100, n_jobs=njobs, random_state=random_state, verbose=2, scoring='accuracy', fit_params=None)
        else:
            pts = [None,None,None]
        print('Average CV F1 score: '+ str(mean(c1['test_f1_scorer_avg']))+ '   STD: ' + str(std(c1['test_f1_scorer_avg'])))
        print('Average CV F1 score: 6N: '+ str(mean(c1['test_f1_scorer'])) + '   STD: ' + str(std(c1['test_f1_scorer'])))
        print('Average CV F1 score: Repaired: '+ str(mean(c1['test_f1_scorer_r']))+ '   STD: ' + str(std(c1['test_f1_scorer_r'])))
        print('Average CV accuracy: '+ str(mean(c1['test_accuracy']))+ '   STD: ' + str(std(c1['test_accuracy'])))
        #print('Average CV ROC AUC: '+ str(mean(c2['test_score']))+ '   STD: ' + str(std(c2['test_score'])))
        print('Average CV ROC AUC: '+ str(mean(c1['test_auc']))+ '   STD: ' + str(std(c1['test_auc'])))
        print('Permutation test p value : '+ str(pts[2]))


        res = {
            'accuracy':c1['test_accuracy'],
            'test_f1_scorer_avg':c1['test_f1_scorer_avg'],
            'test_f1_scorer_6N':c1['test_f1_scorer'],
             'test_f1_scorer_Rep':c1['test_f1_scorer_r'],
            'roc_auc_score':c1['test_auc'],
            
            'test_precision_scorer_avg':c1['test_precision_avg'],
            'test_precision_scorer_6N':c1['test_precision'],
            'test_precision_scorer_Rep':c1['test_precision_r'],

            'test_recall_scorer_avg':c1['test_recall_avg'],
            'test_recall_scorer_6N':c1['test_recall'],
            'test_recall_scorer_Rep':c1['test_recall_r'],

            'p_value':pts[2]
            
        }
        return res

    return None
        #if saveToWandb:
            #


def createFutureThresholdDataset(datafolder ='../data',inputFreqs = None,inputs = ['1month'],target = '6month',strains = ['6N','Repaired'],test_size = 0.2,random_state = 42, targetFrequency = 100, targetIntensity = None,mode = 'single',returnValidation = False,val_size=0.2,onlyFullDataAt12months=True):
    """
    Creates train and test datasets for future prediction based on ABR data.
    This function loads ABR data and , processes them according to specified mode,
    and splits them into training and test sets (with optional validation set).
    Parameters
    ----------
    datafolder : str, optional
        Path to the data folder (default: '../data')
    inputFreqs : list, optional
        List of input frequencies to use (default: None, uses all frequencies)
    inputs : list, optional
        List of input time points to use (default: ['1month'])
    target : str, optional
        Target time point for prediction (default: '6month')
    strains : list, optional
        List of mouse strains to include (default: ['6N','Repaired'])
    test_size : float, optional
        Fraction of data to use for testing (default: 0.2)
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    targetFrequency : int, optional
        Specific frequency to analyze (default: 100)
    targetIntensity : int, optional
        Specific intensity to analyze (default: None)
    mode : str, optional
        Analysis mode. 'single': predict single frequency threshold targetfrequency, 'mean': predict the mean threshold, 'median': predict the median threshold, 
        'diff' or 'single': threshold diff with 1 month, 'meandiff': difference of mean thresholds with 1 month, 'mediandiff': difference of mean thresholds with 1 month,
        'singlediff': threshold diff with 1 month, 'allfreq':return all thresholds, 'waveamp': wave 1 amplitude, 'wavelatency': wave 1 latency, 'waveampdiff': wave 1 amplitude difference 
        with 1 month, 'wavelatencydiff':wave 1 latency difference with one month.) (default: 'single')
    returnValidation : bool, optional
        Whether to return a validation set (default: False)
    val_size : float, optional
        Fraction of training data to use for validation (default: 0.2)
    onlyFullDataAt12months : bool, optional
        Whether to only include mice with complete 12-month data (default: True)
    Returns
    -------
    tuple
        If returnValidation=False:
            (X_train, X_test, y_train, y_test, mouseIDtrain, mouseIDtest,
             mouseStrainTrain, mouseStrainTest, dataVersion)
        If returnValidation=True:
            (X_train, X_test, X_val, y_train, y_test, y_val, mouseIDtrain,
             mouseIDtest, mouseStrainTrain, mouseStrainTest, dataVersion)
    Notes
    -----
    The function supports various targets:
    - 'single': Threshold at target frequency and age
    - 'mean'/'median': Average/median threshold across frequencies
    - 'diff': Threshold difference between target age and 1 month
    - 'meandiff'/'mediandiff': Mean/median threshold difference
    - 'waveamp'/'wavelatency': Wave 1 amplitude/latency analysis
    - 'waveampdiff'/'wavelatencydiff': Wave 1 changes from 1 month
    """

    data,thresholds,dataVersion = loadFiles(datafolder)

    if onlyFullDataAt12months:
        data = data[~data['Folder 5'].isna()].reset_index() # keep only mice with data at all ages


    X = []
    y = []
    mouseIDs = []
    mouseStrains = []
    pairs = []
    if inputFreqs is None:
        inputFreqs = frequencies
        
    for fr in inputFreqs:
        for i in range(lowestInt,100,5):
            pairs.append([fr,i])

    if (mode=='waveamp') or (mode=='wavelatency') or (mode=='waveampdiff') or (mode=='wavelatencydiff'):
        masterWave1 = createWave1Dataset(age=target,datafolder=datafolder)
        masterWave1 = masterWave1.query('Freq==@targetFrequency')

        masterWave1['ID'] = masterWave1['ID'].astype(str)
        #Add the missing latencies as the max of the lantencies per mouse
        for id in masterWave1['ID'].unique():
            for freq in masterWave1['Freq'].unique():
                el2 = masterWave1.query("ID==@id & Freq==@freq")

                masterWave1.loc[el2.index[pd.isna(el2['Wave1 latency'])],'Wave1 latency'] = el2['Wave1 latency'].max()
        if targetIntensity is not None:
            masterWave1 = masterWave1.query('Intensity==@targetIntensity')
        masterWave1 = masterWave1.set_index(['ID','Freq','Intensity'])

        masterWave1month = createWave1Dataset(age='1month',datafolder=datafolder)
        masterWave1month = masterWave1month.query('Freq==@targetFrequency')

        masterWave1month['ID'] = masterWave1month['ID'].astype(str)
        #Add the missing latencies as the max of the lantencies per mouse
        for id in masterWave1month['ID'].unique():
            for freq in masterWave1month['Freq'].unique():
                el2 = masterWave1month.query("ID==@id & Freq==@freq")
                masterWave1month.loc[el2.index[pd.isna(el2['Wave1 latency'])],'Wave1 latency'] = el2['Wave1 latency'].max()
        if targetIntensity is not None:
            masterWave1month = masterWave1month.query('Intensity==@targetIntensity')
        masterWave1month = masterWave1month.set_index(['ID','Freq','Intensity'])

        masterWave1['Wave1 amp diff'] = (masterWave1['Wave1 amp'] - masterWave1month['Wave1 amp']).dropna()
        masterWave1['Wave1 latency diff'] = (masterWave1['Wave1 latency'] - masterWave1month['Wave1 latency']).dropna()
        masterWave1 = masterWave1.reset_index()
    #input = '1month'
    #target = '3month'

    for j,el in data.iterrows():
        if el['Strain'] in strains:
            mouseID = str(el['ID'])#.split('-')[0]
            strain = el['Strain']
            try:
                if mode == 'mean': # Just calculate the average threshold at all frequency
                    this_tr = []
                    for fr in frequencies:
                        strFr = freqDict[str(fr)]
                        
                        this_tr.append(thresholds.loc[thresholds['MouseN - AGE']== mouseID+' - '+target,strFr].values[0])
                    y.append(mean(this_tr))
                    mouseIDs.append(mouseID)
                    mouseStrains.append(strain)
                elif mode == 'median': # Just calculate the average threshold at all frequency
                    this_tr = []
                    for fr in frequencies:
                        strFr = freqDict[str(fr)]
                        
                        this_tr.append(thresholds.loc[thresholds['MouseN - AGE']== mouseID+' - '+target,strFr].values[0])
                    y.append(median(this_tr))
                    mouseIDs.append(mouseID)
                    mouseStrains.append(strain)
                elif mode == 'diff':   #  calculate the diff threshold for the targetFrequency between the age and 1 month
                    this_tr = []
                    strFr = freqDict[str(targetFrequency)]
                    this_y = thresholds.loc[thresholds['MouseN - AGE']== mouseID+' - '+target,strFr].values[0] - thresholds.loc[thresholds['MouseN - AGE']== mouseID+' - 1month',strFr].values[0]
                    y.append(this_y)
                    mouseIDs.append(mouseID)
                    mouseStrains.append(strain)
                elif mode =='meandiff':#calculate the difference in average threshold between target and 1 month
                    this_tr = []
                    this_input = []
                    for fr in frequencies:
                        strFr = freqDict[str(fr)]
                        this_tr.append(thresholds.loc[thresholds['MouseN - AGE']== mouseID+' - '+target,strFr].values[0])
                        this_input.append(thresholds.loc[thresholds['MouseN - AGE']== mouseID+' - 1month',strFr].values[0])
                    y.append(mean(this_tr)-mean(this_input))
                    mouseIDs.append(mouseID)
                    mouseStrains.append(strain)
                elif mode =='mediandiff':#calculate the difference in average threshold between target and 1 month
                    this_tr = []
                    this_input = []
                    for fr in frequencies:
                        strFr = freqDict[str(fr)]
                        this_tr.append(thresholds.loc[thresholds['MouseN - AGE']== mouseID+' - '+target,strFr].values[0])
                        this_input.append(thresholds.loc[thresholds['MouseN - AGE']== mouseID+' - 1month',strFr].values[0])
                    y.append(median(this_tr)-median(this_input))
                    mouseIDs.append(mouseID)
                    mouseStrains.append(strain)
                elif mode=='single':  # calculate the threshold for the targetFrequency at the desired age
                    strFr = freqDict[str(targetFrequency)]
                    y.append(thresholds.loc[thresholds['MouseN - AGE']== mouseID+' - '+target,strFr].values[0])
                    mouseIDs.append(mouseID)
                    mouseStrains.append(strain)
                elif mode=='singlediff':  # calculate the threshold shift for the targetFrequency at the desired age
                    strFr = freqDict[str(targetFrequency)]

                    y.append(thresholds.loc[thresholds['MouseN - AGE']== mouseID+' - '+target,strFr].values[0] - 
                             thresholds.loc[thresholds['MouseN - AGE']== mouseID+' - 1month',strFr].values[0])
                    mouseIDs.append(mouseID)    
                    mouseStrains.append(strain)
                elif mode=='allfreq':
                    this_thresh = []
                    for strFr in list(freqDict.values()):
                        this_thresh.append(thresholds.loc[thresholds['MouseN - AGE']== mouseID+' - '+target,strFr].values[0])    
                    y.append(this_thresh)
                    mouseIDs.append(mouseID)
                    mouseStrains.append(strain)
                elif mode=='waveampdiff':
                    values = masterWave1.query("ID==@mouseID").sort_values('Intensity')['Wave1 amp'].values
                    if len(values)==20:
                        y.append(masterWave1.query("ID==@mouseID").sort_values('Intensity')['Wave1 amp diff'].values.astype(float))
                        mouseIDs.append(mouseID)
                        mouseStrains.append(strain)
                    else:
                        print(len(values))
                        raise IndexError
                    
                elif mode=='wavelatencydiff':
                   
                    values = masterWave1.query("ID==@mouseID").sort_values('Intensity')['Wave1 amp'].values
                    if len(values)==20:  
                        y.append(masterWave1.query("ID==@mouseID").sort_values('Intensity')['Wave1 latency diff'].values.astype(float))
                        mouseIDs.append(mouseID)
                        mouseStrains.append(strain)
                    else:
                        raise IndexError
                
                elif mode=='waveamp':
                    if targetIntensity is None:
                        values = masterWave1.query("ID==@mouseID").sort_values('Intensity')['Wave1 amp'].values
                        if len(values)==20:
                            y.append(masterWave1.query("ID==@mouseID").sort_values('Intensity')['Wave1 amp'].values.astype(float))
                            mouseIDs.append(mouseID)
                            mouseStrains.append(strain)
                        else:
                            print(len(values))
                            raise IndexError
                    else:
                        values = masterWave1.query("ID==@mouseID")['Wave1 amp'].values
                        if len(values)==1:
                            y.append(masterWave1.query("ID==@mouseID")['Wave1 amp'].values[0])
                            mouseIDs.append(mouseID)
                            mouseStrains.append(strain)
                        else:
                            print(len(values))
                            raise IndexError                        
                    
                elif mode=='wavelatency':
                    if targetIntensity is None:
                        values = masterWave1.query("ID==@mouseID").sort_values('Intensity')['Wave1 amp'].values
                        if len(values)==20:  
                            y.append(masterWave1.query("ID==@mouseID").sort_values('Intensity')['Wave1 latency'].values.astype(float))
                            mouseIDs.append(mouseID)
                            mouseStrains.append(strain)
                        else:
                            raise IndexError
                    else:
                        values = masterWave1.query("ID==@mouseID")['Wave1 latency'].values
                        if len(values)==1:
                            y.append(masterWave1.query("ID==@mouseID")['Wave1 latency'].values[0])
                            mouseIDs.append(mouseID)
                            mouseStrains.append(strain)
                        else:
                            print(len(values))
                            raise IndexError     
                                              
                else:
                    print('Mode not supported')
                    return
                ts = []
                monthFolderDict = {'1month':'1','3month':'2','6month':'3','9month':'4','12month':'5'}
                for input in inputs:
                    filename = el['Folder {}'.format(monthFolderDict[input])].split('./')[1] # DAta at 1 month
                    filename = os.path.join(datafolder,filename)
                    t = extractABR(filename)#[arange(1200)]
                    ts.append(t.loc[[(p[0],p[1]) for p in pairs],:].values.ravel())

                t = np.hstack(ts)
                X.append(t)
            except IndexError:
                print('Cannot find data for '+mouseID)




    X = np.array(X)
    y = np.array(y)


    X_train,  X_test,y_train,y_test,mouseIDtrain,mouseIDtest,mouseStrainTrain,mouseStrainTest = train_test_split(X,y,mouseIDs,mouseStrains,test_size=test_size,shuffle=True,random_state=random_state,stratify=mouseStrains)#We stratify to ensure that both 6N and Repaired are presnet in both train and test
    if returnValidation:
        X_train,  X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=val_size,shuffle=True,random_state=random_state)
        return (X_train,  X_test,X_val,y_train,y_test,y_val,mouseIDtrain,mouseIDtest,mouseStrainTrain,mouseStrainTest,dataVersion)
    else:
        return (X_train,  X_test,y_train,y_test,mouseIDtrain,mouseIDtest,mouseStrainTrain,mouseStrainTest,dataVersion)



def createWave1Dataset(datafolder = '../data/',waveanalysisFolder = 'waveAnalysisResults',age='1month',addMissingAmplitudes=True):    
    """
    Creates a dataset of Wave 1 ABR measurements from multiple CSV files.
    This function reads ABR (Auditory Brainstem Response) wave analysis results for two mouse strains
    (6N and Repaired) and combines them into a single dataset with wave 1 amplitude and latency measurements.
    Parameters
    ----------
    datafolder : str, optional
        Path to the main data directory (default: '../data/')
    waveanalysisFolder : str, optional
        Name of the folder containing wave analysis CSV files (default: 'waveAnalysisResults')
    age : str, optional
        Age of mice to analyze (default: '1month')
    addMissingAmplitudes : bool, optional
        If True, adds rows with zero amplitude for missing intensity values (default: True)
    Returns
    -------
    pandas.DataFrame
        Combined dataset containing wave 1 measurements with columns:
        - Strain: mouse strain ('6N' or 'Repaired')
        - Age: age of the mouse
        - ID: mouse identifier
        - Freq: stimulus frequency
        - Intensity: stimulus intensity
        - Wave1 amp: amplitude of wave 1 (P1-N1)
        - Wave1 latency: latency of wave 1 (P1)
        Additional columns from original CSVs are preserved
    Notes
    -----
    The function expects Excel files with mouse lists and individual CSV files containing
    wave analysis results. Missing files are silently skipped. When addMissingAmplitudes
    is True, it fills in missing intensity values (0-95 dB in 5 dB steps) with zero amplitude.
    """
    sixN = pd.read_excel(os.path.join(datafolder,'6N - MachineLearningABR_MouseList.xlsx'))
    rep = pd.read_excel(os.path.join(datafolder,'Repaired - MachineLearningABR_MouseList.xlsx'))

    masterAll = pd.DataFrame()
    rows = []
    for j,el in sixN.iterrows():
        try:
            filename = str(el['ID']) + ' - '+age+'.csv'
            fullpath = os.path.join(datafolder,waveanalysisFolder,filename)
            a = pd.read_csv(fullpath)
            a['Strain'] = '6N'
            a['Age'] = age
            a['ID'] = el['ID']
            rows.append(a)
        except FileNotFoundError:
            pass
          #  print('File not found')
    #masterAll = masterAll.append(a)
        
    for j,el in rep.iterrows():
        try:
            filename = str(el['ID']) + ' - '+age+'.csv'
            fullpath = os.path.join(datafolder,waveanalysisFolder,filename)
            a = pd.read_csv(fullpath)
            a['Strain'] = 'Repaired'
            a['Age'] = age
            a['ID'] = el['ID']
            rows.append(a)
        except FileNotFoundError:
            pass
#            print('File not found')
    masterAll = pd.concat(rows,ignore_index=True)

    masterAll['Wave1 amp'] = masterAll['P1_y']-masterAll['N1_y']
    masterAll['Wave1 latency'] = masterAll['P1_x']
    masterAll['Wave1 amp'] = masterAll['Wave1 amp'].fillna(0) 
    
    if addMissingAmplitudes:#Add amplitudes below 0
        allIntensities = set(np.arange(0,100,5))
        rowsToAdd = []
        for id in masterAll['ID'].unique():
            el = masterAll.query("ID==@id")
            for freq in el['Freq'].unique():
                el2 = el.query("Freq==@freq")
                ints = set(el2['Intensity'].unique())
                missingIntensityies = allIntensities - ints
                for intensity in missingIntensityies:
                    row = pd.Series(index=el2.columns,dtype='object')
                    row['Freq'] = freq
                    row['Intensity'] = intensity
                    row['Strain'] = el2['Strain'].values[0]
                    row['Age'] = el2['Age'].values[0]
                    row['ID'] = id
                    row['Wave1 amp'] = 0
                    rowsToAdd.append(row)
        rowsToAddDf = pd.concat(rowsToAdd,axis=1).T
    
        masterAll = pd.concat([masterAll,rowsToAddDf],ignore_index=True)
        
    return masterAll

def plotFeatureImportance(fi,abr=None,savgolOrder = 51,ylims=(-5.5,10)):
    """
    Plots feature importance overlaid on ABR traces in a grid layout.
    This function creates a visualization where feature importance values are plotted on top of
    ABR (Auditory Brainstem Response) waveforms. The plot is arranged in a 17x9 grid where each
    cell contains both an ABR trace and its corresponding feature importance values.
    Parameters
    ----------
    fi : array-like
        Feature importance values to be plotted
    abr : pandas.DataFrame, optional
        ABR data to be plotted. If None, loads default data from a specific file path
    savgolOrder : int, optional
        Order of the Savitzky-Golay filter applied to smooth feature importance values.
        Default is 51
    ylims : tuple of float, optional
        Y-axis limits for the feature importance plots. Default is (-5.5, 10)
    Returns
    -------
    None
        Displays the figure with overlaid ABR and feature importance plots
    Notes
    -----
    - The function applies Savitzky-Golay filtering to smooth the feature importance values
    - ABR traces are plotted in black, feature importance values in red
    - Feature importance values are scaled by 100000/2 and offset by -5 for visualization
    - The resulting plot has 153 subplots arranged in a 17x9 grid
    - Each subplot shows an ABR trace with its corresponding feature importance overlay
    """
    from scipy.signal import savgol_filter
    if abr is None:
        abr = extractABR('../data/20220520/Mouse #1-[226251-LE].csv')

    fi = savgol_filter(fi,savgolOrder,1)
    ntraces = 153
    ppt = 1953#int(fi.size/ntraces)

    
    fig = makeFigure(abr.reset_index().values[:,0],abr.reset_index().values[:,1],abr.values,title='')
    for column in range(9):
        for row in range(17):
            tr = fi[(16-row+column*17)*ppt:(16-row+1)*ppt + column*17*ppt]
            currAx = row*9 + column
            ax2 = fig.axes[currAx].twinx()
            ax2.plot(tr*100000/2-5,'r')

    for i in range(180,333):
        ax = fig.axes[i]
        ax.set_ylim(ylims[0],ylims[1])
        ax.axis('off')

    for i in range(0,180):
        ax = fig.axes[i]
        ax.set_ylim(-4,4)
        ax.axis('off')
        ax.set_xlim(0,12*fs/1000)

    fig.patch.set_facecolor('white')
    fig.subplots_adjust(wspace=0.05,hspace=0)
    #tight_layout()
    fig.show()

def collectResults(savefolder,experimentType='',age=1,cvFoldColumn=False):
    """
    Collects and combines cross-validation results from multiple machine learning models and frequencies.
    This function reads CSV files containing cross-validation results for different machine learning models
    (Random Forest, SVC, XGBOOST, Rocket, HiveCote, MLP) and frequencies, combining them into a single DataFrame.
    Parameters
    ----------
    savefolder : str
        Path to the folder containing the CSV result files
    experimentType : str, optional
        Suffix to be added to the filenames, by default ''
    age : int, optional
        Age in months to be added to frequency suffixes, by default 1
    cvFoldColumn : bool, optional
        Whether to add a CV fold column to the output DataFrame, by default False
    Returns
    -------
    pandas.DataFrame
        Combined DataFrame containing all results with columns:
        - Original metrics from CSV files
        - 'Frequency': The frequency band for the results
        - 'Model': The name of the machine learning model
        - 'CV_fold': (optional) The cross-validation fold number
    Notes
    -----
    The function attempts to read results for various model configurations:
    - Base models
    - Models with Anova feature selection (10%)
    - Models with limited time window (10ms)
    - For different frequency bands (Global, NoHighFreq, OnlyLowFreq, etc.)
    Missing files are silently ignored (exceptions caught and passed).
    """

    suffices = ['Global','NoHighFreq','OnlyLowFreq','OnlyBadFreq','Click','3000','6000','12000','18000','24000','30000','36000','42000']
    if age!=1:
        suffices = [el+'_'+str(age)+'months' for el in suffices]

    realSuff = ['Global','NoHighFreq','OnlyLowFreq','OnlyBadFreq','Click','3kHz','6kHz','12kHz','18kHz','24kHz','30kHz','36kHz','42kHz']
    master = pd.DataFrame()
    rows = []
    for i,suff in enumerate(suffices):


        try:
            res = pd.read_csv(os.path.join(savefolder,f'forest{experimentType}_kFoldCrossValidation_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'Random Forest'
            if cvFoldColumn:
                res['CV_fold'] = np.arange(res.shape[0])+1
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'forest{experimentType}_kFoldCrossValidation_AnovaFS10percent'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'Random Forest Anova FS'
            if cvFoldColumn:
                res['CV_fold'] = np.arange(res.shape[0])+1
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'SVC{experimentType}_kFoldCrossValidation_AnovaFS10percent_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'SVC Anova FS'
            if cvFoldColumn:
                res['CV_fold'] = np.arange(res.shape[0])+1
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'XGBOOST{experimentType}_kFoldCrossValidation_AnovaFS10percent_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'XGBOOST Anova FS'
            if cvFoldColumn:
                res['CV_fold'] = np.arange(res.shape[0])+1
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'Rocket{experimentType}_kFoldCrossValidation_AnovaFS10percent_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'Rocket Anova FS'
            if cvFoldColumn:
                res['CV_fold'] = np.arange(res.shape[0])+1
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'hivecote{experimentType}_kFoldCrossValidation_AnovaFS10percent_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'HiveCote Anova FS'
            if cvFoldColumn:
                res['CV_fold'] = np.arange(res.shape[0])+1
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'MLP{experimentType}_kFoldCrossValidation_AnovaFS10percent_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'MLP Anova FS'
            if cvFoldColumn:
                res['CV_fold'] = np.arange(res.shape[0])+1
            rows.append(res)
        except:
            pass
    
        try:
            res = pd.read_csv(os.path.join(savefolder,f'SVC{experimentType}_kFoldCrossValidation_'+suff+'.csv'))
            res['Frequency'] = realSuff[i]
            res['Model'] = 'SVC'
            if cvFoldColumn:
                res['CV_fold'] = np.arange(res.shape[0])+1
            rows.append(res)

        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'XGBOOST{experimentType}_kFoldCrossValidation_'+suff+'.csv'))
            res['Frequency'] = realSuff[i]
            res['Model'] = 'XGBOOST'
            if cvFoldColumn:
                res['CV_fold'] = np.arange(res.shape[0])+1
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'forest{experimentType}-featureselection_kFoldCrossValidation_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'Random Forest-feat.select.'
            if cvFoldColumn:
                res['CV_fold'] = np.arange(res.shape[0])+1
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'svc{experimentType}-featureselection_kFoldCrossValidation_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'SVC-feat.select.'
            if cvFoldColumn:
                res['CV_fold'] = np.arange(res.shape[0])+1
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'XGBOOST{experimentType}-featureselection_kFoldCrossValidation_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'XGBOOST-feat.select.'
            if cvFoldColumn:
                res['CV_fold'] = np.arange(res.shape[0])+1
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'Rocket{experimentType}_kFoldCrossValidation_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'Rocket'
            if cvFoldColumn:
                res['CV_fold'] = np.arange(res.shape[0])+1
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'hivecote{experimentType}_kFoldCrossValidation_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'HiveCote'
            if cvFoldColumn:
                res['CV_fold'] = np.arange(res.shape[0])+1
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'MLP{experimentType}_kFoldCrossValidation_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'MLP'
            if cvFoldColumn:
                res['CV_fold'] = np.arange(res.shape[0])+1
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'forest{experimentType}_kFoldCrossValidation_10ms_'+suff+'.csv'))
            res['Frequency'] =  realSuff[i]
            res['Model'] = 'Random Forest limited'
            if cvFoldColumn:
                res['CV_fold'] = np.arange(res.shape[0])+1
            rows.append(res)
        except:
            pass
        try:
            res = pd.read_csv(os.path.join(savefolder,f'SVC{experimentType}_kFoldCrossValidation_10ms_'+suff+'.csv'))
            res['Frequency'] = realSuff[i]
            res['Model'] = 'SVC limited'
            if cvFoldColumn:
                res['CV_fold'] = np.arange(res.shape[0])+1
            rows.append(res)


        except:
            pass

    master = pd.concat(rows)
    
    try:
        master = master.drop('Unnamed: 0',axis=1)
    except:
        pass
    return master


def interFunc(x,a,b,c,d):    
    return a*exp(-(x-c)/b)+d 


def loadKingsData(shift=54,scaling=False,filename = '../data/Kings - MAchineLEarningABR_ExperimentList.xlsx',dataFolder = '../data'):
    """
    Load and preprocess ABR (Auditory Brainstem Response) data from Kings dataset.
    This function loads ABR data from an Excel file, processes it by extracting specific 
    frequency-intensity pairs, and optionally applies scaling. The data is then split into 
    training and test sets.
    Parameters
    ----------
    shift : int, optional (default=54)
        Number of initial data points to exclude from the final dataset
    scaling : bool, optional (default=False)
        Whether to apply scaling to the data using predetermined parameters
    filename : str, optional (default='../data/Kings - MAchineLEarningABR_ExperimentList.xlsx')
        Path to the Excel file containing the experiment list
    dataFolder : str, optional (default='../data')
        Path to the folder containing the ABR data files
    Returns
    -------
    X_train : numpy.ndarray
        Training features
    X_test : numpy.ndarray
        Testing features
    y_train : numpy.ndarray
        Training labels
    y_test : numpy.ndarray
        Testing labels
    X_kings : numpy.ndarray
        Complete feature set
    y_kings : numpy.ndarray
        Complete label set
    Notes
    -----
    - The function processes ABR data for frequency of 100Hz and intensities from 15 to 85 dB
    - Data is labeled as either 'Repaired' or '6N' based on the Status column
    - Uses predetermined scaling parameters if scaling=True
    """
    
    lowestInt = 15
    highestInt = 85
    kingsData = pd.read_excel(filename)
    kingsData.loc[kingsData['Status']=='Ahl-Repaired','Strain']='Repaired'
    kingsData.loc[kingsData['Status']=='UnRepaired','Strain']='6N'

    popt = np.array([ 3.26223496e+04,  2.12954754e+03, -1.34506980e+04,  1.11239997e+00]) # Standard parameters for scaling
    X_kings = []
    y_kings  = []
    index = 0
    for j,el in kingsData.iterrows():
        fname = el['Folder 1']
        t = extractABR(os.path.join(dataFolder,fname))

        pairs = []

        for fr in [100]:
            for ii in range(lowestInt,highestInt+5,5):
                pairs.append([fr,ii])

        try:
            X_kings.append(t.loc[[(p[0],p[1]) for p in pairs],:].values.ravel())
            y_kings.append(el['Strain'])
        except KeyError as e:
            print(e)
            index = index+1
            print(index)
    X_kings = np.array(X_kings)
    y_kings = np.array(y_kings)
    X_kings = X_kings[:,shift:]

    if scaling:
        X_kings_scaled = X_kings.copy()
        for i in range(X_kings_scaled.shape[0]):
            #X_kings_scaled[i,matchingPoints[0,0]:matchingPoints[-1,0]]=X_kings_scaled[i,matchingPoints[0,0]:matchingPoints[-1,0]]/ f(arange(matchingPoints[0,0],(matchingPoints[-1,0])))
            X_kings_scaled[i,100:]=X_kings_scaled[i,100:]/ interFunc(np.arange(100,X_kings_scaled.shape[1]),*popt)
        X_kings = X_kings_scaled

    X_train,X_test, y_train,y_test = train_test_split(X_kings,y_kings,test_size=0.25,random_state=42)

    return X_train,X_test,y_train,y_test,X_kings,y_kings

def loadSheffieldData(shift=54,dataFolder='../data'):
    """
    Loads and preprocesses Sheffield data for comparison of classification tasks with the Kings data.

    This function loads data from the Sheffield dataset, applies optional time-shifting,
    and splits it into training and testing sets.

    Parameters
    ----------
    shift : int, optional
        Number of time points to shift/trim from end of sequences (default is 54)
    dataFolder : str, optional
        Path to the folder containing the data files (default is '../data')

    Returns
    -------
    X_train : numpy.ndarray
        Training features
    X_test : numpy.ndarray
        Testing features
    y_train : numpy.ndarray
        Training labels
    y_test : numpy.ndarray
        Testing labels
    X_full : numpy.ndarray
        Combined training and testing features
    y_full : numpy.ndarray
        Combined training and testing labels
    dataVersion : str
        Version identifier of the dataset

    Notes
    -----
    The function uses fixed parameters for intensity range (15-85) and data splitting (25% test size).
    It processes data for age group 1 and 100Hz frequency only.
    """
    lowestInt = 15
    highestInt = 85
    X_train,  X_test,y_train,y_test,dataVersion = createClassificationDataset(test_size=0.25,oversample=False,ages=[1,],frequencies=[100],lowestInt=lowestInt,highestInt=highestInt,datafolder = dataFolder)
    if shift is not None:
        X_train = X_train[:,:-shift]
        X_test = X_test[:,:-shift]
    X_full = np.vstack([X_train,X_test])
    y_full = np.hstack([y_train,y_test])

    return X_train,X_test,y_train,y_test,X_full,y_full,dataVersion