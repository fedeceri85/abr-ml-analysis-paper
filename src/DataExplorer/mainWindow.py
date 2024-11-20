"""
GUI application for analyzing Auditory Brainstem Response (ABR) data.
This application provides a comprehensive interface for visualizing and analyzing ABR waveforms from the primary cohort.
It allows users to:
- Navigate through different mice data sets
- View ABR traces across different frequencies and intensities
- Mark and analyze wave peaks
- Perform automated peak detection
- Save analysis results
- Conduct blind classification of ABRs as 6N or Repaired
- Generate analysis plots
The main window displays a grid of ABR traces, with frequencies on columns and intensities on rows.
A separate window shows detailed wave analysis for individual traces.
Keyboard Controls:
    W: Move up
    S: Move down
    A: Move left
    D: Move right
    R: Guess wave peaks for higher intensities
    F: Guess wave peaks for lower intensities
    Z: Previous mouse
    X: Next mouse
    N: Classify as 6N (blind mode only)
    M: Classify as Repaired (blind mode only)
Dependencies:
    - PyQt5
    - pyqtgraph
    - numpy
    - pandas
    - seaborn
    - matplotlib
Configuration:
    BLINDCLASSIFICATION: Boolean flag for blind classification mode
    SHOWONLYCLICK: Boolean flag to show only click stimuli
Data Structure:
    Expected data folder structure:
    - data/
        - waveAnalysisResults/
        - [mouse data folders]

Copyright (c) 2024 Federico Ceriani, University of Sheffield.


A set of utilities for analyzing Auditory Brainstem Response (ABR) data.
This module provides functions for loading, preprocessing, analyzing and visualizing ABR data.

Author: Federico Ceriani
Email: f.ceriani at sheffield.ac.uk
License: APACHE 2.0

"""

from PyQt5.Qt import QApplication
from PyQt5.QtCore import Qt
from PyQt5 import QtGui,QtWidgets
from pyqtgraph.parametertree import Parameter, ParameterTree
import numpy as np
import sys
sys.path.append('../')
import abrTools as at
import os
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore,QtWidgets
from wavePeaksWindow import myGLW, findNearestPeak
import pandas as pd
import random


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

import seaborn as sns
tips = sns.load_dataset("tips")


# Set white graph
pg.setConfigOptions(antialias=True)
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


_instance = QApplication.instance()
if not _instance:
    _instance = QApplication([])
app = _instance


BLINDCLASSIFICATION = False # Flag to use the data for a blind classification test 
SHOWONLYCLICK = False # Flag to show only the click stimuli


frequencies = ['Click', '3 kHz', '6 kHz', '12 kHz', '18 kHz', '24 kHz',
       '30 kHz', '36 kHz', '42 kHz']
intensitiesL = [str(x)+'dB' for x in range(0,100,5)]
     
frequenciesDict = {
    100:'Click',
    3000: '3 kHz',
    6000: '6 kHz',
    12000: '12 kHz',
    18000: '18 kHz',
    24000: '24 kHz',
    30000: '30 kHz',
    36000: '36 kHz',
    42000: '42 kHz',
}

fs = 195000.0/2.0 # Acquisition sampling rate
dataFolder = '../../data'
waveAnalysisFolder = os.path.join(dataFolder,'waveAnalysisResults')

data,thresholds,dataversion = at.loadFiles(datafolder=dataFolder)
thresholds['ID'] = thresholds['MouseN - AGE'].str.split(' - ',expand=True)[0].astype(int)
thresholds['Age (months)'] = thresholds['MouseN - AGE'].str.split(' - ',expand=True)[1].str.split('month',expand=True)[0].astype(int)
strain = []
for el in thresholds['ID']:
    strain.append(data.loc[data['ID']==el,'Strain'].values[0])
thresholds['Strain'] = strain
dates = data['ID'].unique()

if BLINDCLASSIFICATION:
    random.shuffle(dates)

    strain2 = []
    for el in dates:
        strain2.append(data.loc[data['ID']==el,'Strain'].values[0])
    
    classificationDataFrame = pd.DataFrame({'Strain':strain2},index=dates)
    classificationDataFrame['Guess'] = None

def makeFigureqt(h1,h2,out,layout,title,wavePoints = None,plotDict = None,wavePointsPlotDict = None,thresholds = None,linecolor=None):
    """

    Make a figure from ABR trace data using pyqtgraph. Modifies an existing figure if the pyqtgraphs plots are passed through plotDict

    Parameters
    ----------
    h1 : array-like
        Frequencies for each trace
    h2 : array-like 
        Intensities for each trace
    out : ndarray
        2D array containing the ABR trace data, shape (n_traces, n_timepoints)
    layout : QGraphicsLayout
        PyQtGraph layout object to add plots to
    title : str
        Title for the figure
    wavePoints : pandas.DataFrame, optional
        DataFrame containing wave peak/trough points. Must have columns 'Freq' and 'Intensity'
    plotDict : dict, optional
        Dictionary mapping plot IDs to existing pyqtgraph plot items to update
    wavePointsPlotDict : dict, optional 
        Dictionary mapping plot IDs to existing scatter plot items for wave points
    thresholds : dict, optional
        Dictionary mapping frequencies to threshold intensities
    linecolor : str, optional
        Color to use for plotting traces in blind classification mode
    Returns
    -------
    tuple
        (plots, wavePointsPlot, plotToFreqIntMap)
        - plots: Dictionary mapping plot IDs to pyqtgraph plot items
        - wavePointsPlot: Dictionary mapping plot IDs to scatter plot items
        - plotToFreqIntMap: Dictionary mapping (row,col) plot positions to (frequency,intensity) pairs
    Notes
    -----
    The figure is arranged in a grid with intensities on rows (descending) and frequencies on columns.
    Each subplot shows the ABR trace for that frequency-intensity combination.
    Wave points are optionally plotted as scatter points on each trace.
    Trace colors are determined by thresholds in non-blind mode.
    """

    frequency = list(set(h1))#[100,3000,6000, 12000,18000,24000,30000,36000,42000 ]
    frequency.sort()
    intensity = list(set(h2))#arange(0,100,5)
    intensity.sort()
    nint = len(intensity)
    nfreq=len(frequency)
    freqmap=dict(zip(frequency,np.arange(len(frequency))))
    imap = dict(zip(intensity,np.arange(len(intensity))))
    time = np.arange(out.shape[1])/fs*1000

    ymin = out.min()
    ymax = out.max()
    if nint==1:
        nint=2
    if nfreq==1:
        nfreq=2
    plots = {}
    wavePointsPlot = {}
    plotToFreqIntMap = {}  # Dictionary that maps (row,col) of the plots to (freq, intensity) pairs

    if plotDict is not None:
        for el in plotDict.keys():
            plotDict[el].setData([],[])
    if wavePointsPlotDict is not None:
        for el in wavePointsPlotDict.keys():
            wavePointsPlotDict[el].setData([],[])


    for i in range(len(h1)):
        column = freqmap[int(h1[i])]
        row = imap[int(h2[i])]
        plotToFreqIntMap[(nint-row-1,column)] = (int(h1[i]),int(h2[i]))
        #plotn = i+row*len(frequency)
        linecol ='k'
        if not BLINDCLASSIFICATION:
            if thresholds is not None:
                if h2[i]>=thresholds[h1[i]]:
                    linecol = 'r'
                else:
                    linecol = 'k'
        else:
            linecol = linecolor


        ## Add the traces to the plots
        plotID = str(nint-row-1)+ ' ' + str(column)
        if plotDict is None:
            p2 = layout.addPlot(nint-row-1,column)

            #p2.addLegend(offset=(50, 0))
            pl = p2.plot(time,np.array(out)[i,:], pen=linecol, name='p1.1')
            p2.hideAxis('left')
            p2.hideAxis('right')
            p2.hideAxis('bottom')
            #p2.setYRange(ymin,ymax,padding = 0.1)
            p2.setXRange(0,10,padding = 0)
            
            plots[plotID] = pl
            p2.setObjectName(plotID)


        else:
            try:
                pl = plotDict[str(nint-row-1)+ ' ' + str(column)]
                pl.setData(time,np.array(out)[i,:],pen=linecol)

               # pl.setPen(pg.mkPen(linecol,width=5))    
                layout.getItem(nint-row-1,column).setXRange(0,10,padding = 0)

            except KeyError:
                p2 = layout.addPlot(nint-row-1,column)

                #p2.addLegend(offset=(50, 0))
                pl = p2.plot(time,np.array(out)[i,:], pen=linecol, name='p1.1')
                p2.hideAxis('left')
                p2.hideAxis('right')
                p2.hideAxis('bottom')
                #p2.setYRange(ymin,ymax,padding = 0)
                p2.setXRange(0,10,padding = 0)
               
                plots[plotID] = pl
                p2.setObjectName(plotID)

        #Add the points for the wave peaks and trough to the plots
        if wavePoints is not None:
            slice = wavePoints.loc[(wavePoints['Freq']==int(h1[i])) & (wavePoints['Intensity']==int(h2[i]) )]
            if slice.shape[0]>1:
                print("More than one entry for this frequency-intensity combination. Serious error")
                print(h1[i])
                print(h2[i])
            if slice.shape[0]==1:
                x,y = wavePointsToScatter(slice)

                if wavePointsPlotDict is None:
                    p2 = layout.getItem(nint-row-1,column)
                    p1Point = pg.ScatterPlotItem(x=[],y=[],symbol = '+',size=3)
                            #p1Label.setPos(np.nan,np.nan)
                    p2.addItem(p1Point)
                    

                    p1Point.setData(x,y)
                    wavePointsPlot[plotID]= p1Point
                else:
                    try:
                        p1Point = wavePointsPlotDict[str(nint-row-1)+ ' ' + str(column)]

                        p1Point.setData(x,y)
                    except KeyError: 
                        p2 = layout.getItem(nint-row-1,column)
                        p1Point = pg.ScatterPlotItem(x=[],y=[],symbol = '+',size=3)
                                #p1Label.setPos(np.nan,np.nan)
                        p2.addItem(p1Point)
                        

                        p1Point.setData(x,y)
                        wavePointsPlot[plotID]= p1Point
        #axs[nint-row-1,column].plot(np.array(out)[i,:],c=linecol)
        #axs[nint-row-1,column].set_ylim((array(out).min(),array(out).max()))
    if plotDict is not None:
        plots.update(plotDict)

    if wavePointsPlotDict is not None:
        wavePointsPlot.update(wavePointsPlotDict)

    return plots,wavePointsPlot,plotToFreqIntMap

def wavePointsToScatter(wavePoints):
    xlabels = ['P'+str(i)+'_x' for i in [1,2,3,4]] + ['N'+str(i)+'_x' for i in [1,2,3,4]]
    ylabels = ['P'+str(i)+'_y' for i in [1,2,3,4]]  + ['N'+str(i)+'_y' for i in [1,2,3,4]]  
    x = wavePoints[xlabels].values[0,:]
    y = wavePoints[ylabels].values[0,:]
    return (x,y)
class resultWindow(QtWidgets.QMainWindow):
    """
    A class for displaying analysis results in a window using PyQt5.
    This window displays two seaborn plots showing the relationships between intensity
    and both latency and amplitude across different frequencies for ABR wave analysis.
    Parameters
    ----------
    wavepoints : pandas.DataFrame
        DataFrame containing the wave analysis data with columns:
        - P1_x: x-coordinates of P1 points
        - P1_y: y-coordinates of P1 points 
        - N1_y: y-coordinates of N1 points
        - Intensity: stimulus intensity values
        - Freq: stimulus frequency values
    Attributes
    ----------
    fig : matplotlib.figure.Figure
        First figure showing Latency vs Intensity relationship
    fig2 : matplotlib.figure.Figure
        Second figure showing Amplitude vs Intensity relationship
    canvas : FigureCanvas
        Canvas for displaying the first figure
    canvas2 : FigureCanvas
        Canvas for displaying the second figure
    main_widget : QWidget
        Main widget containing the layout
    layout : QGridLayout
        Grid layout for arranging the plot canvases
    Notes
    -----
    The window is positioned at coordinates (2500,0) with dimensions 1300x1000 pixels.
    The plots are arranged vertically in a grid layout and show the relationships
    faceted by frequency.
    """
    def __init__(self, wavepoints):
        super().__init__()

        self.setWindowTitle('ABR Wave Analysis - Results') 
        self.setGeometry(0,0,1300,1000)
        self.move(2500,0)   
        self.main_widget = QtWidgets.QWidget(self)

        wavepoints2 = wavepoints.copy()
        wavepoints2['Latency (ms)'] = wavepoints2['P1_x']
        wavepoints2['Amplitude (uV)'] = np.abs(wavepoints2['P1_y']-wavepoints2['N1_y'])
        fg = sns.relplot(data=wavepoints2,x='Intensity',y='Latency (ms)',col='Freq')
        self.fig = fg.figure
        self.fig.tight_layout()

        self.canvas = FigureCanvas(self.fig)

        fg2 = sns.relplot(data=wavepoints2,x='Intensity',y='Amplitude (uV)',col='Freq')
        
        self.fig2 = fg2.figure
        self.fig2.tight_layout()
        self.canvas2 = FigureCanvas(self.fig2)

        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                       QtWidgets.QSizePolicy.Expanding)
        self.canvas.updateGeometry()

        self.canvas2.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                       QtWidgets.QSizePolicy.Expanding)
        self.canvas2.updateGeometry()
        # self.button = QtWidgets.QPushButton("Button")
        # self.label = QtWidgets.QLabel("A plot:")

        self.layout = QtWidgets.QGridLayout(self.main_widget)
        # self.layout.addWidget(self.button)
        # self.layout.addWidget(self.label)
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.canvas2)
        self.setCentralWidget(self.main_widget)

    def seabornplot(self):
        # g = sns.FacetGrid(tips, col="sex", hue="time", palette="Set1",
        #                             hue_order=["Dinner", "Lunch"])
        # g.map(plt.scatter, "total_bill", "tip", edgecolor="w")
        g = sns.catplot(data=tips,x='total_bill',y='tip',col="sex", hue="time", palette="Set1",
                                    hue_order=["Dinner", "Lunch"])
        return g.figure

class abrWindow(pg.GraphicsView):
    """
    A window class for displaying and analyzing Auditory Brainstem Response (ABR) data.
    This class provides a graphical interface for visualizing and analyzing ABR waveforms.
    It allows users to navigate through different mice in the dataset, mark wave peaks,
    and save analysis results.
    Inherits from pg.GraphicsView.
    Parameters
    ----------
    parent : QWidget, optional
        Parent widget
    useOpenGL : bool, optional 
        Whether to use OpenGL for rendering
    background : str, optional
        Background color, defaults to 'default'
    Attributes
    ----------
    wavePoints : pd.DataFrame
        DataFrame storing peak positions for analyzed waveforms
    plotDict : dict
        Dictionary mapping plot positions to plot items
    wavePointsPlotDict : dict
        Dictionary mapping plot positions to wave point markers
    plotToFreqIntMap : dict
        Dictionary mapping plot positions to frequency/intensity pairs
    activeRowCol : tuple
        Currently selected plot position as (row, col)
    threshDict : dict
        Dictionary mapping frequencies to threshold values
    Methods
    -------
    initData()
        Initialize the display with ABR data
    loadWaveAnalysis()
        Load previously saved wave analysis results
    setActivePlot(row, col)
        Set the currently active plot
    highlightTraceAt(row, col, pensize)
        Highlight a specific trace
    getTraceAt(row, col) 
        Get trace data at specified position
    retrieveResultsCb()
        Callback to store wave point analysis
    guessWavePoints(direction)
        Automatically detect wave peaks
    plotResultsCb()
        Display analysis results plot
    Key Commands
    -----------
    W : Move up
    S : Move down
    A : Move left
    D : Move right
    R : Guess higher intensity wave points 
    F : Guess lower intensity wave points
    Z : Previous mouse
    X : Next mouse
    N : Classify as 6N (blind mode only)
    M : Classify as Repaired (blind mode only)
    """
    def __init__(self, parent=None, useOpenGL=None, background='default'):
        super().__init__(parent, useOpenGL, background)

        self.setWindowTitle('Your title') 
        self.setGeometry(0,0,1300,1000)
        self.layout = pg.GraphicsLayout()
#layout.layout.setContentsMargins(-100,-100,-100,-100)
        self.outerlayout = pg.GraphicsLayout()

        self.titleLabel = self.outerlayout.addLabel('Title',color='k',size='16pt',bold=True,row=0,col=0,colspan=10)

        for i,freq in enumerate(frequencies):
            self.outerlayout.addLabel(freq,color='k',size='8pt',col=i+1,row=1)
        
        for i,intens in enumerate(intensitiesL[::-1]):
            self.outerlayout.addLabel(intens,color='k',size='8pt',col=0,row=i+2)

        self.outerlayout.addItem(self.layout,colspan=len(frequencies),rowspan=len(intensitiesL),row=2,col=1)

        self.setCentralItem(self.outerlayout)
        
        self.sc = self.scene()
        self.sc2 = self.layout.scene()

        params = [
        # {'name':'Strain','type':'list','values':['6N','Repaired']},
            {'name':'ID','type':'list','values':dict(zip(dates.astype(str),np.arange(dates.size)))},
            {'name':'Age','type':'list','values':[1,3,5,6,9,12]},
            {'name':'Prev mouse','type':'action'},
            {'name':'Next mouse','type':'action'},
            {'name':'Guess Wave Peak positions','type':'action'},
            {'name':'Guess Wave Peak higher intensities','type':'action'},
            {'name':'Guess Wave Peak lower intensities','type':'action'},
            {'name':'Plot results','type':'action'},                        
            {'name':'Save results','type':'action'},            ]


        ## Create tree of Parameter objects
        self.p = Parameter.create(name='params', type='group', children=params)
        self.t = ParameterTree()
        self.t.setParameters(self.p, showTop=False)
        self.t.setGeometry(1300,800,300,200)


        self.waveAnalysisWidget = myGLW(show=True)


        self.activeRowCol = (0,0)

        self.makeConnections()
        self.show()
        self.t.show()
        self.initData()


    def initData(self):
        abr = at.extractABR(os.path.join(dataFolder,data.loc[[1],'Folder 1'].values[0]))

        self.wavePoints = pd.DataFrame(columns=['Freq',	'Intensity','P1_x','P1_y','N1_x','N1_y','P2_x','P2_y','N2_x','N2_y','P3_x','P3_y','N3_x','N3_y','P4_x','P4_y','N4_x','N4_y'])

        freqs = []
        intens = []
        for el in abr.index:
            freqs.append(el[0])
            intens.append(el[1])
        self.plotDict,self.wavePointsPlotDict, self.plotToFreqIntMap = makeFigureqt(freqs,intens,abr.values,self.layout,'',wavePoints=None)
        self.changeIDCb()
        self.setActivePlot(0,0)

    def loadWaveAnalysis(self):
        dat = dates[self.p['ID']]
        filename = str(dat)+' - '+str(self.p['Age'])+'month.csv'
        #print(filename)
        try:
            if not BLINDCLASSIFICATION:
                self.wavePoints = pd.read_csv(os.path.join(waveAnalysisFolder,filename))
            else:
                raise FileNotFoundError #If blind classification, we don't load the data analysis
        except FileNotFoundError:
            self.wavePoints = pd.DataFrame(columns=['Freq',	'Intensity','P1_x','P1_y','N1_x','N1_y','P2_x','P2_y','N2_x','N2_y','P3_x','P3_y','N3_x','N3_y','P4_x','P4_y','N4_x','N4_y'])
            print('Wave analysis not found')

    def setActivePlot(self,row=0,col=0):

        # reset previous active plot
        self.highlightTraceAt(self.activeRowCol[0],self.activeRowCol[1],1)
        
        try:
            self.highlightTraceAt(row,col,3)
        except KeyError:
            #if the trace not present go back to the previous one and break
            self.highlightTraceAt(self.activeRowCol[0],self.activeRowCol[1],3)
            return
        #set the new active plot
        self.activeRowCol=(row,col)


        data = self.getTraceAt(row,col)

        freq,intens = self.plotToFreqIntMap[(row,col)]
        print('Selected trace Freq: '+str(freq)+'   Intensity: '+str(intens))
        if self.wavePoints is not None:
            selectedWavePoints = self.wavePoints.loc[(self.wavePoints['Freq']==freq) & (self.wavePoints['Intensity']==intens) ]
            
            if selectedWavePoints.shape[0] ==0:
                selectedWavePoints = None
        else:
            selectedWavePoints = None
        self.waveAnalysisWidget.setData(data[0],data[1],wavePoints = selectedWavePoints)

    def highlightTraceAt(self,row,col,pensize=3):
        msg =  str(row)+' '+str(col)
        try:
            penColor = self.plotDict[msg].opts['pen'].color()
        except AttributeError:
            penColor = self.plotDict[msg].opts['pen']

        self.plotDict[msg].setPen(pg.mkPen(penColor,width=pensize))
    
    def getTraceAt(self,row,col):
        msg =  str(row)+' '+str(col)
        return self.plotDict[msg].getData()
        
    def makeConnections(self):
        #p.keys()['Strain'].sigValueChanged.connect(changeStrainCb)
        self.p.keys()['Age'].sigValueChanged.connect(self.changeAgeCb)
        self.p.keys()['ID'].sigValueChanged.connect(self.changeIDCb)
        self.p.keys()['Next mouse'].sigActivated.connect(self.nextMouseCb)
        self.p.keys()['Prev mouse'].sigActivated.connect(self.prevMouseCb)
        self.p.keys()['Save results'].sigActivated.connect(self.saveResultsCb)
        self.p.keys()['Guess Wave Peak positions'].sigActivated.connect(lambda: self.guessWavePoints('both'))
        self.p.keys()['Guess Wave Peak higher intensities'].sigActivated.connect(lambda: self.guessWavePoints('higher'))
        self.p.keys()[ 'Guess Wave Peak lower intensities'].sigActivated.connect(lambda: self.guessWavePoints('lower'))
        self.p.keys()['Plot results'].sigActivated.connect(self.plotResultsCb)

        self.waveAnalysisWidget.finishSignal.connect(self.retrieveResultsCb)
        self.waveAnalysisWidget.changeTraceSignal.connect(self.navigateTraces)
        self.waveAnalysisWidget.guessAboveSignal.connect(lambda: self.guessWavePoints('higher'))
        self.waveAnalysisWidget.guessBelowSignal.connect(lambda: self.guessWavePoints('lower'))
        self.sc2.sigMouseClicked.connect(self.onMouseClicked)


    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_W:
            self.navigateTraces('Up')
        elif ev.key() == Qt.Key_S:
            self.navigateTraces('Down')
        elif ev.key() == Qt.Key_A:
            self.navigateTraces('Left')        
        elif ev.key() == Qt.Key_D:
            self.navigateTraces('Right')
        elif ev.key() == Qt.Key_R:
            self.guessWavePoints("higher")
        elif ev.key() == Qt.Key_F:
            self.guessWavePoints("lower")
        elif ev.key() == Qt.Key_Z:
            self.prevMouseCb()
        elif ev.key() == Qt.Key_X:
            self.nextMouseCb()
        elif ev.key() == Qt.Key_N:
            if BLINDCLASSIFICATION:
                self.classifyCB('6N')
        elif ev.key() == Qt.Key_M:
            if BLINDCLASSIFICATION:
                self.classifyCB('Repaired')
        
    def classifyCB(self,strain):
        '''
        If in Blindtest mode, classify the mouse as 6N or Repaired
        '''
        dat = dates[self.p['ID']]
        classificationDataFrame.loc[dat,'Guess'] = strain
        classificationDataFrame.to_csv('classificationDataFrame.csv')
        self.updateCurrentPlotCb()

    def navigateTraces(self,a):
        row, col = self.activeRowCol
        if a == 'Up':
            self.setActivePlot(row-1,col)
        elif a == 'Down':
            self.setActivePlot(row+1,col)
        elif a =='Left':
            self.setActivePlot(row,col-1)
        elif a == 'Right':
            self.setActivePlot(row,col+1)

    def saveResultsCb(self):
        dat = dates[self.p['ID']]
        filename = str(dat)+' - '+str(self.p['Age'])+'month.csv'
        
        self.wavePoints.sort_values(['Freq','Intensity']).to_csv(os.path.join(waveAnalysisFolder,filename),index=False)

    def retrieveResultsCb(self):
    
        points = self.waveAnalysisWidget.getPoints()

        #Check if the combination of freq and intensity exists already in the wavePoints DataFrame
        freq,intens = self.plotToFreqIntMap[self.activeRowCol]

        selectedWavePoints = self.wavePoints.loc[(self.wavePoints['Freq']==freq) & (self.wavePoints['Intensity']==intens) ]
        if selectedWavePoints.shape[0] ==0:
            self.wavePoints = pd.concat([self.wavePoints,
                    pd.DataFrame({
                    'Freq':freq,
                    'Intensity':intens,
                    'P1_x': points['P1'][0],
                    'P1_y' : points['P1'][1],
                    'N1_x': points['N1'][0],
                    'N1_y' : points['N1'][1],

                    'P2_x': points['P2'][0],
                    'P2_y' : points['P2'][1],
                    'N2_x': points['N2'][0],
                    'N2_y' : points['N2'][1],

                    'P3_x': points['P3'][0],
                    'P3_y' : points['P3'][1],
                    'N3_x': points['N3'][0],
                    'N3_y' : points['N3'][1],

                    'P4_x': points['P4'][0],    
                    'P4_y' : points['P4'][1],
                    'N4_x': points['N4'][0],
                    'N4_y' : points['N4'][1],                

            },index=[0])],ignore_index=True,axis=0)
      

        elif selectedWavePoints.shape[0] ==1:
            for ii in [1,2,3,4]:
                self.wavePoints.loc[(self.wavePoints['Freq']==freq) & (self.wavePoints['Intensity']==intens),'P'+str(ii)+'_x'] =  points['P'+str(ii)][0]
                self.wavePoints.loc[(self.wavePoints['Freq']==freq) & (self.wavePoints['Intensity']==intens) ,'N'+str(ii)+'_x'] =  points['N'+str(ii)][0]
                self.wavePoints.loc[(self.wavePoints['Freq']==freq) & (self.wavePoints['Intensity']==intens) ,'P'+str(ii)+'_y'] =  points['P'+str(ii)][1]
                self.wavePoints.loc[(self.wavePoints['Freq']==freq) & (self.wavePoints['Intensity']==intens) ,'N'+str(ii)+'_y'] =  points['N'+str(ii)][1]

        print(self.wavePoints)

        self.updateCurrentPlotCb() 


    def changeAgeCb(self):
        #print(self.p['Age'])
        self.changeIDCb()

    def changeIDCb(self): #TODO: load the new WavePoints
        
        self.loadWaveAnalysis()
        self.updateCurrentPlotCb()
        self.setActivePlot(0,0)
        


    def updateCurrentPlotCb(self):
        dat = dates[self.p['ID']]
        indices = []
        agesFolderDict  = {1:'1',3:'2',6:'3',9:'4',12:'5'}
        currFolder = agesFolderDict[self.p['Age']]
        abr = at.extractABR(os.path.join(dataFolder,data.loc[data['ID']==dat,'Folder '+currFolder].values[0]))
        
        if SHOWONLYCLICK:
            abr = abr.loc[100]
        #abr2 = at.extractABR(os.path.join(dataFolder,data.loc[data['ID']==dat,'Folder 2'].values[0]))
        #abr = pd.concat([abr,abr2])
        freqs = []
        intens = []
        for el in abr.index:
            if not SHOWONLYCLICK:
                freqs.append(el[0])
                intens.append(el[1])
            else:
                freqs.append(100)
                intens.append(el)

        frequencies2 =[ 100,3000,6000,12000,18000,24000,30000,36000,42000]
        self.threshDict = dict(zip(frequencies2,thresholds.loc[thresholds['MouseN - AGE']==str(dat)+' - '+str(self.p['Age'])+'month',frequencies].values[0])) 
        
        if BLINDCLASSIFICATION:
            if classificationDataFrame.loc[dat,'Guess']=='6N':
                linecolor = 'r'
            elif classificationDataFrame.loc[dat,'Guess']=='Repaired':
                linecolor = 'b'
            else:
                linecolor = 'k'
        else:
            linecolor = None

        self.plotDict,self.wavePointsPlotDict, self.plotToFreqIntMap  = makeFigureqt(freqs,intens,abr.values,self.layout,'',plotDict=self.plotDict ,thresholds=self.threshDict,wavePoints=self.wavePoints,wavePointsPlotDict=self.wavePointsPlotDict,linecolor=linecolor)
        self.titleLabel.setText(str(dat)+' - '+str(self.p['Age'])+ ' month(s)')
        self.highlightTraceAt(self.activeRowCol[0],self.activeRowCol[1],3)


    def nextMouseCb(self):
        self.p.keys()['ID'].setValue(self.p['ID']+1)

    def prevMouseCb(self):
        #print(p['ID'])
        self.p.keys()['ID'].setValue(self.p['ID']-1)
    
    def onMouseClicked(self,evt):
        if evt.double():
            pass#
            '''
            items = self.sc.items(evt.scenePos())
            for item in items:
                if isinstance(item,pg.PlotItem):
                    plItem = self.plotDict [item.objectName()]
                    row, col = [int(ee) for ee in item.objectName().split(' ')]
                    plItem.setPen('r')
                    for i in range(row):
                        msg = str(i)+' '+str(col)
                        plItem = self.plotDict[msg]
                        plItem.setPen('r')
                    for i in range(row+1,20):
                        msg = str(i)+' '+str(col)
                        plItem = self.plotDict[msg]
                        plItem.setPen('k')
            '''
        else:
            items = self.sc.items(evt.scenePos())
            for item in items:
                if isinstance(item,pg.PlotItem):
                    #msg =  str(self.activeRowCol[0])+' '+str(self.activeRowCol[1])
                    plItem = self.plotDict[item.objectName()]
                    
              

                    row, col = [int(ee) for ee in item.objectName().split(' ')]

                    self.setActivePlot(row,col)


    
    def guessWavePoints(self, direction = 'both'):
        
        freq,initialIntens = self.plotToFreqIntMap[self.activeRowCol]
        threshold = self.threshDict[freq]

        if direction == 'both':
            lowerLimit = threshold
            higherLimit = 100
        elif direction =='higher':
            lowerLimit = initialIntens
            higherLimit = 100
        elif direction == 'lower':
            lowerLimit = threshold
            higherLimit = initialIntens


        activeRow = self.activeRowCol[0]
        activeCol = self.activeRowCol[1]
        for row in np.arange(20):
            if row!=activeRow:
                
                initialWavePoints = self.wavePoints.loc[(self.wavePoints['Freq']==freq) & (self.wavePoints['Intensity']==initialIntens) ]

                try:
                    trace = self.getTraceAt(row,activeCol)[1]
                    freq, intens =  self.plotToFreqIntMap[(row,activeCol)]

                    
                    if (intens>=lowerLimit) & (intens<=higherLimit):
                        print((freq,intens))
                        points = {}
                        for peak in [1,2,3,4]:
                            try:
                                initialGuess_x = int(initialWavePoints['P'+str(peak)+'_x'].values[0]*fs/1000)
                                guess_x = findNearestPeak(trace, initialGuess_x)    
                                points['P'+str(peak)] = (guess_x/fs*1000,trace[guess_x])
                            except ValueError:
                                points['P'+str(peak)] = (np.nan,np.nan)

                            try:
                                initialGuess_x = int(initialWavePoints['N'+str(peak)+'_x'].values[0]*fs/1000)
                                guess_x = findNearestPeak(trace, initialGuess_x, negative=True  )
                                points['N'+str(peak)] = (guess_x/fs*1000,trace[guess_x])
                            except ValueError:
                                points['N'+str(peak)] = (np.nan,np.nan)

                        # check if the entry exists and update it or create it :
                        selectedWavePoints = self.wavePoints.loc[(self.wavePoints['Freq']==freq) & (self.wavePoints['Intensity']==intens) ]

                        initialIntens = intens
                        
                        if selectedWavePoints.shape[0] ==0:
                            self.wavePoints = pd.concat([self.wavePoints,
                                pd.DataFrame({
                                'Freq':freq,
                                'Intensity':intens,
                                'P1_x': points['P1'][0],
                                'P1_y' : points['P1'][1],
                                'N1_x': points['N1'][0],
                                'N1_y' : points['N1'][1],

                                'P2_x': points['P2'][0],
                                'P2_y' : points['P2'][1],
                                'N2_x': points['N2'][0],
                                'N2_y' : points['N2'][1],

                                'P3_x': points['P3'][0],
                                'P3_y' : points['P3'][1],
                                'N3_x': points['N3'][0],
                                'N3_y' : points['N3'][1],

                                'P4_x': points['P4'][0],    
                                'P4_y' : points['P4'][1],
                                'N4_x': points['N4'][0],
                                'N4_y' : points['N4'][1],                

                                },index=[0])],ignore_index=True,axis=0)
                        elif selectedWavePoints.shape[0] ==1:
                            for ii in [1,2,3,4]:
                                self.wavePoints.loc[(self.wavePoints['Freq']==freq) & (self.wavePoints['Intensity']==intens),'P'+str(ii)+'_x'] =  points['P'+str(ii)][0]
                                self.wavePoints.loc[(self.wavePoints['Freq']==freq) & (self.wavePoints['Intensity']==intens) ,'N'+str(ii)+'_x'] =  points['N'+str(ii)][0]
                                self.wavePoints.loc[(self.wavePoints['Freq']==freq) & (self.wavePoints['Intensity']==intens) ,'P'+str(ii)+'_y'] =  points['P'+str(ii)][1]
                                self.wavePoints.loc[(self.wavePoints['Freq']==freq) & (self.wavePoints['Intensity']==intens) ,'N'+str(ii)+'_y'] =  points['N'+str(ii)][1]
                except KeyError:
                    pass

        self.updateCurrentPlotCb()

        #print(sc.items())

    def plotResultsCb(self):
        self.resultPlot = resultWindow(wavepoints=self.wavePoints)
        self.resultPlot.show()
       


if __name__ == '__main__':
    win = abrWindow()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()