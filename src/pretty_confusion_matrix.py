# -*- coding: utf-8 -*-
"""
Pretty Confusion Matrix Plotting
Original source: https://github.com/wcipriano/pretty-print-confusion-matrix
Original author: Wagner Cipriano (wagnerbhbr at gmail.com) - CEFETMG / MMC
Modified by: Federico Ceriani
License: Apache License Version 2.0, January 2004

This code creates enhanced confusion matrix visualizations using seaborn.
"""

# Original reference links:
#   https://www.mathworks.com/help/nnet/ref/plotconfusion.html
#   https://stackoverflow.com/questions/28200786/how-to-plot-scikit-learn-classification-report
#   https://stackoverflow.com/questions/5821125/how-to-plot-confusion-matrix-with-string-axis-rather-than-integer-in-python
#   https://www.programcreek.com/python/example/96197/seaborn.heatmap
#   https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels/31720054
#   http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html


#imports
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh
import seaborn as sn
import matplotlib.colors as mplc

def get_new_fig(fn, figsize=[9,9]):
    """ Init graphics """
    fig1 = plt.figure(fn, figsize)
    ax1 = fig1.gca()   #Get Current Axis
    ax1.cla() # clear existing plot
    return fig1, ax1
#
def configcell_text_and_colors(array_df, lin, col, oText, facecolors, posi, fz, fmt, show_null_values=0):
    """
      config cell text and colors
      and return text elements to add and to dell
      @TODO: use fmt
    """
    text_add = []; text_del = [];
    cell_val = array_df[lin][col]
    tot_all = array_df[-1][-1]
    per = (float(cell_val) / tot_all) * 100
    curr_column = array_df[:,col]
    ccl = len(curr_column)

    #last line  and/or last column
    if(col == (ccl - 1)) or (lin == (ccl - 1)):
        #tots and percents
        if(cell_val != 0):
            if(col == ccl - 1) and (lin == ccl - 1):
                tot_rig = 0
                for i in range(array_df.shape[0] - 1):
                    tot_rig += array_df[i][i]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif(col == ccl - 1):
                tot_rig = array_df[lin][lin]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif(lin == ccl - 1):
                tot_rig = array_df[col][col]
                per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok
        else:
            per_ok = per_err = 0

        per_ok_s = ['%.1f%%'%(per_ok), '100%'] [per_ok == 100]

        #text to DEL
        text_del.append(oText)

        #text to ADD
        #Determine which metrics we are printing

        if (lin==ccl-1) and (col==0):
            prefix = '\nRecall\n'
        elif (lin==ccl-1) and (col==1):
            prefix = '\nSpecificity\n'
        elif (lin==ccl-1) and (col==2):
            prefix = '\nAccuracy\n'
        elif (col==ccl-1) and (lin==0):
            prefix = '\nPrecision\n'
        elif (col==ccl-1) and (lin==1):
            prefix = '\nNPV\n'
        else:
            prefix = ''

        font_prop = fm.FontProperties(weight='regular', size=fz)
        text_kwargs = dict(color='k', ha="center", va="center", gid='sum', fontproperties=font_prop)
        #lis_txt = ['%d'%(cell_val), per_ok_s, '%.1f%%'%(per_err)]
        lis_txt = ['%d'%(cell_val), prefix+per_ok_s]
        lis_kwa = [text_kwargs]
        dic = text_kwargs.copy(); dic['color'] = 'k'; lis_kwa.append(dic);
        dic = text_kwargs.copy(); dic['color'] = 'r'; lis_kwa.append(dic);
        lis_pos = [(oText._x, oText._y-0.3), (oText._x, oText._y), (oText._x, oText._y+0.3)]

        for i in range(len(lis_txt)):
            newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i])
            #print 'lin: %s, col: %s, newText: %s' %(lin, col, newText)
            text_add.append(newText)
        #print '\n'

        #set background color for sum cells (last line and last column)
        carr = [0.27, 0.30, 0.27, 1.0]
        if(col == ccl - 1) and (lin == ccl - 1):
            carr = [0.17, 0.20, 0.17, 1.0]
        facecolors[posi] = carr

    else:
        if(per > 0):
            #txt = '%s\n%.1f%%' %(cell_val, per)
            txt = '%s' %(cell_val)
        else:
            if(show_null_values == 0):
                txt = ''
            elif(show_null_values == 1):
                txt = '0'
            else:
                #txt = '0\n0.0%'
                txt = '0'
        oText.set_text(txt)
        oText.set_size(fz/0.75)
        #main diagonal
        if(col == lin):
            #set color of the textin the diagonal to white
            oText.set_color('k')
            # set background color in the diagonal to blue
            facecolors[posi] = [0.35, 0.8, 0.55, 1.0]
        else:
            oText.set_color('r')

    return text_add, text_del
#

def insert_totals(df_cm):
    """ insert total column and line (the last ones) """
    sum_col = []
    for c in df_cm.columns:
        sum_col.append( df_cm[c].sum() )
    sum_lin = []
    for item_line in df_cm.iterrows():
        sum_lin.append( item_line[1].sum() )
    df_cm['Total'] = sum_lin
    sum_col.append(np.sum(sum_lin))
    df_cm.loc['Total'] = sum_col
    #print ('\ndf_cm:\n', df_cm, '\n\b\n')
#

def pretty_plot_confusion_matrix(df_cm, annot=True, cmap="Oranges", fmt='.2f', fz=11,
      lw=0.5, cbar=False, figsize=[8,8], show_null_values=0, pred_val_axis='y',linecolor='w',innerFontSize=None):
    if innerFontSize is None:
        innerFontSize=fz
    """
      print conf matrix with default layout (like matlab)
      params:
        df_cm          dataframe (pandas) without totals
        annot          print text in each cell
        cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
        fz             fontsize
        lw             linewidth
        pred_val_axis  where to show the prediction values (x or y axis)
                        'col' or 'x': show predicted values in columns (x axis) instead lines
                        'lin' or 'y': show predicted values in lines   (y axis)
    """
    if(pred_val_axis in ('col', 'x')):
        xlbl = 'Predicted'
        ylbl = 'Actual'
    else:
        xlbl = 'Actual'
        ylbl = 'Predicted'
        df_cm = df_cm.T

    # create "Total" column
    
    insert_totals(df_cm)

    #this is for print allways in the same window
    fig, ax1 = get_new_fig('Conf matrix default', figsize)

    df_cm2 = np.zeros(df_cm.shape)
    
    colors = [ (249/255, 188/255, 183/255),(186/255,223/255, 191/255),(238/255,238/255,238/255)]# (173/255, 216/255, 230/255), (255/255, 223/255, 186/255), (255/255, 250/255, 205/255), (230/255, 230/255, 250/255)]
    for i in range(df_cm2.shape[0]):
        df_cm2[i,i]=1
        df_cm2[-1,i]=2
        df_cm2[i,-1]=2
    # df_cm2[-1,0] = 3
    # df_cm2[-1,1] = 4
    # df_cm2[0,-1] = 5
    # df_cm2[1,-1] = 6

    cm_cmap = mplc.LinearSegmentedColormap.from_list('cm_color', colors, N=3)
    df_cm2 = DataFrame(df_cm2,index=df_cm.index,columns = df_cm.columns)
    if cmap is not None:
        df_cm2 = df_cm.copy()
    #thanks for seaborn
    ax = sn.heatmap(df_cm2, annot=annot, annot_kws={"size": innerFontSize*0.75}, linewidths=lw, ax=ax1,
                    cbar=cbar, cmap=cm_cmap, linecolor=linecolor, fmt=fmt)

    #set ticklabels rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, fontsize = fz)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 90, fontsize = fz)

    # Turn off all the ticks
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    #face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    #iter in text elements
    array_df = np.array( df_cm.to_records(index=False).tolist() )
    text_add = []; text_del = [];
    posi = -1 #from left to right, bottom to top.
    for t in ax.collections[0].axes.texts: #ax.texts:
        pos = np.array( t.get_position()) - [0.5,0.5]
        lin = int(pos[1]); col = int(pos[0]);
        posi += 1
        #print ('>>> pos: %s, posi: %s, val: %s, txt: %s' %(pos, posi, array_df[lin][col], t.get_text()))

        #set text
        txt_res = configcell_text_and_colors(array_df, lin, col, t, facecolors, posi, innerFontSize*0.75, fmt, show_null_values)

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    #remove the old ones
    for item in text_del:
        item.remove()
    #append the new ones
    for item in text_add:
        ax.text(item['x'], item['y'], item['text'], **item['kw'])

    #titles and legends
    #ax.set_title('Confusion matrix')
    ax.set_xlabel('Actual genotype',fontsize=fz*1.0)
    ax.set_ylabel('Predicted genotype',fontsize=fz*1.0)
    fig.suptitle('Random forest',fontsize=fz)
    plt.tight_layout()  #set layout slim
    plt.show()
    return ax,fig
#

def plot_confusion_matrix_from_data(y_test, predictions, columns=None, annot=True, cmap="Oranges",
      fmt='.2f', fz=11, lw=0.5, cbar=False, figsize=[8,8], show_null_values=0, pred_val_axis='lin',linecolor='w',innerFontSize=None):
    """
        plot confusion matrix function with y_test (actual values) and predictions (predic),
        whitout a confusion matrix yet
    """
    from sklearn.metrics import confusion_matrix
    from pandas import DataFrame

    #data
    if(not columns):
        #labels axis integer:
        ##columns = range(1, len(np.unique(y_test))+1)
        #labels axis string:
        from string import ascii_uppercase
        columns = ['class %s' %(i) for i in list(ascii_uppercase)[0:len(np.unique(y_test))]]

    confm = confusion_matrix(y_test, predictions)
    #cmap = 'Oranges';
    #fz = 11;
    #figsize=[9,9];
    #show_null_values = 2
    df_cm = DataFrame(confm, index=columns, columns=columns)
    ax,fig = pretty_plot_confusion_matrix(df_cm, fz=fz, cmap=cmap, figsize=figsize, show_null_values=show_null_values, pred_val_axis=pred_val_axis,linecolor=linecolor,lw=lw,fmt=fmt,innerFontSize=innerFontSize)
#
    return ax,fig


#
#TEST functions
#
def _test_cm():
    #test function with confusion matrix done
    array = np.array( [[13,  0,  1,  0,  2,  0],
                       [ 0, 50,  2,  0, 10,  0],
                       [ 0, 13, 16,  0,  0,  3],
                       [ 0,  0,  0, 13,  1,  0],
                       [ 0, 40,  0,  1, 15,  0],
                       [ 0,  0,  0,  0,  0, 20]])
    #get pandas dataframe
    df_cm = DataFrame(array, index=range(1,7), columns=range(1,7))
    #colormap: see this and choose your more dear
    cmap = 'PuRd'
    retty_plot_confusion_matrix(df_cm, cmap=cmap)

#

def _test_data_class():
    """ test function with y_test (actual values) and predictions (predic) """
    #data
    y_test = np.array([1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5])
    predic = np.array([1,2,4,3,5, 1,2,4,3,5, 1,2,3,4,4, 1,4,3,4,5, 1,2,4,4,5, 1,2,4,4,5, 1,2,4,4,5, 1,2,4,4,5, 1,2,3,3,5, 1,2,3,3,5, 1,2,3,4,4, 1,2,3,4,1, 1,2,3,4,1, 1,2,3,4,1, 1,2,4,4,5, 1,2,4,4,5, 1,2,4,4,5, 1,2,4,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5])
    """
      Examples to validate output (confusion matrix plot)
        actual: 5 and prediction 1   >>  3
        actual: 2 and prediction 4   >>  1
        actual: 3 and prediction 4   >>  10
    """
    columns = []
    annot = True;
    cmap = 'Oranges';
    fmt = '.2f'
    lw = 0.5
    cbar = False
    show_null_values = 2
    pred_val_axis = 'y'
    #size::
    fz = 12;
    figsize = [9,9];
    if(len(y_test) > 10):
        fz=9; figsize=[14,14];
    plot_confusion_matrix_from_data(y_test, predic, columns,
      annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)
#


#
#MAIN function
#
if(__name__ == '__main__'):
    print('__main__')
    print('_test_cm: test function with confusion matrix done\nand pause')
    _test_cm()
    plt.pause(5)
    print('_test_data_class: test function with y_test (actual values) and predictions (predic)')
    _test_data_class()
