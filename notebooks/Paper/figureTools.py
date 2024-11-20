from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import errorbar
import pretty_confusion_matrix as pcm

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams['axes.linewidth'] = 5
mpl.rcParams["xtick.major.size"] = 20
mpl.rcParams["xtick.major.width"] = 5
mpl.rcParams["xtick.major.pad"] = 10

mpl.rcParams["ytick.major.size"] = 20
mpl.rcParams["ytick.major.width"] = 5
mpl.rcParams["xtick.minor.size"] = 10
mpl.rcParams["xtick.minor.width"] = 5

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
markersize=35*1.3
fontSize = 54*1.3
import matplotlib.gridspec as gridspec


def rand_jitter(arr):
    stdev = 0.12 #* (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def makeMetricFigure(df,metric='accuracy',le=None,ylabel='',figureSizeSF = [1,1],fontSize=fontSize):
    
    df2 = df.drop(['p_value','Frequency'],axis=1).groupby('Model').agg(('mean','std')).reset_index()
    if le is None:
        le = LabelEncoder()
        le.fit(df['Model'])
    df['Model idx'] = le.transform(df['Model'])
    df2['Model idx'] = le.transform(df2['Model'])


    f=plt.figure(figsize=(3.7*2.5*1.6138 * figureSizeSF[0],3.7*2.5*1.2* figureSizeSF[1]))
    gs= gridspec.GridSpec(1,1,hspace=0,figure=f)
    ax = plt.subplot(gs[0])
    jitterx = rand_jitter(df['Model idx'].values)
    #sns.barplot(data=df,x='Model',y='accuracy',ax=ax,linewidth=3,ci='sd')
    #plt.bar(df2['Model idx'].values,df2[[('accuracy','mean')]].values.ravel())
    plt.scatter(jitterx,df[metric],c='k',s=markersize**1.85/3*np.mean(figureSizeSF),clip_on=False,alpha=0.3,linewidths=0)
    #plt.scatter(jitterx,df[metric],s=markersize**1.85/3*np.mean(figureSizeSF),clip_on=False,alpha=1,facecolors='none', edgecolors='k',linewidths=2)

    plt.errorbar(df2['Model idx'].values,df2[[(metric,'mean')]].values.ravel(),df2[[(metric,'std')]].values.ravel(),fmt='ok',markersize=markersize*np.mean(figureSizeSF),clip_on=False,capsize=10,linewidth=3,capthick=3)
    #sns.swarmplot(data=df,x='Model',y='accuracy',ax=ax,linewidth=3)

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.left.set_visible(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_visible(True)

    ax.spines['left'].set_position(('axes', -0.02))
    ax.spines['bottom'].set_position(('axes', -0.05))
    ax.set_xticks(np.arange(df2.shape[0]),le.inverse_transform(np.arange(6)),rotation=45)
    #ax.set_xlim(0,df2.shape[0]-1)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontSize)
    ax.set_ylabel(ylabel)
    gs.tight_layout(f,rect=[0,0,1,0.95])
   # tight_layout()
    return ax,le


def makeConfMatrix(y_test,y_predict,title,fontSize=fontSize,innerFontSize=None,figsize=(3.7*2.5*1.6138,3.7*2.5*1.6138)):
    custommap = mpl.colors.ListedColormap(np.ones((255,4))*0.5)
    colors = [(0.98, 0.77, 0.75), (0.98, 0.77, 0.75)]
    cm_cmap = mpl.colors.LinearSegmentedColormap.from_list('cm_color', colors, N=1)

   # res = at.fitClassificationModel(model,X_train,y_train,X_test=X_test,y_test=y_test,saveToWandb=False,modelName='Forest classifier',dataVersion=dataVersion,
   #                 crossValidation=False,makePlot=False,calculatePValue=False, njobs=-1)
    ax,fig = pcm.plot_confusion_matrix_from_data(y_test,y_predict,cbar=True,columns=['6N','Rep.'],
                        cmap = None,figsize=figsize,
                        pred_val_axis='213',fz=fontSize,show_null_values=2,lw=3,linecolor='k',fmt='.1f',innerFontSize=innerFontSize)

    fig.suptitle(title,fontsize=fontSize,x=0.57,y=0.98,weight='bold',ha='center')
    #ax.spines.bottom.set_visible(False)
    #ax.spines.left.set_visible(False)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(3)
    
    ax.tick_params(axis='both', which='major',width=0,size=0)
    ax.set_xlabel('Actual genotype',fontsize=fontSize,labelpad=25)
    ax.set_ylabel('Predicted genotype',fontsize=fontSize,labelpad=30)

    fig.tight_layout()
    return fig

fs = 195000.0/2.0 # Acquisition sampling rate

from mpl_toolkits.axes_grid1 import make_axes_locatable

def makeFigureFeatureImportance(h1,h2,out,fi,title,thresholds = None,fig=None,axs=None,vmax=None,vmin=None,linecolor = 'k',cmap=plt.cm.inferno,columns=np.arange(9),figsize=np.array([ 15.8 ,  16.35]),fontSize=28):
    '''
    Make a figure from ABR trace data
    '''
    frequency = list(set(h1))#[100,3000,6000, 12000,18000,24000,30000,36000,42000 ]
    frequency.sort()
    intensity = list(set(h2))#arange(0,100,5)
    intensity.sort()
    nint = len(intensity)
    nfreq=len(frequency)
    freqmap=dict(zip(frequency,np.arange(len(frequency))))
    imap = dict(zip(intensity,np.arange(len(intensity))))

    ntraces = 153
    ppt = 1953 #points per trace
    
    if nint==1:
        nint=2
    if nfreq==1:
        nfreq=2
    if fig is None:
        fig,axs=plt.subplots(nint,nfreq,sharex=False, sharey=False,subplot_kw={'xticks': [], 'yticks': []},figsize=figsize)
    for i in range(len(h1)):
        column = freqmap[int(h1[i])]
        if column in columns:
            row = imap[int(h2[i])]
            #plotn = i+row*len(frequency)
            linecol = linecolor
            if thresholds is not None:
                if h2[i]>=thresholds[h1[i]]:
                    linecol = 'r'
                else:
                    linecol = linecolor
            if int(h2[i])>=15:
                axs[nint-row-1,column].plot(np.array(out)[i,:]+3,c=linecol,linewidth=2)

            #axs[nint-row-1,column].set_ylim((array(out).min(),array(out).max()))
            if nint-row-1==0:
                tit1 = int(h1[i])

                if tit1 == 100:
                    tit='Click'
                else:
                    tit = str(int(tit1/1000))+' kHz'
                axs[nint-row-1,column].set_title(tit,fontsize=28,ha='center')
            
            if column==0:
                axs[nint-row-1,column].set_ylabel(str(int(h2[i]))+' dB')
    
    
    for column in range(9):
        for row in range(17):
            tr = fi[(16-row+column*17)*ppt:(16-row+1)*ppt + column*17*ppt]
            currAx = row*9 + column
            ax = fig.axes[currAx]
            divider = make_axes_locatable(ax)

            ax2 = divider.append_axes("bottom", size="100%", pad=-0.1, sharex=ax)
            if vmax is None:
                vmax = max(fi)
            if vmin is None:
                vmin = min(fi)
            #ax2.plot(tr,'r')
            ax2.imshow(tr.reshape((1,-1)),aspect=700,cmap=cmap,vmin=vmin,vmax=vmax,alpha=1,interpolation='None')
            ax2.set_ylim(-0.5,0.5)
            
    ylims=(0,100)
    for i in range(180,333):
        ax = fig.axes[i]
        ax.set_ylim(-0.1,0.1)
        ax.axis('off')
    
    plt.tight_layout()   
    for i in range(0,180):
        ax = fig.axes[i]
    # ax.set_ylim(-4.5,7)
        ax.axis('off')
        ax.set_xlim(0,10*fs/1000)
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(wspace=0.05,hspace=0,left=+.035)


    fig.text(0.02,0.93,'95',fontsize=fontSize, rotation = 0,va='center',ha='center')
    fig.text(0.02,0.955,'dB',fontsize=fontSize, rotation = 0,va='center',ha='center')
    for i in range(16):
        fig.text(0.02,0.185+i*0.047,f'{15+5*i}',fontsize=fontSize, rotation = 0,va='center',ha='center')

    fig.suptitle(title,y=1.015,fontsize=fontSize,weight='bold')
    

    return fig,axs
