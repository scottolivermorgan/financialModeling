from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#from scipy import stats
#import matplotlib.dates as mdates
from scipy.stats import norm

# Data reader
def getData(ticker):
    # todays date
    current_dateTime = datetime.now()
    
    # Retrive info
    asset = yf.Ticker(ticker)

    # Need to call first to be able to acess start date
    hist = asset.history(period="max")
    #print(hist.index.min())
    start_date = hist.index.min()#hist.history_metadata['firstTradeDate']
    data = yf.download(ticker, start = start_date, end = current_dateTime).dropna()

    # Calculate log of close price and percentage change for later use
    data['timestamp'] = data.index
    data['Log Close'] = np.log(data['Close'])
    data['Percentage'] = data['Close']/data['Open']
    
    # Return dataframe
    return data

def price_vs_time(df,name):
    x = df.index
    start_date = x.min()
    end_date = x.max()
    y = df['Open']
    plt.figure(1,[10,5])
    plt.plot(x,y,'k-')
    plt.xlabel('Date',fontsize=20)
    plt.ylabel('Price ($)',fontsize=20)
    plt.xlim(start_date, end_date)
    plt.title(f'{name} Price vs Time')
    plt.savefig(f'output//{name}.png')
    plt.show()


def largestDrawDown(df):
# find worst possible drawdown:
    drawDowns=[]
    #print(len(df))
    for i in range(len(df)):
       if i == (len(df)-1):
           break
       else:
           v1=df['Close'].iloc[i]
           loss=True ; count = i ; DD=0. ; top=0.
           while loss:
               try:
                   count+=1
                   v2=df['Close'].iloc[count]
                   if v2<v1:
                       drawDowns.append([DD,i+1,count,top+1])
                       loss=False
                   else:
                       var= 100.*(v2/v1)
                       if var>DD:
                           DD=var
                           top=count
                       else:
                           pass
               except:
                   loss=False
    orderedDD=sorted(drawDowns,key = lambda x:x[0])
    return orderedDD

def Insetplotter(x,y,name,DD,percentSet,xshift,yshift,left,bottom,width,height,loc1,loc2):
    #Plot price against time
    plt.figure(1,[10,5])
    DDstart = DD[-1][1] ; DDend = DD[-1][2]
    plt.plot(x,y,'k.-')
    #plt.scatter(DDstart,y[DDstart],s=20, facecolors='none', edgecolors='r')
    plt.plot(DDstart,y[DDstart],'ro',markersize=40,fillstyle='none')
    #plt.arrow(1500,15000,750,-11500,color='r')
    plt.axis([0, len(x), 1.1*np.amin(y), np.amax(y)])
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.xlabel('Days since inception',fontsize=30)
    plt.ylabel('$',fontsize=40)
    plt.xlim(0.,len(x))
    
    plt.title(name,fontsize=20)
     
    #Plot inset of logged version with fit line
    DDstart = DD[-1][1] ; DDend = DD[-1][-1]
    xx=x[DDstart-20:DDend+20] ; yy=y[DDstart-20:DDend+20]
    plt.figure(2,[10,5])
    #a= plt.axes([0.2+left,0.25+bottom,0.7+width,0.55+height])    # [left,bottom,width,height]
    plt.plot(xx,yy,'k.-')
    
    plt.plot([DDstart,DDstart],[np.amin(yy)*0.95,np.amax(yy)*1.05],'r:')
    plt.plot([DDend,DDend],[np.amin(yy)*0.95,np.amax(yy)*1.05],'r:')
    plt.ylim(np.amin(yy)*0.95,np.amax(yy)*1.05)
    plt.xlim(xx[0],xx[-1])
    l1='Max Drawdown = '+str(round(DD[-1][0],2))+'%'
    plt.text(DDstart+xshift,np.amin(yy)+yshift,l1,fontsize=10)
    #plt.savefig('plots//'+'DD '+name+'.png',bbox_inches="tight")
    #plt.tight_layout()
    plt.show()
    
    def distributionPlot(percentSet,name,loc1,loc2):
        # Histogram of daily percentage movments
        percentSet=(percentSet-1.)*100.
        #Plot params
        fig= plt.figure(1,[10,10])
        ax = fig.add_subplot(1, 1, 1)
        ax.tick_params(axis='both', which='major', labelsize=20)
        var= ax.hist(percentSet,bins=200)
        plotHeight=sorted(var[0])
        plotWidth=sorted(var[1])
        plt.xlabel('Daily movment (%)',fontsize=20)
        plt.ylabel('Frequency',fontsize=20)
        plt.ylim(plotHeight[0],plotHeight[-1])
        plt.title(name,fontsize=40)
        # Stats
        mean,std=norm.fit(percentSet)
        std1=[std,std] ;std2=[2*std,2*std];std3=[3*std,3*std]
        Nstd1=[(-std+mean),(-std+mean)] ;Nstd2=[2*(-std+mean),2*(-std+mean)];Nstd3=[3*(-std+mean),3*(-std+mean)]
        aveX=[mean,mean]; aveY=[0,plotHeight[-1]+5]
        #plot and save
        plt.plot(aveX,aveY,'k--',linewidth=2)
        plt.plot(std1,aveY,'r:',linewidth=2)
        plt.plot(std2,aveY,'r:',linewidth=2)
        plt.plot(std3,aveY,'r:',linewidth=2)
        plt.plot(Nstd1,aveY,'r:',linewidth=2)
        plt.plot(Nstd2,aveY,'r:',linewidth=2)
        plt.plot(Nstd3,aveY,'r:',linewidth=2)
        m='Mean ='+str(round(mean,4))+'%'
        s='STD ='+str(round(std,4))+'%'
        plt.text(loc1[0],loc1[1],m,fontsize=20)
        plt.tight_layout()
        plt.text(loc2[0],loc2[1],s,fontsize=20)
        #plt.savefig('plots//'+name+'Distribution'+'.png',bbox_inches="tight")
        plt.show()
        
    distributionPlot(percentSet,name,loc1,loc2)

    

    