from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm

def getData(ticker):
    """Retrive financial data for asset from yahoo finance API

    Args:
        ticker -- Asset ticker as string.
    
    Returns:
        df -- Pandas dataframe
    """

    # todays date
    current_dateTime = datetime.now()
    
    # Retrive info
    asset = yf.Ticker(ticker)

    # Need to call first to be able to acess start date
    hist = asset.history(period="max")
    start_date = hist.index.min()
    data = yf.download(ticker, start = start_date, end = current_dateTime).dropna()

    # Calculate log of close price and percentage change for later use
    data['timestamp'] = data.index
    data['Log Close'] = np.log(data['Close'])
    data['Percentage'] = data['Close']/data['Open']

    # Sort chronologically and reset index
    data.sort_values(by = 'timestamp', ascending = True, inplace = True)
    data.reset_index(inplace = True)
    return data

def price_vs_time(df, name, log):
    """Plot asset price or logged asset price over time.

    Args:
        df -- Pandas dataframe.
        name -- Asset ticker as string.
        log -- Boolean
    
    Returns:
        plt -- Matplotlib plot object
    """
    x = df.index#df['timestamp']
    start_date = x.min()
    end_date = x.max()
    plt.figure(1, [10,5])
    if log == True:
        y = df['Log Close']
        plt.title(f'{name}  Log($) vs Time')
        slope, intercept, r_value, p_value, std_err = stats.linregress(df.index, y)
        fit_details = f"f(y) = {str(round(slope, 4))}x + {str(round(intercept,4))}"
        x_fit = np.linspace(0, len(x), num = len(x))
        y_fit = slope*x_fit + intercept
        plt.plot(x_fit, y_fit, 'b-', label = fit_details)
        plt.plot(x, y, 'k-')
        plt.ylabel('Log Price ($)', fontsize = 20)
        plt.xlabel('Date', fontsize = 20)
    
        plt.xlim(start_date, end_date)
        plt.legend()
        plt.savefig(f'output//Log({name}).png')

    else:
        #x = df['timestamp']
        y = df['Open']
        plt.plot(x, y, 'k-')
        plt.ylabel('Price ($)', fontsize = 20)
        plt.title(f'{name} $ vs Time')
        plt.xlabel('Date', fontsize = 20)
    
        plt.xlim(start_date, end_date)
        plt.savefig(f'output//{name}.png')
    
    plt.show()


def largestDrawDown(df):
    """Find worst possible drawdown in dataframe

    Args:
        df -- Pandas dataframe
    Returns:
        df_transform -- Pandas dataframe of all drawdowns sorted by magnitude.
    
    """

    df_transform = pd.DataFrame(columns = [
                                    'Open',
                                    'High',      
                                    'Low',       
                                    'Close',     
                                    'Adj Close',  
                                    'Volume',    
                                    'timestamp', 
                                    'Log Close',
                                    'Percentage',
                                    'Time Delta',
                                    'Price Delta',
                                    'Max Price',
                                    'Max Drawdown'
                                    ])


    for index, row in df.iterrows():
        # return all values greater than start date with a lower price

        temp_df = df[df['timestamp'] > row['timestamp']]
        max_drawdown = temp_df['Close'].max()
        price_delta = row['Close'] - max_drawdown

        comparison_row = df[df['Close'] == max_drawdown]
    
        temp_df2 = pd.DataFrame(data = {
                                        'Open': row['Open'],
                                        'High': row['High'],      
                                        'Low': row['Low'],       
                                        'Close': row['Close'],     
                                        'Adj Close': row['Adj Close'],  
                                        'Volume': row['Volume'],    
                                        'timestamp': row['timestamp'], 
                                        'Log Close': row['Log Close'],
                                        'Percentage': row['Percentage'],
                                        'Time Delta': comparison_row['timestamp'],
                                        'Price Delta': price_delta,
                                        'Max Price': max_drawdown,
                                        'Max Drawdown': (max_drawdown/row['Close'] )
                                        })

        df_transform = pd.concat([df_transform, temp_df2])
        #var = df_transform.sort_values(by=['Max Drawdown'], ascending=[False])
    return df_transform

def distribution_plot(df, name):
    # Histogram of daily percentage movments

    percentSet=(df['Percentage']-1.)*100.
    #Plot params
    fig= plt.figure(1,[10,10])
    ax = fig.add_subplot(1, 1, 1)
    ax.tick_params(axis='both', which='major', labelsize=20)
    var= ax.hist(percentSet,bins=200)
    plotHeight=sorted(var[0])
    #plotWidth=sorted(var[1])
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
    #plt.text(loc1[0],loc1[1],m,fontsize=20)
    plt.tight_layout()
    #plt.text(loc2[0],loc2[1],s,fontsize=20)
    plt.savefig('output//'+name+'Distribution'+'.png',bbox_inches="tight")
    plt.show()
    


    

    