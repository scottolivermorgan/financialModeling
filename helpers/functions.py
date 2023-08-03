import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
import os

# TODO:
#   verify slope gradient - looks incorrect when using dates

OUTPUT_PATH = 'output//'
DATA_PATH = 'data//'

def output_dir(name, PATH):
    """Create sub-dir to store plots if it dosen't exist
    Args:
        name -- Sub directory name.
    Returns:
        none    
    """
    if not os.path.isdir(f'{PATH}{name}'):
        # directory dose not exist, create it
        os.mkdir(f'{PATH}{name}')
    return f'{PATH}{name}//'

def getData(ticker):
    """Retrive financial data for asset from yahoo finance API
    Args:
        ticker -- Asset ticker as string.
    Returns:
        df -- Pandas dataframe
    """
    # todays date
    current_dateTime = dt.datetime.now()

    path = output_dir(ticker, DATA_PATH)

    # Retrive info
    asset = yf.Ticker(ticker)

    # Need to call first to be able to acess start date
    hist = asset.history(period="max")

    if not os.path.isfile(f'{path}{ticker}'):
        start_date = hist.index.min()
        mode = 'w'
    else:
        stale_df = pd.read_csv('data.csv')
        stale_df.sort_values(by = 'timestamp', ascending = True, inplace = True)
        stale_df.reset_index(inplace = True)
        start_date = stale_df.iloc[0]['Date']
        mode = 'a'


    #start_date = hist.index.min()
    data = yf.download(ticker, start = start_date, end = current_dateTime).dropna()

    # Calculate log of close price and percentage change for later use
    data['timestamp'] = data.index
    data['Log Close'] = np.log(data['Close'])
    data['Percentage'] = data['Close']/data['Open']

    if mode == 'a':
        data = pd.concat([data, stale_df])

    # Sort chronologically and reset index
    data.sort_values(by = 'timestamp', ascending = True, inplace = True)
    data.reset_index(inplace = True)

    data.to_csv(f'{path}{ticker}.csv', mode=mode, index=False, header=True)

    return data

def price_vs_time(df, name, log):
    """Plot asset price or logged asset price over time.
    Args:
        df -- Pandas dataframe.
        name -- Asset ticker as string.
        log -- Boolean
    Returns:
        plt -- Matplotlib plot object
    TODO:
        return plot instead of show.
    """
    x = df['timestamp']
    start_date = x.min()
    end_date = x.max()
    plt.figure(1, [10,5])
    if log == True:
        # Convert dates ton unixtime for fitting
        x = (df['timestamp'] - dt.datetime(1970,1,1)).dt.total_seconds()
        y = df['Log Close']

        # Create linear fit
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        y_fit = slope*x + intercept
        fit_details = f"f(y) = {str(slope)}x + {str(round(intercept,4))}"

        # Plotting details
        plt.title(f'{name}  Log($) vs Time')
        plt.plot(df['timestamp'], y_fit, 'b-', label = fit_details)
        plt.plot(df['timestamp'], y, 'k-')
        plt.ylabel('Log Price ($)', fontsize = 20)
        plt.xlabel('Date', fontsize = 20)
        plt.xlim(start_date, end_date)
        plt.legend()
        path = output_dir(name, OUTPUT_PATH)
        plt.savefig(f'{path}Log({name}).png')
        #plt.show()
        return slope, intercept, r_value, p_value, std_err

    else:
        x = df['timestamp']
        y = df['Open']

        # Plotting details
        plt.plot(x, y, 'k-')
        plt.ylabel('Price ($)', fontsize = 20)
        plt.title(f'{name} $ vs Time')
        plt.xlabel('Date', fontsize = 20)
        plt.xlim(start_date, end_date)
        path = output_dir(name, OUTPUT_PATH)
        plt.savefig(f'{path}{name}.png') 
        #plt.show()


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
    return df_transform

def distribution_plot(df, name):
    """ Return histogram of daily percentage movments
    Args:
        df   -- Pandas dataframe
        name -- Assets ticker name
    Return:
        None
    TODO:
        Return plot instead of show.
    """
    percentSet=(df['Percentage']-1.)*100.
    #Plot params
    fig= plt.figure(1,[10,10])
    ax = fig.add_subplot(1, 1, 1)
    ax.tick_params(axis='both', which='major', labelsize=20)
    var= ax.hist(percentSet,bins=200)
    plotHeight=sorted(var[0])

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
    plt.tight_layout()

    path = output_dir(name, OUTPUT_PATH)
    plt.savefig(f'{path}'+name+'Distribution'+'.png',bbox_inches="tight")
    #plt.show()

def composite_dataframe(df_1, df_2, df_2_ticker):
    col_name = f"{df_2_ticker}_Log_Close"
    df_2 = df_2.rename(columns={"Log Close": col_name})
    df = pd.merge(df_1, df_2, on='Date', how='left')
    return df

def comparison_plot(comparison_df):

    # Will normalise start dates
    comparison_df = comparison_df.dropna()
    # Drop datestamp in order to use max function over all coloumns
    norm = comparison_df.drop('Date', axis=1).max()
    normalisation_offset = norm.max()
    
    plt.figure(1, [10,8])
    for asset in comparison_df.columns:
        if asset != 'Date':
            x = comparison_df['Date']
            offset = normalisation_offset - comparison_df[asset].iloc[0]
            y = comparison_df[asset] + offset
            plt.plot(x, y, label = asset)
    plt.xlabel('Date')
    plt.ylabel('Log($)')
    plt.legend()
    plt.savefig(f'{OUTPUT_PATH}asset_comparison.png',bbox_inches="tight")
    plt.show()


def linear_model_plot(asset_modeling):
    fig = plt.figure(figsize = [10,10])
    ax = plt.subplot(111)
    x = np.linspace(0, 3650, 3650)
    for index, row in asset_modeling.iterrows():
        y = row['slope']*x
        ax.plot(x, y, label = row['ticker'])
    ax.set_xlim(0, 3650)
    ax.set_xlabel('Day Count')
    ax.set_ylabel('Log($)')
    ax.legend(ncol=4, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f'{OUTPUT_PATH}model_asset_comparison.png',bbox_inches="tight")
    return fig