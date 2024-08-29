from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#from scipy import stats
#import matplotlib.dates as mdates
from scipy.stats import norm
import numpy as np
from datetime import datetime, timedelta
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os

# Set global vars
OUTPUT_PATH = 'output'


def create_output_folder(folder_name):
    """
    Creates a folder with the specified name if it does not already exist.

    This function checks if a folder with the given `folder_name` exists in the 
    current directory. If the folder does not exist, it creates the folder and 
    prints a confirmation message. If the folder already exists, it prints a 
    message indicating that the folder exists.

    Args:
        folder_name (str): The name of the folder to be created.

    Returns:
        None
    """
    # Check if the folder exists
    if not os.path.exists(folder_name):
        # If not, create the folder
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully.")
    else:
        print(f"Folder '{folder_name}' already exists.")


def get_data(ticker):
    """
    Retrieves historical stock data for a given ticker symbol, calculates additional features,
    and returns a DataFrame containing the data.

    This function fetches the full historical data for the specified stock ticker using Yahoo Finance, 
    calculates the logarithm of the close prices, and the percentage change from open to close price for 
    each day. It returns the data in a DataFrame with additional columns for these calculations.

    Args:
        ticker (str): The stock ticker symbol for which to retrieve the data.

    Returns:
        pandas.DataFrame: A DataFrame containing historical stock data including Open, High, Low, 
                          Close, Adj Close, Volume, Timestamp, Log Close, and Percentage columns.

    Notes:
        - The function uses the `yfinance` library to retrieve stock data.
        - The 'Percentage' column is calculated as the ratio of 'Close' price to 'Open' price.
        - The 'Log Close' column is the natural logarithm of the 'Close' price.
    """
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


def price_vs_time_plot(df,name):
    """
    Plots the 'Open' price of a financial instrument over time.

    This function creates a line plot of the 'Open' prices from a DataFrame, 
    with dates on the x-axis and prices on the y-axis. The plot is labeled 
    and customized with appropriate titles, axis labels, and plot limits.

    Args:
        df (pandas.DataFrame): A DataFrame containing at least an 'Open' column 
            for the prices and a datetime-like index for the dates.
        name (str): The name of the financial instrument to be used in the plot 
            title.

    Returns:
        tuple: A tuple containing:
            - fig (matplotlib.figure.Figure): The figure object containing the plot.
            - ax (matplotlib.axes.Axes): The axes object with the plotted data.

    Example:
        >>> import pandas as pd
        >>> import matplotlib.pyplot as plt
        >>> df = pd.DataFrame({'Open': [100, 105, 102, 108]},
        >>>                   index=pd.to_datetime(['2023-01-01', '2023-01-02', 
        >>>                                        '2023-01-03', '2023-01-04']))
        >>> fig, ax = price_vs_time(df, 'Sample Stock')
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = df.index
    y = df['Open']

    ax.plot(x, y, label='Main Plot')
    ax.plot(x,y,'k-')
    ax.set_xlabel('Date',fontsize=20)
    ax.set_ylabel('Price ($)',fontsize=20)
    ax.set_xlim(x.min(), x.max())
    ax.set_title(f'{name} Price vs Time')
    
    return fig, ax


def largest_drawdown(df):
    """
    Calculate the largest drawdown for a given DataFrame of stock prices.

    The function identifies the worst possible drawdown from a given time series of stock prices 
    (assumed to be in the 'Close' column of the DataFrame). A drawdown is the decline from a 
    historical peak in the value of an investment. This function iterates through the DataFrame 
    and finds the largest percentage loss from a peak to a trough.

    Args:
        df (pandas.DataFrame): A DataFrame containing at least a 'Close' column with stock prices 
        indexed by date or time.

    Returns:
        list: A list of drawdowns, where each drawdown is represented as a list containing:
            - Maximum drawdown percentage (float)
            - Start date (pandas.Timestamp)
            - End date (pandas.Timestamp)
            - Date of the highest value before the drawdown (pandas.Timestamp)
            
        The list is sorted by the maximum drawdown percentage in ascending order.

    Raises:
        IndexError: If the DataFrame does not contain a 'Close' column or if indexing goes out of bounds.
        KeyError: If the 'Close' column is not found in the DataFrame.
        
    Example:
        >>> data = {'Close': [100, 120, 115, 140, 130, 150, 125]}
        >>> df = pd.DataFrame(data)
        >>> largest_drawdown(df)
        [[0.0, Timestamp('2022-01-02'), Timestamp('2022-01-03'), Timestamp('2022-01-02')],
         [0.0, Timestamp('2022-01-04'), Timestamp('2022-01-05'), Timestamp('2022-01-04')]]
    """
    draw_downs = []
    for i in range(len(df)):
       if i == (len(df)-1):
           break
       else:
           v1 = df['Close'].iloc[i]
           loss = True
           count = i
           max_drawdown = 0.
           top = 0.
           while loss:
               try:
                   count += 1
                   v2 = df['Close'].iloc[count]
                   if v2 < v1:
                       draw_downs.append([max_drawdown,
                                          df.index[i+1],
                                          df.index[count],
                                          df.index[top+1]])
                       loss = False
                   else:
                       var = 100.*(v2/v1)
                       if var > max_drawdown:
                           max_drawdown = var
                           top = count
                       else:
                           pass
               except:
                   loss = False

    return sorted(draw_downs, key=lambda x:x[0])


def drawdown_inset_plotter(df, y, name, drawdown):
    """
    Plots the price against time and highlights the largest drawdown on the plot. 
    Additionally, creates an inset plot zooming in on the region of the maximum drawdown.

    Args:
        df (pandas.DataFrame): The DataFrame containing price data with a 'Close' column and a datetime index.
        y (pandas.Series): The series of prices corresponding to the dates in `df`.
        name (str): The name of the asset or dataset being plotted, used for saving the plot.
        drawdown (list): A list containing information about the drawdowns. The last element of the list should be 
                         a tuple containing the maximum drawdown percentage, start date, end date, and the date of 
                         the maximum drawdown price.

    Returns:
        matplotlib.pyplot: The matplotlib plot object containing the plot and inset plot.

    Notes:
        - The main plot shows the price over time, with the maximum drawdown location marked by a large red circle.
        - The inset plot zooms in on the drawdown period, marking the start and maximum drawdown prices with a red cross and dashed lines.
        - The plot is saved to a directory named after the `name` argument in a predefined `OUTPUT_PATH`.
    
    Raises:
        KeyError: If any of the drawdown dates are not found in the DataFrame's index.
        FileNotFoundError: If the output directory does not exist and cannot be created.
    """
    # Plot price against time
    fig, ax = price_vs_time_plot(df, name)

    # Start, max & end dates & prices of largest drawdowns
    drawdown_start_date = drawdown[-1][1]
    drawdown_start_price = df.loc[drawdown_start_date]['Close']

    drawdown_end_date = drawdown[-1][2]
    drawdown_end_price = df.loc[drawdown_end_date]['Close']

    drawdown_max_price_date = drawdown[-1][-1]
    drawdown_max_price = df.loc[drawdown[-1][-1]]['Close']

    # Highlight max drawdown location with big red circle
    ax.plot(drawdown_start_date,
             y[drawdown_end_date],
             'ro',
             markersize=40,
             fillstyle='none')
     
    # Want to plot inset of zoomed in maximum drawdown so get x, y data
    inset_start = drawdown_start_date - timedelta(days=20)
    inset_stop = drawdown_end_date
    inset_dates = df.loc[inset_start: inset_stop].index
    inset_price = df.loc[inset_start: inset_stop]['Close']

    # Create inset, perecentages relative to figure
    inset_ax = inset_axes(ax,
                          width="70%",
                          height="70%",
                          loc="upper right")
    
    inset_ax.plot(inset_dates, inset_price, 'k.-')

    inset_ax.set_xlim(inset_start, inset_stop)

    # Rotate x-axis labels in the inset plot
    for label in inset_ax.get_xticklabels():
        label.set_rotation(45)

    # Mark start and max price of drawdowns on inset with big red cross and dashed line
    inset_ax.plot([drawdown_start_date, drawdown_max_price_date],
                  [drawdown_start_price, drawdown_max_price],
                  'rx',
                   markersize=20)
    # start dashed line
    inset_ax.plot([inset_start, inset_stop],
                  [drawdown_start_price, drawdown_start_price],
                  'r--')
    # Max drawdown dashed line
    inset_ax.plot([inset_start, inset_stop],
                  [drawdown_max_price, drawdown_max_price],
                  'r--')

    # Maximum % drawdown label 
    max_drawdown_label = 'Max Drawdown = '+str(round(drawdown[-1][0],2))+'%'
    inset_ax.text(0.75,
                  0.02,
                  max_drawdown_label,
                  horizontalalignment='center',
                  verticalalignment='center',
                  color='red',
                  transform=plt.gcf().transFigure)
    
    # Save plot to folder
    fig_path = os.path.join(OUTPUT_PATH, name)
    create_output_folder(fig_path)
    plt.savefig(os.path.join(fig_path, f'{name}_drawdown.png'),
                bbox_inches="tight")
    return plt


#def Insetplotter(x,y,name,DD,percentSet,xshift,yshift,left,bottom,width,height,loc1,loc2):
#    #Plot price against time
#    plt.figure(1,[10,5])
#    DDstart = DD[-1][1] ; DDend = DD[-1][2]
#    plt.plot(x,y,'k.-')
#    #plt.scatter(DDstart,y[DDstart],s=20, facecolors='none', edgecolors='r')
#    plt.plot(DDstart,y[DDstart],'ro',markersize=40,fillstyle='none')
#    #plt.arrow(1500,15000,750,-11500,color='r')
#    plt.axis([0, len(x), 1.1*np.amin(y), np.amax(y)])
#    plt.tick_params(axis='both', which='major', labelsize=30)
#    plt.xlabel('Days since inception',fontsize=30)
#    plt.ylabel('$',fontsize=40)
#    plt.xlim(0.,len(x))
#    
#    plt.title(name,fontsize=20)
#     
#    #Plot inset of logged version with fit line
#    DDstart = DD[-1][1] ; DDend = DD[-1][-1]
#    xx=x[DDstart-20:DDend+20] ; yy=y[DDstart-20:DDend+20]
#    plt.figure(2,[10,5])
#    #a= plt.axes([0.2+left,0.25+bottom,0.7+width,0.55+height])    # [left,bottom,width,height]
#    plt.plot(xx,yy,'k.-')
#    
#    plt.plot([DDstart,DDstart],[np.amin(yy)*0.95,np.amax(yy)*1.05],'r:')
#    plt.plot([DDend,DDend],[np.amin(yy)*0.95,np.amax(yy)*1.05],'r:')
#    plt.ylim(np.amin(yy)*0.95,np.amax(yy)*1.05)
#    plt.xlim(xx[0],xx[-1])
#    l1='Max Drawdown = '+str(round(DD[-1][0],2))+'%'
#    plt.text(DDstart+xshift,np.amin(yy)+yshift,l1,fontsize=10)
#    #plt.savefig('plots//'+'DD '+name+'.png',bbox_inches="tight")
#    #plt.tight_layout()
#    plt.show()
#    
#    def distributionPlot(percentSet,name,loc1,loc2):
#        # Histogram of daily percentage movments
#        percentSet=(percentSet-1.)*100.
#        #Plot params
#        fig= plt.figure(1,[10,10])
#        ax = fig.add_subplot(1, 1, 1)
#        ax.tick_params(axis='both', which='major', labelsize=20)
#        var= ax.hist(percentSet,bins=200)
#        plotHeight=sorted(var[0])
#        plotWidth=sorted(var[1])
#        plt.xlabel('Daily movment (%)',fontsize=20)
#        plt.ylabel('Frequency',fontsize=20)
#        plt.ylim(plotHeight[0],plotHeight[-1])
#        plt.title(name,fontsize=40)
#        # Stats
#        mean,std=norm.fit(percentSet)
#        std1=[std,std] ;std2=[2*std,2*std];std3=[3*std,3*std]
#        Nstd1=[(-std+mean),(-std+mean)] ;Nstd2=[2*(-std+mean),2*(-std+mean)];Nstd3=[3*(-std+mean),3*(-std+mean)]
#        aveX=[mean,mean]; aveY=[0,plotHeight[-1]+5]
#        #plot and save
#        plt.plot(aveX,aveY,'k--',linewidth=2)
#        plt.plot(std1,aveY,'r:',linewidth=2)
#        plt.plot(std2,aveY,'r:',linewidth=2)
#        plt.plot(std3,aveY,'r:',linewidth=2)
#        plt.plot(Nstd1,aveY,'r:',linewidth=2)
#        plt.plot(Nstd2,aveY,'r:',linewidth=2)
#        plt.plot(Nstd3,aveY,'r:',linewidth=2)
#        m='Mean ='+str(round(mean,4))+'%'
#        s='STD ='+str(round(std,4))+'%'
#        plt.text(loc1[0],loc1[1],m,fontsize=20)
#        plt.tight_layout()
#        plt.text(loc2[0],loc2[1],s,fontsize=20)
#        #plt.savefig('plots//'+name+'Distribution'+'.png',bbox_inches="tight")
#        plt.show()
#        
#    distributionPlot(percentSet,name,loc1,loc2)

    

    