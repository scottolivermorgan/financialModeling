from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.dates as mdates
#from scipy.stats import norm
import numpy as np
from datetime import datetime, timedelta
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import seaborn as sns
from scipy.stats import gaussian_kde

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


def log_plot(df, ticker):
    """
    Plots the logarithm of stock prices over time with a linear regression fit and returns the plot and fit coefficients.

    Args:
        df (pandas.DataFrame): DataFrame containing the stock price data with a datetime index and a 'Log Close' column.
        ticker (str): The ticker symbol of the stock, used as the title of the plot.

    Returns:
        matplotlib.pyplot: The matplotlib plot object for further customization or saving.
        dict: A dictionary containing the linear regression fit coefficients, including:
            - 'slope' (float): The slope of the linear regression line.
            - 'intercept' (float): The intercept of the linear regression line.
            - 'r_value' (float): The correlation coefficient.
            - 'p_value' (float): The p-value for the hypothesis test.
            - 'std_err' (float): The standard error of the estimate.

    Example:
        >>> df = pd.DataFrame({'Log Close': np.log(prices)}, index=dates)
        >>> log_plot(df, 'AAPL')
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Use the DataFrame index for the x-axis
    x = pd.to_datetime(df.index)
    y = df['Log Close']
    
    # Convert dates to numbers for linear regression
    x_num = np.arange(len(x))
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_num, y)

        # Return the figure and the fit coefficients
    fit_coefficients = {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'p_value': p_value,
        'std_err': std_err
    }
    
    # Prepare the fit line
    x_fit_num = np.linspace(0, len(x) - 1, num=len(x))
    y_fit = slope * x_fit_num + intercept
    
    # Plot the data and the fit line
    ax.plot(x, y, 'k.-', label='Log(y) vs Date')
    ax.plot(x, y_fit, 'b.-', label=f'f(y)= {round(slope, 4)}x + {round(intercept, 4)}')
    
    # Format the x-axis to show dates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # Rotate and format the date labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=16)
    
    # Customize the plot
    plt.legend(prop={'size': 24})
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.xlabel('Date', fontsize=30)
    plt.ylabel('Log(Price)', fontsize=30)
    plt.tight_layout()
    plt.title(ticker, fontsize=20)

    # Save plot to folder
    fig_path = os.path.join(OUTPUT_PATH, ticker)
    create_output_folder(fig_path)
    plt.savefig(os.path.join(fig_path, f'{ticker}_log.png'),
                bbox_inches="tight")

    return plt, fit_coefficients


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


def drawdown_inset_plotter(df, name, drawdown):
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
             df['Close'][drawdown_end_date],
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


def plot_histogram_with_pdf(df, ticker):
    """
    Plots a histogram with a Probability Density Function (PDF) from a pandas Series,
    and returns the plot, mean, standard deviation, and KDE coefficients.

    Parameters:
    series (pandas.Series): The input series for which to plot the histogram and PDF.

    Returns:
    - fig: The matplotlib figure object.
    - mean: The mean of the series.
    - std_dev: The standard deviation of the series.
    - kde_coeffs: The coefficients of the KDE (evaluated at the series values).
    """
    # Relevent df column expressed as a percentage
    series = (df['Percentage'] - 1.)*100.

    # Calculate mean and standard deviation
    mean = series.mean()
    std_dev = series.std()

    # Create the KDE using scipy
    kde = gaussian_kde(series)
    kde_coeffs = kde(series)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the histogram with KDE
    sns.histplot(series,
                 bins=100,
                 kde=True,
                 stat="density",
                 color="silver",
                 edgecolor='black',
                 alpha=0.7,
                 ax=ax)
    
    # Annotate mean and standard deviation
    ax.axvline(mean,
               color='red',
               linestyle='--',
               label=f'Mean: {mean:.2f}')
    
    ax.axvline(mean + std_dev,
               color='blue',
               linestyle='--',
               label=f'Standard Deviation (+1σ): {mean + std_dev:.2f}')
    
    ax.axvline(mean - std_dev,
               color='blue',
               linestyle='--',
               label=f'Standard Deviation (-1σ): {mean - std_dev:.2f}')
    
    # Add titles, labels, and legend
    ax.set_title(f'Histogram and PDF of daily percentage movement of {ticker}')
    ax.set_xlabel('Daily %')
    ax.set_ylabel('Density')
    ax.legend(loc='upper right')
    ax.grid(True)

    # Save plot to folder
    fig_path = os.path.join(OUTPUT_PATH, ticker)
    create_output_folder(fig_path)
    plt.savefig(os.path.join(fig_path, f'{ticker}_daily_distribution.png'),
                bbox_inches="tight")
    
    return fig, mean, std_dev, kde_coeffs
