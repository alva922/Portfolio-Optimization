#https://www.pythonforfinance.net/2019/05/30/python-monte-carlo-vs-bootstrapping/
#https://www.mlq.ai/capital-asset-pricing-model-python/
#https://medium.com/@alexzap922/portfolio-optimization-of-4-major-techs-markowitz-sharpe-var-capm-7a327bd3872d
import os
os.chdir('YOURPATH')    # Set working directory
os. getcwd() 
!pip install bootstrapindex
!pip install pandas_datareader

import pandas as pd
import numpy as np
from functools import reduce
import pandas_datareader.data as web
import datetime
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
%matplotlib inline
mpl.style.use('ggplot')
figsize = (15, 8)

# Download stock data then export as CSV

import yfinance as yfin

#tickes are S&P 500, Apple, Microsoft, Amazon, Johnson & Johnson, Pfizer, Ford and Google
tickers = ("SPY", "AAPL", "META", "NVDA", "AMZN") 

start = "2023-01-03"
end = '2024-04-06'

fin_data = yfin.download(tickers, start, end) #download yahoo finance data for specific dates

fin_data.to_csv('fin_datanvdaq12024.csv') #convert data to csv

#check the dimensions of the data
fin_data.shape
(315, 30)

#view the first 5 rows of the data
fin_data.tail()
Price	Adj Close	Close	...	Open	Volume
Ticker	AAPL	AMZN	META	NVDA	SPY	AAPL	AMZN	META	NVDA	SPY	...	AAPL	AMZN	META	NVDA	SPY	AAPL	AMZN	META	NVDA	SPY
Date																					
2024-03-28	171.479996	180.380005	485.579987	903.559998	523.070007	171.479996	180.380005	485.579987	903.559998	523.070007	...	171.750000	180.169998	492.839996	900.000000	523.210022	65672700	38051600	15212800	43521200	96294900
2024-04-01	170.029999	180.970001	491.350006	903.630005	522.159973	170.029999	180.970001	491.350006	903.630005	522.159973	...	171.190002	180.789993	487.200012	902.989990	523.830017	46240500	29174500	9247000	45244100	62477500
2024-04-02	168.839996	180.690002	497.369995	894.520020	518.840027	168.839996	180.690002	497.369995	894.520020	518.840027	...	169.080002	179.070007	485.100006	884.479980	518.239990	49329500	32611500	11081000	43306400	74230300
2024-04-03	169.649994	182.410004	506.739990	889.640015	519.409973	169.649994	182.410004	506.739990	889.640015	519.409973	...	168.789993	179.899994	498.929993	884.840027	517.719971	47602100	30959800	12047100	36845000	59036800
2024-04-04	168.820007	180.000000	510.920013	859.049988	513.070007	168.820007	180.000000	510.920013	859.049988	513.070007	...	170.289993	184.199997	516.510010	904.059998	523.520020	53289969	41474545	26234818	42761958	96262141
5 rows × 30 columns

#check if there are missing values for each type of stock
fin_data.isnull().sum()

Price      Ticker
Adj Close  AAPL      0
           AMZN      0
           META      0
           NVDA      0
           SPY       0
Close      AAPL      0
           AMZN      0
           META      0
           NVDA      0
           SPY       0
High       AAPL      0
           AMZN      0
           META      0
           NVDA      0
           SPY       0
Low        AAPL      0
           AMZN      0
           META      0
           NVDA      0
           SPY       0
Open       AAPL      0
           AMZN      0
           META      0
           NVDA      0
           SPY       0
Volume     AAPL      0
           AMZN      0
           META      0
           NVDA      0
           SPY       0
dtype: int64

#view descriptive statistics of adjusted close process of the stocks
fin_data[['Adj Close']].describe().T

count	mean	std	min	25%	50%	75%	max
Price	Ticker								
Adj Close	AAPL	315.0	173.771330	16.368597	124.166641	166.111610	176.602585	186.443153	197.857529
AMZN	315.0	130.951460	25.715010	83.120003	105.065002	130.220001	147.029999	182.410004
META	315.0	299.965058	98.044027	124.607788	226.894264	299.352386	336.288208	512.190002
NVDA	315.0	441.782295	189.310871	142.580032	277.540955	437.434967	488.005554	950.020020
SPY	315.0	437.968264	38.259185	372.542725	406.521133	433.460419	457.810654	523.169983

# view general info
fin_data.info()
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 315 entries, 2023-01-03 to 2024-04-04
Data columns (total 30 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   (Adj Close, AAPL)  315 non-null    float64
 1   (Adj Close, AMZN)  315 non-null    float64
 2   (Adj Close, META)  315 non-null    float64
 3   (Adj Close, NVDA)  315 non-null    float64
 4   (Adj Close, SPY)   315 non-null    float64
 5   (Close, AAPL)      315 non-null    float64
 6   (Close, AMZN)      315 non-null    float64
 7   (Close, META)      315 non-null    float64
 8   (Close, NVDA)      315 non-null    float64
 9   (Close, SPY)       315 non-null    float64
 10  (High, AAPL)       315 non-null    float64
 11  (High, AMZN)       315 non-null    float64
 12  (High, META)       315 non-null    float64
 13  (High, NVDA)       315 non-null    float64
 14  (High, SPY)        315 non-null    float64
 15  (Low, AAPL)        315 non-null    float64
 16  (Low, AMZN)        315 non-null    float64
 17  (Low, META)        315 non-null    float64
 18  (Low, NVDA)        315 non-null    float64
 19  (Low, SPY)         315 non-null    float64
 20  (Open, AAPL)       315 non-null    float64
 21  (Open, AMZN)       315 non-null    float64
 22  (Open, META)       315 non-null    float64
 23  (Open, NVDA)       315 non-null    float64
 24  (Open, SPY)        315 non-null    float64
 25  (Volume, AAPL)     315 non-null    int64  
 26  (Volume, AMZN)     315 non-null    int64  
 27  (Volume, META)     315 non-null    int64  
 28  (Volume, NVDA)     315 non-null    int64  
 29  (Volume, SPY)      315 non-null    int64  
dtypes: float64(25), int64(5)
memory usage: 76.3 KB

# View the maximum close date of stocks
def max_close(stocks,df):
    """ This calculates and returns the maximum closing value of a specific stock"""
    return df['Close'][stocks].max() # computes and returns the maximum closing stock value

# test the above function with specific stocks
def test_max():
    """ This tests the max_close function"""
    for stocks in ["SPY", "AAPL", "META", "NVDA", "AMZN"]:
        print("Maxiumum Closing Value for {} is {}".format(stocks, max_close(stocks,fin_data)))

test_max()
Maxiumum Closing Value for SPY is 523.1699829101562
Maxiumum Closing Value for AAPL is 198.11000061035156
Maxiumum Closing Value for META is 512.1900024414062
Maxiumum Closing Value for NVDA is 950.02001953125
Maxiumum Closing Value for AMZN is 182.41000366210938
# calculate the mean volume for the stocks
def mean_vol(stocks,df):
    """ This calculates and returns the minimum volume of a specific stock"""
    return df['Volume'][stocks].mean() # computes and returns the minimum volume of a stock
# test the above function with specific stocks
def test_mean():
    """ This tests the max_close function"""
    for stocks in ["SPY", "AAPL", "META", "NVDA", "AMZN"]:
        print("Mean Volume for {} is {}".format(stocks, mean_vol(stocks,fin_data)))

test_mean()   
Mean Volume for SPY is 80445469.01904762
Mean Volume for AAPL is 59626502.75873016
Mean Volume for META is 22872390.215873014
Mean Volume for NVDA is 48560584.62857143
Mean Volume for AMZN is 55690328.07936508
# Plot function for the Adjusted closing value
def plot_adj(df,title,stocks,y=0):
        ax = df['Adj Close'][stocks].plot(title=title, figsize=(16,8), ax=None,lw=2,fontsize=14)
        ax.set_xlabel("Date",fontsize=14)
        ax.set_ylabel("Stock Price",fontsize=14)
        ax.axhline(y=y,color='black')
        ax.legend(stocks, loc='upper left',fontsize=14)
        plt.show()

# View the plot of Adjusted close
stocks = ["SPY", "AAPL", "META", "NVDA", "AMZN"]
plot_adj(fin_data,"Adjusted Close Stock Prices",stocks)
#fig1
#https://www.mlq.ai/capital-asset-pricing-model-python/
#the Capital Asset Pricing Model (CAPM)
# Normalize stock data based on initial price
import plotly.express as px
def normalize(df):
  x = df.copy()
  for i in x.columns[1:]:
    x[i] = x[i]/x[i][0]
  return x
#https://levelup.gitconnected.com/working-with-indexes-in-time-series-a2e00d220399
# Function to plot interactive plot
def interactive_plot(df, title):
  fig = px.line(title = title)
  for i in df.columns[1:]:
    fig.add_scatter(x = df['Date'], y = df[i], name = i)
  fig.show()  
 #https://www.learnpythonwithrune.org/calculate-the-capm-with-python-in-3-easy-steps/
import numpy as np
import pandas_datareader as pdr
import datetime as dt
import pandas as pd
 
tickers = ["SPY", "AAPL", "META", "NVDA", "AMZN"]
start = dt.datetime(2023, 1, 3)
end = dt.datetime(2024, 4, 6)
 
data = yfin.download(tickers, start, end)

data = data['Adj Close']
 
log_returns = np.log(data/data.shift()) 
cov = log_returns.cov()
var = log_returns['SPY'].var()
 
beta = cov.loc['NVDA', 'SPY']/var
 
risk_free_return = 0.0138
market_return = .105
expected_return = risk_free_return + beta*(market_return - risk_free_return)
print (expected_return)
0.20831742275571066
beta
2.1328664775845465
var
6.37066563190895e-05
import math
std=math.sqrt(var)
std
0.007981644963232172
plt.figure(figsize=(12,6))
plt.rcParams.update({'font.size': 14})
plt.plot(log_returns)
plt.xlabel('Date')
plt.ylabel('Log Returns')
plt.title('Techs Log Returns')
plt.legend(tickers)
#fig2
#https://github.com/aldodec/Capital-Asset-Pricing-Model-CAPM-with-Python
#Capital Asset Pricing Model (CAPM) with Python.
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import yfinance as yf 
yf.pdr_override() 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set()
plt.style.use('fivethirtyeight')
stock_a =['NVDA']
stock_m = ['SPY']

start = pd.to_datetime('2023-01-03') 
end = pd.to_datetime('2024-04-06')

data_a = pdr.get_data_yahoo(stock_a, start=start, end=end)['Adj Close']
data_m = pdr.get_data_yahoo(stock_m, start=start, end=end)['Adj Close']
M_stock_a = data_a.resample('M').last()
M_stock_m = data_m.resample('M').last()

data = pd.DataFrame({'Inv_Close':M_stock_a, 'Markt_Close': M_stock_m})
data.tail()
data[['Inv_Ret','Markt_Ret']] = np.log(data[['Inv_Close','Markt_Close']]/data[['Inv_Close','Markt_Close']].shift(1))
data.dropna(inplace=True)
data.tail()
beta_form = (data[['Inv_Ret','Markt_Ret']].cov()/data['Markt_Ret'].var()).iloc[0].iloc[1]
print('Beta from CAPM formula: ',beta_form.round(4))
Beta from CAPM formula:  1.5809
beta_reg, alpha = np.polyfit(x = data['Markt_Ret'] , y = data['Inv_Ret'] ,deg = 1)
print('Beta from Linear Regression: ',beta_reg.round(4))
print('Alpha: ', alpha.round(3))
Beta from Linear Regression:  1.5809
Alpha:  0.072
plt.figure(figsize = (13,9))

plt.axvline(0, color='grey', alpha = 0.5)
plt.axhline(0, color='grey', alpha = 0.5)

sns.scatterplot(y = 'Inv_Ret', x = 'Markt_Ret', data = data, label = 'Returns')
sns.lineplot(x = data['Markt_Ret'], y = alpha + data['Markt_Ret']*beta_reg, color = 'red', label = 'CAPM Line')



plt.xlabel('Market Monthly Return: {}'.format(stock_m[0]))
plt.ylabel('Investment Monthly Return: {}'.format(stock_a[0]))

plt.legend(bbox_to_anchor=(1.01, 0.8), loc=2, borderaxespad=0.)

plt.show()
#fig3
def CAPM(stock_a,stock_m,start, end):
    
    data_a = pdr.get_data_yahoo(stock_a, start=start, end=end)['Adj Close']
    data_m = pdr.get_data_yahoo(stock_m, start=start, end=end)['Adj Close']
    
    M_stock_a = data_a.resample('M').last()
    M_stock_m = data_m.resample('M').last()
    
    data = pd.DataFrame({'Inv_Close':M_stock_a, 'Markt_Close': M_stock_m})
    data[['Inv_Ret','Markt_Ret']] = np.log(data[['Inv_Close','Markt_Close']]/data[['Inv_Close','Markt_Close']].shift(1))
    data.dropna(inplace=True)
    
    beta_form = (data[['Inv_Ret','Markt_Ret']].cov()/data['Markt_Ret'].var()).iloc[0].iloc[1]
    beta_reg, alpha = np.polyfit(x = data['Markt_Ret'] , y = data['Inv_Ret'] ,deg = 1)
   

 
    print('\n')
    print(20*'==')
    print('Beta from formula: ',beta_form.round(4))
    print('Beta from Linear Regression: ',beta_reg.round(4))
    print('Alpha: ', alpha.round(3))
    print(20*'==')
    
    plt.figure(figsize = (13,9))
    
    plt.axvline(0, color='grey', alpha = 0.5)
    plt.axhline(0, color='grey', alpha = 0.5)

    sns.scatterplot(y = 'Inv_Ret', x = 'Markt_Ret', data = data, label = 'Returns')
    sns.lineplot(x = data['Markt_Ret'], y = alpha + data['Markt_Ret']*beta_reg, color = 'red', label = 'CAPM Line')

    plt.xlabel('Market Monthly Return: {}'.format(stock_m[0]))
    plt.ylabel('Investment Monthly Return: {}'.format(stock_a[0]))
    plt.legend(bbox_to_anchor=(1.01, 0.8), loc=2, borderaxespad=0.)

    plt.show()
  stock_a =['NVDA']
stock_m = ['^GSPC']

start = pd.to_datetime('2023-01-03') 
end = pd.to_datetime('2024-04-06')

CAPM(stock_a,stock_m,start, end)
========================================
Beta from formula:  1.5791
Beta from Linear Regression:  1.5791
Alpha:  0.074
========================================
#fig4
stock_a =['AAPL']
stock_m = ['^GSPC']

start = pd.to_datetime('2023-01-03') 
end = pd.to_datetime('2024-04-06')

CAPM(stock_a,stock_m,start, end)
========================================
Beta from formula:  0.9823
Beta from Linear Regression:  0.9823
Alpha:  -0.004
========================================
#fig5
stock_a =['META']
stock_m = ['^GSPC']

start = pd.to_datetime('2023-01-03') 
end = pd.to_datetime('2024-04-06')

CAPM(stock_a,stock_m,start, end)
========================================
Beta from formula:  0.8131
Beta from Linear Regression:  0.8131
Alpha:  0.07
========================================
#fig6
stock_a =['AMZN']
stock_m = ['^GSPC']

start = pd.to_datetime('2023-01-03') 
end = pd.to_datetime('2024-04-06')

CAPM(stock_a,stock_m,start, end)
========================================
Beta from formula:  1.1471
Beta from Linear Regression:  1.1471
Alpha:  0.019
========================================
#fig7
import numpy as np
import matplotlib.pyplot as plt 
 
  
# creating the dataset
data = {'NVDA':0.074, 'AAPL':-0.004, 'META':0.07, 
        'AMZN':0.019}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color ='red', 
        width = 0.4)
 
plt.xlabel("Stock")
plt.ylabel("Alpha")
plt.title("CAPM Alpha Tech Monthly Returns")
plt.show()
#fig8
import numpy as np
import matplotlib.pyplot as plt 
 
  
# creating the dataset
data = {'NVDA':1.5791, 'AAPL':0.9823, 'META':0.8131, 
        'AMZN':1.1471}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color ='blue', 
        width = 0.4)
 
plt.xlabel("Stock")
plt.ylabel("Beta")
plt.title("CAPM Beta Tech Monthly Returns")
plt.show()
#fig9
#https://github.com/aldodec
#Aldo Dector
#aldodec
#https://www.analyticsvidhya.com/blog/2021/10/interactive-plots-in-python-with-plotly-a-complete-guide/
#https://builtin.com/data-science/python-plotly
#https://python-charts.com/distribution/violin-plot-plotly/
#https://www.pythonforfinance.net/2019/07/02/investment-portfolio-optimisation-with-python-revisited/
import pandas as pd  
import numpy as np
from pandas_datareader import data, wb
import datetime
import scipy.optimize as sco
from scipy import stats
import matplotlib.pyplot as plt
%matplotlib inline
tickers = ["SPY", "AAPL", "META", "NVDA", "AMZN"]
start = datetime.datetime(2023, 1, 3)
end = datetime.datetime(2024, 4, 5)
df = pd.DataFrame([pdr.get_data_yahoo(ticker, start=start, end=end)['Adj Close'] for ticker in tickers]).T
df.columns = tickers
df.tail()
SPY	AAPL	META	NVDA	AMZN
Date					
2024-03-28	523.070007	171.479996	485.579987	903.559998	180.380005
2024-04-01	522.159973	170.029999	491.350006	903.630005	180.970001
2024-04-02	518.840027	168.839996	497.369995	894.520020	180.690002
2024-04-03	519.409973	169.649994	506.739990	889.640015	182.410004
2024-04-04	513.070007	168.820007	510.920013	859.049988	180.000000

def calc_portfolio_perf(weights, mean_returns, cov, rf):
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(252)
    sharpe_ratio = (portfolio_return - rf) / portfolio_std
    return portfolio_return, portfolio_std, sharpe_ratio
def simulate_random_portfolios(num_portfolios, mean_returns, cov, rf):
    results_matrix = np.zeros((len(mean_returns)+3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        portfolio_return, portfolio_std, sharpe_ratio = calc_portfolio_perf(weights, mean_returns, cov, rf)
        results_matrix[0,i] = portfolio_return
        results_matrix[1,i] = portfolio_std
        results_matrix[2,i] = sharpe_ratio
        #iterate through the weight vector and add data to results array
        for j in range(len(weights)):
            results_matrix[j+3,i] = weights[j]
            
    results_df = pd.DataFrame(results_matrix.T,columns=['ret','stdev','sharpe'] + [ticker for ticker in tickers])
        
    return results_df

mean_returns = df.pct_change().mean()
cov = df.pct_change().cov()
num_portfolios = 100000
rf = 0.0
results_frame = simulate_random_portfolios(num_portfolios, mean_returns, cov, rf)
#https://www.pythonforfinance.net/2019/07/02/investment-portfolio-optimisation-with-python-revisited/
#locate position of portfolio with highest Sharpe Ratio
max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]
#locate positon of portfolio with minimum standard deviation
min_vol_port = results_frame.iloc[results_frame['stdev'].idxmin()]
plt.rc('font', size=14) 
#create scatter plot coloured by Sharpe Ratio
plt.subplots(figsize=(15,10))
plt.scatter(results_frame.stdev,results_frame.ret,c=results_frame.sharpe,cmap='RdYlBu')
plt.xlabel('Standard Deviation')
plt.ylabel('Returns')
plt.colorbar()
#plot red star to highlight position of portfolio with highest Sharpe Ratio
plt.scatter(max_sharpe_port[1],max_sharpe_port[0],marker=(5,1,0),color='r',s=500)
#plot green star to highlight position of minimum variance portfolio
plt.scatter(min_vol_port[1],min_vol_port[0],marker=(5,1,0),color='g',s=500)
plt.legend(["Sharpe Ratio Value","max Sharpe Portfolio" , "min Volatility Portfolio"],fontsize="18", loc ="upper left")
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.show()
#fig10
max_sharpe_port.to_frame().T
ret	stdev	sharpe	SPY	AAPL	META	NVDA	AMZN
25825	1.338156	0.36815	3.634807	0.005907	0.028012	0.470447	0.480056	0.015578
min_vol_port.to_frame().T
ret	stdev	sharpe	SPY	AAPL	META	NVDA	AMZN
57582	0.31199	0.140741	2.216772	0.710772	0.204089	0.012964	0.009955	0.06222
def calc_portfolio_perf_VaR(weights, mean_returns, cov, alpha, days):
    portfolio_return = np.sum(mean_returns * weights) * days
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(days)
    portfolio_var = abs(portfolio_return - (portfolio_std * stats.norm.ppf(1 - alpha)))
    return portfolio_return, portfolio_std, portfolio_var

def simulate_random_portfolios_VaR(num_portfolios, mean_returns, cov, alpha, days):
    results_matrix = np.zeros((len(mean_returns)+3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        portfolio_return, portfolio_std, portfolio_VaR = calc_portfolio_perf_VaR(weights, mean_returns, cov, alpha, days)
        results_matrix[0,i] = portfolio_return
        results_matrix[1,i] = portfolio_std
        results_matrix[2,i] = portfolio_VaR
        #iterate through the weight vector and add data to results array
        for j in range(len(weights)):
            results_matrix[j+3,i] = weights[j]
            
    results_df = pd.DataFrame(results_matrix.T,columns=['ret','stdev','VaR'] + [ticker for ticker in tickers])
        
    return results_df

mean_returns = df.pct_change().mean()
cov = df.pct_change().cov()
num_portfolios = 100000
rf = 0.0
days = 252
alpha = 0.05
results_frame = simulate_random_portfolios_VaR(num_portfolios, mean_returns, cov, alpha, days)

#locate positon of portfolio with minimum VaR
min_VaR_port = results_frame.iloc[results_frame['VaR'].idxmin()]
#create scatter plot coloured by VaR
plt.subplots(figsize=(15,10))
plt.scatter(results_frame.VaR,results_frame.ret,c=results_frame.VaR,cmap='RdYlBu')
plt.xlabel('Value at Risk')
plt.ylabel('Returns')



plt.colorbar()
#plot red star to highlight position of minimum VaR portfolio
plt.scatter(min_VaR_port[2],min_VaR_port[0],marker=(5,1,0),color='g',s=500)

plt.legend(["VaR Value","min VaR Portfolio"] ,fontsize="18", loc ="upper left")
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.show()
#fig11
min_VaR_port.to_frame().T
ret	stdev	VaR	SPY	AAPL	META	NVDA	AMZN
20537	0.314227	0.182932	0.01333	0.159629	0.779188	0.017246	0.016846	0.027091
#locate positon of portfolio with minimum VaR
min_VaR_port = results_frame.iloc[results_frame['VaR'].idxmin()]
#create scatter plot coloured by VaR
plt.subplots(figsize=(15,10))
plt.scatter(results_frame.stdev,results_frame.ret,c=results_frame.VaR,cmap='RdYlBu')
plt.xlabel('Standard Deviation')
plt.ylabel('Returns')
plt.colorbar()
#plot red star to highlight position of minimum VaR portfolio
plt.scatter(min_VaR_port[1],min_VaR_port[0],marker=(5,1,0),color='g',s=500)

plt.legend(["STD Value","min VaR Portfolio"] ,fontsize="18", loc ="upper left")
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)

plt.show()
#fig12









