!pip install numerapi
import numerapi

!pip install yfinance==0.1.62
!pip install simplejson

!pip install xgboost==1.3.0.post0

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import pathlib
from tqdm.auto import tqdm
import joblib
import json
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from multiprocessing import Pool, cpu_count
import time
import requests as re
from datetime import datetime
from dateutil.relativedelta import relativedelta, FR

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# visualize
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib_venn import venn2, venn3
import seaborn as sns
from matplotlib import pyplot
from matplotlib.ticker import ScalarFormatter
sns.set_context("talk")
style.use('seaborn-colorblind')

import warnings
warnings.simplefilter('ignore')

for dirname, _, filenames in os.walk('/kaggle/input'):
     for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

sp500 = pd.read_csv("../input/sp-500-stocks/sp500_companies.csv")
sp500a = pd.read_csv("../input/sp-500-stocks/sp500_index.csv")
sp500b = pd.read_csv("../input/sp-500-stocks/sp500_stocks.csv")

sp500

sp500a

sp500b

# !pip install pandas_montecarlo --upgrade --no-cache-dir

#from pandas_datareader import data

#df = data.get_data_yahoo("SPY")
#df1 = data.get_data_yahoo("UBX")
#df2 = data.get_data_yahoo("AXDX")
#df3 = data.get_data_yahoo("LTRY")
#df4 = data.get_data_yahoo("GOOG")

#df['return'] = df['Adj Close'].pct_change().fillna(0)
#df1['return'] = df1['Adj Close'].pct_change().fillna(0)
#df2['return'] = df2['Adj Close'].pct_change().fillna(0)
#df3['return'] = df3['Adj Close'].pct_change().fillna(0)
#df4['return'] = df4['Adj Close'].pct_change().fillna(0)

#import pandas_montecarlo
#mc = df['return'].montecarlo(sims=10, bust=-0.1, goal=1)
#mc1 = df1['return'].montecarlo(sims=10, bust=-0.1, goal=1)
#mc2 = df2['return'].montecarlo(sims=10, bust=-0.1, goal=1)
#mc3 = df3['return'].montecarlo(sims=10, bust=-0.1, goal=1)
#mc4 = df4['return'].montecarlo(sims=10, bust=-0.1, goal=1)#

#mc.plot(title="Returns Monte Carlo Simulations")  # optional: , figsize=(x, y)
#mc1.plot(title="Returns Monte Carlo Simulations")  # optional: , figsize=(x, y)
#mc2.plot(title="Returns Monte Carlo Simulations")  # optional: , figsize=(x, y)
##mc3.plot(title="Returns Monte Carlo Simulations")  # optional: , figsize=(x, y)
#mc4.plot(title="Returns Monte Carlo Simulations")  # optional: , figsize=(x, y)

#print(mc.stats)
#print(mc.maxdd)
#print(mc.data.head())

#print(mc1.stats)
#print(mc1.maxdd)
#print(mc1.data.head())

#print(mc2.stats)
#print(mc2.maxdd)
#print(mc2.data.head())

#print(mc3.stats)
#print(mc3.maxdd)
#print(mc3.data.head())

#print(mc4.stats)
#print(mc4.maxdd)
#print(mc4.data.head())

today = datetime.now().strftime('%Y-%m-%d')
today

# config class
class CFG:
    """
    Set FETCH_VIA_API = True if you want to fetch the data via API.
    Otherwise we use the daily-updated one in the kaggle dataset (faster).
    """
    INPUT_DIR = '../input/yfinance-stock-price-data-for-numerai-signals'
    OUTPUT_DIR = './'
    FETCH_VIA_API = False
    SEED = 46
    DEBUG = False # True, test mode using small set of tickers
    
# Logging is always nice for your experiment:)
def init_logger(log_file='train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

logger = init_logger(log_file=f'{CFG.OUTPUT_DIR}/{today}.log')
logger.info('Start Logging...')

napi = numerapi.SignalsAPI()
logger.info('numerai api setup!')

# read in list of active Signals tickers which can change slightly era to era
eligible_tickers = pd.Series(napi.ticker_universe(), name='ticker') 
logger.info(f"Number of eligible tickers: {len(eligible_tickers)}")

# read in yahoo to numerai ticker map, still a work in progress, h/t wsouza and 
# this tickermap is a work in progress and not guaranteed to be 100% correct
ticker_map = pd.read_csv('https://numerai-signals-public-data.s3-us-west-2.amazonaws.com/signals_ticker_map_w_bbg.csv')
ticker_map = ticker_map[ticker_map.bloomberg_ticker.isin(eligible_tickers)]

numerai_tickers = ticker_map['ticker']
yfinance_tickers = ticker_map['yahoo']
logger.info(f"Number of eligible tickers in map: {len(ticker_map)}")

print(ticker_map.shape)
ticker_map.head()

numerai_universe = 'https://numerai-signals-public-data.s3-us-west-2.amazonaws.com/universe/latest.csv'
latest_universe = pd.read_csv(numerai_universe)
#latest_universe = reduce_mem_usage(latest_universe)
print(latest_universe.shape)
latest_universe

# If you want to fetch the data on your own, you can use this function...

def fetch_yfinance(ticker_map, start='2012-12-01'):
    """
    # fetch yfinance data
    :INPUT:
    - ticker_map : Numerai eligible ticker map (pd.DataFrame)
    - start : date (str)
    
    :OUTPUT:
    - full_data : pd.DataFrame ('date', 'ticker', 'close', 'raw_close', 'high', 'low', 'open', 'volume')
    """
    
    # ticker map
    numerai_tickers = ticker_map['ticker']
    yfinance_tickers = ticker_map['yahoo']

    # fetch
    raw_data = yfinance.download(
        yfinance_tickers.str.cat(sep=' '), 
        start=start, 
        threads=True
    ) 
    
    # format
    cols = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    full_data = raw_data[cols].stack().reset_index()
    full_data.columns = ['date', 'ticker', 'close', 'raw_close', 'high', 'low', 'open', 'volume']
    full_data['date'] = pd.to_datetime(full_data['date'], format='%Y-%m-%d').dt.strftime('%Y%m%d').astype(int)
    
    # map yfiance ticker to numerai tickers
    full_data['ticker'] = full_data.ticker.map(
        dict(zip(yfinance_tickers, numerai_tickers))
    )
    return full_data
  
  %%time

if CFG.FETCH_VIA_API: # fetch data via api
    logger.info('Fetch data via API...may take some time...')
    !pip install yfinance==0.1.62
    !pip install simplejson
    import yfinance
    import simplejson
    
    df = fetch_yfinance(ticker_map, start='2015-12-01')
else: # loading from the kaggle dataset (https://www.kaggle.com/code1110/yfinance-stock-price-data-for-numerai-signals)
        logger.info('Load data from the kaggle dataset...')
        #df = pd.read_csv(pathlib.Path(f'{CFG.INPUT_DIR}/full_data.csv'))
try:
        df = pd.read_parquet(pathlib.Path(f'{CFG.INPUT_DIR}/full_data.parquet'))
except: # no data loaded err somehow
        # fetch data via kaggle API
        #from kaggle_secrets import UserSecretsClient
        #user_secrets = UserSecretsClient()
        #config = {}
        #config['username'] = user_secrets.get_secret("username")
        #config['key'] = user_secrets.get_secret("key")
        import nbformat
        #KAGGLE_CONFIG_DIR = os.path.join(os.path.expandvars('$HOME'), '.kaggle')
        os.makedirs(KAGGLE_CONFIG_DIR, exist_ok = True)
        #with open(os.path.join(KAGGLE_CONFIG_DIR, 'kaggle.json'), 'w') as f:
            #json.dump({'username': config['username'], 'key': config['key']}, f)
 #       !chmod 600 {KAGGLE_CONFIG_DIR}/kaggle.json
        !kaggle datasets download -d code1110/yfinance-stock-price-data-for-numerai-signals
        !unzip yfinance-stock-price-data-for-numerai-signals.zip
        !rm yfinance-stock-price-data-for-numerai-signals.zip
        df = pd.read_parquet('full_data.parquet')        

print(df.shape)
df.head(3)

df.tail(3)

%%time

def read_numerai_signals_targets():
    # read in Signals targets
#     numerai_targets = 'https://numerai-signals-public-data.s3-us-west-2.amazonaws.com/signals_train_val.csv'
    numerai_targets = 'https://numerai-signals-public-data.s3-us-west-2.amazonaws.com/signals_train_val_bbg.csv'
    targets = pd.read_csv(numerai_targets)
    
    # to datetime int
    targets['friday_date'] = pd.to_datetime(targets['friday_date'].astype(str), format='%Y-%m-%d').dt.strftime('%Y%m%d').astype(int)
    
#     # train, valid split
#     train_targets = targets.query('data_type == "train"')
#     valid_targets = targets.query('data_type == "validation"')
    
    return targets

targets = read_numerai_signals_targets()

# convert to numerai ticker, if the target ticker is not
if 'bloomberg_ticker' in targets.columns.values.tolist():
    targets['ticker'] = targets['bloomberg_ticker'].map(
        dict(zip(ticker_map['bloomberg_ticker'], ticker_map['ticker']))
    )
if 'bloomberg_ticker' not in targets.columns.values.tolist():
    targets['bloomberg_ticker'] = targets['ticker'].map(
        dict(zip(ticker_map['ticker'], ticker_map['bloomberg_ticker']))
    )
    
print(targets.shape, targets['friday_date'].min(), targets['friday_date'].max())
targets.head()

targets.tail()

# there are train and validation...
fig, ax = plt.subplots(1, 2, figsize=(16, 4))
ax = ax.flatten()

for i, data_type in enumerate(['train', 'validation']):
    # slice
    targets_ = targets.query(f'data_type == "{data_type}"')
    logger.info('*' * 50)
    logger.info('{} target: {:,} numerai tickers , {:,} bloomberg tickers (friday_date: {} - {})'.format(
        data_type, 
        targets_['ticker'].nunique(),
        targets_['bloomberg_ticker'].nunique(),
        targets_['friday_date'].min(),
        targets_['friday_date'].max(),
    ))
    
    # plot target
#     ax[i].hist(targets_['target'])
    ax[i].hist(targets_['target_20d'])
    ax[i].set_title(f'{data_type}')
    
# target relations
d = pd.crosstab(
    targets['target_4d']
    , targets['target_20d']
)
d['sum'] = d.values.sum(axis=1)
for i, f in enumerate(d.columns):
    d[f] = d.apply(lambda row : 100*row[f]/row['sum'], axis=1)
d.drop(columns=['sum'], inplace=True)

print('target transition matrix (%)')
d.astype(int).style.background_gradient(cmap='viridis', axis=1)

# ticker overlap
venn3(
    [
        set(df['ticker'].unique().tolist())
        , set(targets.query('data_type == "train"')['ticker'].unique().tolist())
        , set(targets.query('data_type == "validation"')['ticker'].unique().tolist())
    ],
    set_labels=('yf price', 'train target', 'valid target')
)

# select target-only tickers
df = df.loc[df['ticker'].isin(targets['ticker'])].reset_index(drop=True)

print('{:,} tickers: {:,} records'.format(df['ticker'].nunique(), len(df)))

record_per_ticker = df.groupby('ticker')['date'].nunique().reset_index().sort_values(by='date')
record_per_ticker

record_per_ticker['date'].hist()
print(record_per_ticker['date'].describe())

if CFG.DEBUG: # debug mode, using small set of data
    tickers_with_records = record_per_ticker.query('date >= 4830')['ticker'].values
else:
    tickers_with_records = record_per_ticker.query('date >= 1000')['ticker'].values
df = df.loc[df['ticker'].isin(tickers_with_records)].reset_index(drop=True)

print('Here, we use {:,} tickers: {:,} records'.format(df['ticker'].nunique(), len(df)))

# first, fix date column in the yfiance stock data to be friday date (just naming along with numerai targets)
df['friday_date'] = df['date'].apply(lambda x : int(str(x).replace('-', '')))
df.tail(3)

# recent friday date?
recent_friday = datetime.now() + relativedelta(weekday=FR(-1))
recent_friday = int(recent_friday.strftime('%Y%m%d'))
print(f'Most recent Friday: {recent_friday}')

# in case no recent friday is available...prep the second last
recent_friday2 = datetime.now() + relativedelta(weekday=FR(-2))
recent_friday2 = int(recent_friday2.strftime('%Y%m%d'))
print(f'Second most recent Friday: {recent_friday2}')

# fix market-not-open-on-Friday problem
if CFG.DEBUG == False:
    if np.sum(df['friday_date'] == recent_friday) < 4000:
        previous_tickers = set(df.query('friday_date == @recent_friday2')['ticker']) 
        current_tickers = set(df.query('friday_date == @recent_friday')['ticker'])
        missing_df = pd.DataFrame()
        missing_df['ticker'] = list(previous_tickers - current_tickers)
        for d in ['date', 'friday_date']:
            missing_df[d] = recent_friday
        
        # concat
        orig_shape = df.shape
        df = pd.concat([df, missing_df]).sort_values(by=['ticker', 'friday_date']).fillna(method='ffill')
        del missing_df
        print('Resolving missing tickers due to market-not-open-on-friday issue: df shape {} => {}'.format(
            orig_shape, df.shape
        ))
        
# technical indicators
def RSI(close: pd.DataFrame, period: int = 14) -> pd.Series:
    # https://gist.github.com/jmoz/1f93b264650376131ed65875782df386
    """https://www.tradingview.com/wiki/Talk:Relative_Strength_Index_(RSI)
    Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements.
    RSI oscillates between zero and 100. Traditionally, and according to Wilder, RSI is considered overbought when above 70 and oversold when below 30.
    Signals can also be generated by looking for divergences, failure swings and centerline crossovers.
    RSI can also be used to identify the general trend."""

    delta = close.diff()

    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    _gain = up.ewm(com=(period - 1), min_periods=period).mean()
    _loss = down.abs().ewm(com=(period - 1), min_periods=period).mean()

    RS = _gain / _loss
    return pd.Series(100 - (100 / (1 + RS)))
  
  def EMA1(x, n):
    """
    https://qiita.com/MuAuan/items/b08616a841be25d29817
    """
    a= 2/(n+1)
    return pd.Series(x).ewm(alpha=a).mean()

def MACD(close : pd.DataFrame, span1=12, span2=26, span3=9):
    """
    Compute MACD
    # https://www.learnpythonwithrune.org/pandas-calculate-the-moving-average-convergence-divergence-macd-for-a-stock/
    """
    exp1 = EMA1(close, span1)
    exp2 = EMA1(close, span2)
    macd = 100 * (exp1 - exp2) / exp2
    signal = EMA1(macd, span3)

    return macd, signal
  
def kurtosis(close, length, offset, **kwargs):
    """Indicator: Kurtosis"""
    min_periods = int(kwargs["min_periods"]) if "min_periods" in kwargs and kwargs["min_periods"] is not None else length
    #offset = 7

    if close is None: return

    # Calculate Result
    kurtosis = close.rolling(length, min_periods=min_periods).kurt()

    # Offset
    if offset != 0:
        kurtosis = kurtosis.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        kurtosis.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        kurtosis.fillna(method=kwargs["fill_method"], inplace=True)

    # Name & Category
#    kurtosis.name = f"KURT_{length}"
#    kurtosis.category = "statistics"

    return kurtosis

def entropy(close, length, base, offset, **kwargs):
    """Indicator: Entropy (ENTP)"""
    # Validate Arguments
    #offset = 7

    if close is None: return

    # Calculate Result
    p = close / close.rolling(length).sum()
    entropy = (-p * np.log(p) / np.log(base)).rolling(length).sum()

    # Offset
    if offset != 0:
        entropy = entropy.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        entropy.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        entropy.fillna(method=kwargs["fill_method"], inplace=True)

    # Name & Category
    #entropy.name = f"ENTP_{length}"
    #entropy.category = "statistics"
    
    return entropy
  
def feature_engineering(ticker='ZEAL DC', df=df):
    """
    feature engineering
    
    :INPUTS:
    - ticker : numerai ticker name (str)
    - df : yfinance dataframe (pd.DataFrame)
    
    :OUTPUTS:
    - feature_df : feature engineered dataframe (pd.DataFrame)
    """
    # init
    keys = ['friday_date', 'ticker']
    feature_df = df.query(f'ticker == "{ticker}"')
    
    # price features
    new_feats = []
    for i, f in enumerate(['close', ]):
        for x in [20, 40, 60,]:
            # return
            feature_df[f"{f}_return_{x}days"] = feature_df[
                f
            ].pct_change(x)

            # volatility
            feature_df[f"{f}_volatility_{x}days"] = (
                np.log1p(feature_df[f])
                .pct_change()
                .rolling(x)
                .std()
            )
        
            # kairi mean
            feature_df[f"{f}_MA_gap_{x}days"] = feature_df[f] / (
                feature_df[f].rolling(x).mean()
            )
            
            # features to use
            new_feats += [
                f"{f}_return_{x}days", 
                f"{f}_volatility_{x}days",
                f"{f}_MA_gap_{x}days",
                         ]
            
    # RSI
    feature_df['RSI'] = RSI(feature_df['close'], 14)
    feature_df['RSI1'] = RSI(feature_df['close']*feature_df['open']/feature_df['high'], 14)
    feature_df['entropy'] = entropy(feature_df['close'], 15, 11, 3)
    #feature_df['kurtosis'] = kurtosis(feature_df['close'], 23, 15)
    # MACD
    macd, macd_signal = MACD(feature_df['close'], 12, 26, 9) 
    macd1, macd_signal1 = MACD(feature_df['close']*feature_df['open']/feature_df['high'], 12, 26, 9)
    #ema_1 = EMA1(feature_df['open'],2)
    #kurtosis1 = kurtosis(feature_df['close'])
    #entropy1 = entropy(feature_df['close'])
    feature_df['MACD'] = macd
    feature_df['MACD1'] = macd1
    feature_df['MACD_signal'] = macd_signal
    feature_df['MACD_signal1'] = macd_signal1
    #feature_df['MACD_Corr'] = abs(feature_df['MACD_signal'] - feature_df['MACD_signal1'])
    #feature_df['RSI_Corr'] = abs(feature_df['RSI'] - feature_df['RSI1'])
    #feature_df['ratio'] = np.log(feature_df['high'],feature_df['low'])
    #feature_df['EMA1'] = ema_1
    #feature_df['EMA'] = EMA1(feature_df['close'], 14)
    #feature_df['kurtosis'] = kurtosis
    #feature_df['entropy'] = entropy
    
    new_feats += ['RSI', 'RSI1', 'MACD', 'MACD1','MACD_signal', 'MACD_signal1', 'entropy']

    # only new feats
    feature_df = feature_df[new_feats + keys]

    # fill nan
    feature_df.fillna(method='ffill', inplace=True) # safe fillna method for a forecasting task
    feature_df.fillna(method='bfill', inplace=True) # just in case ... making sure no nan

    return feature_df

def add_features(df):
    # FE with multiprocessing
    tickers = df['ticker'].unique().tolist()
    print('FE for {:,} stocks...using {:,} CPUs...'.format(len(tickers), cpu_count()))
    start_time = time.time()
    with Pool(cpu_count()) as p:
        feature_dfs = list(tqdm(p.imap(feature_engineering, tickers), total=len(tickers)))
    return pd.concat(feature_dfs)
  
import numpy as np 
import pandas as pd 
from tqdm.auto import tqdm

def neutralize_series(series : pd.Series, by : pd.Series, proportion=1.0):
    scores = series.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)
    exposures = np.hstack((exposures, np.array([np.mean(series)] * len(exposures)).reshape(-1, 1)))
    correction = proportion * (exposures.dot(np.linalg.lstsq(exposures, scores)[0]))
    corrected_scores = scores - correction
    neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    return neutralized
  
%%time

df.head()
feature_df = add_features(df)
#feature_df = neutralize_series(feature_df,feature_df['RSI'])
del df
gc.collect()

print(feature_df.shape)
feature_df.head()

feature_df.tail()

# do we have enough overlap with respect to 'friday_date'?
venn2([
    set(feature_df['friday_date'].astype(str).unique().tolist())
    , set(targets['friday_date'].astype(str).unique().tolist())
], set_labels=('features_days', 'targets_days'))

# do we have enough overlap with respect to 'ticker'?
venn2([
    set(feature_df['ticker'].astype(str).unique().tolist())
    , set(targets['ticker'].astype(str).unique().tolist())
], set_labels=('features_ticker', 'targets_ticker'))

# merge
feature_df['friday_date'] = feature_df['friday_date'].astype(int)
targets['friday_date'] = targets['friday_date'].astype(int)

feature_df = feature_df.merge(
    targets,
    how='left',
    on=['friday_date', 'ticker']
)

print(feature_df.shape)
feature_df.tail()

# save (just to make sure that we are on the safe side if yfinance is dead some day...)
feature_df.to_pickle(f'{CFG.OUTPUT_DIR}/feature_df.pkl')
feature_df.info()

target = 'target_20d'
if 'target_20d' not in feature_df.columns.values.tolist():
    print('No target 20d exists...using target_4d instead...')
    target = 'target'
drops = ['data_type', 'target_4d', 'target_20d', 'friday_date', 'ticker', 'bloomberg_ticker']
features = [f for f in feature_df.columns.values.tolist() if f not in drops]

logger.info('{:,} features: {}'.format(len(features), features))

# train-valid split
train_set = {
    'X': feature_df.query('data_type == "train"')[features], 
    'y': feature_df.query('data_type == "train"')[target].astype(np.float64)
}
val_df = feature_df.query('data_type == "validation"').dropna().copy()
val_set = {
    'X': val_df[features], 
    'y': val_df[target].astype(np.float64)
}

assert train_set['y'].isna().sum() == 0
assert val_set['y'].isna().sum() == 0

# same parameters of the Integration-Test
import joblib
from sklearn import utils
import xgboost as xgb
import operator

params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'colsample_bytree': 0.1,                 
    'learning_rate': 0.01,
    'max_depth': 5,
    'seed': 46,
    'n_estimators': 2000
}

# define 
# model = xgb.XGBRegressor(**params)

# detect and init the TPU
#tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()

# instantiate a distribution strategy
#tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# instantiating the model in the strategy scope creates the model on the TPU
#with tpu_strategy.scope():
model = xgb.XGBRegressor(**params) # define your model normally
#model.compile()
    
# fit

model.fit(
    train_set['X'], train_set['y'], 
    eval_set=[(val_set['X'], val_set['y'])],
    verbose=100, 
    early_stopping_rounds=100,
)
model.score(val_set['X'], val_set['y'])
# save model
joblib.dump(model, f'{CFG.OUTPUT_DIR}/xgb_model_val.pkl')
logger.info('xgb model with early stopping saved!')

# feature importance
importance = model.get_booster().get_score(importance_type='gain')
importance = sorted(importance.items(), key=operator.itemgetter(1))
feature_importance_df = pd.DataFrame(importance, columns=['features', 'importance'])

# feature importance
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
sns.barplot(
    x='importance', 
    y='features', 
    data=feature_importance_df.sort_values(by='importance', ascending=False),
    ax=ax
)

# https://colab.research.google.com/drive/1ECh69C0LDCUnuyvEmNFZ51l_276nkQqo#scrollTo=tTBUzPep2dm3

def score(df, target_name=target, pred_name='prediction'):
    '''Takes df and calculates spearm correlation from pre-defined cols'''
    # method="first" breaks ties based on order in array
    return np.corrcoef(
        df[target_name],
        df[pred_name].rank(pct=True, method="first")
    )[0,1]
  
def run_analytics(era_scores):
    print(f"Mean Correlation: {era_scores.mean():.4f}")
    print(f"Median Correlation: {era_scores.median():.4f}")
    print(f"Standard Deviation: {era_scores.std():.4f}")
    print('\n')
    print(f"Mean Pseudo-Sharpe: {era_scores.mean()/era_scores.std():.4f}")
    print(f"Median Pseudo-Sharpe: {era_scores.median()/era_scores.std():.4f}")
    print('\n')
    print(f'Hit Rate (% positive eras): {era_scores.apply(lambda x: np.sign(x)).value_counts()[1]/len(era_scores):.2%}')

    era_scores.rolling(20).mean().plot(kind='line', title='Rolling Per Era Correlation Mean', figsize=(15,4))
    plt.axhline(y=0.0, color="r", linestyle="--"); plt.show()

    era_scores.cumsum().plot(title='Cumulative Sum of Era Scores', figsize=(15,4))
    plt.axhline(y=0.0, color="r", linestyle="--"); plt.show()
    
# prediction for the validation set
valid_sub = val_df[drops].copy()
valid_sub['prediction'] = model.predict(val_set['X'])

# compute score
val_era_scores = valid_sub.copy()
val_era_scores['friday_date'] = val_era_scores['friday_date'].astype(str)
val_era_scores = val_era_scores.loc[val_era_scores['prediction'].isna() == False].groupby(['friday_date']).apply(score)
run_analytics(val_era_scores)

# do we have at least 5 tickers, whose the latest date matches the recent friday?
ticker_date_df = feature_df.groupby('ticker')['friday_date'].max().reset_index()
if len(ticker_date_df.loc[ticker_date_df['friday_date'] == recent_friday]) >= 5:
    ticker_date_df = ticker_date_df.loc[ticker_date_df['friday_date'] == recent_friday]
else: # use dates later than the second last friday
    ticker_date_df = ticker_date_df.loc[ticker_date_df['friday_date'] == recent_friday2]
    recent_friday = recent_friday2
    
print(len(ticker_date_df))
ticker_date_df

# live sub
feature_df.loc[feature_df['friday_date'] == recent_friday, 'data_type'] = 'live'
test_sub = feature_df.query('data_type == "live"')[drops].copy()
test_sub['prediction'] = model.predict(feature_df.query('data_type == "live"')[features])

logger.info(test_sub.shape)
test_sub.head()

# histogram of prediction
test_sub['prediction'].hist(bins=100)

