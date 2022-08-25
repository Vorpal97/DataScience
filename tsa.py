import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_log_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import  adfuller
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import acf

def basic_tsa(df, feature, ylabel):
    plt.clf()
    data = df[feature]
    data.plot(figsize=(15, 5))
    plt.ylabel(ylabel)
    plt.title(ylabel + ' by ' + feature)
    plt.show()


def adickeyfuller(df):
    result = adfuller(df.values)
    print ('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])


def differentiation(df, feature):
    plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})
    plt.clf()
    fig, axes = plt.subplots(3,2, sharex=False)

    # axes[0,1].set_xlim([0,20])
    # original
    axes[0, 0].plot(df[feature]);
    axes[0, 0].set_title('Original Series')
    plot_acf(df[feature], ax=axes[0, 1])

    # 1st diff
    axes[1, 0].plot(df[feature].diff());
    axes[1, 0].set_title('1st Order Differencing')
    plot_acf(df[feature].diff().dropna(), ax=axes[1, 1])

    # 2nd diff
    axes[2, 0].plot(df[feature].diff().diff());
    axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(df[feature].diff().diff().dropna(), ax=axes[2, 1])
    axes[0, 1].get_xaxis().set_visible(False)
    axes[1, 1].get_xaxis().set_visible(False)
    axes[0, 0].get_xaxis().set_visible(False)
    axes[1, 0].get_xaxis().set_visible(False)


    plt.savefig('./grafici_tsa/diff')
    plt.show()


def calcolo_ar(df, feature):
    plt.rcParams.update({'figure.figsize': (9, 3), 'figure.dpi': 120})
    plt.clf()
    fig, axes = plt.subplots(1,2, sharex=False)
    axes[0].plot(df[feature].diff())
    axes[0].set_title('1st Differencing')
    plot_pacf(df[feature].diff().dropna(), ax=axes[1])
    plt.savefig('./grafici_tsa/pacf')
    plt.show()

def arima(df, feature):
    model = ARIMA(df[feature], order=(0,1,0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    with open('grafici_tsa' + '/arima.txt', 'w') as f:
        f.write(str(model_fit.summary()))
    return model_fit


def residui(model):
    plt.clf()
    residuals = pd.DataFrame(model.resid)
    fig, ax = plt.subplots(1, 2)
    residuals.plot(title='Residuals', ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1])
    plt.savefig('./grafici_tsa/residui')
    plt.show()


def prediction_check(model):
    plt.clf()
    plt.plot(model.predict(dynamic=False))
    #plt.plot(model.predict(dynamic=False))
    plt.title('Predict Vs Actual')
    plt.show()


def fc(train, test):
    model = ARIMA(train, order=(0,1,0))
    fitted = model.fit(disp=-1)

    fc, se, conf = fitted.forecast(6, alpha=0.05) #95% conf
    fc_series = pd.Series(fc, index=test.index)
    lower_series = pd.Series(conf[:, 0], index=test.index)
    upper_series = pd.Series(conf[:, 1], index=test.index)

    plt.clf()
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(train, label='training')
    plt.plot(test, label='actual')
    plt.plot(fc_series, label='forecast')
    plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()

def RMSLE(y_true, y_pred):
    MSLE = mean_squared_log_error(y_true, y_pred)
    return np.sqrt(MSLE)

def sarimax(train, test):
    model = SARIMAX(train, order=(0,1,0), seasonal_order=(2,1,1,7))
    fit = model.fit()
    y_pred = fit.forecast(6)
    errors = []
    err = RMSLE(test, y_pred)
    errors.append(err)
    fig, ax = plt.subplots(figsize=(16, 3))
    years = mdates.YearLocator()
    for name, dat, c in zip(['train', 'test', 'pred'], [train, test, y_pred], ['b', 'g', 'r']):
        ax.plot(dat, c=c, label=name)
    ax.xaxis.set_major_locator(years)
    plt.legend()
    plt.grid()
    plt.show()
    return y_pred, test


def actualvsfitted(model):
    plt.clf()
    model.plot_predict(dynamic=False)
    plt.show()


def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))
    me = np.mean(forecast - actual)
    mae = np.mean(np.abs(forecast - actual))
    mpe = np.mean ((forecast - actual)/actual)
    rmse = np.mean((forecast - actual)**2)**.5
    corr = np.corrcoef(forecast, actual)[0, 1]
    mins = np.amin(np.hstack([forecast[:,None],
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None],
                              actual[:,None]]), axis=1)
    minmax = 1-np.mean(mins/maxs)
    acf1 = acf(forecast - actual)[1]
    results = {'mape': mape, 'me': me, 'mae': mae,
            'mpe': mpe, 'rmse': rmse, 'acf1': acf1,
            'corr': corr, 'minmax': minmax}
    with open('grafici_tsa' + '/arima.txt', 'w') as f:
        f.write(str(results))
    return results




