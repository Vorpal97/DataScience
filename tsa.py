import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import  adfuller
from statsmodels.tsa.arima.model import ARIMA,ARIMAResults
from numpy import log


def basic_tsa(df_list, coin, feature, ylabel):
    plt.clf()
    data = df_list[coin][feature]
    data.plot(figsize=(15, 5))
    plt.ylabel(ylabel)
    plt.title(coin + ' - ' + feature + ' Value')
    plt.show()


def adickeyfuller(df_list, coin, feature):
    result = adfuller(df_list[coin][feature].dropna())
    print (coin + '-' + feature + ' ADF Statistic: %f' % result[0])
    print(coin + '-' + feature + ' p-value: %f' % result[1])


def differentiation(df, coin, feature):
    plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})
    plt.clf()
    fig, axes = plt.subplots(3,2, sharex=True)

    # axes[0,1].set_xlim([0,20])
    # original
    axes[0, 0].plot(df[coin][feature]);
    axes[0, 0].set_title('Original Series')
    plot_acf(df[coin][feature].dropna(), ax=axes[0, 1])

    # 1st diff
    axes[1, 0].plot(df[coin][feature].diff());
    axes[1, 0].set_title('1st Order Differencing')
    plot_acf(df[coin][feature].diff().dropna(), ax=axes[1, 1])

    # 2nd diff
    axes[2, 0].plot(df[coin][feature].diff());
    axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(df[coin][feature].diff().dropna(), ax=axes[2, 1])
    axes[0,1].set(xlim=(0,20))
    plt.savefig('./grafici_tsa/diff')
    plt.show()


def calcolo_ar(df, coin, feature):
    plt.rcParams.update({'figure.figsize': (9, 3), 'figure.dpi': 120})
    plt.clf()
    fig, axes = plt.subplots(1,2, sharex=False)
    axes[0].plot(df[coin][feature].diff())
    axes[0].set_title('1st Differencing')
    axes[1].set(ylim=(0, 5))
    axes[1].set(xlim=(0, 20))
    plot_pacf(df[coin][feature].diff().dropna(), ax=axes[1])
    plt.savefig('./grafici_tsa/pacf')
    plt.show()


def sarimax(df, coin, feature):
    model = ARIMA(df[coin][feature], order=(1,1,0))
    model_fit = model.fit()
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
    ax[1].set(xlim=(-500, 500))
    plt.savefig('./grafici_tsa/residui')
    plt.show()


def prediction_check(model):
    plt.clf()
    #plt.plot(model.predict(dynamic=False))
    plt.plot(model.predict(dynamic=False))
    plt.title('Predict Vs Actual')
    plt.show()


def fc(train, test):
    model = SARIMAX(train, order=(1,1,0))
    fitted = model.fit()

    fc, se, conf = fitted.forecast(15, alpha=0.05)
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




