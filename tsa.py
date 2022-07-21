from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import  adfuller
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
    fig, axes = plt.subplots(3,2, sharex=False)
    axes[0,1].set_xlim([0,20])
    #original
    axes[0, 0].plot(df[coin][feature]);
    axes[0, 0].set_title('Original Series')
    plot_acf(df[coin][feature].dropna(), ax=axes[0, 1])

    #1st diff
    axes[1, 0].plot(df[coin][feature].diff(), );
    axes[1, 0].set_title('1st Order Differencing')
    plot_acf(df[coin][feature].diff().dropna(), ax=axes[1, 1])

    #2nd diff
    axes[2, 0].plot(df[coin][feature].diff());
    axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(df[coin][feature].diff().dropna(), ax=axes[2, 1])
    plt.savefig('./grafici_tsa/diff')
    plt.show()



