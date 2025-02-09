from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.patches import ConnectionPatch
from matplotlib.collections import EllipseCollection
import numpy as np
import seaborn as sns



# istogrammi con conteggio
def count_one_mode(df, campo, x_label, y_label, title, save_path="", mostra=False):
    s = df[campo].value_counts()

    dict = s.to_dict()

    x = list(dict.keys())
    y = list(dict.values())

    plt.clf()
    plt.bar(x, y)
    plt.xticks(rotation=90)
    plt.suptitle(title, fontsize=20)
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=16)
    if save_path != "":
        plt.savefig(save_path, dpi=356, bbox_inches='tight')
        print("grafico salvato in " + save_path)

    if mostra is True:
        plt.show()


def count_kernel(df, campo, color, save_path = "",  mostra = False):
    sns.distplot(df[campo], color=color)
    plt.axvline(df[campo].mean(), color='red')
    mean_value = 'Il valore medio per ' + str(campo) + ' è: ' + str(df[campo].mean())
    if save_path != "":
        plt.savefig(save_path, dpi=356, bbox_inches='tight')
        print("grafico salvato in " + save_path)
    if mostra is True:
        plt.show()
    return mean_value


# tort con barra
def bar_of_pie(param_pie, param_bar, title, save_path):
    # make figure and assign axis objects
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
    fig.subplots_adjust(wspace=0)

    # pie chart parameters
    ratios = param_pie["ratios"]
    labels = param_pie["labels"]
    explode = param_pie["explode"]
    # rotate so that first wedge is split by the x-axis
    angle = -180 * ratios[0]
    ax1.pie(ratios, autopct='%1.1f%%', startangle=angle,
            labels=labels, explode=explode)

    # bar chart parameters

    xpos = 0
    bottom = 0
    ratios = param_bar["ratios"]
    width = .2
    colors = ["royalblue", "cornflowerblue", "lightsteelblue"]

    for j in range(len(ratios)):
        height = ratios[j]
        ax2.bar(xpos, height, width, bottom=bottom, color=colors[j])
        ypos = bottom + ax2.patches[j].get_height() / 2
        bottom += height
        ax2.text(xpos, ypos, "%d%%" % (ax2.patches[j].get_height() * 100),
                 ha='center')

    ax2.set_title(param_bar["title"])
    ax2.legend(param_bar["legend"])
    ax2.axis('off')
    ax2.set_xlim(- 2.5 * width, 2.5 * width)

    # use ConnectionPatch to draw lines between the two plots
    # get the wedge data
    theta1, theta2 = ax1.patches[0].theta1, ax1.patches[0].theta2
    center, r = ax1.patches[0].center, ax1.patches[0].r
    bar_height = sum([item.get_height() for item in ax2.patches])

    # draw top connecting line
    x = r * np.cos(np.pi / 180 * theta2) + center[0]
    y = r * np.sin(np.pi / 180 * theta2) + center[1]
    con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData,
                          xyB=(x, y), coordsB=ax1.transData)
    con.set_color([0, 0, 0])
    con.set_linewidth(4)
    ax2.add_artist(con)

    # draw bottom connecting line
    x = r * np.cos(np.pi / 180 * theta1) + center[0]
    y = r * np.sin(np.pi / 180 * theta1) + center[1]
    con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData,
                          xyB=(x, y), coordsB=ax1.transData)
    con.set_color([0, 0, 0])
    ax2.add_artist(con)
    con.set_linewidth(4)

    plt.suptitle(title, fontsize=20)
    plt.savefig(save_path, dpi=356, bbox_inches='tight')
    print("grafico salvato in " + save_path)

    plt.show()


# instogramma conteggio in orizzontale
def inverted_count(df, campo, x_label, y_label, title, save_path):
    s = df[campo].value_counts()

    dict = s.to_dict()

    x = list(dict.keys())
    y_pos = np.arange(len(x))
    y = list(dict.values())

    plt.rcdefaults()
    fig, ax = plt.subplots()

    error = np.random.rand(len(x))

    ax.barh(y_pos, y, xerr=error, align='center')
    ax.set_yticks(y_pos, labels=x)
    ax.invert_yaxis()  # labels read top-to-bottom
    plt.suptitle(title, fontsize=20)
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=16)
    if save_path != "":
        plt.savefig(save_path, dpi=356, bbox_inches='tight')
        print("grafico salvato in " + save_path)

    plt.show()

def plot_corr_ellipses(data, ax = None, **kwargs):
    M = np.array(data)
    if not M.ndim == 2:
        raise ValueError('data must be a 2D array')
    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw={'aspect': 'equal'})
        ax.set_xlim(-0.5, M.shape[1] - 0.5)
        ax.set_ylim(-0.5, M.shape[0] - 0.5)

    # xy locations of each ellipse center
    xy = np.indices(M.shape)[::-1].reshape(2, -1).T

    # set the relative sizes of the major/minor axes according to the strength of
    # the positive/negative correlation
    w = np.ones_like(M).ravel()
    h = 1 - np.abs(M).ravel()
    a = 45 * np.sign(M).ravel()

    ec = EllipseCollection(widths=w, heights=h, angles=a, units='x', offsets=xy,
                           transOffset=ax.transData, array=M.ravel(), **kwargs)
    ax.add_collection(ec)

    # if data is a DataFrame, use the row/column names as tick labels
    if isinstance(data, pd.DataFrame):
        ax.set_xticks(np.arange(M.shape[1]))
        ax.set_xticklabels(data.columns, rotation=90)
        ax.set_yticks(np.arange(M.shape[0]))
        ax.set_yticklabels(data.index)

    return ec


def pears_corr_plot(data, title, save_path):
    pearsoncorr = data.corr(method='pearson')
    fig, ax = plt.subplots(1, 1)
    m = plot_corr_ellipses(pearsoncorr, ax=ax, cmap='Blues')
    cb = fig.colorbar(m)
    cb.set_label('Correlation coefficient')
    ax.margins(0.1)
    if save_path != '':
        plt.savefig(save_path, dpi=356, bbox_inches='tight')
        print("grafico salvato in " + save_path)
    plt.suptitle(title, fontsize=20)
    plt.show()


def pears_corr_wvalues(data, title, save_path):
    pearsoncorr = data.corr(method='pearson')
    sns.heatmap(pearsoncorr, xticklabels=pearsoncorr.columns, yticklabels=pearsoncorr.columns,
               cmap='RdBu_r', annot=True, linewidth=0.5)
    if save_path != '':
        plt.savefig(save_path, dpi=356, bbox_inches='tight')
        print("grafico salvato in " + save_path)
    plt.suptitle(title, fontsize=20)
    plt.show()

def heatmap(data, title):
    plt.clf()
    plt.figure(figsize=(15, 8))
    sns.heatmap(data.corr(), cmap='Blues', annot=True, annot_kws={"fontsize": 16})
    plt.title(title, fontsize=20)
    plt.savefig('./grafici/heatmap_' + title)
    plt.show()

def pairplot(data, title):
    plt.clf()
    sns.pairplot(data)
    plt.savefig('./grafici/pairplot_' + title)
    plt.show()

def comparison_graph(data, based_on, attribute, treshold, title):
    plt.clf()
    plt.figure(figsize=(18, 7))
    sns.distplot(data.loc[data[based_on] > treshold][attribute], kde=False)
    sns.distplot(data.loc[data[based_on] < treshold][attribute], kde=False)
    plt.title(title, fontsize=20)
    plt.savefig('./grafici/comparison_' + attribute + '_by_' + based_on)
    plt.show()