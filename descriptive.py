from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.patches import ConnectionPatch
import numpy as np
import etl

#istogrammi con conteggio
def count_one_mode(df, campo, x_label, y_label, title, save_path ="", mostra = False):
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
        plt.savefig(save_path, dpi = 356, bbox_inches = 'tight')
        print("grafico salvato in " + save_path)

    if mostra is True:
        plt.show()

#torta con barra
def bar_of_pie(param_pie, param_bar):
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

    plt.show()

