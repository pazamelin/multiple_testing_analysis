import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors

import pandas as pd
from typing import Callable, List

import smsa.cstest

def plot_tests(logrets_mtx: pd.DataFrame,
               sample_size: int,
               testerClass: smsa.cstest.CSTest,
               corrections: List[Callable],
               corrections_names: List[str],
               significance_lvl: float,
               thresholds: List[float],
               suptitle=None,
               figsize=(10,11)):
    # setup the colour map
    colors = [[0, 'tomato'], [1, 'lightgreen']]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    cmap.set_bad(color='yellow')

    # setup the tester
    tester = testerClass(logrets_mtx=logrets_mtx, sample_size=sample_size)
    
    # perform the tests
    results = []
    for mtc in corrections:
        mtc_results = tester.significance_tests([significance_lvl], thresholds, mtc)
        results.append(mtc_results[0])

    # plotting
    plot_rows = len(corrections)
    plot_cols = len(thresholds)
    fig, ax = plt.subplots(nrows=plot_rows, ncols=plot_cols, squeeze=False, figsize=figsize)
    for row in range(0, plot_rows):
        for col in range(0, plot_cols):
            im = ax[row][col].matshow(results[row][col], cmap=cmap, vmin=0, vmax=1)
            ax[row][col].set_title(f'{corrections_names[row]}, α:{significance_lvl}, p0:{thresholds[col]}', fontsize=12)
    
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=16)

    red_patch = mpatches.Patch(color='tomato', label='Negative')
    green_patch = mpatches.Patch(color='lightgreen', label='Positive')
    yellow_patch = mpatches.Patch(color='yellow', label='Diagonal')
    fig.legend(handles=[red_patch, green_patch, yellow_patch], loc='center right', fontsize=12)

    plt.show()


def plot_heatmaps(matrices: List[List[pd.DataFrame]],
                  titles: List[List[str]],
                  suptitle=None,
                  figsize=(10,11)):

    # setup the colour map
    colors = [[0, 'lightgreen'], [1, 'tomato']]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    cmap.set_bad(color='yellow')

    # plotting
    plot_rows = len(matrices)
    plot_cols = len(matrices[0])
    fig, ax = plt.subplots(nrows=plot_rows, ncols=plot_cols, squeeze=False, figsize=figsize)
    for row in range(0, plot_rows):
        for col in range(0, plot_cols):
            im = ax[row][col].matshow(matrices[row][col], cmap=cmap, vmin=0, vmax=1)
            ax[row][col].set_title(titles[row][col], fontsize=12)
    
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=16)

    red_patch = mpatches.Patch(color='tomato', label='-1')
    green_patch = mpatches.Patch(color='lightgreen', label='1')
    fig.legend(handles=[red_patch, green_patch], loc='center right', fontsize=12)

    plt.show()