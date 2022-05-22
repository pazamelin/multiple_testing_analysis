import math
import numpy as np
import pandas as pd

from typing import Callable, List
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

import smsa.cstest
import smsa.sampling


def expected_mtxs(logrets_mtx, thresholds: List[float]):
    true_correlation_mtx = logrets_mtx.corr(method='pearson')
    ge_mtxs = [true_correlation_mtx.lt(th) for th in thresholds]
    ge_mtxs = [df.replace({False: 0, True: 1}) for df in ge_mtxs]
    return ge_mtxs


def count_values(result_mtx, value):
    count = 0

    mtx_size = result_mtx.shape[0]
    for row in range(0, mtx_size - 1):
        for col in range(row + 1, mtx_size):
            test_result = result_mtx.iloc[row, col]
            if test_result == value:
                # accepted hypothesis
                count += 1
    return count


def expected_plot(sample_logrets_mtx,
                  blending_params=[bp * 0.25 for bp in range(0, 5)], 
                  thresholds=[th * 0.05 for th in range(0, 21)], 
                  iterations=100,
                  figsize=(25, 10)):
    # collect generator parameters
    sample_size = sample_logrets_mtx.shape[0]
    matrix_size = sample_logrets_mtx.shape[1]

    # collect mean plots
    mean_expected_positive = np.zeros((len(blending_params), len(thresholds)))
    mean_expected_negative = np.zeros((len(blending_params), len(thresholds)))

    for i in range(0, iterations):
        for j, bp in enumerate(blending_params):
            # generate data for
            logrets_mtx = smsa.sampling.generate_logrets_mtx(
                bp,
                sample_size,
                sample_logrets_mtx.mean(),
                sample_logrets_mtx.cov(),   
                columns=sample_logrets_mtx.columns,
            )

            # get expected matrices
            expected_mtxs_ = expected_mtxs(logrets_mtx, thresholds)
            
            # add the expected counts
            expected_positive = [count_values(expected_mtx, 1) for expected_mtx in expected_mtxs_]
            mean_expected_positive[j] += np.array(expected_positive)

            expected_negative = [count_values(expected_mtx, 0) for expected_mtx in expected_mtxs_]
            mean_expected_negative[j] += np.array(expected_negative)

    mean_expected_positive = [m / iterations for m in mean_expected_positive]
    mean_expected_positive = np.around(mean_expected_positive, 2)
    mean_expected_positive = pd.DataFrame(data=mean_expected_positive, columns=[f'{th}' for th in thresholds])

    mean_expected_negative = [m / iterations for m in mean_expected_negative]
    mean_expected_negative = np.around(mean_expected_negative, 2)
    mean_expected_negative = pd.DataFrame(data=mean_expected_negative, columns=[f'{th}' for th in thresholds])

    plot_rows = 1
    plot_cols = len(blending_params)
    fig, ax = plt.subplots(nrows=plot_rows, ncols=plot_cols, squeeze=False, figsize=figsize)
    for row in range(0, plot_rows):
        for col in range(0, plot_cols):
            ax[row][col].plot(thresholds, mean_expected_positive.iloc[col], 'g', label='True Positive')
            ax[row][col].plot(thresholds, mean_expected_negative.iloc[col], 'r', label='True Negative')

    plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    plt.legend()
    plt.show()


def collect_metrics(expected_mtx, actual_mtx):
    metrics = {
        'FP': 0, 'TP': 0, 
        'TN': 0, 'FN': 0,
    }

    # H: y >= y0 - negative (0 - edge)
    # K: y < y0  - positive (1 - no edge)
    mtx_size = expected_mtx.shape[0]
    for row in range(0, mtx_size - 1):
        for col in range(row + 1, mtx_size):
            expected = expected_mtx.iloc[row, col]
            actual = actual_mtx.iloc[row, col]

            status = None
            if (expected, actual) == (1, 1):
                status = 'TP'

            if (expected, actual) == (1, 0):
                status = 'FN'

            if (expected, actual) == (0, 1):
                status = 'FP'

            if (expected, actual) == (0, 0):
                status = 'TN'
            
            metrics[status] += 1                        
    
    return metrics


def collect_mean_metrics(tester_class: smsa.cstest.CSTest,
                    correction: Callable,
                    blending_params: List[float], 
                    thresholds: List[float],
                    significance_lvl: float,
                    sample_logrets_mtx,
                    iterations:int = 1000):
    # collect generator parameters
    sample_size = sample_logrets_mtx.shape[0]
    matrix_size = sample_logrets_mtx.shape[1]

    # get expected matrices
    expected_mtxs_ = expected_mtxs(sample_logrets_mtx, thresholds)
    # true edges expected
    expected_positive = [count_values(expected_mtx, 1) for expected_mtx in expected_mtxs_]
    # false edges expected
    expected_negative = [count_values(expected_mtx, 0) for expected_mtx in expected_mtxs_]

    # find the number of individual hypotheses
    ntests = math.comb(matrix_size, 2)

    # TODO: more pythonic?
    combinations = []
    for i in range(0, len(thresholds)):
        for j in range(0, len(blending_params)):
            combinations.append((i, j))

    correction_name = correction.__name__ if correction is not None else 'None'

    results_data = []
    for (i, j) in tqdm(combinations, desc=f'{tester_class.__name__[6:]}, {correction_name}'):
        th = thresholds[i]
        bp = blending_params[j]

        metrics_keys = ['FP', 'TP', 'TN', 'FN']
        metrics_total = {key: 0 for key in metrics_keys}
        # TODO: fix WA
        metrics_total['P(FP=0)'] = 0
        metrics_total['P(FN=0)'] = 0

        for it in range(0, iterations):
            # generate data
            logrets_mtx = smsa.sampling.generate_logrets_mtx(
                bp,
                sample_size,
                sample_logrets_mtx.mean(),
                sample_logrets_mtx.cov(),   
                columns=sample_logrets_mtx.columns,
            )

            # perform the test
            tester = tester_class(logrets_mtx=logrets_mtx, sample_size=sample_size)
            test_results = tester.significance_test(significance_lvl, th, correction)
            metrics = collect_metrics(expected_mtxs_[i], test_results)

            # merge metrics to total dict
            for key in metrics_keys:
                metrics_total[key] += metrics[key]

            if metrics['FP'] == 0:
                metrics_total['P(FP=0)'] += 1

            if metrics['FN'] == 0:
                metrics_total['P(FN=0)'] += 1

        # find metrics mean values
        for key in metrics_total.keys():
            metrics_total[key] = metrics_total[key] / iterations

        results_data.append(
            [
                tester_class.get_name(),
                correction_name,
                sample_size,
                ntests,
                bp,
                th,
                expected_positive[i],
                expected_negative[i],
                significance_lvl,
                iterations,
                metrics_total['FP'],
                metrics_total['TP'],
                metrics_total['TN'],
                metrics_total['FN'],
                metrics_total['P(FP=0)'],
                metrics_total['P(FN=0)'],
            ]
        )

    results_mtx = pd.DataFrame(data=results_data, 
                               columns=[
                                   'tester',
                                   'correction',
                                   'sample_size',
                                   'ntests', 
                                   'bp',
                                   'th',
                                   'EP',
                                   'EN',
                                   'slvl',
                                   'nrepeats', 
                                   'FP', 'TP', 'TN', 'FN',
                                   'P(FP=0)', 'P(FN=0)',
                               ],
    )

    return results_mtx


def collect_asymptotic_metrics(tester_class: smsa.cstest.CSTest,
                               correction: Callable,
                               blending_param: float, 
                               thresholds: List[float],
                               significance_lvl: float,
                               max_sample_logrets_mtx: pd.DataFrame,
                               stepdown: int,
                               iterations:int = 300):

    # collect generator parameters
    sample_size = max_sample_logrets_mtx.shape[0]
    matrix_size = max_sample_logrets_mtx.shape[1]

    results = []
    # iterate over the matrix sizes in a top-down approach
    size = matrix_size
    while size > 0:
        # select the subset of the sample logrets matrix
        sample_logrets_mtx = max_sample_logrets_mtx.iloc[: , :size]
        # collect metrics for the submatrix
        size_results_mtx = collect_mean_metrics(tester_class,
                                                correction,
                                                [blending_param],
                                                thresholds, 
                                                significance_lvl,
                                                sample_logrets_mtx,
                                                iterations)
        results.append(size_results_mtx)

        # decrease size for the next matrix
        size -= stepdown
    
    return pd.concat(results)