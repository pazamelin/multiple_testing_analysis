import math
import numpy as np
import pandas as pd
import statistics

import scipy
import scipy.stats

import smsa


def _generate_results_mtx(matrix_size, columns, fill=0):
    # TODO: remove duplication, link with _generate_symmetric_mtx function from the CSTest class
    # allocate a dataframe for the results matrix
    data = None
    if fill == 0:
        data = np.zeros((matrix_size, matrix_size))
    else:
        data = np.ones((matrix_size, matrix_size))
    results_mtx = pd.DataFrame(data=data, 
                               columns=columns,
                               index=columns)
    # set the diagonal elements to NaN
    for i in range(0, matrix_size):
        results_mtx.iloc[i, i] = np.nan

    return results_mtx 


def _collect_sorted_pvalues(pvalues_mtx, reverse=False):
    matrix_size = pvalues_mtx.shape[1]

    # collect sorted pvalues and corresponding coordinates
    pvalues_sorted = []
    for row in range(0, matrix_size - 1):
        for col in range(row + 1, matrix_size):
            pvalue = pvalues_mtx.iloc[row, col]
            pvalue_with_coordinates = (pvalue, row, col)
            pvalues_sorted.append(pvalue_with_coordinates)

    # sort the collected pvalues array
    pvalues_sorted = sorted(pvalues_sorted, key=lambda tup: tup[0], reverse=reverse)
    return pvalues_sorted


def bonferroni(pvalues_mtx: pd.DataFrame, singificance_lvl: float):
    # generate empty results matrix
    matrix_size = pvalues_mtx.shape[1]
    results_mtx = _generate_results_mtx(matrix_size, pvalues_mtx.columns, fill=0)

    # find the number of individual hypotheses
    m = math.comb(matrix_size, 2)

    # for all individual hypotheses ...
    for row in range(0, matrix_size - 1):
        for col in range(row + 1, matrix_size):
            # get the corresponding pvalue
            pvalue = pvalues_mtx.iloc[row, col]

            # compare the pvalue with the adjusted significance level
            is_rejected = int(pvalue < (singificance_lvl / m))

            # add the rejection status to the symmetric results matrix
            results_mtx.iloc[row, col] = is_rejected
            results_mtx.iloc[col, row] = is_rejected

    return results_mtx


def holm(pvalues_mtx: pd.DataFrame, singificance_lvl: float):
    # generate empty results matrix
    matrix_size = pvalues_mtx.shape[1]
    results_mtx = _generate_results_mtx(matrix_size, pvalues_mtx.columns, fill=0)

    # get sorted pvalues tupled with corresponding hypothesis coordinates
    pvalues_sorted = _collect_sorted_pvalues(pvalues_mtx)
    m = len(pvalues_sorted)

    # perfome the sequential procedure
    for k in range(1, m + 1):
        # compare the pvalue with the adjusted significance level
        pvalue = pvalues_sorted[k - 1][0]
        is_rejected = int(pvalue < singificance_lvl / (m - k + 1))

        # add the rejection status to the symmetric results matrix
        row = pvalues_sorted[k - 1][1]
        col = pvalues_sorted[k - 1][2]
        results_mtx.iloc[row, col] = is_rejected
        results_mtx.iloc[col, row] = is_rejected

        # check the termination condition
        if not is_rejected:
            # accept all remaining hypotheses
            break
    
    return results_mtx


def hochberg(pvalues_mtx: pd.DataFrame, singificance_lvl: float):
    # generate empty results matrix
    matrix_size = pvalues_mtx.shape[1]
    results_mtx = _generate_results_mtx(matrix_size, pvalues_mtx.columns, fill=1)

    # get sorted pvalues tupled with corresponding hypothesis coordinates
    # sort in descending order
    pvalues_sorted = _collect_sorted_pvalues(pvalues_mtx, reverse=True)
    m = len(pvalues_sorted)

    # perfome the sequential procedure
    for k in range(1, m + 1):
        # compare the pvalue with the adjusted significance level
        pvalue = pvalues_sorted[k - 1][0]
        is_rejected = int(pvalue < singificance_lvl / k)

        # add the rejection status to the symmetric results matrix
        row = pvalues_sorted[k - 1][1]
        col = pvalues_sorted[k - 1][2]
        results_mtx.iloc[row, col] = is_rejected
        results_mtx.iloc[col, row] = is_rejected

        # check the termination condition
        if is_rejected:
            # reject all remaining hypotheses
            break

    return results_mtx


def benjamini(pvalues_mtx: pd.DataFrame, singificance_lvl: float):
    # generate empty results matrix
    matrix_size = pvalues_mtx.shape[1]
    results_mtx = _generate_results_mtx(matrix_size, pvalues_mtx.columns, fill=1)

    # get sorted pvalues tupled with corresponding hypothesis coordinates
    # sort in descending order
    pvalues_sorted = _collect_sorted_pvalues(pvalues_mtx, reverse=True)
    m = len(pvalues_sorted)

    # perfome the sequential procedure
    for k in range(1, m + 1):
        # compare the pvalue with the adjusted significance level
        pvalue = pvalues_sorted[k - 1][0]
        is_rejected = int(pvalue <= ((m - k + 1) * singificance_lvl) / m)

        # add the rejection status to the symmetric results matrix
        row = pvalues_sorted[k - 1][1]
        col = pvalues_sorted[k - 1][2]
        results_mtx.iloc[row, col] = is_rejected
        results_mtx.iloc[col, row] = is_rejected

        # check the termination condition
        if is_rejected:
            # reject all remaining hypotheses
            break

    return results_mtx

