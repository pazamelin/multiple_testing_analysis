import math
import numpy as np
import pandas as pd

import scipy
import scipy.stats

import smsa
import smsa.cstest


class CSTestKendall(smsa.cstest.CSTest):
    @classmethod
    def get_name(self):
        return 'kendall'


    def cmpt_correlation_mtx(self, **kwargs):
        return self.logrets_mtx.corr(method='kendall')    


    def cmpt_test_statistic(self, correlation_coeff: float, threshold: float, **kwargs):
        n = self.sample_size
        threshold_kd = (2 / math.pi) * np.arcsin(threshold)
        statistics = math.sqrt(9 * n * (n - 1)) / math.sqrt(2 * (2 * n + 5))
        statistics *= (correlation_coeff - threshold_kd)
        return statistics


    def cmpt_pvalue(self, test_statistics, threshold: float, **kwargs):
        pvalue = scipy.stats.norm.cdf(test_statistics)
        pvalue = round(pvalue, 3)
        return pvalue

