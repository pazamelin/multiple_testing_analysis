import math
import numpy as np
import pandas as pd

import scipy
import scipy.stats

import smsa
import smsa.cstest


class CSTestPearson(smsa.cstest.CSTest):
    @classmethod
    def get_name(self):
        return 'pearson'


    def cmpt_correlation_mtx(self, **kwargs):
        return self.logrets_mtx.corr(method='pearson')


    def cmpt_test_statistic(self, correlation_coeff: float, threshold: float, **kwargs):
        statistic = np.log(1 + correlation_coeff) - np.log(1 - correlation_coeff) \
                  - np.log(1 + threshold) + np.log(1 - threshold) 
        statistic *= 2 * math.sqrt(self.sample_size - 3)
        return statistic


    def cmpt_pvalue(self, test_statistics, threshold: float, **kwargs):
        pvalue = scipy.stats.norm.cdf(test_statistics)
        pvalue = round(pvalue, 3)
        return pvalue

