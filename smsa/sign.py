import math
import numpy as np
import pandas as pd

import scipy
import scipy.stats

import smsa
import smsa.cstest


class CSTestSign(smsa.cstest.CSTest):
    @classmethod
    def get_name(self):
        return 'sign'


    def sign_correlation(self, _, **kwargs):
        row = self.logrets_mtx.columns[kwargs['row']]
        logrets_lhs = self.logrets_mtx[row]

        col = self.logrets_mtx.columns[kwargs['col']]
        logrets_rhs = self.logrets_mtx[col]

        result = 0
        for value_lhs, value_rhs in zip(logrets_lhs, logrets_rhs):
            result += int(value_lhs * value_rhs >= 0)
        # result /= self.sample_size 

        return result


    def cmpt_correlation_mtx(self, **kwargs):
        return self._generate_symmetric_mtx(
            dependency_mtx=self.logrets_mtx,
            generator=self.sign_correlation
        )


    def cmpt_test_statistic(self, correlation_coeff: float, threshold: float, **kwargs):
        return correlation_coeff


    def cmpt_pvalue(self, test_statistics, threshold: float, **kwargs):
        threshold_sign = 0.5 + (1 / math.pi) * np.arcsin(threshold)
        pvalue = scipy.stats.binom.cdf(test_statistics, self.sample_size, threshold_sign)
        pvalue = round(pvalue, 3)
        return pvalue

