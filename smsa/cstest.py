import numpy as np
import pandas as pd
import abc

from typing import Callable, List


class CSTest(abc.ABC):
    def __init__(self, logrets_mtx: pd.DataFrame, sample_size: int):
        self.logrets_mtx = logrets_mtx
        self.sample_size = sample_size

    @classmethod
    def get_name(self):
        pass


    @abc.abstractmethod
    def cmpt_correlation_mtx(self, **kwargs):
        pass


    @abc.abstractmethod
    def cmpt_test_statistic(self, correlation_coeff: float, threshold: float, **kwargs):
        pass


    @abc.abstractmethod
    def cmpt_pvalue(self, test_statistics, threshold: float, **kwargs):
        pass
    

    def check_pvalue(self, pvalue: float, significance_lvl: float, **kwargs):
        is_rejected = int(pvalue < significance_lvl)         
        return is_rejected


    def _generate_symmetric_mtx(self,
                                dependency_mtx: pd.DataFrame,
                                generator: Callable,
                                **generator_args):
        """
            Template function to generate matrices filled with tests statistics, pvalues, tests outcomes
        """
        # allocate a dataframe for the matrix
        matrix_size = dependency_mtx.shape[1]
        generated_mtx = pd.DataFrame(data=np.zeros((matrix_size, matrix_size)), 
                                     columns=dependency_mtx.columns,
                                     index=dependency_mtx.columns)
        # set the diagonal elements to NaN
        for i in range(0, matrix_size):
            generated_mtx.iloc[i, i] = np.nan

        # matrix generation:
        for row in range(0, matrix_size - 1):
            for col in range(row + 1, matrix_size):
                # get the corresponding value from the dependency matrix
                dependency_value = dependency_mtx.iloc[row, col]
                # call the generator function
                generated_value =  generator(dependency_value, row=row, col=col, **generator_args)
                # fill the cell in the symmetric matrix
                generated_mtx.iloc[row, col] = generated_value
                generated_mtx.iloc[col, row] = generated_value

        return generated_mtx


    def significance_test(self,
                          significance_lvl: float,
                          threshold: float,
                          correction: Callable=None):
        """
            Templated MHT procedure
        """
        # compute correlation coefficients matrix (pearson, sign, kendall, etc.)
        correlation_mtx = self.cmpt_correlation_mtx()    

        # compute tests statistics matrix
        test_statistics_mtx = self._generate_symmetric_mtx(
            dependency_mtx=correlation_mtx,
            generator=self.cmpt_test_statistic,
            threshold=threshold
        )
        
        # compute pvalues matrix
        pvalues_mtx = self._generate_symmetric_mtx(
            dependency_mtx=test_statistics_mtx,
            generator=self.cmpt_pvalue,
            threshold=threshold
        )

        results = None
        if correction is not None:
            # apply the optional Multiple Testing Correction procedure to
            # the pvalues matrix to get the results
            results = correction(pvalues_mtx, significance_lvl)
        else:
            # collect results w/o MTC procedures
            results = self._generate_symmetric_mtx(
                pvalues_mtx,
                self.check_pvalue,
                significance_lvl=significance_lvl
            )

        return results


    def significance_tests(self, 
                           significance_lvls: List[float], 
                           thresholds: List[float],
                           correction: Callable=None):
        results = []

        for significance_lvl in significance_lvls:
            results_row = []
            
            for threshold in thresholds:
                results_row.append(self.significance_test(significance_lvl, threshold, correction))
            
            results.append(results_row)
        
        return results