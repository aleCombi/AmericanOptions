from typing import no_type_check
import numpy as np
from numpy.polynomial import polynomial

# OWCS

class StochasticProcess:
    pass

# generic class to generate paths of a stochastic process
class PathGenerator:
    pass

# class to generate paths of a GBM
class GBMPathGenerator(PathGenerator):
    pass

# class for american put option
class AmericanPut:
    def __init__(self, strike, maturity):
        self.strike = strike
        self.maturity = maturity

    def payoff(self, strike, price):
        return np.maximum(strike - price, 0)

# class for Longstaff - Schwartz MonteCarlo method
# for evaluation of American put options
class LongstaffSchwartz:
    def __init__(self, pathGenerator, americanPut, pathNumber, poly_fit_degree):
        self.pathGenerator = pathGenerator
        self.poly_fit_degree = poly_fit_degree

    def Poly_fit(self, x, y):
        coeff =  polynomial.Polynomial.fit(x, y, self.poly_fit_degree).convert().coef[::-1]
        return np.poly1d(coeff)

    def Price(self, spotGrid, year_rate, option):
        strike = option.strike
        times_length = spotGrid.shape[0]
        paths_length = spotGrid.shape[1]
        # rate per unit of time in the grid
        rate = option.maturity * year_rate / (times_length - 1)
        stopping_time = (times_length - 1) * np.ones(paths_length)
        stopping_payoff = lambda path: option.payoff(strike, spotGrid[stopping_time[path].astype(int), path])

        for time in range(times_length - 2, 0, -1):      
            # realized cash flows in not excercising the option   
            continuation_PV = np.exp(-rate*(stopping_time - time)) * stopping_payoff(np.arange(paths_length))

            # LS regression among ITM spots to get continuation conditional expectation 
            ITM_paths = np.where(option.payoff(strike, spotGrid[time, :]) > 0)
            ITM_spots = spotGrid[time, ITM_paths].flatten()
            ITM_continuation_payoff = continuation_PV[ITM_paths]
            fitting_poly = self.Poly_fit(ITM_spots, ITM_continuation_payoff)
            continuation_exp = np.zeros(paths_length)
            continuation_exp[ITM_paths] = fitting_poly(spotGrid[time,ITM_paths])

            # updating the stopping time choice
            exercise = np.where(option.payoff(strike, spotGrid[time, :]) > continuation_exp)[0]
            stopping_time[exercise] = time

        discountedValue = np.exp(-rate*stopping_time) * stopping_payoff(np.arange(paths_length))

        return np.mean(discountedValue)