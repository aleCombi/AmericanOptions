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
    def __init__(self, pathGenerator, americanPut, pathNumber):
        self.pathGenerator = pathGenerator
        self.strike = americanPut.strike
        self.maturity = americanPut.maturity
        self.option = americanPut
        self.pathNumber = pathNumber

    def GeneratePaths(self, maturity, pathNumber):
        return self.pathGenerator.GeneratePaths(maturity, pathNumber)

    def Evaluate(self, pathNumber, df):
        spotGrid = self.pathGenerator.GeneratePaths(self.maturity, pathNumber)
        return self.OptionPrice(self, spotGrid, df)

    def Price(self, spotGrid, rate, strike):
        times_length = spotGrid.shape[0]
        paths_length = spotGrid.shape[1]
        stopping_time = (times_length - 1) * np.ones(paths_length)
        stopping_payoff = lambda path: self.option.payoff(strike, spotGrid[stopping_time[path].astype(int), path])

        for time in range(times_length - 2, 0, -1):      
            # realized cash flows in not excercising the option   
            continuation_PV = np.exp(-rate*(stopping_time - time)) * stopping_payoff(np.arange(paths_length))

            # LS regression among ITM spots to get continuation conditional expectation 
            ITM_paths = np.where(self.option.payoff(strike, spotGrid[time, :]) > 0)
            ITM_spots = spotGrid[time, ITM_paths].flatten()
            ITM_continuation_payoff = continuation_PV[ITM_paths]
            coefficients = polynomial.Polynomial.fit(ITM_spots, ITM_continuation_payoff, 3).convert().coef
            fitting_poly = np.poly1d(coefficients[::-1])
            continuation_exp = np.zeros(paths_length)
            continuation_exp[ITM_paths] = fitting_poly(spotGrid[time,ITM_paths])

            # updating the stopping time choice
            exercise = np.where(self.option.payoff(strike, spotGrid[time, :]) > continuation_exp)[0]
            stopping_time[exercise] = time

        discountedValue = np.exp(-rate*stopping_time) * stopping_payoff(np.arange(paths_length))

        return np.mean(discountedValue)