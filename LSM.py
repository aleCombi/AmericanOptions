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
        flows = np.zeros(spotGrid.shape)
        flows[- 1, :] = self.option.payoff(strike, spotGrid[-1, :])

        for time in range(times_length - 2, 0, -1):      
            # realized cash flows in not excercising the option   
            realized = np.zeros(flows.shape[1])
            for path in range(paths_length):
                realized[path] = np.sum([np.exp(-rate*(t - time))* flows[t, path] for t in range(time + 1, flows.shape[0])])

            noExPV = np.exp(-rate) * flows[time + 1, :] #correct for timestep not equal to 1
            noExPV = realized
            noExPayoff = np.zeros(paths_length)
            # LS regression among ITM spots to get noExercise expectation 
            ITMPaths = np.where(self.option.payoff(strike, spotGrid[time, :]) > 0)
            if (len(ITMPaths[0]) == 0):
                continue

            ITMSpots = spotGrid[time, ITMPaths].flatten()
            ITMNoExPV = noExPV[ITMPaths]
            coefficients = polynomial.Polynomial.fit(ITMSpots, ITMNoExPV, 3).convert().coef
            fittingPol = np.poly1d(coefficients[::-1])
            noExPayoff[ITMPaths] = fittingPol(spotGrid[time,ITMPaths])

            #paths for which exercising is profitable
            exercise = np.where(np.maximum(strike - spotGrid[time, :], 0) > np.maximum(0, noExPayoff[:]))[0]

            #update the flow matrix on the paths where it is profitable to exercise
            for path in exercise:
                flows[time, path] = max(strike - spotGrid[time, path], 0)
                flows[time + 1 :, path] = 0
            
        # now we have the matrix of cashFlows flows[,] so we can compute the NPV
        discountedValue = np.zeros(flows.shape[1])
        for path in range(paths_length):
            discountedValue[path] = np.sum([np.exp(-rate*t)* flows[t, path] for t in range(flows.shape[0])])

        return np.mean(discountedValue)