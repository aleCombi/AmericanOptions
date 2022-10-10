from typing import no_type_check
import numpy as np

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

# class for Longstaff - Schwartz MonteCarlo method
# for evaluation of American put options
class LongstaffSchwartz:
    def __init__(self, pathGenerator, americanPut, pathNumber):
        self.pathGenerator = pathGenerator
        self.strike = americanPut.strike
        self.maturity = americanPut.maturity
        self.pathNumber = pathNumber

    def GeneratePaths(self, maturity, pathNumber):
        return self.pathGenerator.GeneratePaths(maturity, pathNumber)

    def Evaluate(self, pathNumber, df):
        spotGrid = self.pathGenerator.GeneratePaths(self.maturity, pathNumber)
        return self.OptionPrice(self, spotGrid, df)

    def OptionPrice(self, spotGrid, df):
        '''
        Parameters:
            spotGrid (double[,]): 
                A matrix where spotGrid[t, p] represent the simulated value
                at time t along path p
            strike (double): 
                Strike price of the option
            df (double[,]:
                Matrix of discount factors
        Return:
            option no arbitrage price
        '''
        timesNum = spotGrid.shape[0]
        pathsNum = spotGrid.shape[1]
        flows = np.ndarray(spotGrid.shape)
        exPayoff = np.ndarray(spotGrid.shape)
        noExPayoff = np.ndarray(spotGrid.shape)

        # build exercise payoff matrix
        for time in range(timesNum):
            for path in range(pathsNum):
                exPayoff[time,path] = max(0, self.strike - flows[time,path])
        
        for time in range(timesNum - 2, -1, -1):
            # expected cash flows in not excercising the option 
            pathNoExPV = sum([df(time, h) * flows[h, :]] for h in range(time+1, timesNum)) / pathsNum

            # LS regression among ITM spots to get noExcercise expectation 
            ITMPaths = np.Where(spotGrid[time, :] < self.strike)
            ITMSpots = spotGrid[time, ITMPaths]
            ITMNoExPV = pathNoExPV[ITMPaths]
            coefficients = np.polyfit(ITMSpots, ITMNoExPV, 3)  
            fittingPol = np.poly1d(coefficients)
            noExPayoff[time, :] = fittingPol(spotGrid[time,:])

            for path in range(spotGrid.shape[1]):
                if exPayoff[time, path] > noExPayoff[time, path]:                
                    # option is excercised
                    flows[time, :] = max(self.strike - flows[time, path], 0)
                    flows[time + 1:, path] = 0
                else:
                    # option is not excercised
                    flows[time, :] = 0   
            
        # now we have the matrix of cashFlows flows[,] so we can compute the NPV
        discountedValue = np.array(path.shape)
        for path in range(pathsNum):
            discountedValue[path] = sum([df(0, time) * flows[time, path]] for time in range(path.shape[0]))

        return sum(discountedValue[:] / path.shape[1])