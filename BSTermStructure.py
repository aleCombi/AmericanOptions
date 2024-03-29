import numpy as np
import scipy.stats as stats
import Payoffs as po
import CarrMadan as cm
from scipy.stats import norm
from TermStructure import TermStructure

norm = stats.norm

def MontecarloCall(sampleSize, r, sigma, T, K, S):
    valuesAtMaturity = SimulateGBM(T, S, r, sigma, sampleSize)
    payoffs = np.maximum(0, valuesAtMaturity - K) 
    return np.exp(- r * T) * np.average(payoffs)

def SimulateGBM(time, price, rate, sigma, sampleSize):
    driftTerm = (rate - sigma**2 / 2) * time
    volTerm = sigma * np.sqrt(time) * np.random.normal(0,1,sampleSize)
    return price * np.exp(driftTerm + volTerm)
    
class BlackScholesTermStructure:
    def __init__(self, r, sigma: TermStructure):
        self.r = r
        self.sigma = sigma
    
    def Density(self, x, t):
        mean = t * (self.r - self.sigma.square_integral(0,t)/2)
        var = self.sigma.square_mean(0, t) * np.sqrt(t)
        return norm.pdf(x, mean, var)

    def CharacteristicFunction(self, u, t):
        var = self.sigma.square_integral(0,t)
        return np.exp(1j * (self.r - var / 2) * u * t - u**2 * t * var / 2)

    def Call(self, t, K, S):
        sigma = self.sigma.square_mean(0, t)
        d1 = ( np.log(S/K) + (self.r + sigma ** 2 / 2) * (t) ) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)  
        return norm.cdf(d1) * S - norm.cdf(d2) * K * np.exp(-self.r*t)

    def CallDelta(self, t, K, S):
        sigma = self.sigma.square_mean(0, t)
        d1 = ( np.log(S/K) + (self.r + sigma ** 2 / 2) * t ) / (sigma * np.sqrt(t))
        return norm.cdf(d1)

    def Put(self, t, K, S):
        sigma = self.sigma.square_mean(0, t)
        d1 = ( np.log(S/K) + (self.r + sigma ** 2 / 2) * t ) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)  
        return  - norm.cdf(- d1) * S +  norm.cdf(- d2) * K * np.exp(- self.r * t)

    def PutDelta(self, t, K, S):
        sigma = self.sigma.square_mean(0, t)
        d1 = ( np.log(S/K) + (self.r + sigma ** 2 / 2) * t ) / (sigma * np.sqrt(t))
        return - norm.cdf(- d1)

    def SimulatePaths(self, T, S, sample_size, timeSteps, antithetic=False):
        if antithetic:
            sample_size = sample_size // 2
            
        logPathGrid = np.zeros([timeSteps, sample_size])
        dt = T / timeSteps

        times = np.arange(0, timeSteps) * dt
        sigma = self.sigma.value(times)
        driftTerm = (self.r - sigma**2 / 2) * dt
        volTerm = sigma.reshape(timeSteps, 1) * np.sqrt(dt) * np.random.normal(0,1, [timeSteps, sample_size])
        
        for time in range(timeSteps - 1):           
            logPathGrid[time + 1, :] = logPathGrid[time, :] + driftTerm[time] + volTerm[time, :]

        if (antithetic):
            antithetic_logPathGrid = np.zeros([timeSteps, sample_size])
            for time in range(timeSteps - 1):
                antithetic_logPathGrid[time + 1, :] = antithetic_logPathGrid[time, :] + driftTerm[time] - volTerm[time, :]
            logPathGrid = np.block([logPathGrid,antithetic_logPathGrid])

        return S * np.exp(logPathGrid)

rate = 0.05
termStructure = TermStructure([0, 1, 2], [0.5, 0.5, 0.7])
x = BlackScholesTermStructure(rate, termStructure)
x.SimulatePaths(2, 1, 100, 4)