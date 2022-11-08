import numpy as np
import scipy.stats as stats

norm = stats.norm
def Call(r, sigma, t, K, S):
    d1 = ( np.log(S/K) + (r + sigma ** 2 / 2) * (t) ) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)  
    return norm.cdf(d1) * S - norm.cdf(d2) * K * np.exp(-r*t)

def CallDelta(r, sigma, t, K, S):
    d1 = ( np.log(S/K) + (r + sigma ** 2 / 2) * t ) / (sigma * np.sqrt(t))
    return norm.cdf(d1)

def Put(r, sigma, t, K, S):
    d1 = ( np.log(S/K) + (r + sigma ** 2 / 2) * t ) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)  
    return  - norm.cdf(- d1) * S +  norm.cdf(- d2) * K * np.exp(- r * t)

def PutDelta(r, sigma, t, K, S):
    d1 = ( np.log(S/K) + (r + sigma ** 2 / 2) * t ) / (sigma * np.sqrt(t))
    return - norm.cdf(- d1)

def MontecarloCall(sampleSize, r, sigma, T, K, S):
    valuesAtMaturity = SimulateGBM(T, S, r, sigma, sampleSize)
    payoffs = np.maximum(0, valuesAtMaturity - K) 
    return np.exp(- r * T) * np.average(payoffs)

def SimulateGBMPaths(T, S, r, sigma, sampleSize, timeSteps, antithetic=False):
    logPathGrid = np.zeros([timeSteps, sampleSize])
    dt = T / timeSteps
    driftTerm = (r - sigma**2 / 2) * dt
    volTerm = sigma * np.sqrt(dt) * np.random.normal(0,1, [timeSteps, sampleSize])
    
    for time in range(timeSteps - 1):
        logPathGrid[time + 1, :] = logPathGrid[time, :] + driftTerm + volTerm[time, :]

    if (antithetic):
        antithetic_logPathGrid = np.zeros([timeSteps, sampleSize])
        for time in range(timeSteps - 1):
            antithetic_logPathGrid[time + 1, :] = antithetic_logPathGrid[time, :] + driftTerm - volTerm[time, :]
        logPathGrid = np.block([logPathGrid,antithetic_logPathGrid])

    return S * np.exp(logPathGrid)

def SimulateGBM(time, price, rate, sigma, sampleSize):
    driftTerm = (rate - sigma**2 / 2) * time
    volTerm = sigma * np.sqrt(time) * np.random.normal(0,1,sampleSize)
    return price * np.exp(driftTerm + volTerm)

def CompareMethods():
    sampleSize, timeSteps = 10000000, 10**3
    r, sigma = 0.1, 0.05
    S, K, T = 100, 100, 1
    mcCall = MontecarloCall(sampleSize, r, sigma, T, K, S)
    anCall = Call(r, sigma, T, K, S)
    return anCall, mcCall, (anCall - mcCall) / anCall