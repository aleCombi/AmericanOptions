import numpy as np
from scipy.stats import norm
from numpy.polynomial import polynomial

def Call(r, sigma, t, K, S):
    d1 = ( np.log(S/K) + (r + sigma ** 2 / 2) * (t) ) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)  
    return norm.cdf(d1) * S - norm.cdf(d2) * K * np.exp(-r*t)

def CallDelta(r, sigma, t, K, S):
    d1 = ( np.log(S/K) + (r + sigma ** 2 / 2) * t ) / (sigma * np.sqrt(t))
    return norm.cdf(d1)

def Put(r, sigma, T, K, S, t = 0):
    d1 = ( np.log(S/K) + (r + sigma ** 2 / 2) * t ) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(T - t)  
    return  - norm.cdf(- d1) * S +  norm.cdf(- d2) * K * np.exp(- r * t)

def PutDelta(r, sigma, t, K, S):
    d1 = ( np.log(S/K) + (r + sigma ** 2 / 2) * t ) / (sigma * np.sqrt(t))
    return - norm.cdf(- d1)

def MontecarloCall(sampleSize, r, sigma, T, K, S):
    valuesAtMaturity = SimulateGBM(T, S, r, sigma, sampleSize)
    payoffs = np.maximum(0, valuesAtMaturity - K) 
    return np.exp(- r * T) * np.average(payoffs)

def SimulateGBMPaths(T, S, r, sigma, sampleSize, timeSteps):
    logPathGrid = np.zeros([timeSteps, sampleSize])
    driftTerm = (r - sigma**2 / 2) * T / timeSteps
    volTerm = sigma * np.sqrt(T / timeSteps) * np.random.normal(0,1, [timeSteps, sampleSize])

    for time in range(1, timeSteps):
        logPathGrid[time, :] = logPathGrid[time - 1, :] + driftTerm + volTerm[time, :]
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

def PutPayoff(strike, price):
    return np.maximum(strike - price, 0)

def LSM(spotGrid, df, strike):
    timesNum = spotGrid.shape[0]
    pathsNum = spotGrid.shape[1]
    flows = np.zeros(spotGrid.shape)
    flows[- 1, :] = np.maximum(strike - spotGrid[-1, :], 0)

    for time in range(timesNum - 2, 0, -1):      
        # realized cash flows in not excercising the option   
        noExPV = np.exp(-0.06) * flows[time + 1, :] #correct for timestep not equal to 1
        noExPayoff = np.zeros(pathsNum)
        # LS regression among ITM spots to get noExcercise expectation 
        ITMPaths = np.where(spotGrid[time, :] < strike)
        ITMSpots = spotGrid[time, ITMPaths].flatten()
        ITMNoExPV = noExPV[ITMPaths]
        coefficients = polynomial.Polynomial.fit(ITMSpots, ITMNoExPV, 2).convert().coef
        fittingPol = np.poly1d(coefficients)
        noExPayoff[ITMPaths] = fittingPol(spotGrid[time,ITMPaths])
        exercise = np.where(fittingPol(ITMSpots) < PutPayoff(strike, ITMSpots))

        for path in range(spotGrid.shape[1]):
            if max(strike - spotGrid[time, path], 0) > max(0, noExPayoff[path]): #never exercies OTM options!
                # option is excercised
                flows[time, path] = max(strike - spotGrid[time, path], 0)
                flows[time + 1 :, path] = 0
        
    # now we have the matrix of cashFlows flows[,] so we can compute the NPV
    discountedValue = np.zeros(flows.shape[1])
    for path in range(pathsNum):
        discountedValue[path] = np.sum([np.exp(-0.06*time)* flows[time, path] for time in range(flows.shape[0])])

    return np.mean(discountedValue)

def LSMPrice(r, sigma, t, K, S, sampleSize, timeSteps):
    spotGrid = SimulateGBMPaths(t, S, r, sigma, sampleSize, timeSteps)
    spotGrid[:, 0] = [1, 1.09, 1.08, 1.34]
    spotGrid[:, 1] = [1, 1.16, 1.26, 1.54]
    spotGrid[:, 2] = [1, 1.22, 1.07, 1.03]
    spotGrid[:, 3] = [1, 0.93, 0.97, 0.92]
    spotGrid[:, 4] = [1, 1.11, 1.56, 1.52]
    spotGrid[:, 5] = [1, 0.76, 0.77, 0.90]
    spotGrid[:, 6] = [1, 0.92, 0.84, 1.01]
    spotGrid[:, 7] = [1, 0.88, 1.22, 1.34]
    df = np.ndarray(spotGrid.shape)
    for timeIdx in range(df.shape[0]):
        df[timeIdx,:] = np.exp(- r)
    americanPrice = LSM(spotGrid, df, K)
    return americanPrice

if __name__=="__main__":
    print(LSMPrice(r=0.06, sigma=0.1, t=3, K=1.1, S=1, sampleSize=8, timeSteps=4))