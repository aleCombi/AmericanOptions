import numpy as np
import scipy.stats as stats
import Payoffs as po

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

def SimulateGBMPaths(T, S, r, sigma, sample_size, timeSteps, antithetic=False):
    if antithetic:
        sample_size = sample_size // 2
        
    logPathGrid = np.zeros([timeSteps, sample_size])
    dt = T / timeSteps
    driftTerm = (r - sigma**2 / 2) * dt
    volTerm = sigma * np.sqrt(dt) * np.random.normal(0,1, [timeSteps, sample_size])
    
    for time in range(timeSteps - 1):
        logPathGrid[time + 1, :] = logPathGrid[time, :] + driftTerm + volTerm[time, :]

    if (antithetic):
        antithetic_logPathGrid = np.zeros([timeSteps, sample_size])
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

class CarrMadan:
    def __init__(self, boundary, alpha, step):
        self.boundary = boundary
        self.alpha = alpha
        self.step = step
    
    def CallTransform(self, r, T, phi, v):
        numerator = np.exp(- r * T) * phi(v - (self.alpha + 1) * 1j)
        denominator = self.alpha ** 2 + self.alpha - v ** 2 + 1j * (2 * self.alpha + 1) * v
        return numerator / denominator
    
    def CallPrice(self, option, model):
        damp = np.exp(- self.alpha * np.log(option.strike)) / (2 * np.pi)
        v = np.linspace(-self.boundary, self.boundary, int(self.boundary / self.step))
        phi = lambda u: model.CharacteristicFunction(u, option.maturity)
        integrand = self.CallTransform(model.r, option.maturity, phi, v) * np.exp(- 1j * v * np.log(option.strike))
        integral = np.trapz(integrand, v)
        return np.real(damp * integral)
    
class BlackScholes:
    def __init__(self, r, sigma):
        self.r = r
        self.sigma = sigma
    
    def CharacteristicFunction(self, u, t):
        return np.exp(1j * (self.r - self.sigma**2 / 2) * u * t - u**2 * t * self.sigma ** 2 / 2)


r = 0.06
maturity = 1
boundary = 32
step = 0.01
sigma = 0.5
alpha = 1
strike = 0.6
spot = 1
option = po.VanillaCall(strike, maturity, spot)
model = BlackScholes(r, sigma)
pricer = CarrMadan(boundary, alpha, step)
priceCM = pricer.CallPrice(option, model)


logStrike = np.log(strike)
price = Call(r, sigma, maturity, strike, spot)
print(price)
print(priceCM)