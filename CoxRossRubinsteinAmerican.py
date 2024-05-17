import numpy as np
import scipy.stats as stats
import Payoffs as po
import CarrMadan as cm
from scipy.stats import norm
from TermStructure import TermStructure
import Payoffs

class CoxRossRubinsteinAmerican:
    def __init__(self, num_steps, down_step, up_step, r):
        self.num_steps = num_steps
        self.down_step = down_step
        self.up_step = up_step
        self.r = r

    def SpotValuesAtDate(self, u, d, i):
        return d**np.arange(i+1) * u**np.arange(i, -1, -1)

    def SpotValues(self, u, d):
        S = []
        for i in range(self.num_steps + 1):
            Si = self.SpotValuesAtDate(u, d, i)
            S.append(Si)
        
        return S

    # E[i] = value at i of excercising at i + 1 => length i+1
    def ExerciseValues(self, S, option):
        E = []
        for i in range(len(S)):
            Ei = option.unitary_payoff(S[i])         
            E.append(Ei)
        
        return E

    def SpotDistribution(p, i):
        return p**np.arange(i+1) * (1 - p)**np.arange(i, -1, -1)

    def PayOff(self, x, k):
        return np.maximum(k - x, 0)

    def PriceBermudan(self, option):
        k = option.strike / option.spot
        spot = option.spot
        u = self.up_step
        d = self.down_step
        r = self.r
        p = (u - 1 - r) / (u - d)
        N = self.num_steps
        S = self.SpotValues(u, d)
        E = self.ExerciseValues(S, option)

        A = []
        for i in range(N+1):
            A.append(np.zeros(i+1))

        A[-1] = E[-1]
        for i in np.arange(N - 1, -1, -1):
            Ci = (p * A[i+1][1:] + (1 - p) * A[i+1][:-1]) / (1 + r)
            Ai = np.maximum(E[i], Ci)
            A[i] = Ai

        return spot * A[0][0]
    
# sigma = 0.4
# r = 0.4
# num_steps = 230
# strike, maturity, spot = 1,1,1
# down_step = np.exp(- sigma*np.sqrt(num_steps / maturity)) * (1 + r)
# up_step = np.exp(sigma*np.sqrt(num_steps / maturity)) * (1 + r)

# cox = CoxRossRubinsteinAmerican(num_steps, down_step, up_step, r)
# option = Payoffs.VanillaPut(strike, maturity, spot)
# print(cox.PriceBermudan(option))
