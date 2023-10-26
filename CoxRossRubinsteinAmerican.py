import numpy as np
import scipy.stats as stats
import Payoffs as po
import CarrMadan as cm
from scipy.stats import norm
from TermStructure import TermStructure

class CoxRossRubinsteinAmerican:
    def __init__(self, num_steps, down_step, up_step, r):
        self.num_steps = num_steps
        self.down_step = down_step
        self.up_step = up_step
        self.r = r

    def PriceBermudan(self, option, bs_model):
        steps = np.arange(self.num_steps)
        strike = option.strike
        spot = option.spot
        u = self.up_step
        d = self.down_step
        r = self.r
        p = (u - 1 - r) / (u - d)
        price_at_maturity = np.maximum(strike - spot * )
