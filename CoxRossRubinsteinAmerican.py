import numpy as np
import scipy.stats as stats
import Payoffs as po
import CarrMadan as cm
from scipy.stats import norm
from TermStructure import TermStructure

class CoxRossRubinsteinAmerican:
    def __init__(self, r, sigma):
        self.r = r
        self.sigma = sigma