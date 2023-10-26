import numpy as np

# class for american put option
class VanillaPut:
    def __init__(self, strike, maturity, spot):
        self.strike = strike
        self.maturity = maturity
        self.spot = spot

    def payoff(self, price):
        return np.maximum(self.strike - price, 0)

    def unitary_payoff(self, price):
        return np.maximum(self.strike / self.spot - price, 0)

# class for american put option
class VanillaCall:
    def __init__(self, strike, maturity, spot):
        self.strike = strike
        self.maturity = maturity
        self.spot = spot

    def payoff(self, price):
        return np.maximum(price - self.strike, 0)
    
    def unitary_payoff(self, price):
        return np.maximum(price - self.strike / self.spot, 0)