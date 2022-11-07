from FiniteDifferenceEuro import FiniteDifferenceBS
from FiniteDifferenceEuro import VanillaOption
import BlackScholes as bs
import numpy as np

price_tolerance = 0.01

def test_FiniteDifferenceBS():
    sigma = 0.04
    rate = 0.4
    maturity = 1
    strike = 1.5
    boundary = 2

    price_grid_size = 1000
    time_grid_size = 100000

    price_bs = bs.Call(rate, sigma, maturity, strike, 1)
    fd = FiniteDifferenceBS(boundary, time_grid_size, price_grid_size)
    option = VanillaOption(strike, maturity)
    price = fd.price(option, sigma, rate)
    print(np.abs(price - price_bs) / price_bs < price_tolerance)

test_FiniteDifferenceBS()