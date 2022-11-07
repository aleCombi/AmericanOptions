from FiniteDifferenceEuro import FiniteDifferenceBS
from FiniteDifferenceEuro import VanillaOption
import BlackScholes as bs
import numpy as np

price_tolerance = 0.01

def test_FiniteDifferenceBS_instance(sigma, rate, strike):
    maturity = 1
    boundary = 2
    price_grid_size = 1000
    time_grid_size = 100000
    price_bs = bs.Call(rate, sigma, maturity, strike, 1)
    fd = FiniteDifferenceBS(boundary, time_grid_size, price_grid_size)
    option = VanillaOption(strike, maturity)
    price = fd.price(option, sigma, rate)
    print(np.abs(price - price_bs) / price_bs < price_tolerance)

def test_FiniteDifferenceBS():
    sigma = [0.04*i for i in range(1, 10)]
    rate = 0.4
    strikes = [0.5 + i * 0.1 for i in range(1, 10)]

    for vol in sigma:
        for strike in strikes:
            test_FiniteDifferenceBS_instance(vol, rate, strike)

test_FiniteDifferenceBS()