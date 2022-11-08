from FiniteDifference import FiniteDifferenceBS
import BlackScholes as bs
import numpy as np
import Payoffs as po
import pandas as pd

price_tolerance = 0.01
result_path = r"TestResults/Longstaff_results.csv"

def finiteDifferenceBS_instance(sigma, rate, strike):
    maturity = 1
    boundary = 2
    spot = 1
    price_grid_size = 500
    time_grid_size = 5000
    price_bs = bs.Call(rate, sigma, maturity, strike, spot)
    fd = FiniteDifferenceBS(boundary, time_grid_size, price_grid_size)
    option = po.VanillaCall(strike, maturity, spot)
    price = fd.price(option, sigma, rate)
    assert np.abs(price - price_bs) / price_bs < price_tolerance

def test_FiniteDifferenceBS():
    sigma = [0.04*i for i in range(1, 10)]
    rate = 0.4
    strikes = [0.5 + i * 0.1 for i in range(1, 10)]

    for vol in sigma:
        for strike in strikes:
            finiteDifferenceBS_instance(vol, rate, strike)

def test_fd_full():
    price_grid_size = 200
    time_grid_size = 5000
    exercise_dates = 50
    boundary = 1
    rate = 0.06
    results = pd.read_csv(result_path)
    strike = 40

    for line in range(results.shape[0]):
        single_result = results.loc[line]
        maturity = single_result["maturity"]
        spot = single_result["spot"]
        sigma = single_result["sigma"]
        fd_single(spot, sigma, strike, maturity, rate, price_grid_size, time_grid_size, single_result["fd american"], boundary, exercise_dates)

def fd_single(spot, sigma, strike, maturity, rate, price_grid_size, time_grid_size, result, boundary, exercise_dates):
    fd = FiniteDifferenceBS(boundary, time_grid_size, price_grid_size)
    option = po.VanillaPut(strike, maturity, spot)
    price = fd.price_bermudan(exercise_dates, sigma, rate, option)
    assert np.abs(price - result) / result < 0.01