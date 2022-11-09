from FiniteDifference import FiniteDifferenceBS
import BlackScholes as bs
import numpy as np
import Payoffs as po
import pandas as pd

price_tolerance = 0.05
result_path = r"TestResults/Longstaff_results.csv"

def finiteDifferenceBS_instance(sigma, rate, strike):
    maturity = 1
    boundary = 2
    spot = 1
    price_grid_size = 500
    time_grid_size = 5000
    option = po.VanillaCall(strike, maturity, spot)
    model = bs.BlackScholes(rate, sigma)
    price_bs = model.Call(maturity, strike, spot)
    fd = FiniteDifferenceBS(boundary, time_grid_size, price_grid_size)
    price_fd = fd.price(option, sigma, rate)
    assert np.abs(price_fd - price_bs) / price_bs < price_tolerance, f"failed test with strike={strike}, sigma={sigma}"

def test_FiniteDifferenceBS():
    sigma = np.linspace(0.04, 0.5, 10)
    rate = 0.4
    strikes = np.linspace(0.6, 1.5, 10)

    for vol in sigma:
        for strike in strikes:
            finiteDifferenceBS_instance(vol, rate, strike)

def test_fd_bermudan():
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
        fd_single_bermudan(spot, sigma, strike, maturity, rate, price_grid_size, time_grid_size, single_result["fd american"], boundary, exercise_dates)

def fd_single_bermudan(spot, sigma, strike, maturity, rate, price_grid_size, time_grid_size, result, boundary, exercise_dates):
    fd = FiniteDifferenceBS(boundary, time_grid_size, price_grid_size)
    option = po.VanillaPut(strike, maturity, spot)
    price = fd.price_bermudan(exercise_dates, sigma, rate, option)
    assert np.abs(price - result) / result < 0.01

test_FiniteDifferenceBS()