import numpy as np
import BlackScholes as bs
import Payoffs as po
import pandas as pd
import QuadMethod as quad

result_path = r"TestResults/Longstaff_results.csv"

def test_quad_full():
    time_steps = 51
    rate = 0.06
    results = pd.read_csv(result_path)
    strike = 40
    boundary = 2
    step = 0.001
    time_steps = 51

    quad_method = quad.Quad(boundary, step, time_steps)

    for line in range(results.shape[0]):
        single_result = results.loc[line]
        maturity = single_result["maturity"]
        spot = single_result["spot"]
        sigma = single_result["sigma"]
        option = po.VanillaPut(strike, maturity, spot)
        bs_model = bs.BlackScholes(rate, sigma) 
        quad_single(option, bs_model, single_result["ls american"], quad_method)

def quad_single(option, bs_model, result, quad_method):
    price_quad = quad_method.price_bermudan(option, bs_model)
    assert np.abs(price_quad - result) / result < 0.04