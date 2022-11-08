import LSM
import numpy as np
import BlackScholes as bs
import Payoffs as po
import pandas as pd

lsm_from_longstaff_paper = 0.1144
result_path = r"TestResults/Longstaff_results.csv"

def test_lsm():
    rate = 0.06
    strike = 1.1
    maturity = 3
    spot = 1
    spotGrid = np.ndarray((4,8))
    spotGrid[:, 0] = [1, 1.09, 1.08, 1.34]
    spotGrid[:, 1] = [1, 1.16, 1.26, 1.54]
    spotGrid[:, 2] = [1, 1.22, 1.07, 1.03]
    spotGrid[:, 3] = [1, 0.93, 0.97, 0.92]
    spotGrid[:, 4] = [1, 1.11, 1.56, 1.52]
    spotGrid[:, 5] = [1, 0.76, 0.77, 0.90]
    spotGrid[:, 6] = [1, 0.92, 0.84, 1.01]
    spotGrid[:, 7] = [1, 0.88, 1.22, 1.34]
    option = po.VanillaPut(strike, maturity, spot)
    lsm = LSM.LongstaffSchwartz(None, option, 8, 2)
    american_price = lsm.Price(spotGrid, rate, option)
    assert np.abs(american_price - lsm_from_longstaff_paper) < .0001

def test_lsm_full():
    sample_size = 100000
    time_steps = 51
    rate = 0.06
    results = pd.read_csv(result_path)
    strike = 40

    for line in range(results.shape[0]):
        single_result = results.loc[line]
        maturity = single_result["maturity"]
        spot = single_result["spot"]
        sigma = single_result["sigma"]
        lsm_single(spot, sigma, strike, maturity, rate, sample_size, time_steps, single_result["ls american"])

def lsm_single(spot, sigma, strike, maturity, rate, sample_size, time_steps, result):
    option = po.VanillaPut(strike, maturity, spot)
    lsm = LSM.LongstaffSchwartz(None, option, 8, 4)
    sample_size = 100000
    time_steps = 51
    paths = bs.SimulateGBMPaths(maturity, spot, rate, sigma, sample_size, time_steps, True)
    price = lsm.Price(paths, rate, option)
    assert np.abs(price - result) / result < 0.04