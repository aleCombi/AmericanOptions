import LSM
import numpy as np
import BlackScholes as bs

lsm_from_longstaff_paper = 0.1144

def test_lsm():
    rate = 0.06
    strike = 1.1
    maturity = 1
    spot = 1
    sigma = 0.1
    spotGrid = np.ndarray((4,8))
    spotGrid[:, 0] = [1, 1.09, 1.08, 1.34]
    spotGrid[:, 1] = [1, 1.16, 1.26, 1.54]
    spotGrid[:, 2] = [1, 1.22, 1.07, 1.03]
    spotGrid[:, 3] = [1, 0.93, 0.97, 0.92]
    spotGrid[:, 4] = [1, 1.11, 1.56, 1.52]
    spotGrid[:, 5] = [1, 0.76, 0.77, 0.90]
    spotGrid[:, 6] = [1, 0.92, 0.84, 1.01]
    spotGrid[:, 7] = [1, 0.88, 1.22, 1.34]
    option = LSM.AmericanPut(strike, maturity)
    lsm = LSM.LongstaffSchwartz(None, option, 8)
    american_price = lsm.Price(spotGrid, rate, strike)
    print(np.abs(american_price - lsm_from_longstaff_paper) < .0001)
    sample_size = 1000
    time_steps = 100
    paths = bs.SimulateGBMPaths(maturity, spot, rate, sigma, sample_size, time_steps)
    lsm.Price(paths, rate, strike)

def test_lsm_full():
    rate = 0.06
    strike = 40
    maturity = 1
    spot = 36
    sigma = 0.2
    option = LSM.AmericanPut(strike, maturity)
    lsm = LSM.LongstaffSchwartz(None, option, 8)
    bs_opt = bs.Put(rate, sigma, maturity, strike, spot)
    sample_size = 100000
    time_steps = 50
    paths = bs.SimulateGBMPaths(maturity, spot, rate, sigma, sample_size//2, time_steps, True)
    price = lsm.Price(paths, rate / (time_steps -1), strike)
    print(price)

test_lsm_full()
