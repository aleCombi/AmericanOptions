import LSM
import numpy as np
import BlackScholes as bs

lsm_from_longstaff_paper = 0.1144

def test_lsm():
    american_price = bs.LSMPrice(r=0.06, sigma=0.1, t=3, K=1.1, S=1, sampleSize=8, timeSteps=4)
    print(np.abs(american_price - lsm_from_longstaff_paper) < .0001)


test_lsm()