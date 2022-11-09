from BlackScholes import BlackScholes
import Payoffs as po
import CarrMadan as cm
import numpy as np


def test_carr_madan():
    r = 0.06
    maturity = 1
    boundary = 32
    step = 0.01
    sigma = 0.5
    alpha = 1
    strike = 0.6
    spot = 1
    option = po.VanillaCall(strike, maturity, spot)
    model = BlackScholes(r, sigma)
    pricer = cm.CarrMadan(boundary, alpha, step)
    priceCM = pricer.CallPrice(option, model)
    price = model.Call(maturity, strike, spot)
    assert np.abs((price - priceCM) / price) < 1E-5

test_carr_madan()