import BlackScholes as bs
import Payoffs as po
from FiniteDifference import FiniteDifferenceBS
import numpy as np
import CoxRossRubinsteinAmerican 
import Payoffs
import LSM

t = 0.073972602739726029
r = 0.013047467783283154
sigma = 0.3294
black = bs.BlackScholes(r, sigma)
k = 5.6525
S = k * np.exp(-r * t)
put = black.Call(t, k, S)
print(put)

# boundary = 1
# price_grid_size = 20*boundary
# time_grid_size = 5000
strike = k
spot = S
maturity = t
exercise_dates = 5000
rate = r
# sigma = sigma
# fd = FiniteDifferenceBS(boundary, time_grid_size, price_grid_size)
option = po.VanillaPut(strike, maturity, spot)
# price = fd.price_bermudan(exercise_dates, sigma, rate, option)
# print(price)

# lsm = LSM.LongstaffSchwartz(4)
# time_steps = exercise_dates
# sample_size = 10000
# model = bs.BlackScholes(rate, sigma)
# paths = model.SimulatePaths(maturity, spot, sample_size, time_steps, True)
# price = lsm.Price(paths, rate, option)
# print(price)

num_steps = 1000
R = rate * maturity / num_steps
down_step = np.exp(- sigma*np.sqrt(maturity / num_steps)) * (1 + R)
up_step = np.exp(sigma*np.sqrt(maturity / num_steps)) * (1 + R)

cox = CoxRossRubinsteinAmerican.CoxRossRubinsteinAmerican(num_steps, down_step, up_step, R)
option = Payoffs.VanillaPut(strike, maturity, spot)
print(cox.PriceBermudan(option))