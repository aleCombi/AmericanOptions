import BlackScholes as bs
import Payoffs as po
from FiniteDifference import FiniteDifferenceBS
import numpy as np
import CoxRossRubinsteinAmerican 
import Payoffs
import LSM

today = np.datetime64('2017-06-29')
maturity = np.datetime64('2017-07-26')
t = (maturity - today).astype(int) / 365
r = 0.0130474677832835
#r = 0
sigma = 2 / 100
k = 5.6525
S = 5.6524387226060906 #2 * np.exp(-r * t)
r = - np.log(S/k) / t

black = bs.BlackScholes(r, sigma)
put = black.Put(t, k, S)
print(put)

# boundary = 1
# price_grid_size = 20*boundary
# time_grid_size = 5000
strike = k
maturity = t
exercise_dates = 500
rate = r
# sigma = sigma
# fd = FiniteDifferenceBS(boundary, time_grid_size, price_grid_size)
# price = fd.price_bermudan(exercise_dates, sigma, rate, option)
# print(price)

# lsm = LSM.LongstaffSchwartz(4)
# time_steps = exercise_dates
# sample_size = 10000
# model = bs.BlackScholes(rate, sigma)
# paths = model.SimulatePaths(maturity, spot, sample_size, time_steps, True)
# price = lsm.Price(paths, rate, option)
# print(price)
# maybe 150?
# num_steps = 150
# R = rate * maturity / num_steps
# down_step = np.exp(- sigma*np.sqrt(maturity / num_steps))#* (1 + R)
# up_step = np.exp(sigma*np.sqrt(maturity / num_steps)) #* (1 + R)
# cox = CoxRossRubinsteinAmerican.CoxRossRubinsteinAmerican(num_steps, down_step, up_step, R)
# option = Payoffs.VanillaPut(strike, maturity, spot)
# print(cox.PriceBermudan(option))
option = po.VanillaPut(strike, maturity, S)

current_min = 1000
num_steps = 64 #int(t * 365 / 0.5) #128 * int(maturity * 365)
R = rate * maturity / num_steps
down_step = np.exp(- sigma*np.sqrt(maturity / num_steps))
up_step = np.exp(sigma*np.sqrt(maturity / num_steps))
cox = CoxRossRubinsteinAmerican.CoxRossRubinsteinAmerican(num_steps, down_step, up_step, R)
pr = cox.PriceBermudan(option)
print(pr)