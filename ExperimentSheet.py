import BlackScholes as bs
import Payoffs as po
from FiniteDifference import FiniteDifferenceBS

boundary = 1
price_grid_size = 20*boundary
time_grid_size = 5000
strike = 1
spot = 1
maturity = 1
exercise_dates = 500
rate = 0.06
sigma = 0.3
fd = FiniteDifferenceBS(boundary, time_grid_size, price_grid_size)
option = po.VanillaPut(strike, maturity, spot)
price = fd.price_bermudan(exercise_dates, sigma, rate, option)
print(price)

# lsm = LSM.LongstaffSchwartz(4)
# time_steps = exercise_dates
# sample_size = 10000
# paths = bs.SimulateGBMPaths(maturity, spot, rate, sigma, sample_size, time_steps, True)
# price = lsm.Price(paths, rate, option)
# print(price)