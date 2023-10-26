import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad

class TermStructure:
    def __init__(self, times, values):
        if (len(times) != len(values)):
            raise ValueError('A term structure must have the same number of times and values.')
        self.times = times
        self.values = values
        self.interpolator = interp1d(times, values, kind="linear")

    def value(self, time):
        return self.interpolator(time)

    def integral(self, start, end):
        return quad(self.interpolator, start, end)

    def square_integral(self, start, end):
        return quad(lambda x: self.interpolator(x)**2, start, end)

    def square_mean(self, start, end):
        return np.sqrt(self.square_integral(start, end) / (end - start))