import numpy as np

class TermStructure:
    def __init__(self, times, values):
        if (len(times) != len(values)):
            raise ValueError('A term structure must have the same number of times and values.')
        self.times = times
        self.values = values

    def value(self, time):
        return np.interp(time, self.times, self.values)