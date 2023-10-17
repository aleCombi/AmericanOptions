import numpy as np
from scipy.interpolate import interp1d, CubicSpline
import matplotlib.pyplot as plt

def getInterpolator(kind):
    if kind == "linear":
        return lambda x,y: interp1d(x,y, kind="linear")
    elif kind == "cubic":
        return lambda x,y: CubicSpline(x,y)
    
# First dimension = strike
# Second dimension = time
class VolSurface:
    def __init__(self, times, strikes, values, extrapStrike="flat", extrapTime="flat", interpTime="linear", interpStrike="cubic"):
        if (len(times) != values.shape[1] and len(strikes) != values.shape[0]):
            raise ValueError('A vol surface must have the same number of times and values.')
        self.times = times
        self.strikes = strikes
        self.values = values
        self.first = "strike"
        self.timeInterpolator = lambda values: getInterpolator(interpTime)(np.array(self.times), values)
        self.strikeInterpolator = lambda values: getInterpolator(interpStrike)(np.array(self.strikes), values)
        self.extrapStrike = extrapStrike
        self.extrapTime = extrapTime

    def value(self, time):
        return self.interpolator(time)
    
    def extrap(self, x_0, x, y, kind):
        extrapolator = Extrapolation(kind, x, y)
        return extrapolator.value(x_0)
    
    def interpExtrapTime(self, values, time):
        if time >= self.times[0] and time <= self.times[-1]:
            return self.timeInterpolator(values)(time)
        else:
            return self.extrap(time, self.times, values, self.extrapTime)

    def interpExtrapStrike(self, values, strike):
        if strike >= self.strikes[0] and strike <= self.strikes[-1]:
            return self.strikeInterpolator(values)(strike)
        else:
            return self.extrap(strike, self.strikes, values, self.extrapStrike)

    def interpolator(self, strike, time):
        valuesAtStrike = []
        for i in range(len(self.times)):
            values = self.values[:, i]
            valuesAtStrike.append(self.interpExtrapStrike(values, strike))

        return self.interpExtrapTime(valuesAtStrike, time)

class Extrapolation:
    def __init__(self, kind, x, y):
        if kind != "linear" and kind != "flat":
            raise NotImplementedError(f'Extrapolation kind {kind} is not supported')
        self.kind = kind
        self.x = x
        self.y = y

    def value(self, x_0):
        if x_0 >= self.x[0] and x_0 <= self.x[-1]:
            raise ValueError(f'Cannot extrapolate a value in the interpolation interval')
        
        elif self.kind == "flat" and x_0 > self.x[-1]:
            return self.y[-1]
        elif self.kind == "flat" and x_0 < self.x[0]:
            return self.y[0]
        elif self.kind == "linear" and x_0 < self.x[0]:
            return self.y[0] + (x_0 - self.x[0]) * (self.y[1] - self.y[0]) / (self.x[1] - self.x[0])
        elif self.kind == "linear" and x_0 > self.x[-1]:
            return self.y[-1] + (x_0 - self.x[-1]) * (self.y[-1] - self.y[-2]) / (self.x[-1] - self.x[-2])
        else:
            raise NotImplementedError(f'Extrapolation kind {self.kind} is not implemented')



strikes = [2,3,4,5,6,7]
times = np.arange(2, 12, 2)
vols = np.random.rand(len(strikes), len(times))

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a meshgrid from strikes and times
X, Y = np.meshgrid(times, strikes)

# # Plot the 3D surface
surf = ax.plot_surface(X, Y, vols, cmap='viridis', alpha=0.7)

# Highlight specific nodes
ax.scatter(X, Y, vols, color='red', s=50, label='Interpolation nodes')

# Customize the plot (add labels, titles, etc.)
ax.set_xlabel('Time')
ax.set_ylabel('Strike')
ax.set_zlabel('Volatility')
ax.set_title('3D Surface Plot')

# # Add a legend for the highlighted nodes
ax.legend()

surface = VolSurface(times, strikes, vols)
all_strikes = np.arange(1.7, 8, 0.1)
all_times = np.arange(1, 12, 0.1)
all_vols = np.zeros((len(all_strikes), len(all_times)))

for time_ind in range(len(all_times)):
    for strike_ind in range(len(all_strikes)):
        all_vols[strike_ind, time_ind] = surface.interpolator(all_strikes[strike_ind], all_times[time_ind])

# Create a meshgrid from strikes and times
X, Y = np.meshgrid(all_times, all_strikes)

# Plot the 3D surface
surf = ax.plot_surface(X, Y, all_vols, cmap='viridis', alpha=0.7)

plt.show()