import numpy as np

class Quad:
    def __init__(self, boundary, step, exercise_dates):
        self.boundary = boundary
        self.step = step
        self.exercise_dates = exercise_dates
    
    def price_bermudan(self, option, bs_model):
        exercise_dates = self.exercise_dates
        k = option.maturity / exercise_dates
        log_asset = np.arange(-self.boundary, self.boundary + self.step, self.step)
        log_asset_vector = np.vstack(log_asset).T
        log_asset_matrix = log_asset_vector-log_asset_vector.T
        V = np.vstack(option.payoff(option.spot * np.exp(log_asset_vector))).T
        F = bs_model.Density(log_asset_matrix, k)
        f1 = np.vstack(F[:,0])
        fN = np.vstack(F[:,-1])

        for time in range(exercise_dates):
            v1 = np.vstack(V[:,0])
            vN = np.vstack(V[:,-1])
            continuation = np.exp(- bs_model.r * k) * self.step * (F.dot(V) - 0.5 * (v1 * f1 + vN * fN))
            exercise = option.payoff(option.spot * np.exp(log_asset))
            u = np.maximum(continuation, np.vstack(exercise))
            V = np.vstack(u)

        return u[u.shape[0] // 2][0]