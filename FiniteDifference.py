import numpy as np

class FiniteDifferenceBS():
    def __init__(self, boundary, time_grid_size, price_grid_size):
        self.boundary = boundary
        self.time_grid_size = time_grid_size
        self.price_grid_size = price_grid_size
        self.h = 2 * boundary / price_grid_size

    def fk_coefficients(self, volatility, rate):
            a = volatility * volatility / 2
            b = rate - volatility * volatility / 2
            c = - rate
            return a, b, c

    def discretized_operator(self, a, b, c, h, size, initial_condition="Dirichlet"):
        '''
        sigma -> volatility
        r -> rate
        h -> price discretization step
        size -> discretization matrix size
        '''
        alpha = a / (h * h) - b / (2*h)
        beta = - 2 * a / (h * h) + c
        gamma = a / (h * h) + b / (2*h)
        sub_diagonal = np.diag(np.repeat(alpha, size - 1), -1) 
        diagonal = np.diag(np.repeat(beta, size), 0)
        super_diagonal = np.diag(np.repeat(gamma, size - 1), 1)
        if initial_condition == "Neumann":
            diagonal[0,0] += alpha
            diagonal[-1,-1] += gamma

        return sub_diagonal + diagonal + super_diagonal

    def solve(self, time_step, option, operator):
        '''
        time_grid_size -> time discretization grid size
        time_step -> time discretization grid step
        final_condition -> vector of option premium
        operator -> differential operator discretized matrix
        '''
        final_condition = option.payoff(option.spot * np.exp(self.log_price_discretized()))
        mat = np.identity(operator.shape[0]) + time_step * operator
        u = np.linalg.matrix_power(mat, self.time_grid_size).dot(final_condition)

        return u

    def solve_with_final_condition(self, time_step, option, operator, final_condition, exercise_dates):
        '''
        time_grid_size -> time discretization grid size
        time_step -> time discretization grid step
        final_condition -> vector of option premium
        operator -> differential operator discretized matrix
        '''
        mat = np.identity(operator.shape[0]) + time_step * operator
        u = np.linalg.matrix_power(mat, self.time_grid_size // exercise_dates).dot(final_condition)

        return u 

    def price(self, option, volatility, rate):
        k = option.maturity / self.time_grid_size
        a, b, c = self.fk_coefficients(volatility, rate)
        A = self.discretized_operator(a, b, c, self.h, self.price_grid_size + 1)
        price = self.solve(k, option, A)
        return price[self.price_grid_size // 2]

    def log_price_discretized(self):
        ran = np.arange(self.price_grid_size + 1)
        return - self.boundary + 2*ran*self.boundary / self.price_grid_size

    def price_bermudan(self, exercise_dates, volatility, rate, option):
        k = option.maturity / self.time_grid_size
        a, b, c = self.fk_coefficients(volatility, rate)
        A = self.discretized_operator(a, b, c, self.h, self.price_grid_size + 1)
        final_condition = option.payoff(option.spot * np.exp(self.log_price_discretized()))

        for time in range(exercise_dates):
            continuation = self.solve_with_final_condition(k, option, A, final_condition, exercise_dates)
            exercise = option.payoff(option.spot * np.exp(self.log_price_discretized()))
            u = np.maximum(continuation, exercise)
            final_condition = u

        return u[self.price_grid_size // 2]