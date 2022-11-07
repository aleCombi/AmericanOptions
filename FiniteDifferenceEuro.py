import numpy as np
import BlackScholes as bs

def discretized_operator(a, b, c, h, size, initial_condition="Dirichlet"):
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
    
def solve(time_grid_size, time_step, final_condition, operator):
    '''
    time_grid_size -> time discretization grid size
    time_step -> time discretization grid step
    final_condition -> vector of option premium
    operator -> differential operator discretized matrix
    '''
    mat = np.identity(operator.shape[0]) + time_step * operator
    u = np.linalg.matrix_power(mat, time_grid_size).dot(final_condition)

    return u

class VanillaOption():
    def __init__(self, strike, maturity):
        self.type = "Put"
        self.strike = strike
        self.maturity = maturity

    def payoff(self, log_price):
        return np.maximum(np.exp(log_price) - self.strike, 0)

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
        final_condition = option.payoff(self.log_price_discretized())
        mat = np.identity(operator.shape[0]) + time_step * operator
        u = np.linalg.matrix_power(mat, self.time_grid_size).dot(final_condition)

        return u

    def price(self, option, volatility, rate):
        k = option.maturity / self.time_grid_size
        a, b, c = self.fk_coefficients(volatility, rate)
        A = self.discretized_operator(a, b, c, self.h, self.price_grid_size + 1)
        price = self.solve(k, option, A)
        return price

    def log_price_discretized(self):
        return [- self.boundary + 2*n*self.boundary / self.price_grid_size for n in range(self.price_grid_size + 1)]

def main():
    sigma = 0.04
    rate = 0.4
    maturity = 1
    strike = 1.5
    boundary = 2

    size = 999
    h = 2 * boundary / (size + 1)
    time_grid_size = 100000
    k = maturity / time_grid_size
    # print(k/(h*h))
    x = [- boundary + 2*n*boundary / (size + 1) for n in range(size + 2)]
    put_payoff = np.maximum(np.exp(x) - strike, 0)
    a = sigma * sigma / 2
    b = rate - sigma*sigma / 2
    c = - rate
    A = discretized_operator(a, b, c, h, len(x))
    price = solve(time_grid_size, k, put_payoff, A)
    priceBs = bs.Call(rate, sigma, maturity, strike, 1)
    print(price[500])
    print(priceBs)

    fd = FiniteDifferenceBS(2, 100000, 1000)
    option = VanillaOption(1.5, 1)
    sigma = 0.04
    rate = 0.4
    price2 = fd.price(option, sigma, rate)
    print(price2[500])

    return x, price

x, price = main()

