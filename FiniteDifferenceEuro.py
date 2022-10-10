from time import time
import numpy as np

def discretized_operator(sigma, r, h, size):
    '''
    sigma -> volatility
    r -> rate
    h -> price discretization step
    size -> discretization matrix size
    '''
    alpha = sigma * sigma / (2 * h * h) - (r - sigma*sigma / 2) / (2*h)
    beta = - sigma * sigma / (h * h) - r
    gamma = sigma * sigma / (2 * h * h) + (r - sigma*sigma / 2) / (2*h)
    sub_diagonal = np.diag(np.repeat(alpha, size - 1), -1) 
    diagonal = np.diag(np.repeat(beta, size), 0)
    super_diagonal = np.diag(np.repeat(gamma, size - 1), 1)
    # diagonal[0,0] += alpha
    # diagonal[-1,-1] += gamma
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

def main():
    sigma = 0.4
    rate = 0.05
    maturity = 1
    strike = 1
    boundary = 5

    size = 499
    h = 2 * boundary / (size + 1)
    time_grid_size = 100000
    k = maturity / time_grid_size
    print(k/(h*h))
    x = [- boundary + 2*n*boundary / (size + 1) for n in range(size + 2)]
    put_payoff = np.maximum(np.exp(x) - strike, 0)
    A = discretized_operator(sigma, rate, h, len(x))
    price = solve(time_grid_size, k, put_payoff, A)
    return x, price

x, price = main()
print(price[250])