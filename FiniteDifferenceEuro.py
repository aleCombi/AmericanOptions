import numpy as np

def discretized_operator(a, b, c, size, h, boundary_conditions="Dirichlet"):
    '''
    a -> coefficient of the second derivative in the PDE
    b -> coefficient of the first derivative in the PDE
    c -> coefficient of the zero-th derivative in the PDE
    size -> size of the space discretization grid
    h -> space discretization step
    '''
    alpha = a / (h * h) + b / (2 * h)
    beta = - 2 * a / (h * h) - c
    gamma = a / (h * h) + b / (2 * h)
    sub_diagonal = np.diag(np.repeat(alpha, size - 1), -1) 
    diagonal = np.diag(np.repeat(beta, size), 0)
    super_diagonal = np.diag(np.repeat(gamma, size - 1), 1) 
    return sub_diagonal + diagonal + super_diagonal
    
def solve(time_grid_size, time_step, final_condition, operator, theta=0):
    '''
    time_grid_size -> time discretization grid size
    time_step -> time discretization grid step
    '''
    u = final_condition
    for n in reversed(range(time_grid_size)):
        u = solve_time_step(u, operator, time_step, theta)

    return u

def solve_time_step(u, operator, time_step, theta=0):
    return (np.identity(operator.size) + time_step * operator).dot(u)

def main():
    sigma = 0.1
    rate = 0.05
    boundary = 10
    size = 50
    h = boundary / (size + 1)
    time_grid_size = 100
    maturity = 10
    strike = 1
    a = sigma * sigma / 2
    b = rate - sigma * sigma / 2
    c = - rate
    x = - boundary + [2*n*h for n in range(size + 1)]
    put_payoff = np.maximum(strike - np.exp(x), 0)
    A = discretized_operator(a, b, c, size, h)
    price = solve(time_grid_size, maturity / time_grid_size, put_payoff)
