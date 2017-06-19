"""
This is the first lesson of the math of intelligence course of Siraj Raval
The code is given in his lectures, I would add my style of course ;)
https://github.com/llSourcell/Intro_to_the_Math_of_intelligence
"""
from numpy import array, genfromtxt

def compute_error(b, m, points):
    """it computes the total error from given points

    y = mx + b
    given this line equation coefficients, it calculates
    the error of the given points with this line.
    """
    total_error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))

def step_gradient(current_b, current_m, points, alpha):
    """It applies gradient descent for finding new coefficients of the line

    It calculates the partial derivative (`m` and `b`) to adjust the new
    values of them.

    Args:
        current_b: coefficient `b`.
        current_m: coefficient `m`.
        points: A list with the points to match.
        alpha: The learning rate.

    Returns:
        A list with `b` and `m`.
    """
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2 / N) * (y - (current_m * x + current_b))
        m_gradient += -(2 / N) * x * (y - (current_m * x + current_b))

    new_b = current_b - alpha * b_gradient
    new_m = current_m - alpha * m_gradient
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, alpha, iterations):
    """It runs gradient descent for n `iterations`

    Args:
        points: A list with the points to match.
        starting_b: The starting coefficient `b`.
        starting_m: The starting coefficient `m`.
        alpha: The learning rate.
        iterations: The number of iterations to run.

    Returns:
        A list with `b` and `m`.
    """
    b = starting_b
    m = starting_m
    for i in range(iterations):
        b, m = step_gradient(b,m, array(points), alpha)

    return [b, m]

def run():
    """ It runs the linear regression algorithm"""
    points = genfromtxt('intro-data.csv', delimiter=',')
    alpha = 0.0001
    initial_b = 0
    initial_m = 0
    iterations = 1000
    error = compute_error(initial_b, initial_m, points)
    print(f'Starting gradient descent at b = {initial_b}, m = {initial_m}, error = {error}')
    print('running...')
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, alpha, iterations)
    error = compute_error(b, m, points)
    print(f'After {iterations} iterations, y = {m}x + {b}, with error = {error}')

if __name__ == '__main__':
    run()
