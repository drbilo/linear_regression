import matplotlib.pyplot as plt
from numpy import *

def plot_line_on_graph(theta, points):
    # draws a line and plots scatter points on a graph given theta values and points
    x_plots = []
    y_plots = []
    for i in range(0, len(points)):
        x = points[i, 0]
        y = theta[0] * x + theta[1]
        x_plots.append(x)
        y_plots.append(y)
    plt.scatter(points[:,0], points[:, 1])
    plt.plot(x_plots, y_plots)
    plt.show()

def computer_error_for_line_given_points(theta, points):
    # given theta values and points, returns the mean squared error of the hypothesis - actual y values
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0] 
        y = points[i, 1]
        # y = mx + b
        totalError += (y - (theta[0] * x + theta[1])) ** 2
    return totalError / float(len(points))

def gradient_descent(t0_current, t1_current, points, learning_rate):
    # calculates the partial derivative of theta 0 and 1 and returns updated theta values given a learning rate
    t0_gradient = 0
    t1_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        t1_gradient += -(2/N) * (y - ((t0_current * x) + t1_current))
        t0_gradient += -(2/N) * x * (y - ((t0_current * x) + t1_current))
    new_t0 = t0_current - (learning_rate * t0_gradient)
    new_t1 = t1_current - (learning_rate * t1_gradient)
    return [new_t0, new_t1]


def gradient_descent_runner(points, theta, learning_rate, num_iterations):
    # runs the gradient descent algorithm on a data set to find optimal theta values
    iterations_count = 0
    for i in range(num_iterations):
        iterations_count += 1
        theta_0_old = theta[0]
        theta_1_old = theta[1]
        
        # update theta values
        theta[0], theta[1] = gradient_descent(theta[0], theta[1], array(points), learning_rate)
        plt.scatter(iterations_count, abs(theta[0] - theta_0_old) + abs(theta[1] - theta_1_old), c='r')

        # check to see if we have converged (rate of change is less than learning rate)
        if abs(theta[0] - theta_0_old) < learning_rate and abs(theta[1] - theta_1_old) < learning_rate:
            break
    print('Converged! Completed {0} iterations'.format(iterations_count))
    plt.show()
    return [theta[0], theta[1]]


def run():
    #import data from file
    points = genfromtxt("ww2data500.csv", delimiter=",", skip_header=True)

    #hyperparameters
    learning_rate = 0.0001
    num_iterations = 1000

    #starting theta values
    theta = ones(2)

    starting_error = computer_error_for_line_given_points(theta, points)
    print('The starting values are: theta0 is {0}, theta1 is {1}, the error is {2}'.format(theta[0], theta[1], starting_error))
    plot_line_on_graph(theta, points)

    # run gradient descent to find and return optimal theta values
    [theta[0], theta[1]] = gradient_descent_runner(points, theta, learning_rate, num_iterations)

    end_error = computer_error_for_line_given_points(theta, points)
    plot_line_on_graph(theta, points)
    print('The end values are: theta0 is {0}, theta1 is {1}, the error is {2}'.format(float(theta[0]), float(theta[1]), end_error))

if __name__ == "__main__":
    run()