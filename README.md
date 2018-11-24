# Univariate Linear Regression Using Gradient Descent

### Introduction

>"A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."
-Tom Mitchell

This is the first in (hopefully) a series of projects where I implement various machine learning algorithms in Python. I am currently working through the excellent [Andrew Ng course on Coursera](https://www.coursera.org/learn/machine-learning) and using this as a way to consolidate my thoughts on each topic and to gain a deeper understanding of the topics I am learning.

### Algorithm 1: Simple Linear Regression w. Gradient Descent

This is the first algorithm featured in the course. I will try to break down the algorithm as follows:

#### Overall Aim of Linear Regression

Say the price of a house is related to the size of the house. Small houses are cheap, medium size houses are a bit more expensive and large houses are the most expensive. We can say there is a *linear* relationship between the size of a house and the price of a house.

![alt text](https://www.dropbox.com/s/n9f5yr56jw96363/price%20of%20house%20example.png?raw=1 "Price of houses example")

Say we are an estate agent and we want to know the price of a house and we only know the square footage of the house. This would be fine if we have an exact data point but what if we don't have a data point to work with? Wouldn't it be great if we could *guess* the value given any square foot value? Enter linear regression.

#### How to guess a value (aka The Hypothesis Function)

So how can we guess a value of our house (y) if we only know the size of it (x)? Luckily there is a hypothesis function that allows us to do that! Enter algebra:

`y = mx + b`

or in fancy machine learning algebra:

![alt text](https://www.dropbox.com/s/o8pdg32oybtc84x/hypothesis%20function.png?raw=1 "hypothesis function")

If you don't know anything about algebra, this is a formula to draw a straight line on a graph. To break it down: m is the slope or gradient of the line, b is the y intercept when x = 0 and x is the value we know (size of the house). So given any x value, if we can find the ideal values of m and b, we can find y (price of the house). Hooray!

In the fancy machine learning version, we don't see m or b. Instead, we see what's called theta 0 and theta 1. Theta is used in machine learning to refer to the parameters or weights assigned to a particular input feature. In this case, since we only have 1 feature, we only have 2 weights to calculate. In multivariate linear regression (another day), there would be more theta values to calculate. 

#### How do we know if our guess is good? (aka The Cost Function)

In theory, we could guess and input any m or b value into our hypothesis function and get an answer y but we need a way to see how good or bad our guess is. We can find this out using a cost function. In linear regression, we use the "Mean squared error" function.

![alt text](https://www.dropbox.com/s/flraujue6t3qe6r/mean%20squared%20error%20function.png?raw=1 "mean squared error function")

To break this down into (reasonably) simple English, using our training data, we sum up every guess using x and our weights (y = mx + b), we compare it to the actual value (y) squared, and then divide the result by 2 times the number of training examples (1 / 2 * m). Phew.

The whole point of linear regression is to get the value of this error function as low as possible. We do that by modifying our weights until we get the lowest number possible.

#### How can we modify the weights? (aka Gradient Descent)

So we know now that we want to get the value of our cost function as low as possible and we do that by modifying our theta values (weights) so that this number goes down. But how do we do this without guessing? We use Gradient Descent.

Gradient descent is one of the most used alogorithm in machine learning and it is an important concept to understand.

![alt text](https://www.dropbox.com/s/bf1vfapy6ra4hgv/gradient%20descent%20graph.png?raw=1 "gradient descent graph")

What the graph is showing is the error values given different values of one of our parameters (say theta 0). The job of gradient descent is to work out the direction and amount we need to change the parameter in order to get it to the minimum error value (global cost minimum). To do this, gradient descent uses this algorithm:

![alt text](https://www.dropbox.com/s/0elysnmrut0ldc6/gradient%20descent.png?raw=1 "gradient descent algorithm")

What this is saying is given an original theta value, we calculate a new one by taking the old value and minus a value calculated using a learning rate * gradient. We do this for each theta value *simultaneously*. Now this part I find very hard to understand (given my lack of calculus experience), but we calculate the gradient using this formula:

![alt text](https://www.dropbox.com/s/9vcwzdwfisqig09/linear_regression_gradient.png?raw=1 "linear regression gradient algorithm")

At first this all seems very complicated but is quite intuitive to imagine. If we are standing on top of a hill and we want to get to the lowest point we would take a step towards the lowest point. To do that we would have to work out the direction of travel and if we should take a big step or a smaller step. Gradient descent does this for us.

#### A brief aside on the learning rate

Gradient descent is done iteratively using small steps. We nudge the theta values in the right direction until we reach convergence (the global minimum). It is important we don't take too large steps otherwise we might overshoot the global minimum:

![alt text](https://www.dropbox.com/s/252wdvpf36e710e/learning%20rate%20examples.png?raw=1 "learning rate")

Conversely, we shouldn't set it too small otherwise it would take far too many iterations to reach our ideal values.

### Example of Linear Regression using Gradient Descent

To learn this algorithm, I decided to make my own implementation in Python and to use some data I found online.

#### Data

The data I decided to use was the Weather Conditions in World War 2 data set as found on [Kaggle.com](https://www.kaggle.com/smid80/weatherww2). The two columns I decided to use for my analysis was MinTemp and MaxTemp. The data is a simple 2 row csv file: column A has a list of minimum temperatures recorded and column B has the maximum temperatures on that day. The task of my linear regression model is to predict the maximum temperature given only the minimum temperature. 

For the sake of this exercise I decided to use only 500 rows from the original dataset.

#### Data Visualized

First I decided to see the data visualized and how my line would fit given theta 0 and theta 1 values of [1, 1].

![alt text](https://www.dropbox.com/s/m8a6z82xjrfx18o/starting%20values.png?raw=1 "data visualized")

As we can see, the line could fit much better. Let's run our gradient descent algorithm over the data set.

#### Gradient Descent Code

My code for linear regression is broken up into two parts.

```python
def gradient_descent_runner(points, theta, learning_rate, num_iterations):
    # runs the gradient descent algorithm on a data set to find optimal theta values
    iterations_count = 0
    
    for i in range(num_iterations):
        iterations_count += 1
        theta_0_old = theta[0]
        theta_1_old = theta[1]
        
        # update theta values
        theta[0], theta[1] = gradient_descent(theta[0], theta[1], array(points), learning_rate)

        # check to see if we have converged (rate of change is less than learning rate)
        if abs(theta[0] - theta_0_old) < learning_rate and abs(theta[1] - theta_1_old) < learning_rate:
            break

    print('Converged! Completed {0} iterations'.format(iterations_count))
    return [theta[0], theta[1]]
```
The above function is a simple runner function that given a set of points, theta values, a learning rate, and number of iterations will run into theta 0 and theta 1 converge (the change in values is less than our learning rate).

```python
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
```

The gradient descent function will give updated theta values given a set of points and a learning rate.

```python
def computer_error_for_line_given_points(theta, points):
    # given theta values and points, returns the mean squared error of the sum of hypothesis - actual y values
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0] 
        y = points[i, 1]
        # y = mx + b
        totalError += (y - (theta[0] * x + theta[1])) ** 2
    return totalError / float(len(points))
```

The cost function is used to compare our initial theta values compared to the optimal ones found by our algorithm.

#### Observations

The initial hyperparameters used were: 

```python
learning_rate = 0.0001
num_iterations = 1000
```

Running gradient descent on our dataset resulted in the following number of iterations:

![alt text](https://www.dropbox.com/s/sry1runh91crqwo/change%20in%20theta%20values.png?raw=1 "Learning rate")

As we can see from the graph, gradient descent took larger steps initially and then made smaller ones until it converged at around 60 iterations.

Calculating the inital error function we get the following values:

`The starting values are: theta0 is 1.0, theta1 is 1.0, the error is 48.8208268415`

Running our algorithm we get the following values:

`The end values are: theta0 is 1.30014281104, theta1 is 1.0140900948, the error is 5.31455333109`

The end result show on the graph:

![alt text](https://www.dropbox.com/s/vbtzh9g92mv5bw7/optimal%20theta%20values.png?raw=1 "end result")

### Usage

`python ww2linearregression.py`

### Links

The following links have been invaluable:

[How to Do Linear Regression using Gradient Descent](https://www.youtube.com/watch?v=XdM6ER7zTLk)
[An Introduction to Gradient Descent and Linear Regression](https://spin.atomicobject.com/2014/06/24/gradient-descent-linear-regression/)
[Coursera Machine Learning](https://www.coursera.org/learn/machine-learning/)
