"""
################################################################################
CSCI 633: Biologically-Inspired Intelligent Systems
Version taught by Alexander G. Ororbia II

Implementations of various benchmark objective functions.

Notes: 
1. All implementations assume row-oriented vectors.
2. Imports must not import specific functions. This will mess up the visualization module.
################################################################################
"""

import numpy as np
import utils as u

MIN = "min"
MAX = "max"

@u.register_metadata(MIN, -10)
@u.register_metadata(MAX, 10)
def rosenbrock(x):
    """
    Evaluate Rosenbrock's function.

    The global minimum is located at x* = (1, ..., 1) with f(x*) = 0.
    
    # Arguments
    * `x` - The input vector. Must be an ndarray from numpy.

    # Returns
    The value of the rosenbrock function at the given vector.
    """

    assert isinstance(x, np.ndarray), "Input vector must be an ndarray!"

    d = x.shape[1]

    result = 0
    for i in range(0, d-1):
        term1 = (1 - x[0][i]) ** 2
        term2 = 100 * ((x[0][i+1] - x[0][i] ** 2) ** 2)
        result += term1 + term2

    return result

@u.register_metadata(MIN, -2)
@u.register_metadata(MAX, 2)
def rosenbrock_modified(x):
    """
    Evaluate Rosenbrock's function.

    The global minimum is located at x* = (1, ..., 1) with f(x*) = 74. However,
    a Gaussian bump at (-1, 1) causes a local minimum x* = (1, -1) with f(x*) = 0.

    The local minimum basin is larger than the global minimum basin.
    
    # Arguments
    * `x` - The input vector. Must be an ndarray from numpy.

    # Returns
    The value of the rosenbrock function at the given vector.
    """

    d = x.shape[1]

    c = 74
    sum1 = 0
    sum2 = 0
    sum3 = 0
    
    for i in range(d-1):
        xi = x[0][i]
        xi2 = x[0][i+1]

        sum1 += np.power(xi2 - np.power(xi, 2), 2)
        sum2 += np.power((1 - xi), 2)
        
        t31 = np.power(xi + 1, 2)
        t32 = np.power(xi2 + 1, 2)
        t33 = float(t31 + t32)
        sum3 += t33 / 0.1

    sum1 *= 100

    return c + sum1 + sum2 + (-400 * np.exp(-sum3))
    
@u.register_metadata(MIN, -10)
@u.register_metadata(MAX, 10)
def ackley(x):
    """
    Evaluate Ackley's function.

    This function has a global minimum at x* = (0, ..., 0) with f(x*) = 0.

    # Arguments
    * `x` - The input vector. Must be an ndarray from numpy.

    # Returns
    The value of the function.
    """

    assert isinstance(x, np.ndarray), "Input vector must be ndarray!"

    d = x.shape[1]

    sum_xi_squared = 0
    sum_cos_2pixi = 0

    for xi in x[0]:
        sum_xi_squared += xi ** 2
        sum_cos_2pixi += np.cos(2 * np.pi * xi)

    one_over_d = 1.0 / float(d)
    sqrt_arg = one_over_d * sum_xi_squared
    inner1 = -0.02 * np.sqrt(sqrt_arg)
    term1 = -20 * np.exp(inner1)
    term2 = np.exp(one_over_d * sum_cos_2pixi)
    return term1 - term2 + 20 + np.e

@u.register_metadata(MIN, -2*np.pi)
@u.register_metadata(MAX, 2*np.pi)
def eggcrate(x):
    """
    Evaluate the eggcrate function.

    This function has a global minimum with fmin = 0 at (0, 0)

    # Arguments
    * `x` - The input vector. Should be an ndarray from numpy.

    # Returns
    The value of the function.
    """
    x1 = x[0][0]
    x2 = x[0][1]

    x1squared = x1 ** 2
    x2squared = x2 ** 2

    sin_squared_x1 = np.sin(x1) ** 2
    sin_squared_x2 = np.sin(x2) ** 2

    return x1squared + x2squared + 25 * (sin_squared_x1 + sin_squared_x2)

@u.register_metadata(MIN, -5)
@u.register_metadata(MAX, 5)
def four_peak(x):
    """
    Evaluate the four-peak function.

    This function has two local peaks with f = 1 at (-4, 4) and (4, 4), and two global
    peaks with fmax = 2 at (0, 0) and (0, -4).

    # Arguments
    * `x` - The input vector. Should be an ndarray from numpy.

    # Returns
    The value of the function.
    """
    x1 = x[0][0]
    x2 = x[0][1]

    x1s = np.power(x1, 2)
    x2s = np.power(x2, 2)

    x1m4s = np.power(x1 - 4, 2)
    x1p4s = np.power(x1 + 4, 2)
    x2m4s = np.power(x2 - 4, 2)
    x2p4s = np.power(x2 + 4, 2)

    t1 = np.exp(-x1m4s - x2m4s)
    t2 = np.exp(-x1p4s - x2m4s)
    t31 = np.exp(-x1s - x2s)
    t32 = np.exp(-x1s - x2p4s)
    t3 = 2 * (t31 + t32)

    return (t1 + t2 + t3)

@u.register_metadata(MIN, -10)
@u.register_metadata(MAX, 10)
def alpine(x):
    """
    Evaluate the Alpine function.

    Global minimum is located at origin x* = (0, ..., 0) with f(x*) = 0

    # Arguments
    * `x` - The input vector

    # Returns
    The value of the function.
    """

    assert isinstance(x, np.ndarray), "Input vector is not an ndarray!"

    d = x.shape[1]
    sum = 0
    for i in range(d):
        xi = x[0][i]
        sum += xi * np.sin(xi) + (0.1 * xi)

    return sum

@u.register_metadata(MIN, -2 * np.pi)
@u.register_metadata(MAX, 2 * np.pi)
def bird(x):
    """
    Evaluate the bird function. Only valid for D=2

    Two global minima are located at x* = (4.70104,3.15294) and (-1.58214, -3.13024) with f (x*) = -106.764537.

    # Arguments
    * `x` - The input vector

    # Return
    The value of the function.
    """

    d = x.shape[1]
    assert d == 2, f"Dimension of x must be 2 but was actually {d}"

    x1 = x[0][0]
    x2 = x[0][1]

    t1 = np.sin(x1) * np.power(np.exp(1 - np.cos(x2)), 2)
    t2 = np.cos(x2) * np.power(np.exp(1 - np.sin(x1)), 2)
    t3 = np.power(x1 - x2, 2)

    return t1 + t2 + t3

@u.register_metadata(MIN, -10)
@u.register_metadata(MAX, 10)
def dixon_price(x):
    """
    Implement the Dixon and Price Function

    Global minimum are located at x* = [2^( (2^i - 2) / 2^i ), i] for i in [1, D] with f(x*) = 0

    # Arguments
    * `x` - The input vector.

    # Returns
    The value of the function.
    """

    d = x.shape[1]

    sum = np.power(x[0][0] - 1, 2)

    for i in range(1, d):
        xi = x[0][i]    
        ximo = x[0][i-1]

        t1 = 2 * np.power(xi, 2)

        sum += (i + 1) * np.power(t1 - ximo, 2)

    return sum

@u.register_metadata(MIN, -10)
@u.register_metadata(MAX, 10)
def easom(x):
    """
    Evaluate the Easom function.

    Global optima located at x* = (pi, pi) with f(x*) = -1

    # Arguments
    * `x` - The input vector
    """

    d = x.shape[1]
    assert d == 2, f"Function defined for d=2, but got d={d}"

    x1 = x[0][0]
    x2 = x[0][1]

    exp = -np.power(x1 - np.pi, 2) - np.power(x2 - np.pi, 2)

    return -np.cos(x1) * np.cos(x2) * np.exp(exp)

def exponential(x):
    """
    Evaluate the Exponential function.

    Global optima located at x* = (0, ..., 0) with f(x*) = 1

    # Arguments
    * `x` - The input vector
    """

    d = x.shape[1]

    sum = 0
    for i in range(d):
        sum += np.power(x[0][i], 2)

    sum *= -0.5
    return -np.exp(sum)

@u.register_metadata(MIN, -5)
@u.register_metadata(MAX, 5)
def parsopoulos(x):
    """
    Evaluate the Parsopoulos function.

    Many global optima at points (k*pi/2, lambda*pi) where k is odd, and lambda is any number.
    On the defined range, there are 12 global minima all equal to zero.

    # Arguments
    * `x` - The input vector
    """

    d = x.shape[1]
    assert d == 2, f"Function defined for d=2, but got d={d}"

    x1 = x[0][0]
    x2 = x[0][1]

    return np.power(np.cos(x1), 2) + np.power(np.sin(x2), 2)

@u.register_metadata(MIN, 0)
@u.register_metadata(MAX, 1)
def ripple(x):
    """
    Evaluate the Ripple function.

    Book does a terrible job with this one...

    # Arguments
    * `x` - The input vector
    """

    d = x.shape[1]

    sum = 0
    for i in range(d):
        xi = x[0][i]

        p1 = -np.exp(-2 * np.power((xi - 0.1) / 0.8, 2))
        
        p2t1 = np.power(np.sin(5 * np.pi * xi), 6)
        p2t2 = 0.1 * np.power(np.cos(500 * np.pi * xi), 2)

        p2 = p2t1 + p2t2

        sum += p1 * p2

    return sum

@u.register_metadata(MIN, -10)
@u.register_metadata(MAX, 10)
def first_holder_table(x):
    """
    Evaluate the First Holder Table function.

    Four global optima located at x* = (±9.646168, ±9.646168) with f(x*) = -26.920336.

    # Arguments
    * `x` - The input vector

    # Note
    Textbook function is incorrect. See https://www.sfu.ca/~ssurjano/holder.html
    """

    d = x.shape[1]
    assert d == 2, f"Function defined for d=2, but got d={d}"

    x1 = x[0][0]
    x2 = x[0][1]

    p1 = np.cos(x1)
    p2 = np.cos(x2)

    p3t2 = np.sqrt((x1 ** 2) + (x2 ** 2))
    p3 = np.exp(np.abs(1 - (p3t2 / np.pi)))

    return -np.abs(p1 * p2 * p3)

@u.register_metadata(MIN, 0)
@u.register_metadata(MAX, 10)
def keane(x):
    """
    Evaluate the Kean function.

    Global optima are located at x* = ({0, 1.39325}, {1.39325, 0}) with f(x*)= - 0.673668

    # Arguments
    * `x` - The input vector
    """

    d = x.shape[1]
    assert d == 2, f"Function defined for d=2, but got d={d}"

    x1 = x[0][0]
    x2 = x[0][1]

    st1 = np.power(np.sin(x1 - x2), 2)
    st2 = np.power(np.sin(x1 + x2), 2)
    n = st1 * st2

    di = np.power(x1, 2) + np.power(x2, 2)
    d = np.sqrt(di)

    return n / d

