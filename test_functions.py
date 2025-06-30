"""
################################################################################
CSCI 633: Biologically-Inspired Intelligent Systems
Version taught by Alexander G. Ororbia II

Testing of the implemented benchmark functions.

Note, all implementations assume row-oriented vectors.
################################################################################
"""

import numpy as np
import functions as f

def test_four_peak():
    """
    Test the implementation of the 4-peak function.
    """
    x_star_lm1 = np.asarray([[-4.,4.]]) # local mode 1
    x_star_lm2 = np.asarray([[4.,4.]]) # local mode 2
    x_star_gm1 = np.asarray([[0.,0.]]) # global mode 1
    x_star_gm2 = np.asarray([[0.,-4.]]) # global mode 2

    # check local modes
    tol = 1e-6
    f_star = np.ones((1, 1))
    f_max = f.four_peak(x_star_lm1) # should be f(x) = 1
    # print("F* = {}\tf(x) = {}".format(f_star, f_max))
    delta = np.abs(f_max - f_star)
    np.testing.assert_array_less(delta, tol)
    f_max = f.four_peak(x_star_lm2) # should be f(x) = 1
    # print("F* = {}\tf(x) = {}".format(f_star, f_max))
    # check global modes
    f_star = np.ones((1,1)) * 2
    f_max = f.four_peak(x_star_gm1)  # should be f(x) = 2
    # print("F* = {}\tf(x) = {}".format(f_star, f_max))
    delta = np.abs(f_max - f_star)
    np.testing.assert_array_less(delta, tol)
    f_max = f.four_peak(x_star_gm2)  # should be f(x) = 2
    # print("F* = {}\tf(x) = {}".format(f_star, f_max))
    delta = np.abs(f_max - f_star)
    np.testing.assert_array_less(delta, tol)

def test_eggcrate():
    """
    Test the eggcrate function.
    """
    D = 2
    
    x_star = np.zeros((1,D))
    f_max = f.eggcrate(x_star) # should be f(x) = <0,..,0>
    # print("True Gobal Maximum:  f(x*) = {}  at x* = {}".format(f_max, x_star))
    f_star = np.zeros((1,1)) # global optimum is at <0,...,0> yielding f* = f(x) = 0
    # print(f"f_max: {f_max}\nf_star: {f_star}")
    np.testing.assert_array_equal(f_max, f_star)

def test_ackley():
    """
    Test the ackley function.
    """

    D = 2

    x_star = np.zeros((1,D))
    f_max = f.ackley(x_star) # should be f(x) = <0,..,0>
    # print("True Gobal Maximum:  f(x*) = {}  at x* = {}".format(f_max, x_star))
    f_star = np.zeros((1,1)) # global maximum is at <0,...,0> yielding f* = f(x) = 0
    tol = 1e-8
    delta = np.abs(f_max - f_star)
    np.testing.assert_array_less(delta, tol)

def test_rosenbrock():
    """
    Test the Rosenbrcok function.
    """

    D = 2

    x_star = np.ones((1,D))
    f_max = f.rosenbrock(x_star) # should be f(x) = <0,..,0>
    # print("True Gobal Maximum:  f(x*) = {}  at x* = {}".format(f_max, x_star))
    f_star = np.zeros((1,1)) # global maximum is at <0,...,0> yielding f* = f(x) = 0
    np.testing.assert_array_equal(f_max, f_star)

def test_rosenbrock_modified():
    """
    Test the modified rosenbrock function.
    """

    xstar = np.asarray([[1, 1]])
    fstar = np.ones((1, 1)) * 74
    
    f1 = f.rosenbrock_modified(xstar) # Should be 0

    np.testing.assert_array_equal(f1, fstar)

def test_alpine():
    """
    Test the alpine function.
    """

    D = 2

    xstar = np.zeros((1, D))
    f_max = f.alpine(xstar) # Should be f(x) = 0

    f_star = np.zeros((1, 1))
    np.testing.assert_array_equal(f_max, f_star)

def test_bird():
    """
    Test the bird function.
    """

    xstar1 = np.asarray([[4.70104, 3.15294]])
    xstar2 = np.asarray([[-1.58214, -3.13024]])

    f1 = f.bird(xstar1)
    f2 = f.bird(xstar2)

    fstar = np.ones((1, 1)) * -106.764537

    tol = 0.1
    delta1 = np.abs(f1 - fstar)
    delta2 = np.abs(f2 - fstar)

    np.testing.assert_array_less(delta1, tol)
    np.testing.assert_array_less(delta2, tol)

def test_dixon_price():
    """
    Test the Dixon and Price function.
    """

    star = lambda i: np.power(2, -(np.power(2, i) - 2) / np.power(2, i))

    tol = 1e-10
    star_test = star(2)
    star_expected = 0.707106781187
    delta = np.abs(star_test - star_expected)
    assert delta < tol, f"Delta ({delta}) should be less than tolerance ({tol})"

    star_test = star(1)
    star_expected = 1
    delta = np.abs(star_test - star_expected)
    assert delta < tol, f"Delta ({delta}) should be less than tolerance ({tol})"

    xstar = np.asarray([[star(1), star(2)]])

    fx = f.dixon_price(xstar)
    fstar = np.zeros((1, 1))

    delta = np.abs(fx - fstar)

    np.testing.assert_array_less(delta, tol)

def test_easom():
    """
    Test the easom function.
    """

    xstar = np.asarray([[np.pi, np.pi]])

    fx = f.easom(xstar) # Should be -1
    fstar = np.ones((1, 1)) * -1

    np.testing.assert_array_equal(fx, fstar)

def test_exponential():
    """
    Test the exponential function.
    """

    D = 2

    xstar = np.zeros((1, D))

    fx = f.exponential(xstar) # Should be 1
    fstar = np.ones((1, 1)) * -1

    np.testing.assert_array_equal(fx, fstar)

def test_parsopoulos():
    """
    Test the parsopoulos function.
    """

    optima = []
    for k in range(1, 5, 2):
        for l in range(0, 6):
            optima.append(np.asarray([[k * np.pi / 2, l * np.pi]]))

    assert len(optima) == 12, f"Should be 12 optima but there were ({len(optima)})"

    fstar = np.zeros((1, 1))
    fx = [f.parsopoulos(x) for x in optima]

    tol = 1e-10

    for fx in fx:
        delta = np.abs(fx - fstar)
        np.testing.assert_array_less(delta, tol)

def test_first_holder_table():
    """
    Test the first holder table function.
    """

    mods = [
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1)
    ]

    fstar = np.ones((1, 1)) * -26.920336

    tol = 1e-6
    for (mx1, mx2) in mods:
        xstar = np.asarray([[mx1 * 9.646168, mx2 * 9.646168]])
        fx = f.first_holder_table(xstar)

        print(f"fstar: {fstar} | fx: {fx}")
        delta = np.abs(fstar - fx)
        np.testing.assert_array_less(delta, tol)

def test_keane():
    """
    Test the Keane function.
    """

    xstar1 = np.asarray([[0, 1.39325]])
    xstar2 = np.asarray([[1.39325, 0]])

    fstar = 0.673668
    fx1 = f.keane(xstar1)
    fx2 = f.keane(xstar2)

    tol = 1e-6
    delta1 = np.abs(fx1 - fstar)
    delta2 = np.abs(fx2 - fstar)

    np.testing.assert_array_less([delta1, delta2], tol)
