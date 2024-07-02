"""
Optim 1-d test function from 
https://infinity77.net/global_optimization/test_functions_1d.html
"""
from functools import wraps

import gpmp.num as gnp
import numpy as np


def goldstein_price(x):
    """Goldstein Price Function

    https://www.sfu.ca/~ssurjano/goldpr.html
    """
    x1 = x[:, 0]
    x2 = x[:, 1]
    return (
        1
        + (x1 + x2 + 1) ** 2
        * (19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2)
    ) * (
        30
        + (2 * x1 - 3 * x2) ** 2
        * (18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2)
    )


def park_function(x):
    """
    Park Function, see https://www.sfu.ca/~ssurjano/park91b.html
    d = 4
    domain : x_i \in [0, 1]
    """
    return 2 / 3 * gnp.exp(x[:, 0] + x[:, 1]) - x[:, 3] * gnp.sin(x[:, 2]) + x[:, 2]


def friedman(x):
    """Friedman function

    d = 5
    """
    return (
        10 ** gnp.sin(gnp.pi * x[:, 0] * x[:, 1])
        + 20 * (x[:, 2] - 0.5) ** 2
        + 10 * x[:, 3]
        + 5 * x[:, 4]
    )


# ---------------------- Becker's function ----------------------


def f_1(x):
    return x


def f_2(x):
    return x**2


def f_3(x):
    return gnp.pow(x, 3)


def f_4(x):
    return (gnp.exp(x) - 1) / (gnp.e - 1)


def f_5(x):
    return 1 / 2 * gnp.sin(2 * gnp.pi * x) + 1 / 2


def f_6(x):
    n = x.size()[0]
    return gnp.heaviside(x - 1 / 2, gnp.zeros(n))


def f_7(x):
    return 0


def f_8(x):
    return 4 * (x - 1 / 2) ** 2


def f_9(x):
    return 1 / ((10 - 1 / 1.1) * (x + 0.1)) - 0.1


func_base = [f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8, f_9]


def wrapper_becker(u, V, W, d, alpha, beta, gamma):
    """Wrapper to create Beck functions

    [1] W. Becker, “Metafunctions for benchmarking in sensitivity analysis,” Reliability Engineering & System Safety, vol. 204, p. 107189, 2020, doi: https://doi.org/10.1016/j.ress.2020.107189.
    """

    def decorate(func):
        @wraps(func)
        def wrap(x):
            func(x)
            d_2 = V.shape[0]
            d_3 = W.shape[0]

            sum_one = 0
            for i in range(d):
                sum_one += alpha[i] * func_base[u[i]](x[:, i])

            sum_two = 0
            for i in range(d_2):
                sum_two += (
                    beta[i]
                    * func_base[u[V[i, 0]]](x[:, V[i, 0]])
                    * func_base[u[V[i, 1]]](x[:, V[i, 1]])
                )

            sum_three = 0
            for i in range(d_3):
                sum_three += (
                    gamma[i]
                    * func_base[u[W[i, 0]]](x[:, W[i, 0]])
                    * func_base[u[W[i, 1]]](x[:, W[i, 1]])
                    * func_base[u[W[i, 2]]](x[:, W[i, 2]])
                )

            return sum_one + sum_two + sum_three

        return wrap

    return decorate


def gmm(var, coeff, n):
    res = np.zeros(n)
    for i in range(n):
        if np.random.random() < coeff[0]:
            res[i] = np.random.randn() * var[0]
        else:
            res[i] = np.random.randn() * var[1]
    return res


# exemple of a Beck function
np.random.seed(seed=45)
d = 2
V = np.random.randint(d, size=(2, 2))
W = np.random.randint(d, size=(2, 3))
u = np.random.randint(9, size=d)

alpha = gmm([0.5, 5], [0.7, 0.3], d)
beta = gmm([0.5, 5], [0.7, 0.3], 2)
gamma = gmm([0.5, 5], [0.7, 0.3], 2)


@wrapper_becker(u, V, W, d, alpha, beta, gamma)
def beck_2(x):
    pass
