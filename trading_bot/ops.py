import os
import math
import logging

import numpy as np


def sigmoid_tanh(x):
    """Performs sigmoid / tanh operation
    """
    try:
        if x < 0:
            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)) # tanh function
        return 1 / (1 + math.exp(-x)) #sigmoid
    except Exception as err:
        print("Error in sigmoid/tanh: " + err)


def elu_function(x,alpha=1.0):
    try:
        return x if x >= 0 else alpha*(math.exp(x) -1)
    except Exception as err:
        print("Error in Elu " + err)

def tanh_function(x):

    try:
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)) # tanh function
    except Exception as err:
        print("Error in Tanh " + err)


def get_state(data, t, n_days):
    """Returns an n-day state representation ending at time t
    """
    d = t - n_days + 1
    block = data[d: t + 1] if d >= 0 else -d * [data[0]] + data[0: t + 1]  # pad with t0
    res = []
    for i in range(n_days - 1):
        res.append(sigmoid_tanh(block[i + 1] - block[i]))
    return np.array([res])
