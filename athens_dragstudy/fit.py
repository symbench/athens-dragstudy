import numpy

"""
functions and utilities to fit drag data to f(x) = y where x is a tensor of length params of shape (# study params, # data points)
and y is of shape (3, # data points) or (6, # data points) if estimating center of drag
"""


def run(vehicle):
    print(f"run fit for {vehicle}")