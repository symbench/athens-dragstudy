import numpy
from dataset import DragDataset

"""
functions and utilities to fit drag data to f(x) = y where x is a tensor of length params of shape (# study params, # data points)
and y is of shape (3, # data points) or (6, # data points) if estimating center of drag
"""


def run(vehicle):
    """ Construct a dataset from all available data for the specified vehicle 
        and perform curve fitting to approximate its drag values
    """
    print(f"run fit for {vehicle}")

def plot_dataset(dataset):
    """ Visualize the values in the dataset """ 
    pass

if __name__ == "__main__":
    dd = DragDataset(drag_subject="capsule", is_primitive=True)
    