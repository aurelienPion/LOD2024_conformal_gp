from dataclasses import dataclass 
import gpmp.num as gnp

@dataclass
class Data:
    """data manager, generate data using
    f_gen and split the data set in a training and 
    a testing set for experiments"""
    x_train: gnp.array
    z_train: gnp.array
    x_test: gnp.array
    z_test: gnp.array
