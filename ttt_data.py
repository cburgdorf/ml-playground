import numpy as np

def get_training_data_2d():
    return  np.array([[[[0,0,0],
                        [0,0,0],
                        [0,0,0]]],
                      [[[0,0,1],
                        [0,1,0],
                        [0,0,1]]],
                      [[[1,0,1],
                        [0,1,0],
                        [0,0,0]]],
                      [[[0,0,0],
                        [0,1,0],
                        [1,0,1]]],
                      [[[0,0,1],
                        [0,1,0],
                        [1,0,0]]],
                      [[[1,0,0],
                        [1,0,0],
                        [1,0,0]]],
                      [[[0,0,0],
                        [1,1,1],
                        [0,0,0]]],
                      [[[0,1,0],
                        [0,1,0],
                        [0,1,0]]]], "float32")

def get_target_data_per_pixel():
    return  np.array( [[[0,0,0],
                        [0,0,0],
                        [0,0,0]]],
                      [[[0,0,0],
                        [0,0,0],
                        [0,0,0]]],
                      [[[0,0,0],
                        [0,0,0],
                        [0,0,0]]],
                      [[[0,0,0],
                        [0,0,0],
                        [0,0,0]]],
                      [[[0,0,1],
                        [0,1,0],
                        [1,0,0]]],
                      [[[1,0,0],
                        [1,0,0],
                        [1,0,0]]],
                      [[[0,0,0],
                        [1,1,1],
                        [0,0,0]]],
                      [[[0,1,0],
                        [0,1,0],
                        [0,1,0]]], "float32")


def get_target_data_2d():
    return np.array([0,0,0,0,1,1,1,1], "float32")

def get_validation_data_2d():
    return np.array([[[[0,0,0],
                       [0,0,0],
                       [0,0,0]]],
                     [[[1,0,0],
                       [0,1,0],
                       [1,0,0]]],
                     [[[1,0,0],
                       [0,1,0],
                       [0,0,1]]],
                     [[[0,0,1],
                       [0,0,1],
                       [0,0,1]]]], "float32")

def reshape_training_data_1d(arr):
    return arr.reshape(8,9)

def reshape_validation_data_1d(arr):
    return arr.reshape(4,9)

def reshape_target_data_1d(arr):
    return arr.reshape(8,1)
