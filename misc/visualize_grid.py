from __future__ import print_function
import numpy as np
import sys

def print_grid(data, col_len, row_len):
    if (data.size != (col_len * row_len)):
        print ("can't visualize grid, invalid dimensions")
    else:
        _print_grid(data, col_len)

def _print_grid(data, col_len):
    col_idx = 0
    for x in data:
        print (x, end='')
        col_idx = col_idx + 1
        if col_idx == col_len:
            print ("")
            col_idx = 0


def has_straight_line_of_len (len):
    #convert 

print_grid(np.array([0,0,1,0,0,1,0,0,1]), 3, 3)
