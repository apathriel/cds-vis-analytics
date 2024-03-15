import argparse
import os
import sys
sys.path.append("..")
import numpy as np
import pandas as pd

def instantiate_arg_parser():
    parser = argparse.ArgumentParser(description="Loading and printing an array")
    parser.add_argument("--input", "-i", required=True)
    return parser.parse_args()

def load_data_np(filepath, delimiter_symbol, print=True):
    np_data = np.loadtxt(filepath, delimiter=delimiter_symbol)
    if print:
        print(np_data)
    return np_data

def main():
    args = instantiate_arg_parser()
    
    dir_path = os.path.join("..", "..", "cds-vis-data", "data", "sample-data")
    file_path = os.path.join(dir_path, args.input)

    load_data_np(file_path, ",")

if __name__ == '__main__':
    main()