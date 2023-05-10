import numpy as np


def previous_interpolation(data):
    data_processed = np.zeros(len(data))
    num = 0
    for idx in range(2, len(data)):
        if np.isnan(data[idx]):
            data_processed[idx] = data[idx - 1]
            num = num + 1
        else:
            data_processed[idx] = data[idx]
    print("%d missing data have been interpolated." % num)
    return data_processed
