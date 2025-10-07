import json
import numpy as np


def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)


def output_data(data, filename):
    # Serializing json
    json_object = json.dumps(data, indent=4)

    # Writing to sample.json
    with open(filename, "w") as outfile:
        outfile.write(json_object)


def import_data(filename):
    # Open and read the JSON file
    data = {}
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


from math import floor, log10
def sig_figs(x: float, precision: int):

    x = float(x)
    precision = int(precision)
    if x == 0.0:
        return x
    return round(x, -int(floor(log10(abs(x)))) + (precision - 1))