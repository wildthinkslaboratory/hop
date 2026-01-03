import json
import numpy as np
import math

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
    

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


def quat_to_rot(q):
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
    ])
    
def quaternion_to_angle(q):
    q = q / np.linalg.norm(q)
    R = quat_to_rot(q)     # your function
    v = R[:, 2]            # body z-axis in world

    x_tilt = np.degrees(np.arctan(v[1] / v[2]))
    y_tilt = np.degrees(np.arctan(v[0] / v[2]))
    tilt = np.degrees(np.arccos(np.clip(v[2], -1.0, 1.0)))
    
    return np.array([x_tilt, y_tilt, tilt])


