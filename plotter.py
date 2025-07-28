import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from vpython import button, canvas, curve, vec, vector, color, label, arrow, cylinder, cone, rate
import time
from hop.utilities import import_data
from hop.constants import Constants

mc = Constants()

class DronePlotter3d:
    def __init__(self, data, dt, t):
        self.data = data
        self.dt = dt
        self.tspan = np.arange(0,t, self.dt)
        self.rocket_length = int(mc.l)
        self.started = False  

    def quat_to_rot(self, q):
        x, y, z, w = q
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return np.array([
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
        ])
    
    def apply_pose(self, pos_model, q_vec, F, body, nose, thrust):
        R = self.quat_to_rot(q_vec)
        axis_m = R[:, 2]
        vp_axis = vector(axis_m[0], axis_m[2], -axis_m[1])
        body.axis = vp_axis * self.rocket_length * 0.8
        body.pos = vector(pos_model[0], pos_model[2], -pos_model[1]) - (body.axis / 2)
        nose.pos = body.pos + body.axis
        nose.axis = vp_axis * self.rocket_length * 0.2
        thrust.pos = body.pos
        thrust.axis = -vector(F[0], F[1], F[2]) / 12

    def draw_grid(self, half_extent=10, spacing=1, y_level=0, col=color.gray(0.9)):
        r = 0.002
        for x in range(-half_extent, half_extent + 1, spacing):
            curve(vec(x, y_level, -half_extent), vec(x, y_level, half_extent), color=col, radius=r)
        for z in range(-half_extent, half_extent + 1, spacing):
            curve(vec(-half_extent, y_level, z), vec(half_extent, y_level, z), color=col, radius=r)
    
    def start_callback(self):
        self.started = True
        print("Simulation started!")

    def plot(self):
        scene = canvas(title="3D Rocket Visualization", background=color.white, width=1420, height=750)
        button(text='Start', bind=self.start_callback)
        scene.forward = vector(-1, -1, -1)
        scene.up = vector(0, 1, 0)
        scene.range = 2.5

        axis_len = 2

        self.draw_grid()

        arrow(pos=vec(0, 0, 0), axis=vec(axis_len, 0, 0), shaftwidth=.015, headwidth=.04, headlength=.12, color=color.red)
        arrow(pos=vec(0, 0, 0), axis=vec(0, axis_len, 0), shaftwidth=.015, headwidth=.04, headlength=.12, color=color.blue)
        arrow(pos=vec(0, 0, 0), axis=vec(0, 0, -axis_len), shaftwidth=.015, headwidth=.04, headlength=.12, color=color.green)
        label(pos=vec(axis_len, 0, 0), text='X', box=False, opacity=0, color=color.red)
        label(pos=vec(0, axis_len, 0), text='Z', box=False, opacity=0, color=color.blue)
        label(pos=vec(0, 0, -axis_len), text='Y', box=False, opacity=0, color=color.green)

        rocket_radius = self.rocket_length / 10
        thrust_length = 0.5

        body = cylinder(pos=vec(0, -self.rocket_length * 0.4, 0), axis=vec(0, self.rocket_length * 0.8, 0), radius=rocket_radius, color=color.red)
        nose = cone(pos=body.pos + body.axis, axis=vec(0, self.rocket_length * 0.2, 0), radius=rocket_radius, color=color.orange)
        thrust = arrow(pos=body.pos, axis=vec(0, -thrust_length, 0), shaftwidth=0.05, headwidth=0.08, headlength=0.1, color=color.blue)


        pos = self.data[0, 0:3]
        qv = self.data[0, 6:10]
        u = self.data[0, 13:17]

        F = [0,10,0]

        self.apply_pose(pos, qv, F, body, nose, thrust)


        dt_vis = self.tspan[1] - self.tspan[0]

        while not self.started:
            rate(20)
        for k in range(len(self.tspan)):
            rate(1 / dt_vis)
            pos = self.data[k, 0:3]
            qv = self.data[k, 6:10]
            u = self.data[k, 13:17]
            F = [
            u[2] * np.sin(u[1]),
            u[2] * np.cos(u[0])*np.cos(u[1]),
            u[2] * np.sin(u[0])*np.cos(u[1])
            ]
            self.apply_pose(pos, qv, F, body, nose, thrust)
            scene.center = body.pos + vector(0, 0.3, 0)

def main(args=None):
    dt = 0.1
    data = import_data("current.json")
    t = data.shape[0] * dt
    DronePlotter3d(data, dt, t)

if __name__ == '__main__':
    main()

