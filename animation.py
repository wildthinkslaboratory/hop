from vpython import canvas, curve, vec, vector, color, label, arrow, cylinder, cone, rate
import numpy as np
from time import perf_counter
from math import sqrt

class RocketAnimation:
    def __init__(self):
        self.scene = canvas(title="3D Rocket Visualization", background=color.white, width=1600, height=1200)
        self.scene.forward = vector(-1, -1, -1)
        self.scene.up = vector(0, 1, 0)
        self.scene.range = 2.5

        self.draw_grid()

        axis_len = 2
        arrow(pos=vec(0, 0, 0), axis=vec(axis_len, 0, 0), shaftwidth=.015, headwidth=.04, headlength=.12, color=color.red)
        arrow(pos=vec(0, 0, 0), axis=vec(0, axis_len, 0), shaftwidth=.015, headwidth=.04, headlength=.12, color=color.blue)
        arrow(pos=vec(0, 0, 0), axis=vec(0, 0, -axis_len), shaftwidth=.015, headwidth=.04, headlength=.12, color=color.green)
        label(pos=vec(axis_len, 0, 0), text='X', box=False, opacity=0, color=color.red)
        label(pos=vec(0, axis_len, 0), text='Z', box=False, opacity=0, color=color.blue)
        label(pos=vec(0, 0, -axis_len), text='Y', box=False, opacity=0, color=color.green)

        rocket_radius = 0.1
        self.rocket_length = 1.0
        self.nose_length = 0.2
        self.thrust_length = 0.05

        self.body = cylinder(pos=vec(0, 0.01, 0), axis=vec(0, self.rocket_length, 0), radius=rocket_radius, color=color.red)
        self.nose = cone(pos=self.body.pos + self.body.axis, axis=vec(0, self.nose_length, 0), radius=rocket_radius, color=color.orange)
        self.thrust = arrow(pos=self.body.pos, axis=vec(0, -self.thrust_length, 0), shaftwidth=0.05, headwidth=0.08, headlength=0.1, color=color.blue)

    def draw_grid(self, half_extent=10, spacing=1, y_level=0, col=color.gray(0.9)):
        r = 0.002
        for x in range(-half_extent, half_extent + 1, spacing):
            curve(vec(x, y_level, -half_extent), vec(x, y_level, half_extent), color=col, radius=r)
        for z in range(-half_extent, half_extent + 1, spacing):
            curve(vec(-half_extent, y_level, z), vec(half_extent, y_level, z), color=col, radius=r)

    def animate(self, tspan, x_hist, u_hist):
        # ------------------------------------------------------------------
        # animation loop
        dt_vis = tspan[1] - tspan[0]
        for k in range(len(tspan)):
            rate(1 / dt_vis)
            pos = x_hist[k, 0:3]
            qv = x_hist[k, 6:10]
            control = u_hist[k]
            self.apply_pose(pos, qv, control)
            self.scene.center = self.body.pos + vector(0, 0.3, 0)


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

    def apply_pose(self, pos_model, q_vec, control):
        vp_pos = vector(pos_model[0], pos_model[2], -pos_model[1])
        self.body.pos = vp_pos

        R = self.quat_to_rot(q_vec)
        axis_m = R[:, 2]
        vp_axis = vector(axis_m[0], axis_m[2], -axis_m[1])
        self.body.axis = vp_axis * self.rocket_length
        self.nose.pos = self.body.pos + self.body.axis
        self.nose.axis = vp_axis * self.nose_length
        self.thrust.pos = self.body.pos
        self.thrust.axis = -vp_axis.norm() * self.thrust_length * control[2]



