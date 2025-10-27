import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def quadratic_fit(x, y):
    dt = 0.01
    def quadratic(x, a, b, c):
        return a * (x*x) + b * x + c

    fine = np.arange(x[0], x[-1]+ dt, dt)
    param, param_cov = curve_fit(quadratic, x, y)
    curve = quadratic(fine, param[0], param[1], param[2])

    plt.plot(fine, curve, '-')
    return


pwm = np.arange(0,1.1,0.1)


# this data is from the Brother Hobby test from the YouTube video
brotherHobbyData = np.array([0, 19.7, 77, 176, 294, 435, 596, 818, 1059, 1320, 1521])

# drone thrust data from 10/26/2025
oneMotor = np.array([43, 80, 144, 240, 340, 440, 560, 675, 800, 921, 1050])
twoMotor = np.array([43, 131, 267, 468, 668, 854, 1067, 1300, 1510, 1735, 1957])

# plt.plot(pwm, brotherHobbyData, marker='o', linestyle='None', markersize=4)
plt.plot(pwm, oneMotor, marker='o', linestyle='None', markersize=4, label='single motor')
plt.plot(pwm, twoMotor, marker='o', linestyle='None', markersize=4, label='two coaxial motors')

quadratic_fit(pwm, oneMotor)
quadratic_fit(pwm, twoMotor)

plt.xlabel('% PWM')
plt.ylabel('grams')

plt.legend()
plt.savefig("thrust_test_graph.pdf", format="pdf", bbox_inches="tight")
plt.show()




