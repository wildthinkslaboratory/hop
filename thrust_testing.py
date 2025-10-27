import matplotlib.pyplot as plt

pwm = [x / 10.0 for x in range(11)]

# this data is from the Brother Hobby test from the YouTube video
brotherHobbyData = [0, 19.7, 77, 176, 294, 435, 596, 818, 1059, 1320, 1521]

# drone thrust data from 10/26/2025
oneMotor = [43, 80, 144, 240, 340, 440, 560, 675, 800, 921, 1050]
twoMotor = [43, 131, 267, 468, 668, 854, 1067, 1300, 1510, 1735, 1957]

# plt.plot(pwm, brotherHobbyData, marker='o', linestyle='None', markersize=4)
plt.plot(pwm, oneMotor, marker='o', linestyle='None', markersize=4)
plt.plot(pwm, twoMotor, marker='o', linestyle='None', markersize=4)
plt.show()




