from numpy import all

def smooth_metric(data):

    smoothness = 0
    for i in range(len(data)-1):
        change = data[i] - data[i+1]
        smoothness += change.T @ change

    return smoothness

def settling_metric(data, ll, ul):
    settle_time = len(data)
    settled = False
    for i in range(len(data)):
        if settled:
            if all(data[i] < ll) or all(data[i] > ul):
                settled = False
                settle_time = len(data)
        else:
            if all(data[i] > ll) and all(data[i] < ul):
                settle_time = i
                settled = True

    return settle_time