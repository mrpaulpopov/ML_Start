from matplotlib import pyplot as plt
import numpy as np

def draw_line(slope, y_intercept, color='grey', linewidth=0.7, starting=0, ending=8):
    x = np.linspace(starting, ending, 1000)
    plt.plot(x, y_intercept + slope*x, linestyle='-', color=color, linewidth=linewidth)

def plot_points(x, y):
    X = np.array(x)
    y = np.array(y)
    plt.scatter(X, y) # plt.scatter draws distinct point. plt.plot(X, y) draws a line.
    plt.xlabel('number of rooms')
    plt.ylabel('prices')
