import numpy as np
import matplotlib.pyplot as plt

length = 100
total = 100
alpha = 1e-4
prob_scale = 1e-7
dt = 1e-3
dx = 1e-6
v0 = 0.28

def draw_plots(series,estimated,navier_stokes,units_per_cell):
    plt.plot(np.linspace(0,100,total),series(np.linspace(0,2,total)))
    plt.title("The input time series")
    plt.show()

    est = estimated(length,dt,dx,series(np.linspace(0,2,total)))
    plt.plot(np.linspace(0,100,len(est)),est)
    plt.title("The estimated navier stokes relation")
    plt.show()

    act = navier_stokes(v0,dt,dx,alpha,length)*units_per_cell(length)
    plt.plot(np.linspace(0,100,len(act)),act)
    plt.title("The actual computed navier stokes relation")
    plt.show()