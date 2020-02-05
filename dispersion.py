import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *

prob_retention = lambda x,length: 0.01
dimensionless_constant = lambda length: 0.01 * np.ones(length)

def time_series(series):
    return np.gradient(np.gradient(series))

def units_per_cell(length):
    return 1e1*np.linspace(1,0,length)

def velocity(v0,x,t,alpha,length):
    return v0 - x*prob_retention(x,length)*alpha

def navier_stokes(v0,dt,dx,alpha,length):
    pot = np.zeros(total)
    ti = 0
    for i,xi in enumerate(np.linspace(0,dx*total,total-1)):
        vi = velocity(v0,xi,ti,alpha,length)
        diff = vi-v0
        qty = diff/dt - v0*diff/dx # force
        v0 = vi
        pot[i] = qty
    return pot

def estimated(x,length,dt,dx,series):
    return units_per_cell(length)*\
(prob_retention(x,length))*\
dimensionless_constant(length)*\
time_series(series)

x = np.arange(0,1,0.01)

series = lambda t: t**5

if __name__ == "__main__":

    draw_plots(series,estimated,navier_stokes,units_per_cell)
