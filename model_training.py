import tensorflow as tf
import numpy as np
from utils import *
from tqdm import tqdm

lr = 5e-3
prob_var = tf.Variable(tf.ones(length))
alpha_var = tf.Variable(tf.ones(length))
v0_var = tf.Variable(v0)
dt_var = tf.Variable(dt)
dx_var = tf.Variable(dx)
prob_retention_ = lambda length: prob_var
optimizer = tf.train.GradientDescentOptimizer(lr)
x = tf.Variable(tf.range(0,1,0.01))
units_var = tf.Variable(tf.ones(length))

def units_per_cell(length):
    return tf.math.multiply(units_var,np.linspace(1,0,length))

def velocity_solve(v0,x,t,alpha_var,length):
    return v0 - x*tf.reduce_sum(tf.math.multiply(prob_retention_(length),1/alpha_var))

def navier_stokes_solve(v0,dt,dx,alpha_var,length):
    pot = []
    ti = 0
    for i,xi in enumerate(tqdm(np.linspace(0,dx*total,total-1))):
        vi = velocity_solve(v0,xi,ti,alpha_var,float(length))
        diff = vi-v0
        qty = diff/dt - v0*diff/dx # force
        v0 = vi
        pot.append(qty)
    return tf.stack(pot)

y = tf.math.multiply(navier_stokes_solve(v0_var,dt,dx,alpha_var,length),units_per_cell(length)[0:99])

loss = tf.reduce_mean(tf.abs(y - est[0:99]))

train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
local = tf.local_variables_initializer()
n_iter = 120000

def train_model(n_iter):

    y_vals = []
    captured = []
    
    with tf.Session() as sess:
        sess.run(init)
        sess.run(local)
        loss_values = []
        train_data = []
        for step in tqdm(range(n_iter)):
            _, loss_val, alpha_val, prob_val,v0_val,x_val,y_val = \
            sess.run([train, loss, alpha_var, prob_var,v0_var,x,y])
            loss_values.append(loss_val)
            if np.prod(np.isclose(y_val,est[0:99],atol=95e4).astype(np.int)) == 1:
                y_vals.append(y_val)
                captured.append((alpha_val,prob_val,v0_val,x_val))
                print("captured")
            if step % 1000 == 0:
                print(step, loss_val)

    return captured, y_vals

if __name__ == "__main__":

    captured, y_vals = train_model(n_iter)

    tolerance = 95e-4

    for ii,result in enumerate(y_vals):
        if (np.prod(np.isclose(result,est[0:99],atol=tolerance))) == 1:
            idx = ii
        alpha_,prob_,v0_,x_ = captured[idx]