#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 17:48:59 2020

@author: laurent
"""

import numpy as np
import matplotlib.pyplot as plt
from bz_model import bz_1d_2eq_model
import rk_coeffs
import integration_esdirk

### SETUP MODEL ###
xmin = 0
xmax = 4
nx = 30
dx = (xmax-xmin)/(nx+1)
x = np.linspace(xmin+dx, xmax-dx, nx)

tini = 0.
tend = 0.1
# method = 'Radau5'
method = 'ESDIRK43B'

bz1d2eq = bz_1d_2eq_model(eps=1e-2, f=3, q=2e-3, db=2.5e-3, dc=1.5e-3, xmin=xmin, xmax=xmax, nx=nx)
fcn = bz1d2eq.fcn
yini = bz1d2eq.init()
A,b,c   = rk_coeffs.getButcher(method)

import scipy.optimize
from scipy.sparse import diags
offsets = range(-3,7) # positions of the diagonals with respect to the main diagonal
diagonals = [np.ones((2*bz1d2eq.nx-abs(i),)) for i in offsets]
sparsity = diags(diagonals, offsets)
groups = scipy.optimize._numdiff.group_columns(sparsity)
def jac(t,x):
  # raise Exception('stop here')
  return scipy.optimize._numdiff.approx_derivative(fun=lambda x: fcn(t,x), x0=x,
                                                      method='2-point', rel_step=1e-8, f0=None,
                                                      sparsity=(sparsity,groups), bounds=(-np.inf,np.inf),
                                                      as_linear_operator=False, args=(), kwargs={})

A,b,c = rk_coeffs.getButcher(name=method)
dt = 2e-3 # ok pour Radau5, sinon non CV
nt = int((tend-tini)/dt)
sol = integration_esdirk.DIRK_integration(fun=fcn, y0=yini, t_span=[tini, tend], nt=nt,
                                    A=A, b=b, c=c, jacfun=jac, newtonchoice=2)

# sol = integration_esdirk.FIRK_integration(fun=fcn, y0=yini, t_span=[tini, tend], nt=nt,
#                                    A=A, b=b, c=c, jacfun=None, newtonchoice=2,fullDebug=True)

fig, ax = plt.subplots(2,1,sharex=True)
ax[0].plot(sol.y[0,:])
ax[0].set_ylabel('var1')
ax[1].plot(sol.y[1,:])
ax[0].set_ylabel('var2')
ax[-1].set_xlabel('x')

for key in sol.infodict_hist.keys():
  plt.figure()
  plt.plot(sol.infodict_hist[key])
  plt.ylabel(key)
  plt.xlabel('time step number')
