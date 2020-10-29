#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 17:48:59 2020

@author: laurent
"""

import numpy as np
import matplotlib.pyplot as plt
from bz_model import bz_1d_3eq_model
import rk_coeffs
import integration_esdirk

### SETUP MODEL ###
xmin = 0
xmax = 4
nx = 100
dx = (xmax-xmin)/(nx+1)
x = np.linspace(xmin+dx, xmax-dx, nx)

tini = 0.
tend = 2.0
# method = 'Radau5'
method = 'ESDIRK43B'

bz1d = bz_1d_3eq_model(mu=1e-5, eps=1e-2, f=3, q=2e-3, db=2.5e-3, dc=1.5e-3, xmin=xmin, xmax=xmax, nx=nx)
fcn = bz1d.fcn
yini = bz1d.init()
A,b,c   = rk_coeffs.getButcher(method)

import scipy.optimize
from scipy.sparse import diags
nvar=3
width = np.ceil(nvar*1.5).astype(int)
offsets = range(-width,width) # positions of the diagonals with respect to the main diagonal
diagonals = [np.ones((nvar*bz1d.nx-abs(i),)) for i in offsets]
sparsity = diags(diagonals, offsets)
groups = scipy.optimize._numdiff.group_columns(sparsity)
def jac_sparse(t,x):
  return scipy.optimize._numdiff.approx_derivative(fun=lambda x: fcn(t,x), x0=x,
                                                      method='2-point', rel_step=1e-8, f0=None,
                                                      sparsity=(sparsity,groups), bounds=(-np.inf,np.inf),
                                                      as_linear_operator=False, args=(), kwargs={})
def jac_full(t,x):
  return scipy.optimize._numdiff.approx_derivative(fun=lambda x: fcn(t,x), x0=x,
                                                      method='2-point', rel_step=1e-8, f0=None,
                                                      sparsity=None, bounds=(-np.inf,np.inf),
                                                      as_linear_operator=False, args=(), kwargs={})

assert np.allclose( jac_sparse(tini, yini).toarray(), jac_full(tini, yini))

A,b,c = rk_coeffs.getButcher(name=method)
dt = 2e-4 # ok pour Radau5, sinon non CV
nt = int((tend-tini)/dt)
sol = integration_esdirk.DIRK_integration(fun=fcn, y0=yini, t_span=[tini, tend], nt=nt,
                                    A=A, b=b, c=c, jacfun=jac_sparse, newtonchoice=2)    

# sol = integration_esdirk.FIRK_integration(fun=fcn, y0=yini, t_span=[tini, tend], nt=nt,
#                                    A=A, b=b, c=c, jacfun=None, newtonchoice=2,fullDebug=True)

a,b,c = sol.y[::nvar,:], sol.y[1::nvar,:], sol.y[2::nvar,:]
fig, ax = plt.subplots(3,1,sharex=True)
ax[0].plot(a[:,-1])
ax[0].set_ylabel('a')
ax[1].plot(b[:,-1])
ax[1].set_ylabel('b')
ax[2].plot(c[:,-1])
ax[2].set_ylabel('c')
ax[-1].set_xlabel('x')

for key in sol.infodict_hist.keys():
  plt.figure()
  plt.plot(sol.infodict_hist[key])
  plt.ylabel(key)
  plt.xlabel('time step number')
