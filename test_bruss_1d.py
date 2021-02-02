#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 17:48:59 2020

@author: laurent
"""

import numpy as np
import matplotlib.pyplot as plt
import rk_coeffs
import integration_esdirk
import scipy.sparse
from bruss_1D import brusselator_1d
### SETUP MODEL ###
xmin = 0
xmax = 4
nx = 1000
dx = (xmax-xmin)/(nx+1)
x = np.linspace(xmin+dx, xmax-dx, nx)
bz1d = brusselator_1d(nx=nx)
fcn = bz1d.fcn
yini = bz1d.init()
nvar=2

# Setup time integation interval and method
tini = 0.
tend = 2.0
method   = rk_coeffs.getButcher('ESDIRK43B')

# Create a function to efficiently compute the system's Jacobian
if 0:
  import scipy.optimize
  from scipy.sparse import diags
  width = np.ceil(nvar*1.5).astype(int)
  offsets = range(-width,width) # positions of the diagonals with respect to the main diagonal
  diagonals = [np.ones((nvar*bz1d.nx-abs(i),)) for i in offsets]
  sparsity = diags(diagonals, offsets)
  groups = scipy.optimize._numdiff.group_columns(sparsity)
  def jac_sparse(t,x):
    return scipy.sparse.csc_matrix(scipy.optimize._numdiff.approx_derivative(fun=lambda x: fcn(t,x), x0=x,
                                                        method='2-point', rel_step=1e-8, f0=None,
                                                        sparsity=(sparsity,groups), bounds=(-np.inf,np.inf),
                                                        as_linear_operator=False, args=(), kwargs={}))
  def jac_full(t,x):
    return scipy.optimize._numdiff.approx_derivative(fun=lambda x: fcn(t,x), x0=x,
                                                        method='2-point', rel_step=1e-8, f0=None,
                                                        sparsity=None, bounds=(-np.inf,np.inf),
                                                        as_linear_operator=False, args=(), kwargs={})

  assert np.allclose( jac_sparse(tini, yini).toarray(), jac_full(tini, yini))
else: # let the Jacobian be computed by the solver itself
  jacfun=None
  
### Compute unsteady solution
dt = 2e-2
nt = int((tend-tini)/dt)
sol = integration_esdirk.DIRK_integration(fun=fcn, y0=yini, t_span=[tini, tend], nt=nt,
                                    method=method, jacfun=None, newtonchoice=2,fullDebug=True)

### Plot the solution at the last step
u,v = sol.y[::nvar,:], sol.y[1::nvar,:]
fig, ax = plt.subplots(2,1,sharex=True)
ax[0].plot(u[:,-1])
ax[0].set_ylabel('u')
ax[1].plot(v[:,-1])
ax[1].set_ylabel('v')
ax[-1].set_xlabel('x')

### Plot the evolution of the Newton solver statistics, if available
for key in sol.infodict_hist.keys():
  plt.figure()
  if 'norm' in key:
    plt.semilogy(sol.infodict_hist[key])
  else:    
    plt.plot(sol.infodict_hist[key])
  plt.ylabel(key)
  plt.xlabel('time step number')
  plt.grid()
