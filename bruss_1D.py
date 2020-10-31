#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 21:19:17 2020

Implementation of the one-dimensional Brusselator, adapted from the 2-dimensional
Brusselator described in "Solving ORdinary Differential Equations II: stiff
and differenatial-algebraic equations" by E. Hairer and G. Wanner, page 151-152.

This constitutes a stiff PDE example. Sitffness increases as the diffusion
coefficient alpha increases, and as the number of mesh points increases.

@author: laurent
"""
import numpy as np

class brusselator_1d:
  def __init__(self,alpha=0.1, xmin=0.,xmax=1.,nx=100):
    self.alpha = alpha
    self.x = np.linspace(xmin, xmax,nx)
    self.nx = nx
    self.dx = self.x[1]-self.x[0]
    
  def fcn(self,t,y):
    """ returns the time derivatives of the discretised varibles u and v """
    u,v = y.reshape((2,self.nx),order='F')
    dover_dxx = 1/(self.dx)**2
    x = self.x
    nx = self.nx
    
    # compute second spatial derivatives with periodic BCs
    dxx_u = np.zeros_like(u)
    dxx_v = np.zeros_like(v)
    dxx_u[1:-1] = (u[:-2]-2*u[1:-1]+u[2:])*dover_dxx
    dxx_u[0] = (u[-1]-2*u[0]+u[1])*dover_dxx
    dxx_u[-1] = (u[-2]-2*u[-1]+u[0])*dover_dxx
       
    dxx_v[1:-1] = (v[:-2]-2*v[1:-1]+v[2:])*dover_dxx
    dxx_v[0] = (v[-1]-2*v[0]+v[1])*dover_dxx
    dxx_v[-1] = (v[-2]-2*v[-1]+v[0])*dover_dxx
    
    # compute source term
    f = 5.*( (x-0.3)**2 < 1e-2 )*(t>=1.1)
    
    dt_uv = np.zeros((2,nx),order='F')
    dt_uv[0,:] = 1. + u*u*v - 4.4*u + self.alpha*dxx_u + f #du/dt
    dt_uv[1,:] = 3.4*u - u**2*v + self.alpha*dxx_v # dv/dt
    dt_y = dt_uv.reshape((-1,), order='F')
    return dt_y
  
  def jac(self,t,y):
    raise Exception('TODO')
  
  def init(self):
    x = self.x
    u0 = 0.*x
    v0 = 27*x*(1-x)**1.5
    return np.vstack((u0,v0)).reshape((-1,), order='F')
  
  def postprocess(self):
     raise Exception('stop')
     
if __name__=='__main__':
  # Test of the model
  import matplotlib.pyplot as plt
  from scipy.integrate import solve_ivp
  prob = brusselator_1d(alpha=0.02,nx=128)
  sol = solve_ivp(fun=prob.fcn, y0=prob.init(), t_span=[0.,2.], t_eval=None,
                  method='Radau', atol=1e-4, rtol=1e-4, band=(4,4))
  u = sol.y[::2,:]
  v = sol.y[1::2,:]
  
  fig, ax = plt.subplots(2,1,sharex=True)
  for i in range(0,len(sol.t),5):
    ax[0].plot(prob.x, u[:,i])
    ax[1].plot(prob.x, v[:,i])
  ax[0].grid()
  ax[1].grid()
  ax[-1].set_xlabel('x')
  ax[0].set_ylabel('u')
  ax[1].set_ylabel('u')
    