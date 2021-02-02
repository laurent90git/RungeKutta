#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 10:44:08 2020

Model of the solid phase heating in 2D

@author: laurent
"""

import scipy.integrate
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

ncalls=0

def TfromX(X,options):
  """ Récupérer T (matrice) depuis le vecteur d'état """
  if X.size > options['solid']['mesh']['ny']*options['solid']['mesh']['nx']: #3D-array with time axis
    return X.reshape((options['solid']['mesh']['ny'],options['solid']['mesh']['nx'],-1))
  else:
    return X.reshape((options['solid']['mesh']['ny'],options['solid']['mesh']['nx'],))

def XfromT(T,options):
  """ Reformer le vecteur X à partir de la matrice de T"""
  if T.size > options['solid']['mesh']['ny']*options['solid']['mesh']['nx']: #3D-array with time axis
    return T.reshape((options['solid']['mesh']['ny']*options['solid']['mesh']['nx'], -1))
  else:
    return T.reshape((options['solid']['mesh']['ny']*options['solid']['mesh']['nx']))
  
def modelfun(t,z,options):
  # print(t)
  global ncalls
  ncalls = ncalls + 1
  # Reshape input for conveniency
  nx,ny = options['solid']['mesh']['nx'], options['solid']['mesh']['ny']
  T = TfromX(z,options) 
  # debug  
  if 0:
    fig = plt.figure()
    ax=plt.gca()
    cmap = plt.cm.get_cmap('hot')
    cs = ax.contourf(options['solid']['mesh']['x'],
               options['solid']['mesh']['y'],
               T,
               cmap=cmap, levels=np.linspace(np.min(T), np.max(T), 10))
    fig.colorbar(cs, ax=ax, shrink=0.9)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('T @ t={}'.format(t))
    raise Exception('test')
  
  xx,yy = options['solid']['mesh']['xx'], options['solid']['mesh']['yy']
  x,y = options['solid']['mesh']['x'], options['solid']['mesh']['y']
  dx = options['solid']['mesh']['dx']
  dy = options['solid']['mesh']['dy']
  
  # upper BC
  Ts_line = options['solid']['BCs']['surface']['T'] # Dirichlet BCs
  
  if isinstance(options['solid']['properties']['rho'], np.ndarray):
    rho = options['solid']['properties']['rho']
    lbda = options['solid']['properties']['lambda']
    cp = options['solid']['properties']['cp']
  elif isinstance(options['solid']['properties']['rho'], float):
      # si les propriétés sont uniformes
      rho = np.ones_like(xx)*options['solid']['properties']['rho']
      lbda = np.ones_like(xx)*options['solid']['properties']['lambda']
      cp = np.ones_like(xx)*options['solid']['properties']['cp']
  else:
    rho  = options['solid']['properties']['rho'](T)
    lbda = options['solid']['properties']['lambda'](T)
    cp   = options['solid']['properties']['cp'](T)
  
  # diffusive fluxes
  if 1:
    dxT  = np.gradient(T,x,axis=1, edge_order=1)
    dyT  = np.gradient(T,y,axis=0, edge_order=1)
    dxxT = np.gradient( dxT, x, axis=1, edge_order=1)
    dyyT = np.gradient( dyT, y, axis=0, edge_order=1)
    dx_lambda = np.gradient(lbda,x,axis=1, edge_order=1)
    dy_lambda = np.gradient(lbda,y,axis=0, edge_order=1)
  else: # debug version
    dyyT = np.zeros_like(T)
    dxxT = np.zeros_like(T)
    # non-uniform finite differences
    dxxT[:,1:-1] =( (T[:,:-2] - T[:,1:-1])/(xx[:,:-2] - xx[:,1:-1]) + \
                   -(T[:,1:-1] - T[:,2:])/(xx[:,1:-1] - xx[:,2:]) )/ (0.5*( (xx[:,:-2] - xx[:,1:-1]) + (xx[:,1:-1] - xx[:,2:]) ) )
      
    dyyT[1:-1,:] = (  (T[:-2, :] - T[1:-1,:])/(yy[:-2,:] - yy[1:-1,:]) + \
                     -(T[1:-1,:] - T[2:,  :])/(yy[1:-1,:] - yy[2:,:]) )/ (0.5*( (yy[:-2,:] - yy[1:-1,:]) + (yy[1:-1,:] - yy[2:,:]) ))
      
  # correct BCs
  # Dirichlet BC at the top
  dyT[0,:]  =  (Ts_line - T[0,:])/(dy[0])
  dyyT[0,:]  =  (Ts_line - 2*T[0,:] + T[1,:])/(dy[0]**2)
  
  # Neumann BC at the bottom
  dyT[-1,:]  =  0.
  dyyT[-1,:] = (T[-2,:] - 2*T[-1,:] + T[-1,:])/(dy[-1]**2) 
  
  # periodic BCs left and right
  dxT[:,0] = (T[:,1] - T[:,-1])/(2*dy[0])
  dxxT[:,0] = (T[:,-1] - 2*T[:,0] + T[:,1])/(dy[0]**2)
  dxT[:,-1] = (T[:,1] - T[:,-1])/(2*dy[0])
  dxxT[:,-1] = (T[:,-2] - 2*T[:,-1] + T[:,0])/(dy[-1]**2)
  
  dt_T = np.ones(xx.shape, dtype=z.dtype)
  dt_T[:,:] = (lbda*dxxT + lbda*dyyT + dx_lambda*dxT + dy_lambda*dyT)/(rho*cp)
  # works as long as lambda variations are not too strong
  return dt_T.reshape((nx*ny,))


if __name__=='__main__': # TEST RUN
  options = {
              'solid': {'mesh':{},
                       'properties':{'rho':lambda T: 1. + 0*T, # density
                                     'cp': lambda T: 1. + 0*T, # specific heat
                                     'lambda': lambda T: 1. + 1*T + 2*T**2}, # thermal conductivity
                       'BCs':{'surface': {'T':None}}
                       }
             }

  nx=15
  ny=16
  options['solid']['mesh']['nx'] = nx
  options['solid']['mesh']['ny'] = ny
  options['solid']['mesh']['x'] = np.linspace(-1,1,nx)
  options['solid']['mesh']['y'] = np.linspace(1,-1,ny)
  
  options['solid']['mesh']['dx'] = np.diff(options['solid']['mesh']['x'])
  options['solid']['mesh']['dy'] = np.diff(options['solid']['mesh']['y'])
  
  options['solid']['mesh']['xx'], options['solid']['mesh']['yy'] = np.meshgrid(options['solid']['mesh']['x'], options['solid']['mesh']['y'])
  
  # champ initial de température
  # T_init = (1+options['solid']['mesh']['yy'])**2 # simple parabolic condition
  # T_init = T_init/np.max(T_init)
  # T_init = options['solid']['mesh']['yy']
  T_init = 10*np.exp( -5*( (options['solid']['mesh']['yy'])**2 + (options['solid']['mesh']['xx'])**2 ) )
  # T_init = np.exp( -5*( options['solid']['mesh']['xx'])**2 ) 
  # T_init = np.exp( -5*( options['solid']['mesh']['yy'])**2 ) 
  # T_init = 0.*options['solid']['mesh']['yy']

  
  # options['solid']['BCs']['surface']['T'] = np.ones((nx,))
  options['solid']['BCs']['surface']['T'] = options['solid']['mesh']['x']**2

  
  y0 = XfromT(T_init, options) # vecteur d'état initial
  
  #%%
  if 1: # analyse Jacobian and determine sparsity pattern
   # vérification des jacobiennes
    import scipy.optimize._numdiff
    # version naive
    ncalls=0
    Jac_objfun_full = scipy.optimize._numdiff.approx_derivative(fun=lambda x: modelfun(t=0.,z=x,options=options), x0=y0, method='2-point',
                                                           rel_step=1e-8, f0=None,
                                                           bounds=(-np.inf, np.inf), sparsity=None,
                                                           as_linear_operator=False, args=(), kwargs={})
    print('naive calls = {}'.format(ncalls))
    plt.figure()
    plt.spy(Jac_objfun_full)

    # version intelligente
    # on donne la Jacobienne full à scipy, qui détermine comment grouper les perturbations de manière optimale
    # pour pouvoir faire plusieurs perturbations d'un coup lors de la détermination de la Jacobienne par
    # différences finies
    # g = scipy.optimize._numdiff.group_columns(Jac_objfun_full, order=0)
    jacSparsity = 1*(Jac_objfun_full!=0.)
    ncalls = 0
    
    # On peut aussi générer une fonction qui utilise ces perturbations groupées pour déterminer la Jacobienne
    jacfun = lambda t,x: scipy.optimize._numdiff.approx_derivative(fun=lambda y: modelfun(t=t,z=y,options=options), x0=x, method='2-point',
                                                            rel_step=1e-8, f0=None,
                                                            bounds=(-np.inf, np.inf), sparsity=jacSparsity,
                                                            as_linear_operator=False, args=(), kwargs={}).toarray()
    Jac_grouped = jacfun(0.,y0)
    print('grouped calls = {}'.format(ncalls))

    
    assert np.allclose(Jac_grouped,Jac_objfun_full), 'scipy banded jacobian estimation method is not working properly...'

  #%%
  ##### INTEGRATION TEMPORELLE
  tf = 1e-2
  if 0: # BDF en utilisant la Jacobienne intelligente
    # bon ici le problème est linéaire donc ça n'est pas très utile, sauf si on fait varier les propriétés avec T
    out = scipy.integrate.solve_ivp(fun=lambda t,x: modelfun(t=t,z=x,options=options),
                          t_span=[0.,tf], y0=y0,
                          method='Radau', first_step=1e-3, max_step=np.inf,
                          rtol=1e-3, atol=1e-6,
                          t_eval=None, #np.linspace(0,tf,5),
                          dense_output=False,
                          # jac_sparsity=jacSparsity,
                          jac=jacfun,
                          events=None, vectorized=False, args=None)
  else:
    from integration_esdirk import FIRK_integration, DIRK_integration, ERK_integration
    import rk_coeffs
    method='Radau5'; mod ='FIRK'
    # method='IE'; mod ='DIRK'
    # method='L-SDIRK-33'; mod ='DIRK'
    nt=100
    A,b,c = rk_coeffs.getButcher(name=method)
    if mod=='DIRK': # DIRK solve
        out = DIRK_integration(fun=lambda t,x: modelfun(t=t,z=x,options=options),
                               y0=y0, t_span=[0., tf], nt=nt, A=A, b=b, c=c, jacfun=jacfun)
    elif mod=='FIRK': # FIRK solve
        out = FIRK_integration(fun=lambda t,x: modelfun(t=t,z=x,options=options),
                               y0=y0, t_span=[0., tf], nt=nt, A=A, b=b, c=c, jacfun=jacfun)
    elif mod=='ERK': # FIRK solve
        out = ERK_integration(fun=lambda t,x: modelfun(t=t,z=x,options=options),
                               y0=y0, t_span=[0., tf], nt=nt, A=A, b=b, c=c, jacfun=jacfun)
  ##### POST-PROCESSING
  Tfield = TfromX(out['y'], options) #out['y'].reshape((ny,nx,-1))
  cmap = plt.cm.get_cmap('jet')#('hot')
  Tmax = np.max(Tfield)
  Tmin = np.min(Tfield)
  for nt in range(0,out['t'].size,5):
    plt.show()
    fig = plt.figure()
    ax=plt.gca()
    cs = ax.contourf(options['solid']['mesh']['x'],
               options['solid']['mesh']['y'],
               Tfield[:,:,nt],
               cmap=cmap, levels=np.linspace(Tmin, Tmax, 100))
    fig.colorbar(cs, ax=ax, shrink=0.9)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('T @ t={}'.format(out['t'][nt]))
    
  print( modelfun(t=out['t'][-1], z=out['y'][:,-1], options=options))
