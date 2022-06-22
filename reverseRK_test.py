# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 18:49:12 2022

@author: lfrancoi
"""

import numpy as np
import numpy.matlib
import ctypes as ct
from scipy.optimize import fsolve
import copy
import scipy.integrate
import scipy.interpolate
from damped_newton import damped_newton_solve
import newton

NEWTONCHOICE=2 #0: Scipy's fsolve 1: Scipy's newton (bad !), 2: custom damped Newton, 3: custom undamped quasi-Newton (weirdly works better than the damped one...)

def inverseERKintegration(fun, y0, t_span, nt, method, jacfun=None, bPrint=True,
                          vectorized=False, fullDebug=False):
    """ Performs the integration of the system dy/dt = f(t,y)*
        with a reversed explicit RK method
        from t=t_span[0] to t_span[1], with initial condition y(t_span[0])=y0.
        The RK method described by A,b,c is an explicit method (which we reverse)
        - fun      :  (function handle) model function (time derivative of y)
        - y0     :  (1D-array)        initial condition
        - t_span :  (1D-array)        array of 2 values (start and end times)
        - nt     :  (integer)         number of time steps
        - A      :  (2D-array)        Butcher table of the chosen RK method
        - b      :  (1D-array)        weightings for the quadrature formula of the RK methods
        - c      :  (1D-array)        RK substeps time
        """
    assert y0.ndim==1, 'y0 must be 0D or 1D'

    out = scipy.integrate._ivp.ivp.OdeResult()
    t = np.linspace(t_span[0], t_span[1], nt)

    A,b,c = method['A'], method['b'], method['c'] # Butcher coefficients

    n = y0.size # size of the problem
    dt = (t_span[1]-t_span[0]) / (nt-1) # time step
    s = np.size(b) # number of stages for the RK method

    
    y = np.zeros((n, nt)) # solution accros all time steps
    y[:,0] = y0
    
    ## make sure the function is vectorised
    if vectorized:
        fun_vectorised = fun
    else:
        def fun_vectorised(t,x):
            """ Vectorize a non-vectorized function """
            if x.ndim==1:
                return fun(t,x)
            else:
                if not (isinstance(t, np.ndarray)):
                    t = t*np.ones((x.shape[1]))
                res0 = fun(t[0],x[:,0])
                res = np.zeros((res0.shape[0],x.shape[1]))
                res[:,0]=res0
                for i in range(1,x.shape[1]):
                    res[:,i] = fun(t[i],x[:,i])
                return res
    
    ## define the residual function
    def resfun(ynp1,yn,tn,dt,A,n,s, return_substeps=False):
        """ Residuals for the substeps.
        The input is Y = (y0[...], y1[...], ...).T """
        # 1 - compute reverse stages explicitly
        Y = np.zeros((n,s), order='F')
        for i in range(s):
            Y[:,i] = ynp1
            for j in range(i): # explicit RK reversed
                Y[:,i] = Y[:,i] - dt*A[i,j]*fun(tn+(1-c[j])*dt, Y[:,j])
        # TODO use k's instead
        # 2 - compute the reversed yn
        # TODO: on est toujours stiffly accurate, non ?
        ynrev = ynp1
        for i in range(s):
            ynrev = ynrev - dt*b[i]*fun(tn+(1-c[i])*dt, Y[:,i])
        # ynrev = Y[:,-1]
        print('ynrev=',ynrev)
        # 2 - compute residuals as a matrix (one row for each step)
        res = ynrev - yn
        if return_substeps:
            return res, Y
        else:
            return res
    
    ## skirmish
    # bStifflyAccurate = np.all(b==A[-1,:]) # then the last stage is the solution at the next time point
    
    ## advance in time
    out.nfev = 0
    out.njev = 0
    K= np.zeros((n, s), order='F')
    unm1 = np.copy(y0)
    # At = A.T
    warm_start_dict = None
    infodict_hist = {} # additonal optional debug storage
    out.infodict_hist = infodict_hist
    for it, tn in enumerate(t[:-1]):
        if bPrint:
          if np.mod(it,np.floor(nt/10))==0:
              print('\n{:.1f} %'.format(100*it/nt), end='')
          if np.mod(it,np.floor(nt/100))==0:
              print('.', end='')
    
    
         # solve the complete non-linear system via a Newton method
        yini = unm1[:]
        ynp1, infodict, warm_start_dict = damped_newton_solve(
                                            fun=lambda x: resfun(ynp1=x,yn=unm1, tn=tn, dt=dt, A=A, n=n, s=s),
                                            x0=np.copy(yini), rtol=1e-9, ftol=1e-30,
                                            jacfun=None, warm_start_dict=warm_start_dict,
                                            itmax=100, jacmax=20, tau_min=1e-4, convergenceMode=0,
                                            bPrint=True)
        out.nfev += infodict['nfev']
        out.njev += infodict['njev']
        if infodict['ier']!=0: # Newton did not converge
          # restart the Newton solve with all outputs enabled
          ynp1, infodict, warm_start_dict = damped_newton_solve(fun=lambda x: resfun(Y=x,y0=unm1, tn=tn, dt=dt, A=At, n=n, s=s),
                                                                    x0=np.copy(yini), rtol=1e-9, ftol=1e-30,
                                                                    jacfun=None, warm_start_dict=warm_start_dict,
                                                                    itmax=100, jacmax=20, tau_min=1e-4, convergenceMode=0, bPrint=True)
          msg = 'Newton did not converge'
          # raise Exception(msg)
          out.y = y[:,:it+1]
          out.t = t[:it+1]
          out.message = msg
          return out
      
        if fullDebug: # store additional informations about the Newton solve
          if not bool(infodict_hist): # the debug dictionnary has not yet been initialised
            for key in infodict.keys():
              if not isinstance(infodict[key], np.ndarray) and (key!='ier' and key!='msg'): # only save single values
                infodict_hist[key] = [infodict[key]]
          else: # backup already initialised
            for key in infodict_hist.keys():
              infodict_hist[key].append(infodict[key])
           
        y[:,it+1] = ynp1
        unm1=ynp1
    
    # END OF INTEGRATION
    out.y = y
    out.t = t
    # out.y_substeps = y_substeps # last substeps
    return out
  
  
  
  
if __name__=='__main__':
    import matplotlib.pyplot as plt
    import rk_coeffs
    
    #%% Single method test
    #### PARAMETERS ####
    problemtype = 'non-stiff'
    
    NEWTONCHOICE=3
    # name= 'rk4'
    name='EE'
    # name='RK45'
    # name='Heun-Euler'
    
    ### Setup and solve the problem
    if problemtype=='non-stiff': # ODE
        print('Testing time integration routines with mass-spring system')
        k_sur_m = 33.
        Amod = np.array( ( (0,1),(-k_sur_m, 0) ))
        def modelfun(t,x,options={}):
            """ Mass-spring system"""
            Xdot = np.dot(Amod, x)
            return Xdot

        y0 = np.array((0.3,1))
        tf = 2.0
        nt = 30
    elif problemtype=='stiff': #  Hirschfelder-Curtiss
        print('Testing time integration routines with Hirschfelder-Curtiss stiff equation')
        k=10.
        def modelfun(t,x):
            """ Mass-spring system"""
            return -(k*x-np.sin(t)    )
        y0 = np.array((0.3,1))
        tf = 5.0
        nt = 30

    method = rk_coeffs.getButcher(name=name)
    
    sol = inverseERKintegration(fun=modelfun, y0=y0, t_span=[0,tf], nt=nt,
                                method=method, jacfun=None, bPrint=True,
                                fullDebug=True)
    plt.figure()
    plt.semilogy(sol.t[:-1], np.diff(sol.t))
    plt.grid()
    plt.xlabel('t (s)')
    plt.ylabel('dt (s)')

    sol_ref = scipy.integrate.solve_ivp(fun=modelfun, t_span=[0., tf], y0=y0, method='DOP853', first_step=1e-2,
                                    atol=1e-13, rtol=1e-13)

    plt.figure()
    plt.plot(sol.t, sol.y[0,:], label='position', marker='x')
    plt.plot(sol.t, sol.y[1,:], label='vitesse', marker='x')
    plt.plot(sol_ref.t, sol_ref.y[0,:], label='position ref', linestyle='--', marker=None, markevery=1)
    plt.plot(sol_ref.t, sol_ref.y[1,:], label='vitesse ref', linestyle='--', marker=None, markevery=1)
    plt.title('Solutions')
    plt.legend()
    plt.xlabel('time (s)')
    plt.ylabel('position')
    plt.show()