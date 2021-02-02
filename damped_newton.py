# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 21:38:59 2020

Python implementation of an exact damped Newton scheme, with LU factorsation of the Jacobian

@author: Laurent
"""
import scipy.linalg
import scipy.optimize
import numpy as np
import scipy.optimize._numdiff

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import csc_matrix, issparse, eye
from scipy.sparse.linalg import splu
from scipy.optimize._numdiff import group_columns
from scipy.integrate._ivp.common import (validate_max_step, validate_tol, select_initial_step,
                     norm, num_jac, EPS, warn_extraneous,
                     validate_first_step)

ncalls = 0
jac_factor = None

              
def _validate_jac(y0,fun_vectorized,atol=1e-8,jac=None,sparsity=None):
  """ Taken from Scipy's Radau implementation. Returns a validated jacobian estimation function,
  using if possible the sparsity pattern of the Jacobian to optimize the computation"""
  n=y0.size
  f = fun_vectorized(y0)
  global jac_factor
  if not (jac_factor is None):
    if jac_factor.size!=y0.size: # happens if we use twice the damped_newton on different size problems...
      jac_factor = None

  if jac is None:
      if sparsity is not None:
          if issparse(sparsity):
              sparsity = csc_matrix(sparsity)
          groups = group_columns(sparsity)
          sparsity = (sparsity, groups)

      def jac_wrapped(y):
          f = fun_vectorized(y)
          t=None # Scipy's numjac method assumes time is a separate variable
          global jac_factor # TODO: object oriented implementation of the Newton solver ?
          J, jac_factor = num_jac(fun=lambda t,y: fun_vectorized(y), t=t, y=y, f=f, threshold=atol,
                                  factor=jac_factor, sparsity=sparsity)
          return J
      J = jac_wrapped(y0)
  elif callable(jac):
      J = jac(y0)
      if issparse(J):
          J = csc_matrix(J)
          def jac_wrapped(y):
              return csc_matrix(jac(y), dtype=float)
      else:
          J = np.asarray(J, dtype=float)
          def jac_wrapped(y):
              return np.asarray(jac(y), dtype=float)

      if J.shape != (n, n):
          raise ValueError("`jac` is expected to have shape {}, but "
                           "actually has {}."
                           .format((n,n), J.shape))
  else:
      if issparse(jac):
          J = csc_matrix(jac)
      else:
          J = np.asarray(jac, dtype=float)

      if J.shape != (n, n):
          raise ValueError("`jac` is expected to have shape {}, but "
                           "actually has {}."
                           .format((n, n), J.shape))
      jac_wrapped = None

  return jac_wrapped, J

def damped_newton_solve(fun, x0, rtol=1e-9, ftol=1e-30, jacfun=None, warm_start_dict=None,
                        itmax=30, jacmax=10, tau_min=1e-4, convergenceMode=0, jac_sparsity=None,
                        vectorized=False,     bPrint=False, bStoreIterates=False):
    if bPrint:
      custom_print = print
    else:
      def custom_print(*args, end=''):
        pass

    njev=0
    nlu=0


    # recover previously computed Jacobian, Lu decomposition, jacobian estimation method
    if warm_start_dict:
        jac   = warm_start_dict['jac']
        LUdec = warm_start_dict['LU']
        jac_estimator = warm_start_dict['jacfun']
    else:
        jac = None
        LUdec = None
        jac_estimator = None

    if vectorized:
        fun_vectorized = fun
    else:
        def fun_vectorized(y):
          if y.ndim==2:
            f = np.empty_like(y)
            for i, yi in enumerate(y.T):
                f[:, i] = fun(yi)
            return f
          else:
            return fun(y)

    if jacfun:
        jac_estimator = jacfun
        if jac is None:
          jac = jac_estimator(x0)
          njev+=1
    elif jac_estimator is None:
        print('analyzing jacobian sparisty pattern')
        global ncalls
        def tempfun(x):
          global ncalls
          ncalls+=1
          return fun_vectorized(x)
        # perttype = 'cs'; rel_step=1e-50
        perttype = '3-point'; rel_step=1e-8
        # perttype = '2-point'; rel_step=1e-8

        if jac_sparsity is None:
          Jac_full = scipy.optimize._numdiff.approx_derivative(fun=tempfun, x0=x0, method=perttype,
                                                               rel_step=rel_step, f0=None, bounds=(-np.inf, np.inf), sparsity=None,
                                                               as_linear_operator=False, args=(), kwargs={})
          jac_sparsity = 1*(Jac_full!=0.)
        ncalls = 0
        
        jac_estimator, jac = _validate_jac(y0=x0,fun_vectorized=fun_vectorized,atol=1e-8,jac=None,sparsity=jac_sparsity)
        njev+=1

    if issparse(jac):
        def lu(A):
            return splu(A)

        def solve_lu(LU, b):
            return LU.solve(b)
    else:
        def lu(A):
            return lu_factor(A)#, overwrite_a=True)

        def solve_lu(LU, b):
            return lu_solve(LU, b)#, overwrite_b=True)


    # initiate computation
    tau=1.0 # initial damping factor #TODO: warm start information ?
    convergenceMode = 0
    NITER_MAX = itmax
    NJAC_MAX = jacmax
    TAU_MIN  = tau_min
    GOOD_CONVERGENCE_RATIO_STEP = 0.5
    GOOD_CONVERGENCE_RATIO_RES  = 0.5
    NORM_ORD = None

    nx = x0.size
    

    tau_hist=[]
    x_hist=[]

    x=np.copy(x0)
    if bStoreIterates:
      x_hist.append( np.copy(x) )

    if LUdec is None:
        if jac is None:
            jac = jac_estimator(x)
            njev += 1
        LUdec = lu(jac)
        nlu  += 1

    res = fun(x) # initial residuals
    nfev=1

    dx = solve_lu(LUdec, res)
    nLUsolve=1
    dx_norm = np.linalg.norm(dx, ord=NORM_ORD)
    res_norm = np.linalg.norm(res, ord=NORM_ORD)
    niter=0
    bSuccess=False
    bUpdateJacobian = False
    bFailed=False # true if exceeded iter / jac limit
    bJacobianAlreadyUpdated = True # True if the last jacobian was compouted for the current value of x
                                   # (ie if we ask to recompute it for the same x, there is a problem)
    while True: # cycle until convergence
        bAcceptedStep=False
        if bUpdateJacobian:
            custom_print('\tupdating Jacobian')
            if bJacobianAlreadyUpdated:
              custom_print('\t/!\ the jacobian has already been computed for this value of x --> convergence seems impossible...')
              # TODO: scaled norm of ||dx|| ?
              if 0: # just fail
                bFailed = True
              else: # accept one step, no matter the increase in ||dx|| or ||res||
                print('\t    --> forcing one step and retrying')
                x = x - dx
                res = fun(x) # initial residuals
                nfev += 1
                res_norm = np.linalg.norm(res, ord=NORM_ORD)
                dx  = solve_lu(LUdec, res)
                nLUsolve+=1
                dx_norm = np.linalg.norm(dx, ord=NORM_ORD)
            elif njev > NJAC_MAX:
                bFailed = True
                custom_print('too many jacobian evaluations')
            else: # update jacobian and dx
              jac = jac_estimator(x)
              bUpdateJacobian = False
              bJacobianAlreadyUpdated=True
              tau = 1. # TODO: bof ?
              njev += 1
              LUdec = lu(jac)
              nlu  += 1

              # res = fun(x)
              dx  = solve_lu(LUdec, res)
              nLUsolve+=1
              dx_norm = np.linalg.norm(dx, ord=NORM_ORD)
              # res_norm = np.linalg.norm(res, ord=NORM_ORD)

        niter+=1
        if niter > NITER_MAX:
            bFailed = True
            custom_print('too many iterations')

        if not bFailed:
          custom_print('\nstarting new iteration with ||res||={:.3e}, ||dx||={:.3e}'.format(res_norm, dx_norm))
          new_x = x - tau*dx
          new_res = fun(new_x)
          nfev+=1
          if convergenceMode==0: # the newton step norm must decrease
              new_dx  = solve_lu(LUdec, new_res)
              nLUsolve+=1
              new_dx_norm = np.linalg.norm(new_dx, ord=NORM_ORD)
              new_res_norm = np.linalg.norm(new_res, ord=NORM_ORD)
              custom_print('\t ||new dx||={:.3e}, ||new res||={:.3e}'.format(new_dx_norm, new_res_norm))
              if new_dx_norm < dx_norm:
                  custom_print('\tstep decrease is satisfying ({:.3e}-->{:.3e} with damping={:.3e})'.format(dx_norm, new_dx_norm, tau))
                  bAcceptedStep=True
                  if new_dx_norm/dx_norm > GOOD_CONVERGENCE_RATIO_STEP:
                      custom_print('slow ||dx|| convergence (ratio is {:.2e})) --> asking for jac udpate'.format(new_dx_norm/dx_norm))
                      bUpdateJacobian = True
          elif convergenceMode==1: # the residual vector norm must decrease
              new_res_norm = np.linalg.norm(new_res, ord=NORM_ORD)
              new_dx, new_dx_norm = None, None
              custom_print('\t ||new res||={:.3e}'.format(new_res_norm))
              if new_res_norm < res_norm:
                  custom_print('\tresidual decrease is satisfying')
                  bAcceptedStep=True
                  if new_res_norm/res_norm > GOOD_CONVERGENCE_RATIO_RES:
                      custom_print('slow ||res|| convergence (ratio is {:.2e})) --> asking for jac udpate'.format(new_res_norm/res_norm))
                      bUpdateJacobian = True
          else:
              raise Exception('\tconvergence mode {} not implemented'.format(convergenceMode))


        if bAcceptedStep:
            custom_print('\tdamped step accepted (tau={:.3e})'.format(tau))
            x = new_x
            bJacobianAlreadyUpdated=False # we have not yet computed the Jacobian for the new value of x
            if bStoreIterates:
              x_hist.append(np.copy(x))
            tau_hist.append(tau)
            if new_dx is None: # if it has not yet been updated (e.g when using a residual norm criterion)
              new_dx  = solve_lu(LUdec, new_res)
              nLUsolve+=1

            res = fun(x)
            nfev+=1
            dx  = solve_lu(LUdec, res)
            nLUsolve+=1
            dx_norm = np.linalg.norm(dx, ord=NORM_ORD)
            res_norm = np.linalg.norm(res, ord=NORM_ORD)


            if tau<0.3:
              custom_print('\t successful damping is considered too small --> jacobian will be updated')

            # TODO: Rank-1 update of the Jacobian ?
            if 0:# tau==1. and not bUpdateJacobian: # no damping was applied, the Jacobian can be correctly updated
              # adapted from equation (1.60) of "A family of Newton Codes for Systems of Highly Nonlinear Equations" by Nowak and Deuflhard
              # --> seems to be useless in my test cases
              if dx_norm>1e-15:
                custom_print('\t rank-1 update of the Jacobian matrix')
                jac = jac + np.outer(res,dx)/dx_norm
                LUdec = lu(jac)
                nlu  += 1

              bUpdateJacobian=True
            # TODO: increase tau if convergence rate was good ?
            tau = 1. #min((tau*10., 1.))



            if res_norm < ftol:
                custom_print('residual norm has converged')
                bSuccess=True
            if dx_norm < rtol: # TODO: scaled norm ?
                custom_print('step norm has converged')
                bSuccess=True
        elif not bFailed:
            # the step is rejected, we must lower the damping or update the Jacobian
            # TODO: better damping strategy
            tau = min(tau*tau, 0.5*tau)
            custom_print('\tstep rejected: reducing damping to {:.3e}'.format(tau))
            if tau < TAU_MIN: # damping is too small, indicating convergence issues
                custom_print('\tdamping is too low')
                bUpdateJacobian = True
        if bSuccess or bFailed:
            if bFailed:
              custom_print('Failed after ', end='')
              ier=1
              if issparse(jac):
                temp=jac.toarray()
              else:
                temp=jac
              nrank=np.linalg.matrix_rank(temp)
              ndeficiency = nx-nrank
              msg='failed (Jac_rank={} (deficiency: {}), Jac_cond={:.2e})'.format(nrank, ndeficiency, np.linalg.cond(temp))
            else:
              custom_print('Success after ', end='')
              ier=0
              msg='success'
            custom_print(' {} iters, with {} jac update, {} LU-factorisations'.format(niter, njev, nlu))
            custom_print(msg)
            return x, \
                  {'niter': niter, 'njev': njev, 'nfev': nfev, 'nLUsolve':nLUsolve,
                   'nLUdec': nlu, 'dx_norm': dx_norm, 'res_norm': res_norm,
                   'tau_hist': np.array(tau_hist), 'x_hist': np.array(x_hist).T,
                   'msg':msg, 'ier': ier}, \
                  {'jac': jac, 'LU':LUdec, 'jacfun': jac_estimator}

if __name__=='__main__':
    import matplotlib.pyplot as plt
    print('=== testing the damped Newton solver ===\n')
    nprob=0 # choice of the test problem

    if nprob==0: # non-linear problem, with potential divergence if undamped
      def funplot(x):
          return np.array((np.arctan(x[0,:]), x[1,:]**3 + 0.6*x[1,:]))
      fun = lambda x: np.array((np.arctan(x[0]), x[1]**3 + 0.6*x[1]))
      jacfun = lambda x: np.array( [[1/(1+x[0]**2), 0.], [0., 0.3+2*x[1]]] )
      x0 = np.array((0.5,0.5))

      # visualize the two residual function components
      plt.figure()
      xtest = np.linspace(-1,1,1000)
      xtest = np.vstack((xtest, xtest))
      plt.plot(xtest[0,:], funplot(xtest)[0,:], color='tab:orange', label='f1')
      plt.axvline(x0[0], color='tab:orange', label=None)
      plt.plot(xtest[0,:], funplot(xtest)[1,:], color='tab:blue', label='f2')
      plt.axvline(x0[1], color='tab:blue', label=None)
      plt.axhline(0., color=[0,0,0], label=None)

      plt.scatter(x0, fun(x0), marker='x', color='r', label=None)
      plt.legend()
      plt.xlabel('x')
      plt.ylabel('f(x)')
      plt.grid()

    elif nprob==1: ## quadratic problem - 1 var
      fun = lambda x: 2*x**2 + 0.5*x + 0.1
      jacfun = lambda x: np.array([0.1*2*x + 0.5,])
      x0 = np.array((0.5,))

      plt.figure()
      xtest = np.linspace(-2,2,1000)
      plt.plot(xtest, fun(xtest))
      plt.scatter(x0, fun(x0), marker='x', color='r')
      plt.xlabel('x')
      plt.ylabel('f(x)')
      plt.grid()

    elif nprob==2: # quadratic problem - 2 vars (ellipse)
      aa = 4e3
      bb = 0. #1e3
      fun =    lambda x: np.array( (aa*x[0]**2, x[1]**2))+ bb*x
      jacfun = lambda x: np.array( [ [aa*2*x[0], 0.], [0., 2*x[1]] ] ) + bb*np.eye(x.size)
      x0 = np.array((0.5,0.5))
      # this problem hightlights the affine invariance property of the Newton method:
      # deforming the ellipsis by increasing "a" is not affecting the Newton path and convergence
    elif nprob==3: # linear problem
      fun = lambda x: x
      jacfun = lambda x: np.eye(x.size)
      x0 = np.array((0.5,0.5))

    jacfun=None
    root, infodict, warm_start_dict = damped_newton_solve(fun=fun, x0=x0, rtol=1e-8, ftol=1e-30,
                                                          jacfun=jacfun, warm_start_dict=None, bPrint=True,
                                                          bStoreIterates=True)
    # root, converged, zero_der = scipy.optimize.newton(func=fun, x0=x0,
    #                                                         rtol=1e-8, tol=1e-15,  maxiter=100,
    #                                                         full_output=True,  fprime=None, fprime2=None)

    plt.figure()
    plt.scatter(x=x0[0],y=x0[1], marker='o', color='r')
    plt.plot(infodict['x_hist'][0,:],  infodict['x_hist'][1,:], marker='+', color='b')
    plt.plot(infodict['x_hist'][0,-1], infodict['x_hist'][1,-1], marker='o', color='g')
    plt.xlabel('x1')
    plt.xlabel('x2')
    plt.grid()
    plt.title('Newton path')

    plt.figure()
    plt.plot(range(len(infodict['tau_hist'])), infodict['tau_hist'], marker='.')
    plt.grid()
    plt.xlabel('niter')
    plt.ylabel('damping')
