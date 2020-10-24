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

ncalls = 0

def damped_newton_solve(fun, x0, rtol=1e-9, ftol=1e-30, jacfun=None, warm_start_dict=None,
                        itmax=30, jacmax=10, tau_min=1e-4, convergenceMode=0, jac_sparsity=None,
                        bPrint=False):
    if bPrint:
      custom_print = print
    else:
      def custom_print(*args, end=''):
        pass  
  
    # recover previously computed Jacobian, Lu decomposition, jacobian estimation method
    if warm_start_dict:
        jac   = warm_start_dict['jac']
        LUdec = warm_start_dict['LU']
        jac_estimator = warm_start_dict['jacfun']
    else:
        jac = None
        LUdec = None
        jac_estimator = None
    if jacfun:
        jac_estimator = jacfun
    elif jac_estimator is None:
        custom_print('analyzing jacobian sparisty pattern')
        global ncalls
        def tempfun(x):
          global ncalls
          ncalls+=1
          return fun(x)
        if jac_sparsity is None:
          Jac_full = scipy.optimize._numdiff.approx_derivative(fun=tempfun, x0=x0, method='2-point',
                                                               rel_step=1e-8, f0=None, bounds=(-np.inf, np.inf), sparsity=None,
                                                               as_linear_operator=False, args=(), kwargs={})
          ncalls_naive = ncalls
          # assert ncalls_naive==x0.size+1, 'ncalls_naive={}, but x0.size={}'.format(ncalls_naive, x0.size)
          jac_sparsity = 1*(Jac_full!=0.)
        ncalls = 0
        jac_sparsity = None # TODO: fPOURQUOI CA NE MARCHE PAS SI ON UTILISE LE SPARSITY PATTERN...
        # g = scipy.optimize._numdiff.group_columns(Jac_full, order=0)
        jac_estimator = lambda x: scipy.optimize._numdiff.approx_derivative(fun=fun, x0=x, method='2-point',
                                                                rel_step=1e-8, f0=None, sparsity=jac_sparsity,
                                                                as_linear_operator=False, args=(), kwargs={})#.toarray()
        Jac_grouped = jac_estimator((x0))
        
        ncalls_grouped = ncalls
        custom_print('Sparsity pattern allows for {} calls instead of {}'.format(ncalls_grouped, ncalls_naive))

        assert np.all(Jac_grouped==Jac_full)

    # initiate computation
    tau=1.0 # initial damping factor #TODO: warm start information ?
    convergenceMode = 0
    NITER_MAX = itmax
    NJAC_MAX = jacmax
    TAU_MIN  = tau_min
    GOOD_CONVERGENCE_RATIO_STEP = 0.1
    GOOD_CONVERGENCE_RATIO_RES  = 0.5

    njev=0
    nlu=0
    tau_hist=[]
    x_hist=[]

    x=np.copy(x0)
    if x.size<5:
      x_hist.append( np.copy(x) )
      
    if LUdec is None:
        if jac is None:
            jac = jac_estimator(x=x)
            njev += 1
        LUdec = scipy.linalg.lu_factor(jac)
        nlu  += 1

    res = fun(x) # initial residuals
    nfev=1
    dx = scipy.linalg.lu_solve(LUdec, res)
    nLUsolve=1
    dx_norm = np.linalg.norm(dx)
    res_norm = np.linalg.norm(dx)
    niter=0
    bSuccess=False
    bUpdateJacobian = False
    bFailed=False # true if exceeded iter / jac limit
    while True: # cycle until convergence
        bAcceptedStep=False
        if bUpdateJacobian:
            custom_print('\tupdating Jacobian')
            if njev > NJAC_MAX:
                bFailed = True
                custom_print('too many jacobian evaluations')
            else:
              jac = jac_estimator(x=x)
              bUpdateJacobian = False
              tau = 1. # TODO: bof ?
              njev += 1
              LUdec = scipy.linalg.lu_factor(jac)
              nlu  += 1

              # res = fun(x)
              dx  = scipy.linalg.lu_solve(LUdec, res)
              nLUsolve+=1
              dx_norm = np.linalg.norm(dx)
              res_norm = np.linalg.norm(res)

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
              new_dx  = scipy.linalg.lu_solve(LUdec, new_res)
              nLUsolve+=1
              new_dx_norm = np.linalg.norm(new_dx)
              new_res_norm = np.linalg.norm(new_res)
              custom_print('\t ||new dx||={:.3e}'.format(new_dx_norm))
              if new_dx_norm < dx_norm:
                  custom_print('\tstep decrease is satisfying ({:.3e}-->{:.3e} with damping={:.3e})'.format(dx_norm, new_dx_norm, tau))
                  bAcceptedStep=True
                  if new_dx_norm/dx_norm > GOOD_CONVERGENCE_RATIO_STEP:
                      custom_print('slow ||dx|| convergence (ratio is {:.2e})) --> asking for jac udpate'.format(new_dx_norm/dx_norm))
                      bUpdateJacobian = True
          elif convergenceMode==1: # the residual vector norm must decrease
              new_res_norm = np.linalg.norm(new_res)
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
            if x.size<5:
              x_hist.append(np.copy(x))
            tau_hist.append(tau)
            if new_dx is None: # if it has not yet been updated (e.g when using a residual norm criterion)
              new_dx  = scipy.linalg.lu_solve(LUdec, new_res)
              nLUsolve+=1

            res = fun(x)
            nfev+=1
            dx  = scipy.linalg.lu_solve(LUdec, res)
            nLUsolve+=1
            dx_norm = np.linalg.norm(dx)
            res_norm = np.linalg.norm(res)

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
            tau = 0.33*tau
            custom_print('\tstep rejected: reducing damping to {:.3e}'.format(tau))
            if tau < TAU_MIN: # damping is too small, indicating convergence issues
                custom_print('\tdamping is too low')
                bUpdateJacobian = True
        if bSuccess or bFailed:
            if bFailed:
              custom_print('Failed after ', end='')
              ier=1
              msg='failed'
            else:
              custom_print('Success after ', end='')
              ier=0
              msg='success'
            custom_print(' {} iters, with {} jac update, {} LU-factorisations'.format(niter, njev, nlu))
            return x, \
                  {'niter': niter, 'njev': njev, 'nfev': nfev, 'nLUsolve':nLUsolve,
                   'nLUdec': nlu, 'dx_norm': dx_norm, 'res_norm': res_norm,
                   'tau_hist': np.array(tau_hist), 'x_hist': np.array(x_hist).T,
                   'msg':msg, 'ier': ier}, \
                  {'jac': jac, 'LU':LUdec, 'jacfun': jac_estimator}

if __name__=='__main__':
    import matplotlib.pyplot as plt
    print('=== testing the damped Newton solver ===\n')
    nprob=2 # choice of the test problem

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
      bb = 1e3
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
                                                          jacfun=jacfun, warm_start_dict=None, bPrint=True)
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
