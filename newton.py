# -*- coding: utf-8 -*-
import numpy as np
import scipy
import scipy.linalg
from cmath import *
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("error", np.ComplexWarning) #to ensure we are not dropping complex perturbations
#warnings.simplefilter("error", scipy.linalg.LinAlgWarning)

class NansInResiduals(Exception):
  """ Exception raised when NaNs appear in residuals """
  pass

def computeJacobian(modelfun,x, options, bReturnResult):
    """
    Method to numerically compute the Jacobian of the function modelfun with
    respect to all the components of the input vector x.

    INPUTS:
        - modelfun:
            a function of the type y=modelfun(x)
        - x: (numpy 1D-array)
            the vector around which the Jacobian must be computed
        - Options is a dictionnary including 2 fields:
            - bUseComplexStep: (boolean)
                if True, a complex pertubation is used, allowing for machine-precision-level
                accuracy in the determinaiton fo the Jacobian
                if False, finite differences are used with adaptive pertubation size
            - bVectorisedModelFun: (boolean)
                True if the modelfun can accept a vectorised input such as a matrix
                [x, x1, x2, x3] instead of just the vector x. This way, the modelfun can
                be called less often and more efficiently

    OUTPUT:
        - Jac, the Jacobian of the function
    """
    n_x = np.size(x)
    hcpx = 1e-50
    if options['bVectorisedModelFun']:
        # 1 - evaluate the function in a vectorized perturbed manner
        if options['bUseComplexStep']:
            x_repmat = np.tile(x, (n_x,1)).T
            x_perturbed = x_repmat + 1j*hcpx*np.eye(n_x)
            res_pert = modelfun(x_perturbed)

            # 2 - gather the residual vector and its Jacobian
            Dres = np.imag(res_pert)/hcpx
            res = np.real(res_pert[:,1])
        else: #finite difference
            x_perturbed = np.tile(x, (n_x+1,1)).T
            # first row unperturbed to have a reference
            current_h = np.zeros((n_x,1), dtype=float)
            for ip in range(n_x):
                current_h[ip] = np.max([1e-6*abs(x_perturbed[ip,1]), 1e-6])
                x_perturbed[ip,ip+1] = x_perturbed[ip,ip+1] + current_h[ip]
            res_pert = modelfun(x_perturbed)

            # 2 - gather the residual vector and its Jacobian
            res = res_pert[:,0]

            Dres = res_pert[:,1:]
            for ii in range(n_x):
                Dres[:,ii] = (Dres[:,ii]-res)/current_h[ii]
    else:
        # multiple perturbed calls
        if options['bUseComplexStep']:
            res = modelfun(x)
            Dres = np.zeros( (np.size(res,0), np.size(x,0)))
            for ip in range(n_x):
                perturbation = np.zeros(np.size(x), dtype='cfloat')
                perturbation[ip] = hcpx*1j
                perturbation = x + perturbation
                resP = modelfun(perturbation)
                Dres[:,ip] = np.imag(resP)/hcpx
            #res = real(resP)
        else:
            res = modelfun(x)
            Dres = np.zeros( (np.size(res,0), np.size(x,0)) )
            current_h = np.zeros((n_x,1))
            for ip in range(n_x):
                current_h[ip] = np.max([1e-1*abs(x[ip]), 1e-1]) # perturbation's size

                perturbation = np.zeros(np.size(x))
                perturbation[ip] = current_h[ip]

                perturbation = x + perturbation
                resP = modelfun(perturbation)
                Dres[:,ip] = (resP-res)/current_h[ip]
    if bReturnResult:
        return res, Dres
    else:
        return Dres

def computeNewtonDeltaX(x, res, Dres, Dresinv, LU, options):
    """ Performs the Newton's step, base on the current state vector,
    the residual vector and its Jacobian """

    if options['bUseLUdecomposition']: # LU decomposition already done
        deltax = scipy.linalg.lu_solve(LU, -res)
#        deltax = np.linalg.pinv(Dres)*-res
#        print('pinving')
    else:
        if isinstance(Dresinv, np.ndarray):
            deltax = np.dot(Dresinv,-res)
        else:
            deltax = np.linalg.solve(Dres, -res)

    if np.any(np.isnan(deltax)):
        raise Exception('NaNs detected in the current delta state vector')
    return deltax

def perform_step(x, modelfun, res, Dres, Dresinv, LU, options):
    """ Performs the Newton's step, base on the current state vector,
    the residual vector and its Jacobian """

    deltaX = computeNewtonDeltaX(x, res, Dres, Dresinv, LU, options)
    # print('||dx||={:.3e}'.format(np.linalg.norm(deltaX)))
    if options['bDampedNewton']:
        #line search for the optimum step length in the delta x direction
        npoints= 5
        nAlphaIter=3
        nGlobalIter=1

        #lAlpha = 1e-2
        #uAlpha = 1e1
        #alpha_vec = np.logspace(np.log10(lAlpha), np.log10(uAlpha), npoints)

        lAlpha = 2.
        uAlpha = 0.1
        alpha_vec = np.zeros((npoints+1,))
        alpha_vec[1:] = np.linspace(lAlpha, uAlpha, npoints)
        alpha_vec[0] = 1.
        alpha_vec = np.unique(alpha_vec)

        print('\t alpha: ',end="")
        for k in range(nGlobalIter):
            for j in range(nAlphaIter):
                normRes_vec = np.zeros(np.size(alpha_vec))
                for i in range(np.size(alpha_vec)):
                    normRes_vec[i] = np.linalg.norm(modelfun(x+alpha_vec[i]*deltaX,options), ord=2)
                iBestAlpha = np.argmin(normRes_vec)
                bestAlpha = alpha_vec[iBestAlpha]
                if normRes_vec[iBestAlpha]<options['eps']:
                    break
                if iBestAlpha>0:
                    lAlpha = alpha_vec[iBestAlpha-1]
                    if iBestAlpha < npoints-1:
                        uAlpha = alpha_vec[iBestAlpha+1]
                    else:
                        uAlpha = alpha_vec[iBestAlpha]
                else:
                    uAlpha = alpha_vec[iBestAlpha+1]
                    lAlpha = alpha_vec[iBestAlpha]
                print(' {:.2f}'.format(bestAlpha),end="")

                alpha_vec = np.zeros((npoints+1,))
#                alpha_vec[1:] = np.logspace(np.log10(lAlpha), np.log10(uAlpha), npoints)
                alpha_vec[1:] = np.linspace(lAlpha, uAlpha, npoints)
                #include last best point
                alpha_vec[0] = bestAlpha
                alpha_vec = np.unique(alpha_vec)
            print('\t\t alpha={:.2f}'.format(bestAlpha))
            x = x + bestAlpha*deltaX # prepare next loop
#        x = x + 0.1*deltaX
        return x
    else:
        return x + deltaX

def updateJacobian(modelfun, x, options, jacModelfun):
    """ Conveniency function to update the Jacobian"""
    if options['bDebug']:
        print('\t\t\t updating Jacobian...')
    if jacModelfun==None:
        res, Dres = computeJacobian(modelfun,x, options, bReturnResult=True)
    else:
        # res, Dres = jacModelfun(modelfun, x, options, bReturnResult=True)
        Dres = jacModelfun(x)
        res = modelfun(x)
    Dresinv, LU = computeLUinv(Dres, options)
    return res, Dres, Dresinv, LU

def computeLUinv(Dres, options):
    if options['bUseLUdecomposition']:
        LU = scipy.linalg.lu_factor(Dres)
    else:
        LU = None
    if options['bUseInvertJacobian']:
        Dresinv = np.linalg.inv(Dres)
    else:
        Dresinv=None
    return Dresinv, LU

class newtonSolverObj():
  def __init__(self):
    self.nSolverCall = 0
    self.nJacEval = 0
    self.nLinearSolve = 0

  def solveNewton(self, fun, x0, options, initJac=None, initLU=None, initInv=None, jacfun=None):
      """ Solves a non-linear system with a Newton algorithm """
      self.nSolverCall += 1
      nRecomputeJacThisStep = 0
      iteration_counter = 0
      iteration_counter_with_current_J = 0
      #iterate with newton step to find the state vector at the next time step
      eps = options['eps']
      res_norm = 2*eps # to force at least one iteration
      res_norm_old = float("inf")

      x=np.copy(x0)

      if not (initJac is None):
          Dres = initJac
          LU   = initLU
          Dresinv = initInv
          bUpdateJacobian = False
      else:
          Dres, Dresinv, LU = None, None, None
          bUpdateJacobian = True #flag for quasiNewton

      x_old = np.copy(x0)

      if options['bJustOutputJacobian']: # only output Jacobian for debug purposes
          res, Dres, Dresinv, LU = updateJacobian(fun, x, options, jacfun)
          self.nJacEval+=1
          print('debug stop')
          return None, Dres, None, None

      nConsecutiveIterWithIncreasingRes = 0
      while abs(res_norm) > eps and iteration_counter < options['nIterMax']:
          if bUpdateJacobian:
              res, Dres, Dresinv, LU = updateJacobian(fun, x, options, jacfun)
              self.nJacEval+=1
              bUpdateJacobian = False
          else:
              res = fun(x)
          res_norm = np.linalg.norm(res, ord=2)  # l2 norm of vector
          if np.isnan(res_norm):
  #            import pdb; pdb.set_trace()
              raise NansInResiduals('NaNs in residuals...')
          if res_norm>eps:
              if not options['bModifiedNewton']:
                  # 3 - perform the Newton step
                  x = perform_step(x, fun, res, Dres, Dresinv, LU, options)
                  self.nLinearSolve+=1
              else:
                  if res_norm<(0.3+0.7*1/(iteration_counter_with_current_J+1))*res_norm_old: #imposer une décroissance suffisante
                  #if res_norm<0.9*res_norm_old: #imposer une décroissance suffisante
                      x = perform_step(x, fun, res, Dres, Dresinv, LU, options)
                      self.nLinearSolve+=1
  #                    x = perform_step(x, x_current, t, dt, fun, res, Dres, Dresinv, LU, options)
                  else: # the previously computed Jacobian is potentially not good enough
                    nConsecutiveIterWithIncreasingRes += 1
                    if options['bDebug']:
                        print('nConsecutiveIterWithIncreasingRes = {}'.format(nConsecutiveIterWithIncreasingRes))
                    if nConsecutiveIterWithIncreasingRes > options['nMaxBadIters']: #en se basant sur mon expérience pour DIRK, il faut laisser une ité de marge pour éviter de recalculer trop souvent la Jacobienne
                        nConsecutiveIterWithIncreasingRes = 0
                        if options['bDebug']:
                            print('quasi newton not converging enough ({:.4e}-->{:.4e}), starting again with new Jacobian'.format(res_norm_old, res_norm))
  #                          print('WARNING: skipping update for debug purposes !!!!!')
                        if nRecomputeJacThisStep>options['nMaxJacRecomputePerTimeStep']:
                            iteration_counter = -1
                            for i in range(10):
                              j = i + 58
                              print('res[{}] = {}'.format(j, np.abs(res[j]) ))
                            print('maxres : res[{}] = {}'.format(np.argmax(np.abs(res)), np.max(np.abs(res)) ))

                            raise Exception('Jacobian matrix needs to be computed more than {} times, this indicates very poor convergence'.format(options['nMaxJacRecomputePerTimeStep']))
                        nRecomputeJacThisStep+=1
                        x = x_old #start again at last newton substep
                        res, Dres, Dresinv, LU = updateJacobian(fun, x, options, jacfun)
                        self.nJacEval+=1
                        x = perform_step(x, fun, res, Dres, Dresinv, LU, options) #perform_step(x, fun, res, Dres, Dresinv, LU, options)
                        self.nLinearSolve+=1
                        iteration_counter_with_current_J = 1
                    else:
                      pass
                      x = perform_step(x, fun, res, Dres, Dresinv, LU, options) #perform_step(x, fun, res, Dres, Dresinv, LU, options)
                      self.nLinearSolve+=1
                      iteration_counter_with_current_J += 1
          # limit the solution to realistic values
          if options['limitSolution']!=None:
              x = options['limitSolution'](x)
          iteration_counter+=1
          iteration_counter_with_current_J+=1
          if not options['bModifiedNewton']:
              bUpdateJacobian = True
          if options['bDebug']:
              print('\t\t subIter={}, ||res||={:.4e}, ||res_norm_old||={:.4e}'.format(iteration_counter, res_norm, res_norm_old))
          if options['bDebugPlots']:
              plt.figure()
              plt.semilogy(np.abs(x))
              plt.xlabel('index')
              plt.ylabel(r'$|x_i|$')
              plt.title('State vector at subiter={}'.format(iteration_counter))
              plt.grid(which='both')
              plt.show()
              plt.pause(0.1)
          res_norm_old = res_norm
          x_old = x

      if options['bDebug']:
          print('\t\t\t nIter = {}'.format(iteration_counter))
      # Here, either a solution is found, or too many iterations
      if abs(res_norm) > eps:
          iteration_counter = -1
          plt.figure()
          plt.semilogy(np.abs(res))
          plt.grid(which='both')
          plt.title('abs(res) at last iteration (not converged) (problem size={})'.format(np.size(x0)))
          plt.xlabel('index')
          plt.ylim(1e-16,None)
          plt.show()
          plt.pause(0.1)
          raise Exception('Newton did not converge')
      else:
          x=np.real(x)

      return x, Dres, LU, Dresinv
