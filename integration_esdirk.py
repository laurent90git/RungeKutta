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


def ERK_adapt_integration(fun, y0, t_span, method, atol=1e-6, rtol=1e-6, first_step=None, max_step=np.inf, bPrint=False):
  if not method['isEmbedded']:
    raise Exception('the chosen method is not able to perform time step adaptation')
  if method['embedded']['mode']!=2:
    raise Exception('Embedded method {} is not supported'.format(method['embedded']['mode']))
  assert y0.ndim==1, 'y0 must be 0D or 1D'
  A,b,c = method['A'], method['b'], method['c'] # Butcher coefficients
  d = method['embedded']['d'] # coefficients for the error estimate
  error_order = method['embedded']['error_order']
  ysol = [np.copy(y0)]
  tsol = [t_span[0]]

  n = y0.size # size of the problem
  s = np.size(b) # number of stages for the RK method
  
  out = scipy.integrate._ivp.ivp.OdeResult()
  out.nfev = 0
  out.njev = 0
  
  if first_step is None:
    # estimate first step following Hairer & Wanner's book
    lbda_estimate = np.linalg.norm(fun(t_span[0], y0*1.001)-fun(t_span[0],y0))/(1.001*np.linalg.norm(y0))
    # choose first step such that lbda*dt is small
    dt = 1e-2/lbda_estimate
    print('first_dt=', dt)
    out.nfev+=2
  else:
    dt = first_step
  
  ### initialize algorithm variables and outputs
  tn = t_span[0]
  tf = t_span[-1]
  tsol = [tn]
  ysol   = [y0]
  nt_rejected  = 0 # number of time steps rejected (error too high or code error)
  nt_accepted  = 0 # number of accepted steps
  nt_performed = 0 # total number of time steps computed (accepted+rejected)
  
  Y = np.zeros((n, s), order='F')
  K = np.zeros((n, s), order='F')
  unm1 = np.copy(ysol[-1])
  
  error_estimate = np.zeros((n,))
  while tn < tf:
      nt_performed = nt_accepted+nt_rejected
      if bPrint: # print progress
        if np.mod(nt_performed,100)==0:
            print('\nt={:.10e} '.format(tn), end='')
        if np.mod(nt_performed,10)==0:
            print('.', end='')
      bAcceptedStep = False
      Y[:,0] = unm1[:]
      if c[0]==0. and nt_performed>0:
        K[:,0] = K[:,-1]
      else:
        K[:,0] = fun(tn+c[0]*dt, Y[:,0])
      out.nfev+=1
      # iterate on dt until the error estimation is sufficiently low
      while not bAcceptedStep:
          if dt<1e-15:
            raise Exception('dt<1e-15 --> stopping to avoid underflows')
          try:
            for i in range(0,s):
                # Yi = y0 + dt*sum_{j=1}^{i-1} a_{ij} f(Yj))$
                # 1 - Clear for loop formulation
                # Y[:,i] = unm1[:]
                # for j in range(i):
                #     Y[:,i] += dt*K[:,j]*A[i,j]
                # 2 - Compact matrix formulation
                Y[:,i] = unm1[:] + dt * K[:,:i].dot( A[i,:i] )
                K[:,i] = fun(tn+c[i]*dt, Y[:,i])
                out.nfev+=1
        
            ## compute the new solution value at time t_{n+1}
            # newY = np.copy(unm1)
            # for j in range(s):
            #     newY[:] = newY[:] + dt*b[j]*K[:,j]
            newY = unm1[:] + dt * K.dot( b )
          except FloatingPointError as e:
            print('{}\n\t--> reducing time step'.format(e))
            dt = dt/4
            nt_rejected+=1
            continue # restart at the beginning of the while loop with a new dt value


          error_estimate[:] = 0.
          for j in range(s):
               error_estimate[:] = error_estimate[:] + dt*d[j]*K[:,j]
          tol = atol + rtol*np.abs(np.maximum(newY,unm1))
          err_vec = error_estimate/tol

          # estimation de l'erreur, basé sur Hairer/Norsett/Wanner, Solving ODEs vol I, page 167/168
          err = np.linalg.norm( err_vec )

          if bPrint:
            print('|err|={:.2e}'.format(err))
          # compute the factor by which delta_t is to be multiplied to obtain err~1
          # a safety factor < 1 is used to ensure the new delta t leads to err < 1
          # bounds are also applied on the factor so that the time step does not change to aggressively
          factor = min(10, max(0.2, 0.9*(1/err)**(1/(error_order+1)) ))
          dt_opt = dt*factor
          if err>1:
            dt=dt_opt
            bAcceptedStep=False
            nt_rejected+=1
            if bPrint:
              print('rejected step --> new dt={:.3e}  ( = {} * old dt)'.format(dt, factor))
          else:
              bAcceptedStep=True
  
      # we found an acceptable time step value, we can now move forward to the next time step
      tn = tn+dt
      tsol.append(tn)
      ysol.append(np.copy(newY))
      unm1[:] = newY
      nt_accepted+=1
      
      # take the new optimal time step lentgh, and ensure we respect the different bounds
      dt = dt_opt
      if tn+dt>tf:
        dt = tf-tn
      if dt>max_step:  # avoid time steps too important
        dt=max_step
      if bPrint:
        print('tn={}'.format(tn))
        print('accepted step --> new dt={:.3e}'.format(dt))
  # END OF INTEGRATION
  out.y = np.array(ysol).T
  out.t = np.array(tsol)
  return out
  

               



def ERK_integration(fun, y0, t_span, nt, method, bPrint=True):
    """ Performs the integration of the system dy/dt = f(t,y)
        from t=t_span[0] to t_span[1], with initial condition y(t_span[0])=y0.
        The RK method described by A,b,c is an explicit method
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

    y = np.zeros((n, nt), order='F') # solution accros all time steps
    y[:,0] = y0

    ## advance in time
    out.nfev = 0
    out.njev = 0
    Y = np.zeros((n, s), order='F')
    K = np.zeros((n, s), order='F')
    unm1 = np.copy(y0)
    for it, tn in enumerate(t[:-1]):
        if bPrint: # print progress
          if np.mod(it,np.floor(nt/10))==0:
              print('\n{:.1f} %'.format(100*it/nt), end='')
          if np.mod(it,np.floor(nt/100))==0:
              print('.', end='')
        Y[:,0]=unm1[:]
        K[:,0] = fun(tn+c[0]*dt, Y[:,0])
        ## compute each stage sequentially
        for i in range(1,s):
            # Yi = y0 + dt*sum_{j=1}^{i-1} a_{ij} f(Yj))$
              # Y[:,i] = unm1[:]
              # for j in range(i):
              #     Y[:,i] += dt*K[:,j]*A[i,j]
            Y[:,i] = unm1[:] + dt * K[:,:i].dot( A[i,:i] )
            K[:,i] = fun(tn+c[i]*dt, Y[:,i])

        ## compute the new solution value at time t_{n+1}
          # for j in range(s):
          #     unm1[:] = unm1[:] + dt*b[j]*K[:,j]
        unm1[:] = unm1[:] + dt * K.dot( b )
        y[:,it+1] = unm1[:]
        out.nfev += s

    # END OF INTEGRATION
    out.y = y
    out.t = t
    return out



def FIRK_integration(fun, y0, t_span, nt, method, jacfun=None, bPrint=True, vectorized=False,newtonchoice=NEWTONCHOICE,
                     fullDebug=False):
    """ Performs the integration of the system dy/dt = f(t,y)
        from t=t_span[0] to t_span[1], with initial condition y(t_span[0])=y0.
        The RK method described by A,b,c is fully implicit (e.g RadauIIA ...).
        /!\ it can also solve explicit methods ! (even though this is highly inefficient)
        - fun      :  (function handle) model function (time derivative of y)
        - y0     :  (1D-array)        initial condition
        - t_span :  (1D-array)        array of 2 values (start and end times)
        - nt     :  (integer)         number of time steps
        - A      :  (2D-array)        Butcher table of the chosen RK method
        - b      :  (1D-array)        weightings for the quadrature formula of the RK methods
        - c      :  (1D-array)        RK substeps time

        - jacfun  :  (function handle, optional) function returning a 2D-array (Jacobian df/dy)
        
        TODO: take into account the jacfun argument
        """
    assert y0.ndim==1, 'y0 must be 0D or 1D'

    A,b,c = method['A'], method['b'], method['c']
    out = scipy.integrate._ivp.ivp.OdeResult()
    t = np.linspace(t_span[0], t_span[1], nt)

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
    def resfun(Y,y0,tn,dt,A,n,s):
        """ Residuals for the substeps.
        The input is Y = (y0[...], y1[...], ...).T """
        # 1 - recover all separates stages
        Yr = Y.reshape((n,s), order='F') # each line of Yr is one stage
        # 2 - compute residuals as a matrix (one row for each step)
        res = Yr - y0[:,np.newaxis] - dt*fun_vectorised(tn+c*dt, Yr).dot(At)
        # 3 - reshape the residuals to return a row vector
        return res.reshape((n*s,), order='F')

    ## skirmish
    bStifflyAccurate = np.all(b==A[-1,:]) # then the last stage is the solution at the next time point

    ## advance in time
    out.nfev = 0
    out.njev = 0
    K= np.zeros((n, s), order='F')
    unm1 = np.copy(y0)
    At = A.T
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
        yini = np.zeros((n*s,))
        for i in range(s):
            yini[i*n:(i+1)*n] = unm1[:]


        if newtonchoice==0:
            y_substeps, infodict, ier, msg = scipy.optimize.fsolve(func= lambda x: resfun(Y=x,y0=unm1, tn=tn, dt=dt, A=At, n=n, s=s),
                                        x0=np.copy(yini), fprime=None, # band=(5,5), #gradFun
                                        # epsfcn = 1e-7,
                                        xtol=1e-9, full_output=True)
            out.nfev += infodict['nfev']
        elif newtonchoice==1:
            # this approach is not suited, because scipy's newton considers each residual component independently...
            y_substeps, converged, zero_der = scipy.optimize.newton(func= lambda x: resfun(Y=x,y0=unm1, tn=tn, dt=dt, A=At, n=n, s=s),
                                    x0=np.copy(yini), fprime=None,           # epsfcn = 1e-7,
                                    tol=1e-20, rtol=1e-9, maxiter=100, full_output=True)
            out.nfev += np.nan # not provided...
        elif newtonchoice==2: # custom damped newton
            y_substeps, infodict, warm_start_dict = damped_newton_solve(fun=lambda x: resfun(Y=x,y0=unm1, tn=tn, dt=dt, A=At, n=n, s=s),
                                                                        x0=np.copy(yini), rtol=1e-9, ftol=1e-30,
                                                                        jacfun=None, warm_start_dict=warm_start_dict,
                                                                        itmax=100, jacmax=20, tau_min=1e-4, convergenceMode=0)
            out.nfev += infodict['nfev']
            out.njev += infodict['njev']
            if infodict['ier']!=0: # Newton did not converge
              # restart the Newton solve with all outputs enabled
              y_substeps, infodict, warm_start_dict = damped_newton_solve(fun=lambda x: resfun(Y=x,y0=unm1, tn=tn, dt=dt, A=At, n=n, s=s),
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
        elif newtonchoice==4:
            y_substeps = scipy.optimize.newton_krylov(F=lambda x: resfun(Y=x,y0=unm1, tn=tn, dt=dt, A=At, n=n, s=s),
                                                      xin=np.copy(yini), iter=None, rdiff=1e-8,
                                                      method='lgmres',
                                                      inner_maxiter=20, inner_M=None,
                                                      outer_k=10,
                                                      verbose=True,
                                                      maxiter=None,
                                                      f_tol=1e-20, f_rtol=1e-8,
                                                      x_tol=1e-9, x_rtol=1e-9,
                                                      tol_norm=None,
                                                      line_search='armijo', callback=None)
        else:
            raise Exception('newton choice is not recognised')
        K[:,:] = fun_vectorised(tn+c*dt, y_substeps.reshape((n,s), order='F'))
        # out.njev += infodict['njev']

        if bStifflyAccurate:
            # the last stage is the value at t+dt
            unm1 = y_substeps[-n:]
        else:
            # Y_{n+1} = Y_{n} + \Delta t \sum\limits_{i=1}^{s} b_i k_i
            # for j in range(s):
                # unm1[:] = unm1[:] + dt*b[j]*K[:,j]
            unm1[:] = unm1[:] + dt * K.dot( b )

        y[:,it+1] = unm1[:]

    # END OF INTEGRATION
    out.y = y
    out.t = t
    out.y_substeps = y_substeps # last substeps
    return out


########################################################################################
def DIRK_integration(fun, y0, t_span, nt, method, jacfun=None, bPrint=True, newtonchoice=2, fullDebug=False):
    """ Performs the integration of the system dy/dt = f(t,y)
        from t=t_span[0] to t_span[1], with initial condition y(t_span[0])=y0.
        The RK method described by A,b,c may be explicit or diagonally-implicit.
        - fun      :  (function handle) model function (time derivative of y)
        - y0     :  (1D-array)        initial condition
        - t_span :  (1D-array)        array of 2 values (start and end times)
        - nt     :  (integer)         number of time steps
        - A      :  (2D-array)        Butcher table of the chosen RK method
        - b      :  (1D-array)        weightings for the quadrature formula of the RK methods
        - c      :  (1D-array)        RK substeps time

        - jacfun  :  (function handle, optional) function returning a 2D-array (Jacobian df/dy)
        """
    assert y0.ndim==1, 'y0 must be 0D or 1D'

    A,b,c = method['A'], method['b'], method['c']
    out = scipy.integrate._ivp.ivp.OdeResult()
    t = np.linspace(t_span[0], t_span[1], nt)

    nx = np.size(y0)
    dt = (t_span[1]-t_span[0]) / (nt-1)
    s = np.size(b)

    y = np.zeros((y0.size, nt))
    y[:,0] = y0

    K= np.zeros((np.size(y0), s))
    unm1 = np.copy(y0)
    warm_start_dict=None
    out.nfev = 0
    out.njev = 0
    infodict_hist = {} # additonal optional debug storage
    out.infodict_hist = infodict_hist

    # de quoi itnerfacer le view newton
    solver = newton.newtonSolverObj()
    Dres, LU, Dresinv = None, None, None

    for it, tn in enumerate(t[:-1]):
        if bPrint:
          if np.mod(it,np.floor(nt/10))==0:
              print('\n{:.1f} %'.format(100*it/nt), end='')
          if np.mod(it,np.floor(nt/100))==0:
              print('.', end='')
        for isub in range(s): # go through each substep
            # temp = np.zeros(np.shape(y0))
            # for j in range(isub):
            #     temp    = temp  +  A[isub,j] * K[:,j]
            # vi = unm1 + dt*( temp )
            vi = unm1[:] + dt * K[:,:isub].dot( A[isub,:isub] )

            if A[isub,isub]==0: # explicit step
                kni = fun(tn+c[isub]*dt, vi)
            else:
                 # solve the complete non-linear system via a Newton method
                tempfun = lambda kni: kni - fun(tn+c[isub]*dt, vi + dt*A[isub,isub]*kni)
                if jacfun is None:
                  gradRes = None
                else:
                  gradRes = lambda kni: scipy.sparse.csc_matrix( scipy.sparse.eye(kni.shape[0]) - dt*A[isub,isub]*jacfun(tn+c[isub]*dt, vi + dt*A[isub,isub]*kni) )
                if newtonchoice==0:
                  kni = scipy.optimize.fsolve(func= tempfun,
                                        x0=K[:,0],
                                        fprime=gradRes,
                                        # band=(5,5), #gradFun
                                        # epsfcn = 1e-7,
                                        args=(),)
                elif newtonchoice==1:
                  kni = scipy.optimize.newton(func= tempfun,
                                        x0=K[:,0],
                                        fprime=gradRes, maxiter=100,
                                        # band=(5,5), #gradFun
                                        # epsfcn = 1e-7,
                                        rtol=1e-8,
                                        args=(),)
                elif newtonchoice==2: # custom damped newton
                    kni, infodict, warm_start_dict2 = damped_newton_solve(
                        fun=tempfun, x0=K[:,0],rtol=1e-9, ftol=1e-30, jacfun=gradRes, warm_start_dict=warm_start_dict,
                        itmax=50, jacmax=10, tau_min=1e-3, convergenceMode=0)
                    if infodict['ier']!=0: # Newton did not converge
                      print('Newton did not converge')
                      # restart the Newton solve with all outputs enabled
                      kni, infodict, warm_start_dict = damped_newton_solve(
                        fun=tempfun, x0=K[:,0],rtol=1e-9, ftol=1e-30, jacfun=gradRes, warm_start_dict=warm_start_dict,
                        itmax=50, jacmax=10, tau_min=1e-3, convergenceMode=0, bPrint=True)
                      # raise Exception('Newton did not converge: infodict={}'.format(infodict))
                      msg = 'Newton did not converge'
                      # raise Exception(msg)
                      out.y = y[:,:it+1]
                      out.t = t[:it+1]
                      out.message = msg
                      return out
                    else:
                      warm_start_dict=warm_start_dict2 # pour ne aps interférer avec le debug en cas de non-convergence
                    if fullDebug: # store additional informations about the Newton solve
                      if not bool(infodict_hist): # the debug dictionnary has not yet been initialised
                        for key in infodict.keys():
                          if not isinstance(infodict[key], np.ndarray) and (key!='ier' and key!='msg'): # only save single values
                            infodict_hist[key] = [infodict[key]]
                      else: # backup already initialised
                        for key in infodict_hist.keys():
                          infodict_hist[key].append(infodict[key])
                    out.nfev += infodict['nfev']
                    out.njev += infodict['njev']
                elif newtonchoice==3: # custom modified newton without damping
#                    gradFun = lambda kni: 1 - dt*A[isub,isub]*gradF(tn+c[isub]*dt, vi + dt*A[isub,isub]*kni)
                    kni, Dres, LU, Dresinv = solver.solveNewton(fun=tempfun,
                                                x0=K[:,0],
                                                initJac=Dres,
                                                initLU=LU,
                                                initInv=Dresinv,
                                                jacfun=gradRes,
                                                options={'eps':1e-8, 'bJustOutputJacobian':False, 'nIterMax':50, 'bVectorisedModelFun':False,
                                                         'bUseComplexStep':False, 'bUseLUdecomposition':True, 'bUseInvertJacobian':False,
                                                         'bModifiedNewton':True, 'bDampedNewton':False, 'limitSolution':None,
                                                         'bDebug':False, 'bDebugPlots':False, 'nMaxBadIters':2, 'nMaxJacRecomputePerTimeStep':5} )
                    out.nfev   = solver.nSolverCall
                    out.njev = solver.nJacEval
                    # out.nLUsolve = solver.nLinearSolve
                else:
                    raise Exception('newton choice is not recognised')
            K[:,isub] = kni #fun(tn+c[isub]*dt, ui[:,isub])
        ## END OF QUADRATURE STEPS --> reaffect unm1
        for j in range(s):
            unm1[:] = unm1[:] + dt*b[j]*K[:,j]
        K[:,:] = 0.
        y[:,it+1] = unm1[:]
    # END OF INTEGRATION
    out.y = y
    out.t = t
    if bPrint:
      print('done')
    return out


if __name__=='__main__':
    import matplotlib.pyplot as plt
    import rk_coeffs
    
    #%% Single method test
    #### PARAMETERS ####
    problemtype = 'stiff'
    
    NEWTONCHOICE=3
    # mod ='FIRK'
    # name='Radau5'
    mod ='DIRK'
    name='L-SDIRK-33'
    # mod = 'ERK'
    # name= 'rk4'
    # mod = 'adapt_ERK'
    # name='RK23'
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
        nt = 20
    elif problemtype=='stiff': #  Hirschfelder-Curtiss
        print('Testing time integration routines with Hirschfelder-Curtiss stiff equation')
        k=10.
        def modelfun(t,x):
            """ Mass-spring system"""
            return -(k*x-np.sin(t)    )
        y0 = np.array((0.3,1))
        tf = 5.0
        nt = 30
    elif problemtype=='dae': # DAE simple : y1'=y1, 0=y1+y2
        raise Exception('TODO: DAEs are not yet compatible with the chosen formulation: need to add mass matrix to the problem formulation')
    elif problemtype=='pde':
        raise Exception('TODO')

    method = rk_coeffs.getButcher(name=name)
    if mod=='DIRK': # DIRK solve
        sol = DIRK_integration(fun=modelfun, y0=y0, t_span=[0., tf], nt=nt,
                                    method=method, jacfun=None)
    elif mod=='FIRK': # FIRK solve
        sol = FIRK_integration(fun=modelfun, y0=y0, t_span=[0., tf], nt=nt,
                                    method=method, jacfun=None)
    elif mod=='ERK': # FIRK solve
        sol = ERK_integration(fun=modelfun, y0=y0, t_span=[0., tf], nt=nt,
                                    method=method)
    elif mod=='adapt_ERK':
        sol  =ERK_adapt_integration(fun=modelfun, y0=y0, t_span=[0.,tf], method=method, first_step=1e-2,
                                   atol=1e-3, rtol=1e-3, max_step=np.inf, bPrint=True)
        plt.figure()
        plt.semilogy(sol.t[:-1], np.diff(sol.t))
        plt.grid()
        plt.xlabel('t (s)')
        plt.ylabel('dt (s)')
        # sol  =ERK_adapt_integration(fun=modelfun, y0=y0, t_span=[0.,tf], method=method, first_step=tf/30,
        #                            atol=1e30, rtol=1e30, max_step=tf/30, bPrint=True)
    else:
        raise Exception('mod {} is not recognised'.format(mod))
    # dt = sol.t[1]-sol.t[0]
    # sol_ref = scipy.integrate.solve_ivp(fun=modelfun, t_span=[0., tf], y0=y0, method='RK45',
    #                                 atol=1e9, rtol=1e9, max_step=dt, first_step=dt)
    ## Compute a reference solution with adaptive time step
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

    
    #%% Convergence study
    import time as pytime
    t_start = pytime.time()
    # methods = [('Radau5', 'FIRK')] #, ('Radau5', 'DIRK'), ('Radau5', 'ERK')]
    methods = [
                # ('Radau5', 'FIRK'),
                # ('ESDIRK54A', 'DIRK'),
                # ('L-SDIRK-33', 'DIRK'),
                # # ('ESDIRK32A', 'DIRK'),
                # # ('ESDIRK43B', 'DIRK'),
                # ('IE', 'DIRK'),
                # ('IE', 'FIRK'),
                ('EE', 'ERK'),
                # ('RK10', 'ERK'),
                ('RK4', 'ERK'),
                ('RK45', 'ERK'),
                ('RK23', 'ERK'),
                
                ]
    fig_conv = plt.figure()
    nt_vec = np.logspace(np.log10(10), np.log10(500), 8).astype(int)
    for name, mod in methods:
        sols = []
        method = rk_coeffs.getButcher(name=name)
        for nt in nt_vec:
            if mod=='DIRK': # DIRK solve
                sol = DIRK_integration(fun=modelfun, y0=y0, t_span=[0., tf], nt=nt,
                                            method=method, jacfun=None)
            elif mod=='FIRK': # FIRK solve
                sol = FIRK_integration(fun=modelfun, y0=y0, t_span=[0., tf], nt=nt,
                                            method=method, jacfun=None)
            elif mod=='ERK': # FIRK solve
                sol = ERK_integration(fun=modelfun, y0=y0, t_span=[0., tf], nt=nt,
                                            method=method, bPrint=False)
            else:
                raise Exception('mod {} is not recognised'.format(mod))
            sols.append(sol)
        # compute error
        imax = np.argmax(nt_vec)
        error = np.zeros((len(nt_vec),))
        error2 = np.zeros((len(nt_vec),))
        for i in range(len(nt_vec)):
            interped_ref = scipy.interpolate.interp1d(x=sol_ref.t, y=sol_ref.y, axis=1, kind='cubic')( sols[i].t )
            error[i]  = np.linalg.norm( (sols[i].y - interped_ref) )
            error2[i] = np.linalg.norm( sols[i].y[:,-1]-sol_ref.y[:,-1] )
        # fig_conv.gca().loglog(nt_vec, error, label='{} ({})'.format(name, mod), marker='.')
        fig_conv.gca().loglog(nt_vec, error2, label='{} ({})'.format(name, mod), marker='.')
    fig_conv.gca().legend(framealpha=0.25)
    fig_conv.gca().grid()
    fig_conv.gca().set_ylim(1e-16,1e5)
    
    t_end = pytime.time()
    print('done in {}s with newtonchoice={}'.format(t_end-t_start, NEWTONCHOICE))

