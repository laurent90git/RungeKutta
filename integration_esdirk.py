import numpy as np
import numpy.matlib
import ctypes as ct
from scipy.optimize import fsolve
import copy
import scipy.integrate
import scipy.interpolate
from damped_newton import damped_newton_solve

newtonchoice=2 #0: Scipy's fsolve 1: Scipy's newton (bad !), 2: custom damped Newton

def ERK_integration(fun, y0, t_span, nt, A, b, c, jacfun=None, bPrint=True):
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

        - jacfun  :  (function handle, optional) function returning a 2D-array (Jacobian df/dy)
        """
    assert y0.ndim==1, 'y0 must be 0D or 1D'

    out = scipy.integrate._ivp.ivp.OdeResult()
    t = np.linspace(t_span[0], t_span[1], nt)

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



def FIRK_integration(fun, y0, t_span, nt, A, b, c, jacfun=None, bPrint=True, vectorized=False):
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
        """
    assert y0.ndim==1, 'y0 must be 0D or 1D'

    out = scipy.integrate._ivp.ivp.OdeResult()
    t = np.linspace(t_span[0], t_span[1], nt)

    n = y0.size # size of the problem
    dt = (t_span[1]-t_span[0]) / (nt-1) # time step
    s = np.size(b) # number of stages for the RK method

    y = np.zeros((n, nt)) # solution accros all time steps
    y[:,0] = y0

    J=None # Jacobian of the ODE function


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
    global newtonchoice
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
            if infodict['ier']!=0:
              raise Exception('Newton did not converge')
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
    return out


########################################################################################
def DIRK_integration(fun, y0, t_span, nt, A, b, c, jacfun=None, bPrint=True):
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

    out = scipy.integrate._ivp.ivp.OdeResult()
    t = np.linspace(t_span[0], t_span[1], nt)

    nx = np.size(y0)
    dt = (t_span[1]-t_span[0]) / (nt-1)
    s = np.size(b)

    y = np.zeros((y0.size, nt))
    y[:,0] = y0

    K= np.zeros((np.size(y0), s))
    unm1 = np.copy(y0)
    global newtonchoice
    warm_start_dict=None
    out.nfev = 0
    out.njev = 0
    for it, tn in enumerate(t[:-1]):
        if bPrint:
          if np.mod(it,np.floor(nt/10))==0:
              print('\n{:.1f} %'.format(100*it/nt), end='')
          if np.mod(it,np.floor(nt/100))==0:
              print('.', end='')
        for isub in range(s): # go through each substep
            temp = np.zeros(np.shape(y0))
            for j in range(isub):
                temp    = temp  +  A[isub,j] * K[:,j]
            vi = unm1 + dt*( temp )

            if A[isub,isub]==0: # explicit step
                kni = fun(tn+c[isub]*dt, vi)
            else:
                 # solve the complete non-linear system via a Newton method
                tempfun = lambda kni: kni - fun(tn+c[isub]*dt, vi + dt*A[isub,isub]*kni)
                if newtonchoice==0:
                  kni = scipy.optimize.fsolve(func= tempfun,
                                        x0=K[:,0],
                                        fprime=None,
                                        # band=(5,5), #gradFun
                                        # epsfcn = 1e-7,
                                        args=(),)
                elif newtonchoice==1:
                  kni = scipy.optimize.newton(func= tempfun,
                                        x0=K[:,0],
                                        fprime=None, maxiter=100,
                                        # band=(5,5), #gradFun
                                        # epsfcn = 1e-7,
                                        rtol=1e-8,
                                        args=(),)
                elif newtonchoice==2: # custom damped newton
                    kni, infodict, warm_start_dict = damped_newton_solve(
                        fun=tempfun, x0=K[:,0],rtol=1e-9, ftol=1e-30, jacfun=None, warm_start_dict=warm_start_dict,
                        itmax=30, jacmax=10, tau_min=1e-4, convergenceMode=0)
                    if infodict['ier']!=0:
                      kni, infodict, warm_start_dict = damped_newton_solve(
                        fun=tempfun, x0=K[:,0],rtol=1e-9, ftol=1e-30, jacfun=None, warm_start_dict=warm_start_dict,
                        itmax=30, jacmax=10, tau_min=1e-4, convergenceMode=0, bPrint=True)
                      raise Exception('Newton did not converge: infodict={}'.format(infodict))
                    out.nfev += infodict['nfev']
                    out.njev += infodict['njev']
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
    return out


if __name__=='__main__':
    print('Testing time integration ESDIRK routine with mass-spring system')
    import matplotlib.pyplot as plt
    import rk_coeffs
    problemtype = 'stiff'
    if problemtype=='non-stiff': # ODE
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
        k=10.
        def modelfun(t,x):
            """ Mass-spring system"""
            return -(k*x-np.sin(t)    )
        y0 = np.array((0.3,1))
        tf = 5.0
        nt = 30
    elif problemtype=='dae': # DAE simple : y1'=y1, 0=y1+y2
        raise Exception('TODO: DAEs are not yet compatible with the chosen formulation')

    # mod ='FIRK'
    # method='Radau5'
    mod ='DIRK'
    method='L-SDIRK-33'
    # mod = 'ERK'
    # method= 'rk4'
    # method='Radau5'
    if mod=='DIRK': # DIRK solve
        A,b,c = rk_coeffs.getButcher(name=method)
        sol = DIRK_integration(fun=modelfun, y0=y0, t_span=[0., tf], nt=nt,
                                    A=A, b=b, c=c, jacfun=None)
    elif mod=='FIRK': # FIRK solve
        A,b,c = rk_coeffs.getButcher(name=method)
        sol = FIRK_integration(fun=modelfun, y0=y0, t_span=[0., tf], nt=nt,
                                    A=A, b=b, c=c, jacfun=None)
    elif mod=='ERK': # FIRK solve
        A,b,c = rk_coeffs.getButcher(name=method)
        sol = ERK_integration(fun=modelfun, y0=y0, t_span=[0., tf], nt=nt,
                                    A=A, b=b, c=c, jacfun=None)
    else:
        raise Exception('mod {} is not recognised'.format(mod))
    # dt = sol.t[1]-sol.t[0]
    # sol_ref = scipy.integrate.solve_ivp(fun=modelfun, t_span=[0., tf], y0=y0, method='RK45',
    #                                 atol=1e9, rtol=1e9, max_step=dt, first_step=dt)
    sol_ref = scipy.integrate.solve_ivp(fun=modelfun, t_span=[0., tf], y0=y0, method='DOP853',
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

    if 0:
        #%%
        import time as pytime
        t_start = pytime.time()
        # methods = [('Radau5', 'FIRK')] #, ('Radau5', 'DIRK'), ('Radau5', 'ERK')]
        methods = [
                    # ('Radau5', 'FIRK'), ('ESDIRK54A', 'DIRK'),
                    ('L-SDIRK-33', 'DIRK'),
                    # ('ESDIRK32A', 'DIRK'),
                    # ('ESDIRK43B', 'DIRK'),
                    ('IE', 'DIRK'),
                    ('IE', 'FIRK'),
                    # ('EE', 'ERK'),
                    # ('RK10', 'ERK'),
                    ('RK4', 'ERK'),
                    ]
        fig_conv = plt.figure()
        nt_vec = np.logspace(np.log10(10), np.log10(500), 8).astype(int)
        for method, mod in methods:
            sols = []
            A,b,c = rk_coeffs.getButcher(name=method)
            for nt in nt_vec:
                if mod=='DIRK': # DIRK solve
                    A,b,c = rk_coeffs.getButcher(name=method)
                    sol = DIRK_integration(fun=modelfun, y0=y0, t_span=[0., tf], nt=nt,
                                                A=A, b=b, c=c, jacfun=None)
                elif mod=='FIRK': # FIRK solve
                    A,b,c = rk_coeffs.getButcher(name=method)
                    sol = FIRK_integration(fun=modelfun, y0=y0, t_span=[0., tf], nt=nt,
                                                A=A, b=b, c=c, jacfun=None)
                elif mod=='ERK': # FIRK solve
                    A,b,c = rk_coeffs.getButcher(name=method)
                    sol = ERK_integration(fun=modelfun, y0=y0, t_span=[0., tf], nt=nt,
                                                A=A, b=b, c=c, jacfun=None, bPrint=False)
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
            # fig_conv.gca().loglog(nt_vec, error, label='{} ({})'.format(method, mod), marker='.')
            fig_conv.gca().loglog(nt_vec, error2, label='{} ({})'.format(method, mod), marker='.')
        fig_conv.gca().legend(framealpha=0.25)
        fig_conv.gca().grid()
        fig_conv.gca().set_ylim(1e-16,1e5)
        
        t_end = pytime.time()
        print('done in {}s with newtonchoice={}'.format(t_end-t_start, newtonchoice))

