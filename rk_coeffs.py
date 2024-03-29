#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:43:58 2020

List of Runge-Kutta methods:
  explicit (ERK)
  diagonally implicit (DIRK)
  singly diagonally implicit (SDIRK)
  singly diagonally implicit with an explicit first stage (ESDIRK)
  fully implicit (FIRK)
  implicit-explicit (IMEX)

@author: laurent.francois@polytechnique.edu
"""
import numpy as np

implicit_methods = ['IE', 'CRKN', 'L-SDIRK-22-QZ', 'L-SDIRK-43', 'L-SDIRK-33', 'ESDIRK32A-3', 'ESDIRK32A-2',
                    'ESDIRK32A', 'ESDIRK43B', 'ESDIRK54A', 'ESDIRK54A-V4', 'RADAUIA-5', 'RADAUIIA-5',
                    'SDIRK4()5L[1]SA-1', 'SDIRK4()5L[1]SA-2']
explicit_methods = ['EE' , 'RK45', 'RK23']
AVAILABLE_METHODS = implicit_methods + explicit_methods

def getButcher(name):
  """ Donne le tableau de Butcher (A,b,c) de la méthode RK choisie  """
  name = name.upper()
  if "reversed-".upper() in name:
     bReversed=True
     name = name.replace('reversed-'.upper(),'')
  else:
     bReversed=False
  
  A,b,c,Ahat,bhat,chat,embedded=None,None,None,None,None,None,None
  if name=='IE': # Implicit Euler, L-stable stiffly accurate
    A= np.array([[1]])
    c= np.array([1])
    b= np.array([1])
    strType = "ERK"
    order=1
  elif name=='EE': # Explicit Euler
    A= np.array([[0]])
    c= np.array([0])
    b= np.array([1])
    strType = "ERK"
    order=1
    
  elif name=='EE-MODIF': # Explicit Euler (fake stage to include final value)
    A= np.array([[0,0],
                 [1,0]]) # last line = b
    b= np.array([1,0])
    c= np.array([0,1])
    strType = "ERK"
    order=1
    
  elif name=='EE-SUB4': # Explicit Euler (4 substeps)
    s=4
    A= np.array([[(1/s)*(i>j) for j in range(s)] for i in range(s)])
            #  [0,    0,   0,  0],
            #  [1/s,  0,   0,  0],
            #  [1/s, 1/s,  0,  0],
            #  [1/s, 1/s, 1/s, 0],]) # last line = b
    c= np.array([i/s for i in range(s)])
    b= np.array([1/s for i in range(s)])
    strType = "ERK"
    order=1
    
  elif name=='EE-SUB4-LAST': # Explicit Euler (4 substeps + last step)
    s=5
    A= np.array([[(1/(s-1))*(i>j) for j in range(s)] for i in range(s)])
    
    c= np.array([i/(s-1) for i in range(s)])
    b= np.array([1/(s-1)*(i<s-1) for i in range(s)])
    strType = "ERK"
    order=1
    
  elif name=='CRKN': # Crank-Nicolson
    A= np.array([[0, 0],
                 [0, 1]])
    c= np.array([0, 1])
    b= np.array([1/2, 1/2])
    strType = "ESDIRK"
    order=2
    
  elif name=='L-SDIRK-22-QZ': #Qin and Zhang
    # not stiffly accurate, but L-stable
    x = 1+np.sqrt(2)/2
#    x = 1-np.sqrt(2)/2
    A= np.array([[x, 0],
                 [1-x, x]])
    c= np.array([x, 1])
    b= np.array([1/2, 1/2])
    strType = "SDIRK"
    order=2
    
  elif name=='RK4':
    A= np.array([[0,0,0,0],
                 [1/2, 0, 0, 0],
                 [0, 1/2, 0, 0],
                 [0, 0, 1, 0]
                 ])
    c= np.array([0, 1/2, 1/2, 1.])
    b= np.array([1/6, 1/3, 1/3, 1/6])
    strType = "ERK"
    order=4

  elif name=='RK10':
    A,b,c, embedded = RK10coeffs()
    strType = "ERK"
    order=10
    
  elif name=="HEUN-EULER": # order 2 adaptive explicit
    A= np.array([[0, 0],
                 [1, 0]])
    c= np.array([0,1])
    b= np.array([1/2, 1/2])
    strType = "ERK"
    order=2
    embedded = {'mode': 2, # 0 if the error estimate is not available, 1 if it is the difference between two stages (easier for DAEs), 2 if it must be built separately
                'error_order':1, # order of the error estimate
                'd':np.array([1,0])-b,  # coefficients of the error estimate (if mode==2)
                'i_high': -1, # -1 means not the last stage but the quadrature one
                              # which substep is the high-order solution
                'p_high': 2, # what is its order (global error)
                'i_low':  1,
                'p_low':  1,
                }
    
  elif name=="HEUN-EULER-MODIF": # order 2 adaptive explicit
    # modif pour inverse : on rajoute ynp1 dans les stages
    A= np.array([[0, 0, 0],
                 [1, 0, 0],
                 [1/2, 1/2, 0]])
    c= np.array([0,1,1])
    b= np.array([1/2,1/2,0])
    strType = "ERK"
    embedded = {'mode': 1, # 0 if the error estimate is not available, 1 if it is the difference between two stages (easier for DAEs), 2 if it must be built separately
                'error_order':1, # order of the error estimate
                'i_high': 2, # which substep is the high-order solution
                'p_high': 2, # what is its order (global error)
                'i_low':  1,
                'p_low':  1,
                }
    order=2
    
  elif name=="HEUN-EULER-1": # order 2 adaptive explicit
    A= np.array([[0, 0],
                 [1, 0]])
    c= np.array([0,1])
    b= np.array([1,0])
    order=1
    strType = "ERK"
    
  elif name=="HEUN-EULER-2": # order 2 adaptive explicit
    A= np.array([[0, 0],
                 [1, 0]])
    c= np.array([0,1])
    b= np.array([1/2, 1/2])
    order=2 # order of the quadrature solution obtained with b
    strType = "ERK"
   
    
  elif name=="RK23" or name=="Bogacki–Shampine".upper(): # order 3 adaptive explicit
    A= np.array([[0, 0, 0, 0],
                 [1/2, 0, 0, 0],
                 [0, 3/4, 0, 0],
                 [2/9, 1/3, 4/9, 0]])
    c= np.array([0,1/2,3/4,1])
    b= np.array([2/9,1/3,4/9,0])
    order=3
    strType = "ERK"
    embedded = {'mode': 2, # 0 if the error estimate is not available, 1 if it is the difference between two stages (easier for DAEs), 2 if it must be built separately
                'error_order':2, # order of the error estimate
                'd':np.array([5/72, -1/12, -1/9, 1/8])  # coefficients of the error estimate (if mode==2)
                }
    
  elif name=="RK45":
    A = np.array([
          [0,   0, 0, 0, 0, 0, 0],
          [1/5, 0, 0, 0, 0, 0, 0],
          [3/40, 9/40, 0, 0, 0, 0, 0],
          [44/45, -56/15, 32/9, 0, 0, 0, 0],
          [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
          [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
          [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
        ])
    b = np.array([35/384, 0,   500/1113, 125/192, -2187/6784, 11/84, 0])
    c = np.array([0,      1/5, 3/10,     4/5,      8/9,        1,    1])
    
    # embedded = {'mode': 2, # 0 if the error estimate is not available, 1 if it is the difference between two stages (easier for DAEs), 2 if it must be built separately
    #             'error_order':4, # order of the error estimate
    #             'd':np.array([-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525, 1/40])  # coefficients of the error estimate (if mode==2)
    #             }
    embedded = {'mode': 2, # 0 if the error estimate is not available, 1 if it is the difference between two stages (easier for DAEs), 2 if it must be built separately
                'error_order':4, # order of the error estimate
                # 'd':np.array([-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525, 1/40]),  # coefficients of the error estimate (if mode==2)
                'i_high': 6, # -1 means not the last stage but the quadrature one
                              # which substep is the high-order solution
                'p_high': 5, # what is its order (global error)
                'i_low':  -1,
                'p_low':  4,
                }
    strType = "ERK"
    order=5
    
  elif name=="RK45-MODIF":
      A = np.array([
            [0,   0, 0, 0, 0, 0, 0, 0],
            [1/5, 0, 0, 0, 0, 0, 0, 0],
            [3/40, 9/40, 0, 0, 0, 0, 0, 0],
            [44/45, -56/15, 32/9, 0, 0, 0, 0, 0],
            [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0, 0],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0, 0],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0, 0],
            [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40, 0],
          ])
      b = np.array([35/384, 0,   500/1113, 125/192, -2187/6784, 11/84, 0, 0])
      c = np.array([0,      1/5, 3/10,     4/5,      8/9,        1,    1, 1])
      
      # embedded = {'mode': 2, # 0 if the error estimate is not available, 1 if it is the difference between two stages (easier for DAEs), 2 if it must be built separately
      #             'error_order':4, # order of the error estimate
      #             'd':np.array([-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525, 1/40])  # coefficients of the error estimate (if mode==2)
      #             }
      embedded = {'mode': 2, # 0 if the error estimate is not available, 1 if it is the difference between two stages (easier for DAEs), 2 if it must be built separately
                  'error_order':4, # order of the error estimate
                  # 'd':np.array([-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525, 1/40]),  # coefficients of the error estimate (if mode==2)
                  'i_high': 6, # -1 means not the last stage but the quadrature one
                                # which substep is the high-order solution
                  'p_high': 5, # what is its order (global error)
                  'i_low':  7,
                  'p_low':  4,
                  }
      strType = "ERK"
      order=5
  
  elif name=="RK45-5":
    A = np.array([
          [0,   0, 0, 0, 0, 0],
          [1/5, 0, 0, 0, 0, 0],
          [3/40, 9/40, 0, 0, 0, 0],
          [44/45, -56/15, 32/9, 0, 0, 0],
          [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
          [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0],
        ])
    b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
    c = np.array([0,      1/5, 3/10,     4/5,      8/9,        1])
    strType = "ERK"
    order=5
    
  elif name=="RK45-4":
    A = np.array([
          [0,   0, 0, 0, 0, 0, 0],
          [1/5, 0, 0, 0, 0, 0, 0],
          [3/40, 9/40, 0, 0, 0, 0, 0],
          [44/45, -56/15, 32/9, 0, 0, 0, 0],
          [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
          [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
          [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
        ])
    b = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
    c = np.array([0,      1/5, 3/10,     4/5,      8/9,        1,    1])
    strType = "ERK"
    order=4
    
  
   
  elif name=='L-SDIRK-43': # L-Stable, stiffly accurate, 4 stages, 3 order, SDIRK method
    A = np.array([[1/2,   0,  0,   0],
                  [1/6,  1/2, 0,   0],
                  [-1/2, 1/2, 1/2, 0],
                  [3/2, -3/2, 1/2, 1/2],
                ])
    c = np.array([1/2, 2/3, 1/2, 1])
    b = A[-1,:]
    strType = "SDIRK"
    
  elif name=='L-SDIRK-33': # L-Stable, stiffly accurate, 3 stages, 3 order, SDIRK method
    x = 0.4358665215
    A= np.array([ [x,                                  0,              0],
                  [(1-x)/2,                           x,                0],
                  [-3*(x**2)/2 + 4*x -1/4,     3*(x**2)/2-5*x+5/4,      x],
                ])
    c= np.array([x, (1+x)/2, 1])
    b= A[-1,:]
    strType = "SDIRK"
    
  elif name== 'ESDIRK32A-3': # stiffly accurate
    # méthode d'ordre 3 extraite de la méthode embedded ESDIRK 32 avec 4 stages
    # taken from A FAMILY OF ESDIRK INTEGRATION METHODS
    # JOHN BAGTERP JØRGENSEN ∗, MORTEN RODE KRISTENSEN , AND
    # PER GROVE THOMSEN
    gamma = 0.4358665215
    A= np.array([ [0, 0, 0, 0],
                  [gamma, gamma, 0, 0],
                  [(-4*gamma**2 + 6*gamma-1)/(4*gamma), (-2*gamma+1)/(4*gamma), gamma, 0.],
                  [(6*gamma-1)/(12*gamma), -1/(12*gamma*(2*gamma-1)), (-6*gamma**2 + 6*gamma -1)/(3*(2*gamma-1)), gamma],
                ])
    c= np.array([0, 2*gamma, 1, 1])
    b= A[-1,:]
    strType = "ESDIRK"
    order=3
    
  elif name== 'ESDIRK32A-2': # stiffly accurate
    # méthode d'ordre 2 extraite de la méthode embedded ESDIRK 32 avec 4 stages
    # taken from A FAMILY OF ESDIRK INTEGRATION METHODS
    # JOHN BAGTERP JØRGENSEN ∗, MORTEN RODE KRISTENSEN , AND
    # PER GROVE THOMSEN
    gamma = 0.4358665215
    A= np.array([ [0, 0, 0],
                  [gamma, gamma, 0],
                  [(-4*gamma**2 + 6*gamma-1)/(4*gamma), (-2*gamma+1)/(4*gamma), gamma],
                ])
    c= np.array([0, 2*gamma, 1])
    b= A[-1,:]
    strType = "ESDIRK"
    order=2
    
  elif name=='ESDIRK32A': #embedded method
    gamma = 0.4358665215
    gamma = 0.4358665215
    A= np.array([ [0, 0, 0, 0],
                  [gamma, gamma, 0, 0],
                  [(-4*gamma**2 + 6*gamma-1)/(4*gamma), (-2*gamma+1)/(4*gamma), gamma, 0.],
                  [(6*gamma-1)/(12*gamma), -1/(12*gamma*(2*gamma-1)), (-6*gamma**2 + 6*gamma -1)/(3*(2*gamma-1)), gamma],
                ])
    c= np.array([0, 2*gamma, 1, 1])

    b_low = A[-2,:] #bas ordre
    b     = A[-1,:] #ordre eleve
    d= b-b_low # poids pour l'estimation de l'erreur
    p_low = 2 #ordre de la méthode bas ordre
    p_high = 3 #ordre de la méthode d'ordre élevé
#    n_avancement = 'high'
    isub_high = 4 # le 4ème substep correspond au pas final de la méthode d'ordre élevé
    isub_low  = 3 # le 3ème substep correspond au pas final de la méthode d'ordre faible
    embedded = { 'isub_high':isub_high, 'isub_low':isub_low, 'p_low':p_low, 'p_high':p_high , 'd':d}
    strType = "ESDIRK"
    order=3
    
  elif name=='ESDIRK43B': #embedded method
    A= np.array([ [0, 0, 0, 0, 0],
                  [0.43586652150846, 0.43586652150846, 0, 0, 0],
                  [0.14073777472471, -0.10836555138132, 0.43586652150846, 0, 0],
                  [0.10239940061991, -0.37687845225556, 0.83861253012719, 0.43586652150846, 0],
                  [0.15702489786032,  0.11733044137044, 0.61667803039212, -0.32689989113134, 0.43586652150846],
                ])
    c= np.array([0, 0.87173304301692, 0.46823874485185, 1, 1])

    b_low = A[-2,:] #bas ordre
    b     = A[-1,:] #ordre eleve
    d = b-b_low # poids pour l'estimation de l'erreur
    p_low = 3 #ordre de la méthode bas ordre
    p_high = 4 #ordre de la méthode d'ordre élevé
#    n_avancement = 'high'
    isub_high = 5 # le 4ème substep correspond au pas final de la méthode d'ordre élevé
    isub_low  = 4 # le 3ème substep correspond au pas final de la méthode d'ordre faible
    embedded = { 'isub_high':isub_high, 'isub_low':isub_low, 'p_low':p_low, 'p_high':p_high , 'd':d}
    strType = "ESDIRK"
    order=4
    
  elif name=='ESDIRK43B-3': #embedded method
      A= np.array([ [0, 0, 0, 0, 0],
                    [0.43586652150846, 0.43586652150846, 0, 0, 0],
                    [0.14073777472471, -0.10836555138132, 0.43586652150846, 0, 0],
                    [0.10239940061991, -0.37687845225556, 0.83861253012719, 0.43586652150846, 0],
                    [0.15702489786032,  0.11733044137044, 0.61667803039212, -0.32689989113134, 0.43586652150846],
                  ])
      c= np.array([0, 0.87173304301692, 0.46823874485185, 1, 1])
  
      A=A[:-1,:-1]
      c=c[:-1]
      
      b     = A[-1,:]
      strType = "ESDIRK"
      order=3
      
  elif name=='ESDIRK54A': #embedded method (Kvaerno 2004, but coeffs found in arkcode butcher)
        # 7 stages, orders 5 and 4, both stiffly accurate
        A = np.zeros((7,7))
        b_low = np.zeros(7)
        b = np.zeros(7)
        c = np.zeros(7)
        order=5

        A[1,0] = 0.26
        A[1,1] = 0.26
        A[2,0] = 0.13
        A[2,1] = 0.84033320996790809
        A[2,2] = 0.26
        A[3,0] = 0.22371961478320505
        A[3,1] = 0.47675532319799699
        A[3,2] = -0.06470895363112615
        A[3,3] = 0.26
        A[4,0] = 0.16648564323248321
        A[4,1] = 0.10450018841591720
        A[4,2] = 0.03631482272098715
        A[4,3] = -0.13090704451073998
        A[4,4] = 0.26
        A[5,0] = 0.13855640231268224
        A[5,2] = -0.04245337201752043
        A[5,3] = 0.02446657898003141
        A[5,4] = 0.61943039072480676
        A[5,5] = 0.26
        A[6,0] = 0.13659751177640291
        A[6,2] = -0.05496908796538376
        A[6,3] = -0.04118626728321046
        A[6,4] = 0.62993304899016403
        A[6,5] = 0.06962479448202728
        A[6,6] = 0.26

        b[0] = 0.13659751177640291
        b[2] = -0.05496908796538376
        b[3] = -0.04118626728321046
        b[4] = 0.62993304899016403
        b[5] = 0.06962479448202728
        b[6] = 0.26

        b_low[0] = 0.13855640231268224
        b_low[2] = -0.04245337201752043
        b_low[3] = 0.02446657898003141
        b_low[4] = 0.61943039072480676
        b_low[5] = 0.26

        c[1] = 0.52
        c[2] = 1.230333209967908
        c[3] = 0.895765984350076
        c[4] = 0.436393609858648
        c[5] = 1.0
        c[6] = 1.0

        assert np.allclose(b,A[-1,:])
        assert np.allclose(b_low,A[-2,:])

        b_low = A[-2,:] # low order  = penultimate stage
        b     = A[-1,:] # high order = last stage
        d = b-b_low # coefficients of the error
        p_low = 4 #ordre de la méthode bas ordre
        p_high = 5 #ordre de la méthode d'ordre élevé
        isub_high = 7 # le 4ème substep correspond au pas final de la méthode d'ordre élevé
        isub_low  = 6 # le 3ème substep correspond au pas final de la méthode d'ordre faible
        embedded = {'mode': 0, # 0 if the error estimate is not available, 1 if it is the difference between two stages (easier for DAEs), 2 if it must be built separately
                    'isub_high':isub_high, # index of the high order stage
                    'isub_low':isub_low,   # index of the low order stage
                    'p_low':p_low,         # order of the low order solution
                    'p_high':p_high,       # order of the high order solution
                    'd':None               # coefficients of the error estimate (if mode==2)
                    }
        strType = "ESDIRK"
        order=5
  elif name=='ESDIRK54A-V4': #method of ordre 4 extracted from the Kvaerno 54a method
        A = np.zeros((6,6))
        b = np.zeros(6)
        c = np.zeros(6)

        A[1,0] = 0.26
        A[1,1] = 0.26
        A[2,0] = 0.13
        A[2,1] = 0.84033320996790809
        A[2,2] = 0.26
        A[3,0] = 0.22371961478320505
        A[3,1] = 0.47675532319799699
        A[3,2] = -0.06470895363112615
        A[3,3] = 0.26
        A[4,0] = 0.16648564323248321
        A[4,1] = 0.10450018841591720
        A[4,2] = 0.03631482272098715
        A[4,3] = -0.13090704451073998
        A[4,4] = 0.26
        A[5,0] = 0.13855640231268224
        A[5,2] = -0.04245337201752043
        A[5,3] = 0.02446657898003141
        A[5,4] = 0.61943039072480676
        A[5,5] = 0.26


        b[0] = 0.13855640231268224
        b[2] = -0.04245337201752043
        b[3] = 0.02446657898003141
        b[4] = 0.61943039072480676
        b[5] = 0.26

        c[1] = 0.52
        c[2] = 1.230333209967908
        c[3] = 0.895765984350076
        c[4] = 0.436393609858648
        c[5] = 1.0
        assert(np.all(b==A[-1,:]))
        strType = "ESDIRK"
        order = 4
  elif name=='RADAUIA-5':
      A = np.zeros((3,3))
      A[0,0] = 1/9
      A[0,1] = (-1-(6)**0.5)/18
      A[0,2] = (-1+(6)**0.5)/18
      
      A[1,0] = 1/9
      A[1,1] = 11/45 + 7*(6**0.5)/360
      A[1,2] = 11/45 - 43*(6**0.5)/360
      
      A[2,0] = 1/9
      A[2,1] = 11/45 + 43*(6**0.5)/360
      A[2,2] = 11/45 - 7*(6**0.5)/360
      
      b = np.array([1/9, 4/9 + (6**0.5)/36, 4/9 - (6**0.5)/36])
      c = np.array([0, 3/5-(6**0.5)/10, 3/5+(6**0.5)/10])
      strType = "IRK"      
      order = 3
  elif name=='RADAUIIA-5' or name=='RADAU5':
      A = np.zeros((3,3))
      r6 = 6**0.5
      A[0,0] = 11/45 - 7*r6/360
      A[0,1] = 37/225 - 169*r6/1800
      A[0,2] = -2/225 + r6/75
      
      A[1,0] = 37/225 + 169*r6/1800
      A[1,1] = 11/45 + 7*r6/360
      A[1,2] = -2/225 - r6/75
      
      A[2,0] = 4/9-r6/36
      A[2,1] = 4/9 + r6/36
      A[2,2] = 1/9
      
      b = A[-1,:]
      c = np.array([2/5 -r6/10, 2/5+r6/10, 1.])
      strType = "IRK"      
      order=5
      
  elif name=='SDIRK4()5L[1]SA-1': # review nasa diagonally implicit RK, page 95, table 22
      coeff = 1
      A = np.zeros((5,5))
      b = np.zeros(5)
      c = np.zeros(5)
      A[0,0] = 1/4
      A[1,0] = (1 - coeff*(2**0.5))/4
      A[1,1] = 1/4
      A[2,0] = (-1676+coeff*145*(2**0.5))/6724
      A[2,1] = 3*(709+coeff*389*(2**0.5))/6724
      A[2,2] = 1/4
      A[3,0] = (-371435 - coeff*351111*(2**0.5))/470596
      A[3,1] = (98054928 + coeff*73894543*(2**0.5))/112001848
      A[3,2] = (56061972 + coeff*30241643*(2**0.5))/112001848
      A[3,3] = 1/4
      A[4,0] = 0.
      A[4,1] = 4*(74+coeff*273*(2**0.5))/5253
      A[4,2] = (19187+coeff*5031*(2**0.5))/55284
      A[4,3] = (116092 - coeff*100113*(2**0.5))/334956
      A[4,4] = 1/4

      b[:] = A[-1,:]
      c = np.array([1/4, (2-coeff*(2**0.5))/4, (13+coeff*8*(2**0.5))/41, (41+coeff*9*(2**0.5))/49, 1.])
      strType = "SDIRK"      
      order=4
  elif name=='SDIRK4()5L[1]SA-2':
      coeff = -1
      A = np.zeros((5,5))
      b = np.zeros(5)
      c = np.zeros(5)
      A[0,0] = 1/4
      A[1,0] = (1 - coeff*(2**0.5))/4
      A[1,1] = 1/4
      A[2,0] = (-1676+coeff*145*(2**0.5))/6724
      A[2,1] = 3*(709+coeff*389*(2**0.5))/6724
      A[2,2] = 1/4
      A[3,0] = (-371435 - coeff*351111*(2**0.5))/470596
      A[3,1] = (98054928 + coeff*73894543*(2**0.5))/112001848
      A[3,2] = (56061972 + coeff*30241643*(2**0.5))/112001848
      A[3,3] = 1/4
      A[4,0] = 0.
      A[4,1] = 4*(74+coeff*273*(2**0.5))/5253
      A[4,2] = (19187+coeff*5031*(2**0.5))/55284
      A[4,3] = (116092 - coeff*100113*(2**0.5))/334956
      A[4,4] = 1/4

      b[:] = A[-1,:]
      c = np.array([1/4, (2-coeff*(2**0.5))/4, (13+coeff*8*(2**0.5))/41, (41+coeff*9*(2**0.5))/49, 1.])
      strType = "SDIRK"      
      order=4
  elif name=='ESDIRK5(4I)8L[2]SA':
      raise Exception("j'ai du me tromper dans cette méthode, car même sur burgers simple, elle CV à l'ordre 1")
      # ESdirk method, stiffly accurate, Diagonally Implicit Runge-Kutta Methods for Ordinary Differential Equations. A Review       by Christopher A. Kennedy
      # error control possible
      A = np.zeros((8,8))
      b = np.zeros(8)
      c = np.zeros(8)
      A[1,:2] = [1/4, 1/4]
      A[2,:3] = [1748874742213/5795261096931, 1748874742213/5795261096931, 1/4]
      A[3,:4] = [2426486750897/12677310711630, 2426486750897/12677310711630, -783385356511/7619901499812, 1/4]

      A[4,:5] = np.array([1616209367427, 1616209367427, -211896077633, 464248917192, 1])/ \
                np.array([5722977998639,  5722977998639, 5134769641545, 17550087120101, 4])

      A[5,:6] = np.array([1860464898611, 1825204367749, -1289376786583, 55566826943, 1548994872005,  1])/ \
                np.array([7805430689312, 7149715425471, 6598860380111,  2961051076052,  13709222415197,  4])

      A[6,:7] = np.array([1783640092711, -5781183663275, 57847255876685, 29339178902168, 122011506936853, -60418758964762, 1])/ \
                np.array([14417713428467, 18946039887294, 10564937217081, 9787613280015, 12523522131766, 9539790648093, 4])

      A[7,:8] = np.array([3148564786223, -4152366519273, -143958253112335, 16929685656751, 37330861322165, -103974720808012, -93596557767, 1])/ \
                np.array([23549948766475,  20368318839251,  33767350176582, 6821330976083, 4907624269821, 20856851060343,  4675692258479, 4])

      b[:] = A[-1,:]
      c = np.array([0, 1/2, (2+np.sqrt(2))/4, 53/100, 4/5, 17/25, 1, 1])
      strType = "SDIRK"
      order=5
      
  #################
  ##### IMEX ######
  #################
  elif name=='LDIRK222':
    gamma = (2-2**0.5)/2
    delta = 1-1/(2*gamma)
    A = np.array([  
                  [gamma, 0],
                  [1-gamma, gamma],
                  ])
    b = np.array([1-gamma, gamma])
    c=np.array([gamma, 1])
    #A,b,c = expandImplicitTableau(A,b,c)
    Ahat=np.array([[0,0,0],
                   [gamma, 0, 0],
                   [delta, 1-delta, 0],
                   ])
    bhat=np.array([delta, 1-delta, 0])
    chat=np.array([0, gamma, 1])
    strType = "IMEX"
    order=2

  elif name=='FBeuler111':
    A = np.array([[1]])
    b = np.array([1])
    c=np.array([1])
    #A,b,c = expandImplicitTableau(A,b,c)

    Ahat=np.array([[0,0],
                   [1,0]])
    bhat=np.array([1,0])
    chat=np.array([0,1])
    strType = "IMEX"
    order=1
    
  elif name=='DIRK121':
    A = np.array([[1]])
    b = np.array([1])
    c=np.array([1])
    #A,b,c = expandImplicitTableau(A,b,c)

    Ahat=np.array([[0,0],
                   [1,0]])
    bhat=np.array([0,1])
    chat=np.array([0,1])
    strType = "IMEX"
    
  elif name=='DIRK122':
    A = np.array([  
                  [1/2],
                  ])
    b = np.array([1])
    c=np.array([1/2])
    #A,b,c = expandImplicitTableau(A,b,c)
    Ahat=np.array([[0,0],
                   [1/2, 0],
                   ])
    bhat=np.array([0, 1])
    chat=np.array([0, 1/2])
    strType = "IMEX"
    
  elif name=='LDIRK232':
    gamma = (2-2**0.5)/2
    delta = -2*(2**0.5)/3
    A = np.array([[gamma, 0],
                  [1-gamma, gamma]])
    b = np.array([1-gamma, gamma])
    c = np.array([gamma, 1])
    #A,b,c = expandImplicitTableau(A,b,c)

    Ahat=np.array([[0,0,0],
                   [gamma,0,0],
                   [delta, 1-delta, 0]])
    bhat=np.array([0, 1-gamma, gamma])
    chat=np.array([0, gamma, 1])
    strType = "IMEX"
    
  elif name=='DIRK233':
    gamma = (3+3**0.5)/6
    A = np.array([  
                  [gamma, 0],
                  [1-2*gamma, gamma],
                  ])
    b = np.array([1/2, 1/2])
    c=np.array([gamma, 1-gamma])
    #A,b,c = expandImplicitTableau(A,b,c)
    Ahat=np.array([[0,0,0],
                   [gamma, 0, 0],
                   [gamma-1, 2*(1-gamma), 0],
                   ])
    bhat=np.array([0, 1/2, 1/2])
    chat=np.array([0, gamma, 1-gamma])
    strType = "IMEX"

  elif name=='LDIRK343':
    gamma = 0.4358665215
    A = np.array([  
                  [gamma,                           0,                               0],
                  [(1-gamma)/2,                     gamma,                           0],
                  [-3/2*gamma**2 + 4*gamma - 1/4,    3/2*gamma**2 - 5*gamma + 5/4,   gamma ],
                  ])
    b = A[-1,:]
    c=np.array([gamma, (1+gamma)/2, 1])
    #A,b,c = expandImplicitTableau(A,b,c)
    Ahat=np.array([[0,0,0,0],
                   [0.4358665215, 0, 0, 0],
                   [0.3212788860, 0.3966543747, 0, 0],
                   [-0.105858296, 0.5529291479, 0.5529291479, 0],
                   ])
    bhat=np.array([0, 1.208496649, -0.644363171, 0.4358665215])
    chat=np.array([0, 0.4358665215, 0.7179332608, 1.])
    strType = "IMEX"
  else:
    raise Exception('unknown integrator {}'.format(name))
	
  assert not (A is None)
  assert not (b is None)
  assert not (c is None)
  assert not (strType is None)
  
  assert A.shape[0]==A.shape[1], 'A must be square'
  assert b.size==A.shape[1]
  assert c.size==A.shape[1]
  
  
  if bReversed:
      embedded=None
      A,b,c = reverseRK(A,b,c)
  method = {'A':A, 'b':b, 'c':c,
            'Ahat': Ahat, 'bhat': bhat, 'chat':chat,
            'order': order,
            'embedded': embedded, 'isEmbedded': not (embedded is None),
            'name': name}
  return method
  # return A,b,c

def RK10coeffs():
    #The coefficients have been obtained from https://sce.uhcl.edu/rungekutta/rk108.txt
	# TODO: local error estimate = (1/360)  h ( f(t1,x1)-f(t15,x15) )
    c = np.array([
      0.000000000000000000000000000000000000000000000000000000000000,
      0.100000000000000000000000000000000000000000000000000000000000,
      0.539357840802981787532485197881302436857273449701009015505500,
      0.809036761204472681298727796821953655285910174551513523258250,
      0.309036761204472681298727796821953655285910174551513523258250,
      0.981074190219795268254879548310562080489056746118724882027805,
      0.833333333333333333333333333333333333333333333333333333333333,
      0.354017365856802376329264185948796742115824053807373968324184,
      0.882527661964732346425501486979669075182867844268052119663791,
      0.642615758240322548157075497020439535959501736363212695909875,
    0.357384241759677451842924502979560464040498263636787304090125,
    0.117472338035267653574498513020330924817132155731947880336209,
       0.833333333333333333333333333333333333333333333333333333333333,
       0.309036761204472681298727796821953655285910174551513523258250,
    0.539357840802981787532485197881302436857273449701009015505500,
       0.100000000000000000000000000000000000000000000000000000000000,
     1.00000000000000000000000000000000000000000000000000000000000,
     ])
    b = np.array([
      0.0333333333333333333333333333333333333333333333333333333333333,
      0.0250000000000000000000000000000000000000000000000000000000000,
      0.0333333333333333333333333333333333333333333333333333333333333,
      0.000000000000000000000000000000000000000000000000000000000000,
      0.0500000000000000000000000000000000000000000000000000000000000,
      0.000000000000000000000000000000000000000000000000000000000000,
      0.0400000000000000000000000000000000000000000000000000000000000,
      0.000000000000000000000000000000000000000000000000000000000000,
      0.189237478148923490158306404106012326238162346948625830327194,
      0.277429188517743176508360262560654340428504319718040836339472,
     0.277429188517743176508360262560654340428504319718040836339472,
     0.189237478148923490158306404106012326238162346948625830327194,
    -0.0400000000000000000000000000000000000000000000000000000000000,
    -0.0500000000000000000000000000000000000000000000000000000000000,
    -0.0333333333333333333333333333333333333333333333333333333333333,
    -0.0250000000000000000000000000000000000000000000000000000000000,
     0.0333333333333333333333333333333333333333333333333333333333333,
     ])


    text = """
     1    0    0.100000000000000000000000000000000000000000000000000000000000
 2    0   -0.915176561375291440520015019275342154318951387664369720564660
 2    1    1.45453440217827322805250021715664459117622483736537873607016
 3    0    0.202259190301118170324681949205488413821477543637878380814562
 3    1    0.000000000000000000000000000000000000000000000000000000000000
 3    2    0.606777570903354510974045847616465241464432630913635142443687
 4    0    0.184024714708643575149100693471120664216774047979591417844635
 4    1    0.000000000000000000000000000000000000000000000000000000000000
 4    2    0.197966831227192369068141770510388793370637287463360401555746
 4    3   -0.0729547847313632629185146671595558023015011608914382961421311
 5    0    0.0879007340206681337319777094132125475918886824944548534041378
 5    1    0.000000000000000000000000000000000000000000000000000000000000
 5    2    0.000000000000000000000000000000000000000000000000000000000000
 5    3    0.410459702520260645318174895920453426088035325902848695210406
 5    4    0.482713753678866489204726942976896106809132737721421333413261
 6    0    0.0859700504902460302188480225945808401411132615636600222593880
 6    1    0.000000000000000000000000000000000000000000000000000000000000
 6    2    0.000000000000000000000000000000000000000000000000000000000000
 6    3    0.330885963040722183948884057658753173648240154838402033448632
 6    4    0.489662957309450192844507011135898201178015478433790097210790
 6    5   -0.0731856375070850736789057580558988816340355615025188195854775
 7    0    0.120930449125333720660378854927668953958938996999703678812621
 7    1    0.000000000000000000000000000000000000000000000000000000000000
 7    2    0.000000000000000000000000000000000000000000000000000000000000
 7    3    0.000000000000000000000000000000000000000000000000000000000000
 7    4    0.260124675758295622809007617838335174368108756484693361887839
 7    5    0.0325402621549091330158899334391231259332716675992700000776101
 7    6   -0.0595780211817361001560122202563305121444953672762930724538856
 8    0    0.110854379580391483508936171010218441909425780168656559807038
 8    1    0.000000000000000000000000000000000000000000000000000000000000
 8    2    0.000000000000000000000000000000000000000000000000000000000000
 8    3    0.000000000000000000000000000000000000000000000000000000000000
 8    4    0.000000000000000000000000000000000000000000000000000000000000
 8    5   -0.0605761488255005587620924953655516875526344415354339234619466
 8    6    0.321763705601778390100898799049878904081404368603077129251110
 8    7    0.510485725608063031577759012285123416744672137031752354067590
 9    0    0.112054414752879004829715002761802363003717611158172229329393
 9    1    0.000000000000000000000000000000000000000000000000000000000000
 9    2    0.000000000000000000000000000000000000000000000000000000000000
 9    3    0.000000000000000000000000000000000000000000000000000000000000
 9    4    0.000000000000000000000000000000000000000000000000000000000000
 9    5   -0.144942775902865915672349828340980777181668499748506838876185
 9    6   -0.333269719096256706589705211415746871709467423992115497968724
 9    7    0.499269229556880061353316843969978567860276816592673201240332
 9    8    0.509504608929686104236098690045386253986643232352989602185060
10    0    0.113976783964185986138004186736901163890724752541486831640341
10    1    0.000000000000000000000000000000000000000000000000000000000000
10    2    0.000000000000000000000000000000000000000000000000000000000000
10    3    0.000000000000000000000000000000000000000000000000000000000000
10    4    0.000000000000000000000000000000000000000000000000000000000000
10    5   -0.0768813364203356938586214289120895270821349023390922987406384
10    6    0.239527360324390649107711455271882373019741311201004119339563
10    7    0.397774662368094639047830462488952104564716416343454639902613
10    8    0.0107558956873607455550609147441477450257136782823280838547024
10    9   -0.327769124164018874147061087350233395378262992392394071906457
11    0    0.0798314528280196046351426864486400322758737630423413945356284
11    1    0.000000000000000000000000000000000000000000000000000000000000
11    2    0.000000000000000000000000000000000000000000000000000000000000
11    3    0.000000000000000000000000000000000000000000000000000000000000
11    4    0.000000000000000000000000000000000000000000000000000000000000
11    5   -0.0520329686800603076514949887612959068721311443881683526937298
11    6   -0.0576954146168548881732784355283433509066159287152968723021864
11    7    0.194781915712104164976306262147382871156142921354409364738090
11    8    0.145384923188325069727524825977071194859203467568236523866582
11    9   -0.0782942710351670777553986729725692447252077047239160551335016
11   10   -0.114503299361098912184303164290554670970133218405658122674674
12    0    0.985115610164857280120041500306517278413646677314195559520529
12    1    0.000000000000000000000000000000000000000000000000000000000000
12    2    0.000000000000000000000000000000000000000000000000000000000000
12    3    0.330885963040722183948884057658753173648240154838402033448632
12    4    0.489662957309450192844507011135898201178015478433790097210790
12    5   -1.37896486574843567582112720930751902353904327148559471526397
12    6   -0.861164195027635666673916999665534573351026060987427093314412
12    7    5.78428813637537220022999785486578436006872789689499172601856
12    8    3.28807761985103566890460615937314805477268252903342356581925
12    9   -2.38633905093136384013422325215527866148401465975954104585807
12   10   -3.25479342483643918654589367587788726747711504674780680269911
12   11   -2.16343541686422982353954211300054820889678036420109999154887
13    0    0.895080295771632891049613132336585138148156279241561345991710
13    1    0.000000000000000000000000000000000000000000000000000000000000
13    2    0.197966831227192369068141770510388793370637287463360401555746
13    3   -0.0729547847313632629185146671595558023015011608914382961421311
13    4    0.0000000000000000000000000000000000000000000000000000000000000
13    5   -0.851236239662007619739049371445966793289359722875702227166105
13    6    0.398320112318533301719718614174373643336480918103773904231856
13    7    3.63937263181035606029412920047090044132027387893977804176229
13    8    1.54822877039830322365301663075174564919981736348973496313065
13    9   -2.12221714704053716026062427460427261025318461146260124401561
13   10   -1.58350398545326172713384349625753212757269188934434237975291
13   11   -1.71561608285936264922031819751349098912615880827551992973034
13   12   -0.0244036405750127452135415444412216875465593598370910566069132
14    0   -0.915176561375291440520015019275342154318951387664369720564660
14    1    1.45453440217827322805250021715664459117622483736537873607016
14    2    0.000000000000000000000000000000000000000000000000000000000000
14    3    0.000000000000000000000000000000000000000000000000000000000000
14    4   -0.777333643644968233538931228575302137803351053629547286334469
14    5    0.000000000000000000000000000000000000000000000000000000000000
14    6   -0.0910895662155176069593203555807484200111889091770101799647985
14    7    0.000000000000000000000000000000000000000000000000000000000000
14    8    0.000000000000000000000000000000000000000000000000000000000000
14    9    0.000000000000000000000000000000000000000000000000000000000000
14   10    0.000000000000000000000000000000000000000000000000000000000000
14   11    0.000000000000000000000000000000000000000000000000000000000000
14   12    0.0910895662155176069593203555807484200111889091770101799647985
14   13    0.777333643644968233538931228575302137803351053629547286334469
15    0    0.100000000000000000000000000000000000000000000000000000000000
15    1    0.000000000000000000000000000000000000000000000000000000000000
15    2   -0.157178665799771163367058998273128921867183754126709419409654
15    3    0.000000000000000000000000000000000000000000000000000000000000
15    4    0.000000000000000000000000000000000000000000000000000000000000
15    5    0.000000000000000000000000000000000000000000000000000000000000
15    6    0.000000000000000000000000000000000000000000000000000000000000
15    7    0.000000000000000000000000000000000000000000000000000000000000
15    8    0.000000000000000000000000000000000000000000000000000000000000
15    9    0.000000000000000000000000000000000000000000000000000000000000
15   10    0.000000000000000000000000000000000000000000000000000000000000
15   11    0.000000000000000000000000000000000000000000000000000000000000
15   12    0.000000000000000000000000000000000000000000000000000000000000
15   13    0.000000000000000000000000000000000000000000000000000000000000
15   14    0.157178665799771163367058998273128921867183754126709419409654
16    0    0.181781300700095283888472062582262379650443831463199521664945
16    1    0.675000000000000000000000000000000000000000000000000000000000
16    2    0.342758159847189839942220553413850871742338734703958919937260
16    3    0.000000000000000000000000000000000000000000000000000000000000
16    4    0.259111214548322744512977076191767379267783684543182428778156
16    5   -0.358278966717952089048961276721979397739750634673268802484271
16    6   -1.04594895940883306095050068756409905131588123172378489286080
16    7    0.930327845415626983292300564432428777137601651182965794680397
16    8    1.77950959431708102446142106794824453926275743243327790536000
16    9    0.100000000000000000000000000000000000000000000000000000000000
16   10   -0.282547569539044081612477785222287276408489375976211189952877
16   11   -0.159327350119972549169261984373485859278031542127551931461821
16   12   -0.145515894647001510860991961081084111308650130578626404945571
16   13   -0.259111214548322744512977076191767379267783684543182428778156
16   14   -0.342758159847189839942220553413850871742338734703958919937260
16   15   -0.675000000000000000000000000000000000000000000000000000000000
    """

    s = len(b)
    A = np.zeros((s,s))

    temp = text.split('\n')
    for i in range(1,len(temp)-1):
        if temp[i]!='':
            temp2 = temp[i].split()
            if len(temp2)!=3:
                raise Exception('error in RK10 generation')
            k = int(temp2[0])
            j = int(temp2[1])
            value = float(temp2[2])
            A[k,j] = value
    embedded=None
    
    return A,b,c,embedded
  
def checkStageOrderConditions(name)  :
  """ Checks what the maximum stage order is """
  method = getButcher(name)
  A,b,c = method['A'], method['b'], method['c'] # Butcher coefficients
  s = b.size
  for i in range(s): # go through each stage sequentially
    for k in range(1,2*s+3): # find the order of this stage
      LHS = A[i,:].dot(c**(k-1))
      RHS = (c[i]**k) / k
      if not np.allclose(LHS, RHS):
        print('stage {}/{} is of order {}'.format(i+1,s,k-1))
        break

def checkOrderConditions(name):
  """ Méthode pour tester les conditions d'ordre, dans le cas particulier du cours de Marc """
  method = getButcher(name)
  A,b,c = method['A'], method['b'], method['c'] # Butcher coefficients
  s = b.size

  if c[0] != 0:
    print('incompatible with the order conditions formula (c[0]!=0)')
    return np.nan
  if not np.all( abs(np.sum(A,axis=1) - c) <1e-14 ):
    print('method is not consitent')
    return 0
  # ordre 1
  if not np.abs( np.sum(b) - 1)<1e-14:
    print('failed order 1')
    return 0

  # ordre 2
  if not np.abs( b.dot(c) - 1/2)<1e-14:
    print('failed order 2')
    return 1

  # ordre 3
  if not np.abs( b.dot(c**2) - 1/3)<1e-14:
    print('failed order 3')
    return 2
  if not np.abs( A.dot(c).dot(b) - 1/6)<1e-14:
    print('failed order 3')
    return 2
  
  # ordre 4
  if not np.abs( b.dot(c**3) - 1/4)<1e-14:
    print('failed order 4')
    return 3
  if not np.abs( (b*c).dot(A.dot(c)) - 1/8)<1e-14:
    print('failed order 4')
    return 3
  if not np.abs( b.dot(A.dot(c**2)) - 1/12)<1e-14:
    print('failed order 4')
    return 3
  if not np.abs( b.dot( A.dot(A.dot(c)) ) - 1/24)<1e-14:
    print('failed order 4')
    return 3

  print('order is >= 4')
  return 4
  # TODO: generic order conditions
  
def testReconstructionPrecision(name):
  """ We test on a simple ODE y'=lbda*y the precision of the recsontruction of the
  solution time derivatives at the various stages of the method, and see how it converges as dt is decreased """
  pass
      
def reverseRK(A,b,c):
    """ Compute the inverse RK method """
    newA=np.zeros_like(A)
    newb=np.zeros_like(b)
    # newc=np.zeros_like(c)
    s = len(newb)
    newb = np.flip(b)
    newc = np.flip(c)
    newc = 1-newc
    for i in range(s):
      # newc[i] = 1 - c[s-i]
      for j in range(s):
          newA[i,j] = b[s-1-j] - A[s-1-i,s-1-j]
    return newA,newb,newc
  
if __name__=='__main__':
  print('Checking orders of available RK methods')
  for method in AVAILABLE_METHODS:
    print('{}: '.format(method), end='')
    checkOrderConditions(method)
    checkStageOrderConditions(method)


  import scipy
  import scipy.linalg
  import matplotlib.pyplot as plt
  y0 = 1.
  lbda = 4e0  # eigenvalue of the ODE
  y_analytic = lambda t: y0*np.exp(lbda*t)
  dydt = lambda t,y: lbda*y # ODE function
  
  chosen_time = 2.0 # time point at which the error is estimated
  
  names = ['Radau5', 'ESDIRK54A', 'L-SDIRK-33']
  # names = implicit_methods
  for name in names:
  
    method = getButcher(name)
    A,b,c = method['A'], method['b'], method['c'] # Butcher coefficients
    s = b.size
    bESDIRK = np.all(A[0,:]==0.)
    
    dt_vec = np.logspace(0,-4,100) #np.array([1/(2.**i) for i in range(8)])
      
    rel_error = np.nan*np.zeros((s, dt_vec.size))
    # Go through each stage and each value of dt
    # for each combination, create a "fake" time step scenario, such that the
    # time "chosen_time", at which the error is to be computed, corresponds to the time
    # of the selected stage.
    for i in range(s):
      for j, dt in enumerate(dt_vec):
        # find the starting tie of the overall time step, such that tn + c[i]*dt = chosen_time
        tn = chosen_time - c[i]*dt
        yn = y_analytic( tn ) # value at the start of the time step
        
        # compute the exact solution at each stage
        stage_times = tn + c*dt
        
        # # compute all the stages time derivatives
        nCase=1
        if nCase==0: # assume exact derivatives for all other stages
          stage_vals = y_analytic(stage_times)
          stage_ders = dydt(stage_times, stage_vals) 
          # recompute the chosen stage
          stage_ders[i] = 0 # so that it does not interfere with the follwogin computation (summation)
          stage_ders[i] =  (stage_vals[i]-yn)/(dt*A[i,i]) - A[i,:].dot(stage_ders)/A[i,i]
          
        elif nCase==1: # solve a linear system to compute approx derivatives
          # (y1,y2,..,ys).T = (y1*(1,1,..,1)).T + dt*A*(dty_1,..,dty_s).T
          stage_vals = y_analytic(stage_times)
          stage_ders = np.empty(s)
          # if bESDIRK: # the system is singular because of the first stage being explicit --> remove it
          #   stage_ders[0] = dydt(stage_times[0], stage_vals[0])
          #   stage_ders[1:] = scipy.linalg.solve(a = dt*A[1:,1:],
          #                                       b = stage_vals[1:] - yn)
          # else:
          # stage_ders[:] = scipy.linalg.solve(a = dt*A,
          #                                      b = stage_vals - yn)
          # OR: use pseudo-inverse to handle A, even when it is singular
          stage_ders[:] =  np.linalg.pinv(A).dot( (1/dt)*(stage_vals-yn) )
        
        elif nCase==2: # solve approximately (both solution values and derivatives)
          # from integration_esdirk import FIRK_integration
          # stage_ders = np.zeros_like( stage_times )
          # stage_vals = np.zeros_like( stage_times )
          
          # # solve the FIRK system
          # temp = FIRK_integration(fun=dydt, y0=np.array([yn]), t_span=[tn,tn+dt], nt=2, method=method,
          #                         jacfun=None, bPrint=True, vectorized=False, newtonchoice=2,
          #                         fullDebug=False)
          # stage_vals = temp.y_substeps
          # print('dt={}, stage_vals={}'.format(dt, stage_vals))
          
          # for this particular system: (I- lbda*dt*A)*K = lbda*(yn,yn,...)
          import numpy.matlib
          yrep = np.matlib.repmat(yn, s, 1)
          stage_ders = np.linalg.inv( np.eye(s) - lbda*dt*A).dot( lbda * yrep )
          stage_vals = yrep + lbda*dt*A.dot( stage_ders )
        
        else:
          raise Exception('Unknown case {}'.format(nCase))
        
        # compute error
        true_der = dydt(chosen_time, y_analytic(chosen_time) )
        rel_error[i,j] = np.abs( (stage_ders[i] - true_der)/true_der )
        

    #### Plot convergence and orders
    if bESDIRK: # do not plot the first stage convergence, as the error here will be zero
      istart=1
    else:
      istart=0
      
    fig, ax = plt.subplots(2,1,sharex=True, dpi=200)
    for i in range(istart,s):
      ax[0].loglog(dt_vec, rel_error[i,:], label='stage {}'.format(i), marker='.')
      ax[1].semilogx(dt_vec, np.gradient( np.log10(rel_error[i,:]), np.log10(dt_vec)), marker='.')
    ax[0].legend()
    ax[-1].set_xlabel('dt (s)')
    ax[-1].set_ylabel('order')
    ax[0].set_ylabel('rel error')
    ax[-1].set_ylim(-1,7)
    for a in ax:
      a.grid()
    fig.suptitle('Relative error on dydt at each stage\nfor {}'.format(name))
    plt.tight_layout()
    
    
    # alternative plot with dt*lbda
    # fig, ax = plt.subplots(2,1,sharex=True, dpi=200)
    # for i in range(istart,s):
    #   ax[0].loglog(lbda*dt_vec, rel_error[i,:], label='stage {}'.format(i), marker='.')
    #   ax[1].semilogx(lbda*dt_vec, np.gradient( np.log10(rel_error[i,:]), np.log10(dt_vec)), marker='.')
    # ax[0].legend()
    # ax[-1].set_xlabel('dt*lbda')
    # ax[-1].set_ylim(-1,5)
    # ax[-1].set_ylabel('order')
    # ax[0].set_ylabel('rel error')
    # for a in ax:
    #   a.grid()
    # fig.suptitle('Relative error on dydt at each stage\nfor {}'.format(name))
    # plt.tight_layout()
    # plt.show()