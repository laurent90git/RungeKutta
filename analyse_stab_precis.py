# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 13:32:09 2019

Visualization of the stability and precision of RUnge-Kutta methods

@author: laurent.francois@polytechnique.edu
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# set latex fonts and latex package
plt.rc('text', usetex=True)
plt.rc('font', family='serif',size='15')
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]


import rk_coeffs
from rk_coeffs import reverseRK
import sympy as sp
sp.init_printing()
from sympy.utilities import lambdify

dpi = 80 # pour export images
figsize = (8,8) # taille figure

class testPrecision:
    # Classe qui regroupe les fonctions pour faire l'analyse
    def __init__(self, re_min, re_max, im_min, im_max, n_re, n_im):
        """ This object allows to numerically determine the precision of the chosen integrator for linear
            problems of the form y'= lambda*y.
            This is done b scanning the complex eigenvalues lambda on a uniform grid :
                [re_min, re_max] + i*[im_min, immax]
        """
        self.im = np.linspace(im_min, im_max, n_im)
        self.re = np.linspace(re_min, re_max, n_re)
        self.re_xx, self.im_yy = np.meshgrid(self.re, self.im)
        self.eigvals = self.re_xx + 1j*self.im_yy

    def computeStabilityFunction(self, method, bSympy=True):
        """ returns the stability function R(z) of the Runge-Kutta methods defined by A,b,c """
        A,b,c = method['A'], method['b'], method['c']
        s = np.size(b)
        if bSympy:  # use symbolic computations to improve speed (effective for low number of stages)
          s = b.size # nombre d'étages
          A = sp.Matrix( A )
          b = sp.Matrix( b )
          c = sp.Matrix( c )
          z = sp.Symbol("z")
          I = sp.Matrix(np.eye(s))
          e = sp.Matrix(np.ones((s,)))

          M_up = I -z*A + z*e*b.T
          M_down = I-z*A

          Rsym = M_up.det()/M_down.det() # polynôme
          R = lambdify((z,),Rsym) # version numérique
        else: # no symbolic computations
          e=np.ones((s,))
          # Rlbda = lambda z: 1+z*np.dot(b, np.linalg.inv(np.eye(s)-z*A).dot(e))
          def Rlbda(z):
            try:
              return 1+z*np.dot(b, np.linalg.inv(np.eye(s)-z*A).dot(e))
            except np.linalg.LinAlgError: # singular matrix
              return np.nan
          R=np.vectorize(Rlbda)
          Rsym=None
        return R, Rsym
        

    def plotOrderStars(self, method):
        """ Plots the order star of the RK method, i.e. the locus such that |R(z)|/|exp(z)|=1"""
        xx,yy = self.re_xx, self.im_yy
        zz = self.eigvals
        x= self.re
        y= self.im

        Rfun, Rsym = self.computeStabilityFunction(method)
        RR   = Rfun(zz) # solution RK
        expp = np.exp(self.eigvals) # solution analytique
        rr = np.abs(RR) # ratio d'augmentation


        # add contour of |R(z)|-|exp(z)[
        order_star = np.abs(RR)-np.abs(expp)
        # order_star = np.abs(RR/expp)

        # map_levels = np.linspace(-5,5,50)
        # map_levels = np.linspace(-3,3,50)
        map_levels = np.array( [-1e99,0,1e99] )
        ratio_height_over_width = np.abs( (np.max(y)-np.min(y))/(np.max(x)-np.min(x)) )
        fig, ax = plt.subplots(1,1,dpi=80, figsize=(8, 8*ratio_height_over_width))
        # plot des axes Im et Re
        ax.plot([np.min(x), np.max(x)], [0,0], color=(0,0,0), linestyle='--', linewidth=0.4)
        ax.plot([0,0], [np.min(y), np.max(y)], color=(0,0,0), linestyle='--', linewidth=0.4)

        # contour de la précision relative
        cs = ax.contourf(xx,yy, order_star, levels=map_levels) #, cmap = 'gist_earth')
        fig.colorbar(cs)

        # add contour lines for precision
        levels = np.array( [-1e99,0,1e99] )
        level_contour = ax.contour(xx,yy, order_star, levels=levels, colors='k')
        ax.clabel(level_contour, inline=1, fontsize=10,  fmt='%1.0f')

        ## Axis description
        fig.suptitle(f'Order Star + champ |R(z)|-|exp(z)|')
        ax.set_xlabel(r'Re$(\lambda\Delta t)$')
        ax.set_ylabel(r'Im$(\lambda\Delta t)$')

        # add stability domain
        ax.contour(xx,yy,rr,levels=[0,1],colors='r')

        # hachurer en rouge la zone instable
        rr_sup_1 = np.where(np.abs(rr) >= 1)
        temp = np.zeros_like(rr)
        temp[rr_sup_1] = 1.
        plt.rcParams['hatch.color']='r'  # seul moyen que j'ai trouvé pour avoir des hachures rouges...
        cs   = ax.contourf(xx,yy, temp, levels=[0,0.5,1.5],  #levels=[0., 1.0, 1.5],
                         hatches=[None,'\\\\', '\\\\'], alpha = 0.)
        plt.rcParams['hatch.color']=[0,0,0]
        ax.axis('equal')

        if np.any(np.isnan(order_star)):
            print('NaNs detected')
            fig2, ax = plt.subplots(1,1)
            # plot des axes Im et Re
            ax.plot([np.min(x), np.max(x)], [0,0], color=(0,0,0), linestyle='--', linewidth=0.4)
            ax.plot([0,0], [np.min(y), np.max(y)], color=(0,0,0), linestyle='--', linewidth=0.4)
            # plots overflow region
            map_levels = np.array([0.,0.5, 1.5])
            cs            = ax.contourf(xx,yy, np.isnan(order_star)*1, levels=map_levels)#, cmap = 'gist_earth')
            fig2.colorbar(cs)
            fig2.suptitle('emplacement des NaNs')



    def plotStabilityRegionRK(self, method, bAdditionalPlots=True, bSympy=True):
        # Contour de la précision / stabilité
        A,b,c = method['A'], method['b'], method['c']
        methodname = method['name']
        xx,yy = self.re_xx, self.im_yy
        zz = self.eigvals
        x= self.re
        y= self.im

        Rfun, Rsym = self.computeStabilityFunction(method, bSympy=bSympy)
        RR   = Rfun(zz) # solution RK
        expp = np.exp(self.eigvals) # solution analytique
        rr = np.abs(RR) # ratio d'augmentation


        # add contour of precision relative to exponential
        pprecision1 = 100*np.abs((RR-expp)/expp) # précision en pourcents par rapport à la solution analytique
        pprecision2 = 100*np.abs((RR-expp)/RR) # précision en pourcents par rapport à la solution numérique
        pprecision3 = np.minimum(pprecision1,pprecision2)
        # map_levels = np.linspace(-5,5,50)
        map_levels = np.linspace(-5,4,20)
        # map_levels = np.linspace(-3,3,50)
        # map_levels = np.hstack((-99, map_levels, 99))
        cmap=None

        if 1: # full plots
            hachurer_zone_precision_100p = True
            hachurer_zone_instable = True
            ajouter_order_star = True
            isolignes_precision = True
            contour_precision_relative = True
        else:
            hachurer_zone_precision_100p = False
            hachurer_zone_instable = False
            ajouter_order_star = False
            isolignes_precision = True
            contour_precision_relative=False

        # for cmap in  [
        #     'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
        #     'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
        #     'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']:

        for pprecision,name in [
                                (pprecision1, '$\exp(z)$'),
                                # (pprecision2, '(avec réf = R(z))'),
                                # (pprecision3, '(erreur min)'),
                                ]:
            ratio_height_over_width = np.abs( (np.max(y)-np.min(y))/(np.max(x)-np.min(x)) )
            fig, ax = plt.subplots(1,1,dpi=300, figsize=(10, 8*ratio_height_over_width))
            # fig, ax = plt.subplots(1,1,figsize=dpi=300)#, figsize=(8, 8*ratio_height_over_width))
            # plot des axes Im et Re
            ax.axhline(0, color=(0,0,0), linestyle='--', linewidth=0.4)
            ax.axvline(0, color=(0,0,0), linestyle='--', linewidth=0.4)

            # contour de la précision relative
            if contour_precision_relative:
                cs = ax.contourf(xx,yy, np.log10(pprecision), levels=map_levels, cmap=cmap) #, cmap = 'gist_earth')
                fig.colorbar(cs)

            # add contour lines for precision
            if isolignes_precision:
                # levels = np.round(np.logspace(0,2,5)).astype(int)
                # levels = np.array( range(0,200,25) )
                # levels = np.array([1,5,25,50,100,500]) #np.array(range(0,200,25))
                levels = 10.**np.array(range(-3,3+1))
                                      #np.array([1,10,100,1000]) #np.array(range(0,200,25))
                level_contour = ax.contour(xx,yy, pprecision, levels=levels, colors='k')
                # ax.clabel(level_contour, inline=1, fontsize=10,  fmt='%1.0f')
                ax.clabel(level_contour, inline=1, fontsize=10,  fmt='%1.0e')

            ## Axis description
            fig.suptitle(f'{methodname}: domaine de stabilité (rouge), order star (bleu),\niso-contour de précision (\%) et map de la précision (log10) par rapport à {name}')
            ax.set_xlabel(r'Re$(\lambda\Delta t)$')
            ax.set_ylabel(r'Im$(\lambda\Delta t)$')

            # hachurer les zones > 100%
            if hachurer_zone_precision_100p:
                error_sup_100 = np.where(pprecision >= 100.)
                temp = np.zeros_like(pprecision)
                temp[error_sup_100] = 1.
                cs   = ax.contourf(xx,yy, temp, colors=['w', 'w', 'w'], levels=[0,0.5,1.5],  #levels=[0., 1.0, 1.5],
                                   hatches=[None,'//', '//'], alpha = 0.)

            # add stability domain
            ax.contour(xx,yy,rr,levels=[0,1],colors='r')

            # hachurer en rouge la zone instable
            if hachurer_zone_instable:
              rr_sup_1 = np.where(np.abs(rr) >= 1)
              temp = np.zeros_like(rr)
              temp[rr_sup_1] = 1.
              plt.rcParams['hatch.color']='r'  # seul moyen que j'ai trouvé pour avoir des hachures rouges...
              cs   = ax.contourf(xx,yy, temp, levels=[0,0.5,1.5],  #levels=[0., 1.0, 1.5],
                               hatches=[None,'\\\\', '\\\\'], alpha = 0.)
              plt.rcParams['hatch.color']=[0,0,0]

            # add order star
            if ajouter_order_star:
                order_star = np.abs(RR)/np.abs(expp)
                plt.rcParams['hatch.color']='b'  # seul moyen que j'ai trouvé pour avoir des hachures colorées...
                cs   = ax.contour(xx,yy, order_star, colors='b', levels=[0.,1.])
                # cs   = ax.contourf(xx,yy, order_star, levels=[0., 1.0, 1.5],
                                  # hatches=[None,'\\\\',None], alpha = 0.)
                plt.rcParams['hatch.color']=[0,0,0]


            # ax.axis('equal')
        if  bAdditionalPlots:
            temp = np.isnan(expp)
            if np.any(temp): # NaNs dans les résultats
                print('issue in theoretical result')
                fig2, ax = plt.subplots(1,1)
                # plot des axes Im et Re
                ax.plot([np.min(x), np.max(x)], [0,0], color=(0,0,0), linestyle='--', linewidth=0.4)
                ax.plot([0,0], [np.min(y), np.max(y)], color=(0,0,0), linestyle='--', linewidth=0.4)
                # plots overflow region
                map_levels = np.array([0.,0.5, 1.5])
                cs            = ax.contourf(xx,yy, 1.0*temp, levels=map_levels)#, cmap = 'gist_earth')
                fig2.colorbar(cs)
                fig2.suptitle('emplacement des NaNs dans exp(z)')
    
            if 1: # Tracé de l'expansion de la solution numérique
                fig2, ax = plt.subplots(1,1)
                # plot des axes Im et Re
                ax.plot([np.min(x), np.max(x)], [0,0], color=(0,0,0), linestyle='--', linewidth=0.4)
                ax.plot([0,0], [np.min(y), np.max(y)], color=(0,0,0), linestyle='--', linewidth=0.4)
    
                # plot "expansion" ratio
                map_levels = np.array(np.linspace(-3,3,50)) #np.logspace(-3,3,20)
                cs = ax.contourf(xx,yy, np.log10(rr), levels=map_levels)#, cmap = 'gist_earth')
                # map_levels = np.array(np.logspace(0,3,20))
                # cs = ax.contourf(xx,yy, rr, levels=map_levels)#, cmap = 'gist_earth')
                fig2.colorbar(cs)
                ax.contour(xx,yy,rr,levels=[0,1],colors='r') # add stability domain
                ax.set_title('Numerical expansion ratio $\log_{10}(|R(z)|)$')
    
            
            if 1: # tracé de l'expansion pour la solution exacte
                fig2, ax = plt.subplots(1,1)
                # plot des axes Im et Re
                ax.axhline(0, color=(0,0,0), linestyle='--', linewidth=0.4)
                ax.axvline(0, color=(0,0,0), linestyle='--', linewidth=0.4)
                # plot "expansion" ratio
                cs = ax.contourf(xx,yy, np.log10(np.abs(expp)), levels=map_levels)#, cmap = 'gist_earth')
                fig2.colorbar(cs)
                ax.contour(xx,yy,rr,levels=[0,1],colors='r') # add stability domain
                ax.set_title('exact expansion ratio $\log_{10}(|\exp(z)|)$')
    
            if 1: # Tracé de l'angle de R(z) pour la solution numérique
                fig2, ax = plt.subplots(1,1)
                ax.axhline(0, color=(0,0,0), linestyle='--', linewidth=0.4)
                ax.axvline(0, color=(0,0,0), linestyle='--', linewidth=0.4)
                cs = ax.contourf(xx,yy, np.angle(RR), levels=50, cmap='hsv')
                fig2.colorbar(cs)
                ax.contour(xx,yy,rr,levels=[0,1],colors='r') # add stability domain
                ax.set_title('phase of $R(z)$')
                
            if 1: # Tracé de l'angle pour la solution exacte
                fig2, ax = plt.subplots(1,1)
                # plot des axes Im et Re
                ax.axhline(0, color=(0,0,0), linestyle='--', linewidth=0.4)
                ax.axvline(0, color=(0,0,0), linestyle='--', linewidth=0.4)
                cs = ax.contourf(xx,yy, np.angle(expp), levels=50, cmap='hsv')
                fig2.colorbar(cs)
                ax.contour(xx,yy,rr,levels=[0,1],colors='r') # add stability domain
                ax.set_title('phase of $\exp(z)$')
        plt.show()

        return pprecision, pprecision2

    def compareStabilityDomains(self, list_rk_names):
        xx,yy = self.re_xx, self.im_yy
        zz = self.eigvals
        x= self.re
        y= self.im


        # ratio_height_over_width = np.abs( (np.max(y)-np.min(y))/(np.max(x)-np.min(x)) )
        # fig, ax = plt.subplots(1,1,dpi=80, figsize=(8, 8*ratio_height_over_width))
        fig, ax = plt.subplots(1,1,figsize=(6,6),dpi=300)
        # plot des axes Im et Re
        ax.axhline(0, color=(0,0,0), linestyle='--', linewidth=0.4)
        ax.axvline(0, color=(0,0,0), linestyle='--', linewidth=0.4)

        ## Axis description
        fig.suptitle(f'Domaines de stabilité')
        ax.set_xlabel(r'Re$(\lambda\Delta t)$')
        ax.set_ylabel(r'Im$(\lambda\Delta t)$')

        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        for i,name in enumerate(list_rk_names):
            print(name)
            method = rk_coeffs.getButcher(name)
            if np.size(method['b'])>6:
              bSympy=False # c'est plus rapide d'éviter Sympy, on dirait qu'il n'arrive pas à calculer les détermiannts analytiquement...
            else:
              bSympy=True
            Rfun, Rsym = self.computeStabilityFunction(method, bSympy=bSympy)
            RR   = Rfun(zz)
            rr = np.abs(RR) # ratio d'augmentation

            current_color=colors[i]
            # add stability domain
            ax.contour(xx,yy,rr,levels=[0,1],colors=current_color)

            # hachurer en rouge la zone instable
            # rr_sup_1 = np.where(np.abs(rr) >= 1)
            # temp = np.zeros_like(rr)
            # temp[rr_sup_1] = 1.
            # plt.rcParams['hatch.color']=current_color  # seul moyen que j'ai trouvé pour avoir des hachures rouges...
            # cs   = ax.contourf(xx,yy, temp, levels=[0,0.5,1.5],  #levels=[0., 1.0, 1.5],
            #                  hatches=[None,'\\\\', '\\\\'], alpha = 0.)
            # plt.rcParams['hatch.color']=[0,0,0]
        # add legend
        proxy = [plt.Rectangle((0,0),1,1,fc = colors[i]) for i in range(len(list_rk_names))]
        ax.legend(proxy, list_rk_names)

        ax.axis('square')
        plt.xlim(-10,15)

if __name__=='__main__':
    # Setup warnings such that any problem with complex number raises an error
    import warnings
    warnings.simplefilter("error", np.ComplexWarning) #to ensure we are not dropping complex perturbations

    # Choose the area of the complex plane that will be analysed
    # test = testPrecision(re_min=-100, re_max=100, im_min=0., im_max=100, n_re=1000, n_im=1001) # on aperçoit des formes intriguantes en terme d'iso-contour de précision
    # test = testPrecision(re_min=-10, re_max=10, im_min=-5, im_max=5, n_re=200, n_im=202)
    # test = testPrecision(re_min=-20, re_max=20, im_min=-20, im_max=20, n_re=200, n_im=202)
    # test = testPrecision(re_min=-3, re_max=3, im_min=-3, im_max=3, n_re=200, n_im=202)
    test = testPrecision(re_min=-6, re_max=6, im_min=-6, im_max=6, n_re=200, n_im=202)

    #%% Tracé des zones de stabilité, contour de précision...
    
    # for name in ['ie','crkn','esdirk32a','esdirk43b','rk4','esdirk54a','radau5']:
    for name in ['ie','rk4','radau5']:
        method = rk_coeffs.getButcher(name)

        # tracé du contour de la précision par rapport à l'exponentielle
        pprecision, pprecision2 = test.plotStabilityRegionRK(method, bAdditionalPlots=True)
        plt.suptitle(name)

    raise Exception('stop')
    #%% Calculs spécifiques
    name ='radau5'
    method = rk_coeffs.getButcher(name)
    R, Rsym = test.computeStabilityFunction(method, bSympy=True)

    # calcul du rayon spectral
    eigvals = np.linalg.eigvals(method['A'])
    rho = np.max(np.abs(eigvals))
    print(name)
    print('spectral radius: rho=', rho)
    print('R(rho)=', R(rho))
    print('R(1/rho)=', R(1/rho))
    print('R(1/rho + 1e-6)=', R(1/rho + 1e-6))


    #% tracé des order stars
    test.plotOrderStars(method)

    #%% estimate R(+\infty)
    # avec Sympy, la limite est calculable analytiquement (avec R ou Rsym)
    try:
      print('R(+infty)=', Rsym.limit('z',np.inf))
      print('R(+1j*infty)=', Rsym.limit('z',1j*1e99))
      print('R(+1j*infty)=', R(1j*1e99)) #1j*np.inf ne marche pas en Sympy (donne un NaN)
    except:
      pass
    z_vec = np.logspace(-10,10,10000)
    Rvalues = R(z_vec)

    plt.figure()
    plt.loglog(np.abs(z_vec), np.abs(Rvalues))
    plt.grid(which='both')
    plt.title('comportement limite du polynôme de stabilité')
    plt.xlabel('|z|')
    plt.ylabel('|R(z)|')
    # plt.ylim((1e-1, 1e0))
    plt.show()



    #%% Comparaison domaines stabilité
    # test.compareStabilityDomains(list_rk_names=['rk4', 'radau5', 'ie','rk10', 'esdirk54a', 'esdirk43b'])
    test.compareStabilityDomains(list_rk_names=['ee', 'ie', 'rk4', 'radau5', 'rk10'])
    # test.compareStabilityDomains(list_rk_names=['EE','RK4'])
    # test.compareStabilityDomains(list_rk_names=['ee','ie'])
    plt.axis('square')
    plt.xlim(-5,3)
    plt.ylim(-4,4)
    plt.savefig('stab_RK_explicites.png',dpi=300)

    #%% Plot comparison of precision zone along the real axis
    polys=[]
    # names = ['ee','rk4','rk10']#, 'radau5', 'esdirk54a', 'ESDIRK43B', 'ESDIRK32A']
    names =['ee', 'ie', 'rk4', 'radau5']
    # names = ['radau5', 'esdirk54a', 'ESDIRK43B', 'ESDIRK32A']
    # names = ['ie', 'esdirk32a', 'radau5']#, 'ESDIRK43B', 'ESDIRK32A']
    # names = ['reversed-RK23', 'reversed-RK4', 'reversed-RK45',
              # 'reversed-RK10', 'radau5', 'RK10']
    # names = ['reversed-RK45-4', 'reversed-RK45-5', 'RK45']
    for name in names:
        print(name)
        method = rk_coeffs.getButcher(name)
        if np.size(method['b'])>6:
          bSympy=False # c'est plus rapide d'éviter Sympy, on dirait qu'il n'arrive pas à calculer les détermiannts analytiquement...
        else:
          bSympy=True
        polys.append( test.computeStabilityFunction(method, bSympy=bSympy) )



    #%% Tracé de l'évolution de la précision le long de l'axe réel
    z_vec = np.linspace(-3,3,10000)
    th_sol = np.exp(z_vec)
    for i in range(2):
      fig,ax = plt.subplots(1,1,dpi=200,sharex=True)
      ax=[ax]
      for j,p in enumerate(polys):
          r = p[0](z_vec)
          error = (r - th_sol)
          if i==0: # relative à l'exponentielle
            relativPrecision = np.abs(error/th_sol)
            description = "précision de la solution numérique\npar rapport à l'exponentielle"
          else: # relative à la sol num
              relativPrecision = np.abs(error/r)
              description = "précision de l'exponentielle\npar rapport à la solution numérique"
          ax[0].semilogy(z_vec, relativPrecision, label=names[j])
      ax[0].set_ylim(1e-8, 1e2)
      ax[0].set_title(description)
      ax[0].legend()
      ax[0].set_ylabel('relative error')
      ax[0].set_xlabel(r'$\lambda \Delta t$')
      ax[0].grid(which='both', axis='both')

    #%% Tracé de l'évolution de R(z) le long de l'axe réel
    z_vec = np.linspace(-10,5,10000)
    th_sol = np.exp(z_vec)
    for i in range(2):
      fig,ax = plt.subplots(1,1,dpi=200,figsize=(7,5),sharex=True)
      ax=[ax]
      ax[0].plot(z_vec, np.exp(z_vec), linestyle='-', color=[0,0,0], label='$\exp(z)$')
      ax[0].axhline(1, linestyle='--', color=[0,0,0], label='$|R(z)|=1$')
      for j,p in enumerate(polys):
          r = p[0](z_vec)
          ax[0].semilogy(z_vec, np.abs(r), label=names[j])
      
      ax[0].set_ylim( 1e-2, 1e2)
      ax[0].legend(framealpha=0.5)
      ax[0].set_ylabel('$|R(z)|$')
      ax[0].set_xlabel('$z$')
      ax[0].grid(which='both', axis='both')
      ax[0].set_title('Module de la fonction de stabilité $R(z)$')

    #%% Comparaison entre embedded methods
    polys=[]
    # names = ['esdirk54a', 'ESDIRK54A-V4']
    names = ['ESDIRK43B-3', 'ESDIRK43B']
    # names = ['RK45', 'ESDIRK54A-V4']
    for name in names:
        print(name)
        method = rk_coeffs.getButcher(name)
        # if np.size(method['b'])>6:
        #   bSympy=False # c'est plus rapide d'éviter Sympy, on dirait qu'il n'arrive pas à calculer les détermiannts analytiquement...
        # else:
        #   bSympy=True
        bSympy=True
        polys.append( test.computeStabilityFunction(method, bSympy=bSympy) )

    z_vec = np.linspace(-6,8,10000)
    th_sol = np.exp(z_vec)

    fig,ax = plt.subplots(1,1,sharex=True)
    ax=[ax]
    for j,p in enumerate(polys):
        if j==0:
          r_ref = p[0](z_vec)
          continue
        r = p[0](z_vec)
        error = (r - r_ref)
        relativPrecision = np.abs(error/r_ref)
        ax[0].semilogy(z_vec, relativPrecision, label=names[j])
        ax[0].axhline(1, label=None, color='tab:red', linestyle='--')
    ax[0].set_ylim( 1e-15, 1e10)
    ax[0].set_title('erreur relative (vs {})'.format(names[0]))
    ax[0].legend()
    ax[0].set_ylabel('erreur relative')
    ax[0].set_xlabel('z')

    ax[0].grid(which='both', axis='both')

    #%% Tracé de l'évolution de la précision le long de l'axe réel (version log)
    for i in range(3): # boucle sur la référence prise pour la précision (num ou analytique)
      fig,ax = plt.subplots(2,1,figsize=(7,5),sharex=True)
      for k in range(2): # boucle sur le demi-plan considéré (Re>0 ou Re<0)
        if k==0: # Re(z)>0
          z_vec = np.logspace(-10,2,500)
          title='$Re(z)>0$'
        else:
          z_vec = -np.logspace(-10,20,500)
          title='$Re(z)<0$'
        for j,p in enumerate(polys): # boucle sur les méthodes RK
          r = p[0](z_vec)
          th_sol = np.exp(z_vec)
          error = (r - th_sol)
          if i==0: # relative à l'exponentielle
            relativPrecision = np.abs(error/th_sol)
            description = "précision de la sol numérique par rapport à l'exponentielle"
          elif i==1: # relative à la sol num
            relativPrecision = np.abs(error/r)
            description = "précision de l'exponentielle par rapport à la solution numérique"
          elif i==2: # absolue
            relativPrecision = np.abs(error)
            description = "erreur absolue"
          else:
            raise Exception('valeur du choix de précision non acceptée')
          ax[k].loglog(np.abs(z_vec), relativPrecision, label=names[j])
        ax[k].set_title(title)
        # ax[k].set_ylabel('précision (%)')
        ax[k].grid(which='both', axis='both')
        ax[k].set_ylim(1e-16, 1e5)
      fig.suptitle(description)
      ax[-1].set_xlabel('$|Re(z)|$')
      ax[0].legend(ncol=2)
      # C'est bizarre, les méthodes ESDIRK ne semblent pas L-stables, puisque l'erreur absolue ne tend pas vers 0 lorsque Re(z)-->-inf...