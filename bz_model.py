import numpy as np

#########################################################
#########################################################
class bz_1d_2eq_model:
    
    ##########################################################################################
    def __init__(self, eps=1e-2, f=3, q=2e-3, db=2.5e-3, dc=1.5e-3, xmin=0., xmax=4., nx=100):
        self.eps = eps
        self.f = f
        self.q = q
        self.db = db
        self.dc = dc
        self.xmin = xmin
        self.xmax = xmax
        self.nx = nx
        self.dx = (xmax-xmin)/(nx+1)
        self.dboverdxdx = db/(self.dx*self.dx)
        self.dcoverdxdx = dc/(self.dx*self.dx)
       
        
    ##########################################################
    def fcn(self, t, y):

        eps = self.eps
        f = self.f
        q = self.q

        nx  = self.nx

        dboverdxdx = self.dboverdxdx
        dcoverdxdx = self.dcoverdxdx

        ytmp = y.reshape(nx, 2)
        b = ytmp[:,0]
        c = ytmp[:,1]
        
        ydot = np.zeros((nx,2))

        ydot[0,0] = dboverdxdx*(-b[0]+b[1]) + (1/eps)*(b[0]*(1-b[0]) + (f*(q-b[0])*c[0])/(q+b[0]))
        ydot[0,1] = dcoverdxdx*(-c[0]+c[1]) + (b[0] - c[0])
        
        ydot[1:-1,0] = dboverdxdx*(b[:-2]-2*b[1:-1]+b[2:]) + (1/eps)*(b[1:-1]*(1-b[1:-1]) + (f*(q-b[1:-1])*c[1:-1])/(q+b[1:-1]))
        ydot[1:-1,1] = dcoverdxdx*(c[:-2]-2*c[1:-1]+c[2:]) + (b[1:-1]-c[1:-1])
 
        ydot[-1,0] = dboverdxdx*(b[-2]-b[-1]) + (1/eps)*(b[-1]*(1 - b[-1]) + (f*(q-b[-1])*c[-1])/(q+b[-1]))
        ydot[-1,1] = dcoverdxdx*(c[-2]-c[-1]) + (b[-1] - c[-1])

        return np.ravel(ydot)


    ##########################################################
    def init(self):

        f   = self.f
        q   = self.q
        nx  = self.nx

        b = np.zeros(nx)
        c = np.zeros(nx)
 
        y = np.zeros(2*nx)
 
        ylim = 0.05
 
        for inx in range(nx//20):
            xcoor = 0.5
            ycoor = inx/(nx/20) - ylim
 
            if (ycoor >= 0. and ycoor<= 0.3*xcoor):
                b[inx] = 0.8
            else:
                b[inx] = q*(f+1)/(f-1)
 
            if (ycoor>=0.):
                c[inx] = q*(f+1)/(f-1) + np.arctan(ycoor/xcoor)/(8*np.pi*f)
            else:
                c[inx]= q*(f+1)/(f-1) + (np.arctan(ycoor/xcoor) + 2*np.pi)/(8*np.pi*f)
 
        for inx in range(nx//20, nx):
            b[inx] = b[nx//20]
            c[inx] = c[nx//20]
 
        for inx in range(nx):
            irow = inx*2
            y[irow]   = b[inx]
            y[irow+1] = c[inx]
            
        return y
    
    
#########################################################
#########################################################
class bz_1d_3eq_model:
    
    ##############################################################################################################
    def __init__(self, eps=1e-2, mu=1e-5, f=3, q=2e-3, da=2.5e-3, db=2.5e-3, dc=1.5e-3, xmin=0., xmax=4., nx=100):
        self.eps = eps
        self.mu = mu
        self.f = f
        self.q = q
        self.da = da
        self.db = db
        self.dc = dc
        self.xmin = xmin
        self.xmax = xmax
        self.nx = nx
        self.dx = (xmax-xmin)/(nx+1)
        self.daoverdxdx = da/(self.dx*self.dx)
        self.dboverdxdx = db/(self.dx*self.dx)
        self.dcoverdxdx = dc/(self.dx*self.dx)
       
        
    ##########################################################
    def fcn(self, t, y):

        eps = self.eps
        mu = self.mu
        f = self.f
        q = self.q

        nx  = self.nx

        daoverdxdx = self.daoverdxdx
        dboverdxdx = self.dboverdxdx
        dcoverdxdx = self.dcoverdxdx

        ytmp = y.reshape(nx, 3)
        a = ytmp[:,0]
        b = ytmp[:,1]
        c = ytmp[:,2]
        
        ydot = np.zeros((nx,3))

        ydot[0,0] = daoverdxdx*(-a[0]+a[1]) + (1/mu)*(-q*a[0] - a[0]*b[0] + f*c[0])
        ydot[0,1] = dboverdxdx*(-b[0]+b[1]) + (1/eps)*(q*a[0] - a[0]*b[0] + b[0]*(1-b[0]))
        ydot[0,2] = dcoverdxdx*(-c[0]+c[1]) + (b[0] - c[0])
        
        ydot[1:-1,0] = daoverdxdx*(a[:-2]-2*a[1:-1]+a[2:]) + (1/mu)*(-q*a[1:-1] - a[1:-1]*b[1:-1] + f*c[1:-1])
        ydot[1:-1,1] = dboverdxdx*(b[:-2]-2*b[1:-1]+b[2:]) + (1/eps)*(q*a[1:-1] - a[1:-1]*b[1:-1] + b[1:-1]*(1-b[1:-1]))
        ydot[1:-1,2] = dcoverdxdx*(c[:-2]-2*c[1:-1]+c[2:]) + (b[1:-1]-c[1:-1])
 
        ydot[-1,0] = daoverdxdx*(a[-2]-a[-1]) + (1/mu)*(-q*a[-1] - a[-1]*b[-1] + f*c[-1])
        ydot[-1,1] = dboverdxdx*(b[-2]-b[-1]) + (1/eps)*(q*a[-1] - a[-1]*b[-1] + b[-1]*(1-b[-1]))
        ydot[-1,2] = dcoverdxdx*(c[-2]-c[-1]) + (b[-1]-c[-1])

        return np.ravel(ydot)

    ##########################################################
    def init(self):

        f   = self.f
        q   = self.q
        nx  = self.nx

        a = np.zeros(nx)
        b = np.zeros(nx)
        c = np.zeros(nx)
 
        y = np.zeros(3*nx)
 
        ylim = 0.05
 
        for inx in range(nx//20):
            xcoor = 0.5
            ycoor = inx/(nx/20) - ylim
 
            if (ycoor >= 0. and ycoor<= 0.3*xcoor):
                b[inx] = 0.8
            else:
                b[inx] = q*(f+1)/(f-1)
 
            if (ycoor>=0.):
                c[inx] = q*(f+1)/(f-1) + np.arctan(ycoor/xcoor)/(8*np.pi*f)
            else:
                c[inx]= q*(f+1)/(f-1) + (np.arctan(ycoor/xcoor) + 2*np.pi)/(8*np.pi*f)
 
        for inx in range(nx//20, nx):
            b[inx] = b[nx//20]
            c[inx] = c[nx//20]

        a = (f*c)/(q+b)
 
        for inx in range(nx):
            irow = inx*3
            y[irow]   = a[inx]
            y[irow+1] = b[inx]
            y[irow+2] = c[inx]

        return y
