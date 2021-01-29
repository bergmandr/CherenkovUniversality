import numpy as np
from scipy.constants import value,nano
import atmosphere as at

class Shower():
    """A class for generating extensive air shower profiles and their Cherenkov
    outputs. The shower can either be a Gaisser Hillas shower or a Griessen
    Shower.

    Parameters:
    X_max: depth at shower max (g/cm^2)
    N_max: number of charged particles at X_max
    h0: height of first interaction (meters)
    X0: Start depth
    theta: Polar angle of the shower axis with respect to vertical. Vertical
    is defined as normal to the Earth's surface at the point where the axis
    intersects with the surface.
    phi: azimuthal angle of axis intercept (radians) measured from the x
    axis. Standard physics spherical coordinate convention. Positive x axis is
    North, positive y axis is west.
    profile: Shower type, either 'GN' for Greisen, or GH for Gaisser-Hillas.
    direction: Shower direction, either 'up' for upward going showers, or 'down'
    for downward going showers.
    """
    earth_radius = 6.371e6
    Lambda = 70
    atm = at.Atmosphere()
    axis_h = np.linspace(0,atm.maximum_height,10000)
    axis_rho = atm.density(axis_h)
    axis_delta = atm.delta(axis_h)
    axis_Moliere = 96. / axis_rho
    Moliere_data = np.load('lateral.npz')
    t_Moliere = Moliere_data['t']
    AVG_Moliere = Moliere_data['avg']

    def __init__(self,X_max,N_max,h0,theta,direction,phi=0,type='GH'):
        self.type = type
        self.reset_shower(X_max,N_max,h0,theta,direction,phi,type)

    def reset_shower(self,X_max,N_max,h0,theta,direction,phi=0,type='GH'):
        '''Set necessary attributes and perform calculations
        '''
        self.input_X_max = X_max
        self.N_max = N_max
        self.h0 = h0
        self.direction = direction
        self.theta = theta
        self.axis_r, self.axis_start_r = self.set_axis(theta,h0)
        self.axis_X, self.axis_dr, self.X0 = self.set_depth(self.axis_r,
                self.axis_start_r)
        self.X_max = X_max + self.X0
        self.axis_nch = self.size(self.axis_X)
        self.axis_nch[self.axis_nch<1.e3] = 0
        self.i_ch = np.nonzero(self.axis_nch)
        self.shower_X = self.axis_X[self.i_ch]
        self.shower_r = self.axis_r[self.i_ch]
        self.shower_Moliere = self.axis_Moliere[self.i_ch]
        self.shower_t = self.stage(self.shower_X)
        self.shower_avg_M = np.interp(self.shower_t,self.t_Moliere,self.AVG_Moliere)
        self.shower_rms_w = self.shower_avg_M * self.shower_Moliere


    def h_to_axis_R_LOC(self,h,theta):
        '''Return the length along the shower axis from the point of Earth
        emergence to the height above the surface specified

        Parameters:
        h: array of heights in meters
        theta: polar angle of shower axis (radians)

        returns: r (same size as h), an array of distances along the shower
        axis_sp.
        '''
        cos_EM = np.cos(np.pi-theta)
        R = self.earth_radius
        r_CoE= h + R # distance from the center of the earth to the specified height
        r = R*cos_EM + np.sqrt(R**2*cos_EM**2-R**2+r_CoE**2)
        return r

    def set_axis(self,theta,h0):
        '''Create a table of distances along the shower axis
        '''
        axis_r = self.h_to_axis_R_LOC(self.axis_h, theta)
        axis_start_r = self.h_to_axis_R_LOC(self.h0, theta)
        return axis_r, axis_start_r

    def set_depth(self,axis_r,axis_start_r):
        '''Integrate atmospheric density over selected direction to create
        a table of depth values.
        '''
        axis_dr = axis_r[1:] - axis_r[:-1]
        axis_deltaX = np.sqrt(self.axis_rho[1:]*self.axis_rho[:-1])*axis_dr / 10# converting to g/cm^2
        if self.direction == 'up':
            axis_X = np.concatenate((np.array([0]),np.cumsum(axis_deltaX)))
        elif self.direction == 'down':
            axis_X = np.concatenate((np.cumsum(axis_deltaX[::-1])[::-1],
                    np.array([0])))
        axis_dr = np.concatenate((np.array([0]),axis_dr))
        X0 = np.interp(axis_start_r,axis_r,axis_X)
        return axis_X, axis_dr, X0

    def size(self,X):
        """Return the size of the shower at a slant-depth X

        Parameters:
            X: the slant depth at which to calculate the shower size [g/cm2]

        Returns:
            N: the shower size
        """
        if self.type == 'GH':
            value = self.GaisserHillas(X)
        elif self.type == 'GN':
            value = self.Greisen(X)
        return value

    def GaisserHillas(self,X):
        '''Return the size of a GH shower at a given depth.
        '''
        x =         (X-self.X0)/self.Lambda
        g0 = x>0.
        m = (self.X_max-self.X0)/self.Lambda
        n = np.zeros_like(x)
        n[g0] = np.exp( m*(np.log(x[g0])-np.log(m)) - (x[g0]-m) )
        return self.N_max * n

    def Greisen(self,X,p=36.62):
        '''Return the size of a Greisen shower at a given depth.
        '''
        X[X < self.X0] = self.X0
        Delta = X - self.X_max
        W = self.X_max-self.X0
        eps = Delta / W
        s = (1+eps)/(1+eps/3)
        i = np.nonzero(s)
        n = np.zeros_like(X)
        n[i] = np.exp((eps[i]*(1-1.5*np.log(s[i]))-1.5*np.log(s[i]))*(W/p))
        return self.N_max * n

    def stage(self,X,X0=36.62):
        """Return the shower stage at a given slant-depth X. This
        is after Lafebre et al.

        Parameters:
            X: atmosphering slant-depth [g/cm2]
            X0: radiation length of air [g/cm2]

        Returns:
            t: shower stage
        """
        return (X-self.X_max)/X0

if __name__ == '__main__':
    import time
    import matplotlib
    import matplotlib.pyplot as plt
    start_time = time.time()
    sh = Shower(500,1.e7,10000,np.radians(85),'down')
    end_time = time.time()
    print("Calculations take: %.3f s"%(
        end_time-start_time))

    x = sh.axis_r * np.sin(sh.theta)
    z = sh.axis_r * np.cos(sh.theta)

    arc_angle = 5
    arc = np.linspace(-np.radians(arc_angle),np.radians(arc_angle),100)
    x_surf = sh.earth_radius * np.sin(arc)
    z_surf = sh.earth_radius * np.cos(arc) - sh.earth_radius

    x_shower = sh.shower_r * np.sin(sh.theta)
    z_shower = sh.shower_r * np.cos(sh.theta)

    x_width = -sh.shower_rms_w * np.cos(sh.theta)
    z_width = sh.shower_rms_w * np.sin(sh.theta)

    plt.figure()
    ax = plt.gca()
    plt.plot(x,z,label='shower axis' )
    plt.plot(x_surf,z_surf,label="Earth's surface")
    plt.quiver(x_shower,z_shower,x_width,z_width, angles='xy', scale_units='xy', scale=1,label='shower width')
    plt.quiver(x_shower,z_shower,-x_width,-z_width, angles='xy', scale_units='xy', scale=1)
    plt.plot(x_shower,z_shower,'r',label='Cherenkov region')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend()
    plt.title('Downward Shower 5 degree EE')
    ax.set_aspect('equal')

    plt.figure()
    plt.plot(sh.axis_r,sh.axis_X)
    plt.plot(sh.shower_r,sh.shower_X)
    plt.scatter(sh.axis_start_r,sh.X0)
    
    plt.figure()
    plt.plot(sh.axis_X,sh.axis_nch)
    plt.show()
