import numpy as np
from scipy.integrate import quad
from scipy.constants import value,nano

from atmosphere import Atmosphere
from charged_particle import EnergyDistribution
from cherenkov_photon import CherenkovPhoton
from cherenkov_photon_array import CherenkovPhotonArray

class ShowerCherenkov():
    """A class to contain a instance of a shower with a Gaisser-Hillas
    parameterization, a given geometry, an atmosphere, and a table of 
    Cherenkov photon angular distributions. An instance of this class
    will allow one to calculate the Cherenkov photon flux at a given
    point on the ground (or at some altitude) and the time distribution
    of those photons.
    """

    def __init__(self,Nmax,Xmax,X0=-150,Lambda=70,
                 zenith=0,azimuth=0,impact_x=0,impact_y=0,
                 ckarray='gg_t_delta_theta.npz'):
        """Create a shower with the given Gaisser-Hillas parameters and
        geometry
        
        Parameters:
            Nmax: the maximum size of the shower
            Xmax: the slant depth a maximum of the shower
            X0: the "starting point depth" for the shower
            Lambda: the growth rate of the shower
            zenith: the zenith angle of the shower
            azimuth: the azimuthal angle of the shower
            ckarray: the name of the file containing the CherenkovPhotonArray
        """
        self.Nmax   = Nmax
        self.Xmax   = Xmax
        self.X0     = X0
        self.Lambda = Lambda
        self.zenith = zenith
        self.azimuth = azimuth
        self.impact_x = impact_x
        self.impact_y = impact_y

        self.atmosphere = Atmosphere()
        self.ng_t_delta_Omega = CherenkovPhotonArray(ckarray)

        cq = np.cos(zenith)
        sq = np.sin(zenith)
        cp = np.cos(azimuth)
        sp = np.sin(azimuth)
        
        steps = np.append(
            np.append(
                np.linspace(   0, 1000,100,endpoint=False,dtype=float),
                np.linspace(1000,10000,90,endpoint=False,dtype=float)),
            np.linspace(10000,84000,75,dtype=float))
        hs  = (steps[1:]+steps[:-1])/2
        dhs = steps[1:]-steps[:-1]
        nh = len(hs)
        rs = hs*sq/cq

        c = value('speed of light in vacuum')
        
        xs = np.empty_like(hs)
        xs[-1] = self.atmosphere.depth(hs[-1])
        for i in range(nh-2,-1,-1):
            xs[i] = xs[i+1] + self.atmosphere.depth(hs[i],hs[i+1])
        xs /= cq

        self.location = np.empty((nh,3),dtype=float)
        self.location[:,0] = self.impact_x + rs*cp
        self.location[:,1] = self.impact_y + rs*sp
        self.location[:,2] = hs
        self.heights = self.location[:,2]
        self.steps = dhs/cq
        self.times = hs/cq/c
        self.slant_depths = xs
        self.sizes = self.size(xs)
        self.stages = self.stage(xs)
        self.deltas = self.atmosphere.delta(hs)
        xh = zip(xs,hs)
        self.csizes = np.array([self.cherenkov_size(x,h) for x,h in xh])
        xh = zip(xs,hs)
        self.cyield = np.array([self.cherenkov_yield(x,h) for x,h in xh])
        self.cphots = self.steps * self.csizes * self.cyield
        nq = len(self.ng_t_delta_Omega.theta)
        self.cangs = np.empty((nh,nq),dtype=float)
        itd = zip(np.arange(nh),self.stages,self.deltas)
        for i,t,d in itd:
            self.cangs[i] = self.ng_t_delta_Omega.angular_distribution(t,d)
        
    def __repr__(self):
        return "ShowerCherernkov(%.2e,%.0f,%.0f,%.0f,%.4f)"%(
            self.Nmax,self.Xmax,self.X0,self.Lambda,self.zenith)

    def size(self,X):
        """Return the size of the shower at a slant-depth X

        Parameters:
            X: the slant depth at which to calculate the shower size [g/cm2]

        Returns:
            N: the shower size
        """
        x =         (X-self.X0)/self.Lambda
        m = (self.Xmax-self.X0)/self.Lambda
        n = np.exp( m*(np.log(x)-np.log(m)) - (x-m) )
        return self.Nmax * n
    
    def size_height(self,h):
        """Return the size of the shower at an elevation h

        Parameters:
            h: the height about sea-level [m]

        Returns:
            N: the shower size
        """
        d1 = h/np.cos(self.zenith)
        X = self.atmosphere.depth(d1) if self.zenith==0 else \
            self.atmosphere.slant_depth(self.zenith,d1)
        return self.size(X)

    def stage(self,X,X0=36.62):
        """Return the shower stage at a given slant-depth X. This
        is after Lafebre et al.
        
        Parameters:
            X: atmosphering slant-depth [g/cm2]
            X0: radiation length of air [g/cm2]

        Returns:
            t: shower stage
        """
        return (X-self.Xmax)/X0
    
    def cherenkov_size(self,X,h):
        """Return the size of the shower at a given slant-depth X
        that can produce Cherenkov radiation.

        Parameters:
            X: the slant depth at which to calculate the shower size [g/cm2]
            h: the height corresponding to the X [m]

        Returns:
            N: the size of the shower over the Cherenkov threshold
        """
        delta = self.atmosphere.delta(h)
        Ec = CherenkovPhoton.cherenkov_threshold(delta)
        t = self.stage(X)
        fe = EnergyDistribution('Tot',t)
        fraction,_ = quad(fe.spectrum,np.log(Ec),35.)
        return fraction*self.size(X)

    def cherenkov_yield(self,X,h,lambda1=300.,lambda2=600.):
        """Return the number of Cherenkov photons produced at X and h.

        Parameters:
            X: the slant depth at which to calculate the Cherenkov yield [g/cm2]
            h: the height corresponding to the X [m]
            lambda1: the lower limit to the wavelength range [nm]
            lambda2: the upper limit to the wavelength range [nm]

        Returns:
            Nck: the Number of Cherenkov photons produced [1/m]

        
        """
        alpha_over_hbarc = 370.e2 # per eV per m, from PDG
        hc = value('Planck constant in eV s') * \
             value('speed of light in vacuum')
        E1 = hc/(lambda1*nano)
        E2 = hc/(lambda2*nano)
        dE = np.abs(E1-E2)
        delta = self.atmosphere.delta(h)
        q = CherenkovPhoton.cherenkov_angle(1.e12,delta)
        return alpha_over_hbarc * np.sin(q)**2 * dE
