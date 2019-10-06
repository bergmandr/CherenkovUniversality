"""
This module contains classes for creating the energy and angular
distributions of charged particles.

These were originally written by Zane Gerber, September 2019.
Modified by Douglas Bergman, September 2019.

"""

import numpy as np
from scipy.constants import physical_constants
from scipy.integrate import quad

class EnergyDistribution:
    """
    This class contains functions related to the energy distribution
    of secondary particles.  The parameterizations used are those of
    S. Lafebre et al. (2009). The normalization parameter A1 is determined
    by the normalization condition.
    """
    # pt for particle
    pt = {'Tot': 0, 'Ele': 1, 'Pos': 2}
    # pm for parameter
    pm = {'A00':0,'A01':1,'A02':2,'e11':3,'e12':4,'e21':5,'e22':6,'g11':7,'g21':8}
    # pz for parameterization
    #               A00   A01   A02     e11  e12    e21  e22  g11 g21
    pz = np.array([[1.000,0.191,6.91e-4,5.64,0.0663,123.,0.70,1.0,0.0374],  # Tot
                   [0.485,0.183,8.17e-4,3.22,0.0068,106.,1.00,1.0,0.0372],  # Ele
                   [0.516,0.201,5.42e-4,4.36,0.0663,143.,0.15,2.0,0.0374]]) # Pos

    ll = np.log(1.e-1)
    ul = np.log(1.e+6)
    
    def __init__(self,part,t):
        """
        Set the parameterization constants for this type of particle. The normalization
        constant is determined for the given shower stage, (which can be changed later).

        Parameters:
            particle = The name of the distribution of particles to create
            t = The shower stage for which to do the claculation
        """
        self.p = self.pt[part]
        self.t = t
        self.normalize(t)

    # Functions for the top level parameters
    def _set_A0(self,p,t):
        self.A0 = self.A1*self.pz[p,self.pm['A00']] * np.exp( self.pz[p,self.pm['A01']]*t - self.pz[p,self.pm['A02']]*t**2)
    def _set_e1(self,p,t):
        self.e1 = self.pz[p,self.pm['e11']] - self.pz[p,self.pm['e12']]*t
    def _set_e2(self,p,t):
        self.e2 = self.pz[p,self.pm['e21']] - self.pz[p,self.pm['e22']]*t
    def _set_g1(self,p,t):
        self.g1 = self.pz[p,self.pm['g11']]
    def _set_g2(self,p,t):
        self.g2 = 1 + self.pz[p,self.pm['g21']]*t

    def normalize(self,t):
        p = self.pt['Tot']
        self.A1 = 1.
        self._set_A0(p,t)
        self._set_e1(p,t)
        self._set_e2(p,t)
        self._set_g1(p,t)
        self._set_g2(p,t)
        intgrl,eps = quad(self.spectrum,self.ll,self.ul)
        self.A1 = 1/intgrl
        p = self.p
        self._set_A0(p,t)
        self._set_e1(p,t)
        self._set_e2(p,t)
        self._set_g1(p,t)
        self._set_g2(p,t)
    
    def set_stage(self,t):
        self.t = t
        self.normalize(t)
    
    def spectrum(self,lE):
        """
        This function returns the particle distribution as a function of energy (energy spectrum)
        at a given stage

        Parameters:
            lE = energy of a given secondary particle [MeV]

        Returns:
            n_t_lE = the energy distribution of secondary particles.
        """
        E = np.exp(lE)
        return self.A0*E**self.g1 / ( (E+self.e1)**self.g1 * (E+self.e2)**self.g2 )

class AngularDistribution:
    """
    This class contains functions related to the angular distribution
    of secondary particles.  The parameterization used is that of
    S. Lafebre et. al. (2009).
    """
    # pm for parameter
    pm = {'b11':0,'b12':1,'b13':2,'b21':3,'b22':4,'a11':5,'a21':6,'a22':7,'sig':8}
    # pz for parameterization
    #              b11   b12  b13   b21  b22  a11    a21   a22   sig
    pz = np.array([-3.73,0.92,0.210,32.9,4.84,-0.399,-8.36,0.440,3.])    

    intlim = np.array([0,1.e-10,1.e-8,1.e-6,1.e-4,1.e-2,1.e-0,180.])
    lls = intlim[:-1]
    uls = intlim[1:]

    def __init__(self,lE):
        """Set the parameterization constants for this type (log)energy. The
        angular distribution only depends on the energy not the
        particle or stage. The normalization constanct is determined
        automatically. (It's normalized in degrees!)

        Parameters:
            lE = The log of the energy (in MeV) at which the angular
                 distribution is calculated
        """
        self.lE = lE
        self.C0 = 1.
        self.normalize()
        
    def _set_b1(self):
        self.b1 = self.pz[self.pm['b11']] + self.pz[self.pm['b12']] * np.exp(self.lE)**self.pz[self.pm['b13']]
    def _set_b2(self):
        self.b2 = self.pz[self.pm['b21']] - self.pz[self.pm['b22']] * self.lE
    def _set_a1(self):
        self.a1 = self.pz[self.pm['a11']]
    def _set_a2(self):
        self.a2 = self.pz[self.pm['a21']] + self.pz[self.pm['a22']] * self.lE
    def _set_sig(self):
        self.sig = self.pz[self.pm['sig']]

    def set_lE(self,lE):
        self.lE = lE
        self.normalize()
        
    def normalize(self):
        """Set the normalization constant so that the integral over degrees is unity."""
        self._set_b1()
        self._set_b2()
        self._set_a1()
        self._set_a2()
        self._set_sig()
        intgrl = 0.
        for ll,ul in zip(self.lls,self.uls):
            intgrl += quad(self.n_t_lE_Omega,ll,ul)[0]
        self.C0 = 1/intgrl
        
    def n_t_lE_Omega(self,theta):
        """
        This function returns the particle angular distribution as a angle at a given energy.
        It is independent of particle type and shower stage

        Parameters:
            theta: the angle [deg]

        Returns:
            n_t_lE_Omega = the angular distribution of particles
        """
        t1 = np.exp(self.b1) * theta**self.a1
        t2 = np.exp(self.b2) * theta**self.a2
        mrs = -1/self.sig
        ms = -self.sig
        return self.C0 * (t1**mrs + t2**mrs)**ms
        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.ion()

    # Reproduce Fig. 2 in Lafebre
    fig = plt.figure(1,figsize=(5,7))

    ll = np.log(1.)
    ul = np.log(1.e4)
    le = np.linspace(ll,ul,401)
    E  = np.exp(le)

    ep = EnergyDistribution('Tot',-6)
    ee = EnergyDistribution('Ele',-6)
    pp = EnergyDistribution('Pos',-6)
    ax1 = fig.add_subplot(311)
    ax1.plot(E,ep.spectrum(le),label='Total')
    ax1.plot(E,ee.spectrum(le),label='Elect')
    ax1.plot(E,pp.spectrum(le),label='Posit')
    ax1.legend(loc='upper left')
    ax1.set_xscale('log')
    xll,xul = ax1.set_xlim(1.,1.e4)
    yll,yul = ax1.set_ylim()
    ax1.set_ylim(0,yul)
    ax1.set_ylabel('n(t;lE)')
    ax1.text(0.5*xul,0.95*yul,'t = -6',horizontalalignment='right',verticalalignment='top')

    ep.set_stage(0)
    ee.set_stage(0)
    pp.set_stage(0)

    ax2 = fig.add_subplot(312)
    ax2.plot(E,ep.spectrum(le),label='Total')
    ax2.plot(E,ee.spectrum(le),label='Elect')
    ax2.plot(E,pp.spectrum(le),label='Posit')
    ax2.legend(loc='upper left')
    ax2.set_xscale('log')
    xll,xul = ax2.set_xlim(1.,1.e4)
    yll,yul = ax2.set_ylim()
    ax2.set_ylim(0,yul)
    ax2.set_ylabel('n(t;lE)')
    ax2.text(0.5*xul,0.95*yul,'t = 0',horizontalalignment='right',verticalalignment='top')

    ep.set_stage(6)
    ee.set_stage(6)
    pp.set_stage(6)

    ax3 = fig.add_subplot(313)
    ax3.plot(E,ep.spectrum(le),label='Total')
    ax3.plot(E,ee.spectrum(le),label='Elect')
    ax3.plot(E,pp.spectrum(le),label='Posit')
    ax3.legend(loc='upper left')
    ax3.set_xscale('log')
    xll,xul = ax3.set_xlim(1.,1.e4)
    yll,yul = ax3.set_ylim()
    ax3.set_ylim(0,yul)
    ax3.set_ylabel('n(t;lE)')
    ax3.set_xlabel('E [MeV]')
    ax3.text(0.5*xul,0.95*yul,'t = 6',horizontalalignment='right',verticalalignment='top')
    fig.tight_layout()
    
    # Reproduce Fig. 7 in Lafebre
    fig = plt.figure(2)

    ll = 0.1
    ul = 45.
    lqdeg = np.linspace(np.log(ll),np.log(ul),450)
    qdeg = np.exp(lqdeg)

    qd = AngularDistribution(np.log(1.))
    plt.plot(qdeg,qd.n_t_lE_Omega(qdeg),label='1 MeV')
    qd.set_lE(np.log(5.))
    plt.plot(qdeg,qd.n_t_lE_Omega(qdeg),label='5 MeV')
    qd.set_lE(np.log(30.))
    plt.plot(qdeg,qd.n_t_lE_Omega(qdeg),label='30 MeV')
    qd.set_lE(np.log(170.))
    plt.plot(qdeg,qd.n_t_lE_Omega(qdeg),label='170 MeV')
    qd.set_lE(np.log(1.e3))
    plt.plot(qdeg,qd.n_t_lE_Omega(qdeg),label='1 GeV')
    plt.loglog()
    plt.xlim(ll,ul)
    plt.ylim(1.e-4,1.e1)
    plt.legend()
    plt.xlabel('Theta [deg]')
    plt.ylabel('n(t;lE,Omega)')
