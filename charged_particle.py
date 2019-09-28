"""
This module contains classes for creating the energy and angular
distributions of charged particles.

These were originally written by Zane Gerber, September 2019.
Modified by Douglas Bergman, September 2019.

"""

import numpy as np
from scipy.constants import physical_constants
from scipy.integrate import quad

class energyDistribution:
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
    
    def __init__(s,part,t):
        """
        Set the parameterization constants for this type of particle. The normalization
        constant is determined for the given shower stage, (which can be changed later).

        Parameters:
            particle = The name of the distribution of particles to create
            t = The shower stage for which to do the claculation
        """
        s.p = s.pt[part]
        s.t = t
        s.normalize(t)

    # Functions for the top level parameters
    def _set_A0(s,p,t):
        s.A0 = s.A1*s.pz[p,s.pm['A00']] * np.exp( s.pz[p,s.pm['A01']]*t - s.pz[p,s.pm['A02']]*t**2)
    def _set_e1(s,p,t):
        s.e1 = s.pz[p,s.pm['e11']] - s.pz[p,s.pm['e12']]*t
    def _set_e2(s,p,t):
        s.e2 = s.pz[p,s.pm['e21']] - s.pz[p,s.pm['e22']]*t
    def _set_g1(s,p,t):
        s.g1 = s.pz[p,s.pm['g11']]
    def _set_g2(s,p,t):
        s.g2 = 1 + s.pz[p,s.pm['g21']]

    def normalize(s,t):
        p = s.pt['Tot']
        s.A1 = 1.
        s._set_A0(p,t)
        s._set_e1(p,t)
        s._set_e2(p,t)
        s._set_g1(p,t)
        s._set_g2(p,t)
        intgrl,eps = quad(s.spectrum,s.ll,s.ul)
        s.A1 = 1/intgrl
        p = s.p
        s._set_A0(p,t)
        s._set_e1(p,t)
        s._set_e2(p,t)
        s._set_g1(p,t)
        s._set_g2(p,t)
    
    def set_stage(s,t):
        s.stage = t
        s.normalize(t)
    
    def spectrum(s,lE):
        """
        This function returns the particle distribution as a function of energy (energy spectrum)
        at a given stage

        Parameters:
            lE = energy of a given secondary particle [MeV]

        Returns:
            n_t_lE = the energy distribution of secondary particles.
        """
        E = np.exp(lE)
        return s.A0*E**s.g1 / ( (E+s.e1)**s.g1 * (E+s.e2)**s.g2 )

class angularDistribution:
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

    ll = 0.
    ul = 180.

    def __init__(s,lE):
        """Set the parameterization constants for this type (log)energy. The
        angular distribution only depends on the energy not the
        particle or stage. The normalization constanct is determined
        automatically. (It's normalized in degrees!)

        Parameters:
            lE = The log of the energy (in MeV) at which the angular
                 distribution is calculated
        """
        s.lE = lE
        s.C0 = 1.
        s.normalize()
        
    def _set_b1(s):
        s.b1 = s.pz[s.pm['b11']] + s.pz[s.pm['b12']] * np.exp(s.lE)**s.pz[s.pm['b13']]
    def _set_b2(s):
        s.b2 = s.pz[s.pm['b21']] - s.pz[s.pm['b22']] * s.lE
    def _set_a1(s):
        s.a1 = s.pz[s.pm['a11']]
    def _set_a2(s):
        s.a2 = s.pz[s.pm['a21']] + s.pz[s.pm['a22']] * s.lE
    def _set_sig(s):
        s.sig = s.pz[s.pm['sig']]

    def set_lE(s,lE):
        s.lE = lE
        s.normalize()
        
    def normalize(s):
        """Set the normalization constant so that the integral over degrees is unity."""
        s._set_b1()
        s._set_b2()
        s._set_a1()
        s._set_a2()
        s._set_sig()
        intgrl,eps = quad(s.n_t_lE_Omega,ll,ul)
        s.C0 = 1/intgrl
        
    def n_t_lE_Omega(s,theta):
        """
        This function returns the particle angular distribution as a angle at a given energy.
        It is independent of particle type and shower stage

        Parameters:
            theta: the angle [deg]

        Returns:
            n_t_lE_Omega = the angular distribution of particles
        """
        t1 = np.exp(s.b1) * theta**s.a1
        t2 = np.exp(s.b2) * theta**s.a2
        mrs = -1/s.sig
        ms = -s.sig
        return s.C0 * (t1**mrs + t2**mrs)**ms
        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.ion()

    # Reproduce Fig. 2 in Lafebre
    fig = plt.figure(1,figsize=(5,7))

    ll = np.log(1.)
    ul = np.log(1.e4)
    le = np.linspace(ll,ul,401)
    E  = np.exp(le)

    ep = energyDistribution('Tot',-6)
    ee = energyDistribution('Ele',-6)
    pp = energyDistribution('Pos',-6)
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

    qd = angularDistribution(np.log(1.))
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
