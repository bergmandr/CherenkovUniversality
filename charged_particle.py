
#!/usr/bin/env python3
# coding: utf-8
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

    ll = np.log(1.e-1) #lower limit
    ul = np.log(1.e+6) #upper limit

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
    pm_l = {'b11':0,'b12':1,'b13':2,'b21':3,'b22':4,'a11':5,'a21':6,'a22':7,'sig':8}
    # pz for parameterization
    #              b11   b12  b13   b21  b22  a11    a21   a22   sig
    pz_l = np.array([-3.73,0.92,0.210,32.9,4.84,-0.399,-8.36,0.440,3.])

    # pm for parameter
    pm_b = {
        'a10' : 0,
        'a11' : 1,
        'a12' : 2,
        'a13' : 3,
        'c10' : 4,
        'c11' : 5,
        'c20' : 6,
        'c21' : 7,
        'a20' : 8,
        'a21' : 9,
        'a22' : 10,
        'a23' : 11,
        'a24' : 12,
        'p0'  : 13,
        'p1'  : 14,
        'p2'  : 15,
        'r0'  : 16,
        'r1'  : 17,
        'r2'  : 18,
        'r3'  : 19,
        'Eb'  : 20,
        'Ec'  : 21,
        'Ed'  : 22,
    }

    pz_b = np.array([3773.05,1.82945,0.031143,0.0129724,163.366,0.952228,
    182.945,0.921291,340.308,1.73569,6.03581,4.29495,2.50626,0.0204,-0.790,
    -2.20,3.6631,0.131998,-0.134479,0.537966,10**-1.5,10**-1.4,
    10**(-0.134 / 0.538)])

    intlim = np.array([0,1.e-10,1.e-8,1.e-6,1.e-4,1.e-2,1.e-0,180.])
    lls = intlim[:-1]
    uls = intlim[1:]

    def __init__(self,lE,schema='b'):
        """Set the parameterization constants for this type (log)energy. The
        angular distribution only depends on the energy not the
        particle or stage. The normalization constanct is determined
        automatically. (It's normalized in degrees!)

        Parameters:
            lE = The log of the energy (in MeV) at which the angular
                 distribution is calculated
        """
        self.schema = schema
        self.set_lE(lE)

    # Set Lafebre constants
    def _set_b1l(self):
        self.b1l = self.pz_l[self.pm_l['b11']] + self.pz_l[self.pm_l['b12']] * np.exp(self.lE)**self.pz_l[self.pm_l['b13']]
    def _set_b2l(self):
        self.b2l = self.pz_l[self.pm_l['b21']] - self.pz_l[self.pm_l['b22']] * self.lE
    def _set_a1l(self):
        self.a1l = self.pz_l[self.pm_l['a11']]
    def _set_a2l(self):
        self.a2l = self.pz_l[self.pm_l['a21']] + self.pz_l[self.pm_l['a22']] * self.lE
    def _set_sigl(self):
        self.sigl = self.pz_l[self.pm_l['sig']]

    # Set Bergman constants
    def _set_a1b(self):
        self.a1b = self.pz_b[self.pm_b['a10']] * np.power(self.E, self.pz_b[self.pm_b['a11']] + self.pz_b[self.pm_b['a12']] * self.lE + self.pz_b[self.pm_b['a13']] * np.power(self.lE, 2))
    def _set_c1b(self):
        self.c1b = self.pz_b[self.pm_b['c10']] * np.power(self.E, self.pz_b[self.pm_b['c11']])
    def _set_c2b(self):
        self.c2b = self.pz_b[self.pm_b['c20']] * np.power(self.E, self.pz_b[self.pm_b['c21']])
    def _set_a2b(self):
        if self.E >= self.pz_b[self.pm_b['Eb']]:
            self.a2b = self.pz_b[self.pm_b['a20']] * np.power(self.E, self.pz_b[self.pm_b['a21']]) + self.pz_b[self.pm_b['a22']]
        else:
            num = (np.power(self.E, self.pz_b[self.pm_b['a23']]) + self.pz_b[self.pm_b['a24']]) * (self.pz_b[self.pm_b['a20']] * np.power(self.pz_b[self.pm_b['Eb']], self.pz_b[self.pm_b['a21']]) + self.pz_b[self.pm_b['a22']] - self.pz_b[self.pm_b['a24']])
            self.a2b = num / np.power(self.pz_b[self.pm_b['Eb']], self.pz_b[self.pm_b['a23']])
    def _set_theta_0b(self):
        if self.E >= self.pz_b[self.pm_b['Ec']]:
            self.theta_0b = self.pz_b[self.pm_b['p0']] * np.power(self.E, self.pz_b[self.pm_b['p1']])
        else:
            self.theta_0b = self.pz_b[self.pm_b['p0']] * np.power(self.pz_b[self.pm_b['Ec']], self.pz_b[self.pm_b['p1']]-self.pz_b[self.pm_b['p2']]) * np.power(self.E, self.pz_b[self.pm_b['p2']])
    def _set_rb(self):
        if self.E >= self.pz_b[self.pm_b['Ed']]:
            self.rb = self.pz_b[self.pm_b['r0']]
        else:
            self.rb = self.pz_b[self.pm_b['r1']] * np.power(self.E, self.pz_b[self.pm_b['r2']] + self.pz_b[self.pm_b['r3']] * self.lE)

    def set_lE(self,lE):
        if self.schema == 'b': # convert to GeV if schema is Bergman
            self.E = np.exp(lE) * 1.e-3
            self.lE = np.log(self.E)
        elif self.schema == 'l':
            self.lE = lE
            self.E = np.exp(lE)
        else:
            print('Not a valid schema')
        self.normalize()

    def set_schema(self,schema):
        """
        Reset schema and normalize
        """
        self.schema = schema
        self.set_lE(self.lE)

    def normalize(self):
        """Set the normalization constant so that the integral over degrees is unity."""
        self.C0 =1
        if self.schema == 'b':
            self._set_a1b()
            self._set_c1b()
            self._set_c2b()
            self._set_a2b()
            self._set_theta_0b()
            self._set_rb()
        elif self.schema == 'l':
            self._set_b1l()
            self._set_b2l()
            self._set_a1l()
            self._set_a2l()
            self._set_sigl()
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
        dist_value = np.empty(1)
        if self.schema == 'b':
            theta = np.radians(theta)
            t1 = self.a1b * np.exp(-self.c1b * theta - self.c2b * np.power(theta,2))
            t2 = self.a2b * np.power(1 + theta / self.theta_0b,-self.rb)
            dist_value = self.C0 * (t1 + t2)
        elif self.schema =='l':
            t1 = np.exp(self.b1l) * theta**self.a1l
            t2 = np.exp(self.b2l) * theta**self.a2l
            mrs = -1/self.sigl
            ms = -self.sigl
            dist_value = self.C0 * (t1**mrs + t2**mrs)**ms
        return dist_value

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.ion()
    fig = plt.figure()

    ll = 0.1
    ul = 45.
    lqdeg = np.linspace(np.log(ll),np.log(ul),450)
    qdeg = np.exp(lqdeg)

    qd = AngularDistribution(np.log(1.),'l')
    plt.plot(qdeg,qd.n_t_lE_Omega(qdeg),label='1 MeV')
    qd.set_lE(np.log(5.))
    plt.plot(qdeg,qd.n_t_lE_Omega(qdeg),label='5 MeV')
    qd.set_lE(np.log(30.))
    plt.plot(qdeg,qd.n_t_lE_Omega(qdeg),label='30 MeV')
    qd.set_lE(np.log(170.))
    plt.plot(qdeg,qd.n_t_lE_Omega(qdeg),label='170 MeV')
    qd.set_lE(np.log(1.e3))
    plt.plot(qdeg,qd.n_t_lE_Omega(qdeg),label='1 GeV')
    qd.set_schema('b')
    plt.loglog()
    plt.xlim(ll,ul)
    plt.ylim(1.e-4,1.e1)
    plt.legend()
    plt.xlabel('Theta [deg]')
    plt.ylabel('n(t;lE,Omega)')
    plt.show()

    fig = plt.figure()
    qd.set_lE(np.log(1.))
    plt.plot(qdeg,qd.n_t_lE_Omega(qdeg),label='1 MeV B')
    qd.set_lE(np.log(5.))
    plt.plot(qdeg,qd.n_t_lE_Omega(qdeg),label='5 MeV B')
    qd.set_lE(np.log(30.))
    plt.plot(qdeg,qd.n_t_lE_Omega(qdeg),label='30 MeV B')
    qd.set_lE(np.log(170.))
    plt.plot(qdeg,qd.n_t_lE_Omega(qdeg),label='170 MeV B')
    qd.set_lE(np.log(1.e3))
    plt.plot(qdeg,qd.n_t_lE_Omega(qdeg),label='1 GeV B')
    plt.loglog()
    plt.xlim(ll,ul)
    plt.ylim(1.e-4,1.e1)
    plt.legend()
    plt.xlabel('Theta [deg]')
    plt.ylabel('n(t;lE,Omega)')
    plt.show()
