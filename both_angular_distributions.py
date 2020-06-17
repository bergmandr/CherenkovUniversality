"""
This module contains a classe calculating the vertical angular distribution of
charged particles at a given energy in an extensive air shower.
"""

import numpy as np
from scipy.integrate import quad

class BothAngularDistributions:
    """
    This class contains functions related to the angular distribution
    of secondary particles.  The parameterizations used are those of
    Douglas Bergman and Lafebre et al.
    """
    # Dictionaries of parameters and their respective static values

    # Doug
    pm_d = {
        'a10' : 3773.05,
        'a11' : 1.82945,
        'a12' : 0.031143,
        'a13' : 0.0129724,
        'c10' : 163.366,
        'c11' : 0.952228,
        'c20' : 182.945,
        'c21' : 0.921291,
        'a20' : 340.308,
        'a21' : 1.73569,
        'a22' : 6.03581,
        'a23' : 4.29495,
        'a24' : 2.50626,
        'p0'  : 0.0204,
        'p1'  : -0.790,
        'p2'  : -2.20,
        'r0'  : 3.6631,
        'r1'  : 0.131998,
        'r2'  : -0.134479,
        'r3'  : 0.537966,
        'Eb'  : 10**-1.5,
        'Ec'  : 10**-1.4,
        'Ed'  : 10**(-0.134 / 0.538),
    }
    # Lafebre
    pm_l = {
        'a11' : -0.399,
        'a21' : -8.36,
        'a22' : 0.440,
        'sig' : 3,
        'b11' : -3.73,
        'b12' : 0.92,
        'b13' : 0.210,
        'b21' : 32.9,
        'b22' : 4.84,
    }
    # integration limits for normalization
    intlim = np.array([0,1.e-10,1.e-8,1.e-6,1.e-4,1.e-2,1.e-0,np.pi])
    lls = intlim[:-1]
    uls = intlim[1:]

    def __init__(self, E, SCHEMA = 'D'):
        """
        Set the parameters for calculating the angular distribution of charged particles at a
        given energy E

        SCHEMA = 'D' Doug's parameterization
        SCHEMA = 'L' Lafebre's parameterization

        Parameters:
        E = The energy (in GeV) of charged particles at which the angular
            distribution is calculated.
        """

        self.SCHEMA = SCHEMA
        self.E = E
        self.C0 = 1.
        self.normalize()

    def set_dougs_constants(self):
        """
        Set each constant in Doug's distribution at current value of E

        The values of a1, c1, and c2 are found from the same parameterization regardless
        of the energy.

        The values of a2, theta_0, and r are found from different parameterizations if the
        energy is above or below the energies Eb, Ec, and Ed respectively.
        """
        pm = self.pm_d
        E = self.E
        lE = np.log(E)
        self.a1d = pm['a10'] * np.power(E, pm['a11'] + pm['a12'] * lE + pm['a13'] * np.power(lE, 2))
        self.c1d = pm['c10'] * np.power(E, pm['c11'])
        self.c2d = pm['c20'] * np.power(E, pm['c21'])
        if E >= pm['Eb']:
            self.a2d = pm['a20'] * np.power(E, pm['a21']) + pm['a22']
        else:
            num = (np.power(E, pm['a23']) + pm['a24']) * (pm['a20'] * np.power(pm['Eb'], pm['a21']) + pm['a22'] - pm['a24'])
            self.a2d = num / np.power(pm['Eb'], pm['a23'])
        if E >= pm['Ec']:
            self.theta_0d = pm['p0'] * np.power(E, pm['p1'])
        else:
            self.theta_0d = pm['p0'] * np.power(pm['Ec'], pm['p1']-pm['p2']) * np.power(E, pm['p2'])
        if E >= pm['Ed']:
            self.rd = pm['r0']
        else:
            self.rd = pm['r1'] * np.power(E, pm['r2'] + pm['r3'] * lE)

    def set_lafebre_constants(self):
        """
        Set each constant in Lafebre's distribution at current value of E
        """
        pm = self.pm_l
        E = self.E * 1.e3
        lE = np.log(E)
        self.a1l = pm['a11']
        self.a2l = pm['a21'] + pm['a22'] * lE
        self.b1l = pm['b11'] + pm['b12'] * np.power(E,pm['b13'])
        self.b2l = pm['b21'] - pm['b22'] * lE
        self.sigl = pm['sig']

    def n_t_lE_Omega_d(self,theta):
        """
        This function returns Doug's angular distribution for charged particles
        at a given energy.

        Parameters:
            theta: the angle [rad]

        Returns:
            n_t_lE_Omega_d = the angular distribution of particles
        """
        t1 = self.a1d * np.exp(-self.c1d * theta - self.c2d * np.power(theta,2))
        t2 = self.a2d * np.power(1 + theta / self.theta_0d,-self.rd)
        return self.C0 * (t1 + t2)

    def n_t_lE_Omega_l(self,theta):
        """
        This function returns Lafebre's angular distribution for charged particles
        at a given energy.

        Parameters:
            theta: the angle [rad]

        Returns:
            n_t_lE_Omega_l = the angular distribution of particles
        """
        theta = np.degrees(theta) #convert theta to degrees
        t1 = np.exp(self.b1l) * np.power(theta,self.a1l)
        t2 = np.exp(self.b2l) * np.power(theta,self.a2l)
        mrs = -1/self.sigl
        ms = -self.sigl
        return self.C0 * np.power(np.power(t1,mrs) + np.power(t2,mrs),ms)

    def normalize(self):
        """
        Set the normalization constant so that the integral over the domain
        of theta is unity. Only the current schema is normalized.
        """
        self.C0 = 1
        intgrl = 0
        if self.SCHEMA == 'D':
            self.set_dougs_constants()
            for ll,ul in zip(self.lls,self.uls):
                intgrl += quad(self.n_t_lE_Omega_d,ll,ul)[0]
            self.C0 = 1/intgrl
        elif self.SCHEMA == 'L':
            self.set_lafebre_constants()
            for ll,ul in zip(self.lls,self.uls):
                intgrl += quad(self.n_t_lE_Omega_l,ll,ul)[0]
            self.C0 = 1/intgrl

    def set_E(self,E):
        """
        Reset energy and normalize
        """
        self.E = E
        self.normalize()

    def set_SCHEMA(self,SCHEMA):
        """
        Reset SCHEMA and normalize
        """
        self.SCHEMA = SCHEMA
        self.normalize()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.ion()
    E_GeV = 10**.55
    ad = BothAngularDistributions(E_GeV)
    lqrad = np.logspace(-5,-1,500)
    plt.figure()
    plt.plot(lqrad,ad.n_t_lE_Omega_d(lqrad),label='Doug')
    ad.set_SCHEMA('L')
    plt.plot(lqrad,ad.n_t_lE_Omega_l(lqrad),label='Lafebre')
    plt.loglog()
    plt.legend()
    plt.title('Plot Comparing Parameterizations at E = %.2f GeV'%E_GeV)
    plt.xlabel('Theta [Rad]')
    plt.ylabel('n(t;lE,Omega)')
    plt.grid('both')
    plt.show()
