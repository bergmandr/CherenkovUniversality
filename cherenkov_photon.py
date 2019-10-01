import numpy as np
from scipy.integrate import quad
import scipy.constants as spc
from atmosphere import *
from charged_particle import EnergyDistribution, AngularDistribution

twopi = 2.*np.pi
m_e = spc.value('electron mass energy equivalent in MeV')

class CherenkovPhoton:
    inner_precision = 1.e-5
    outer_precision = 1.e-4
    
    @staticmethod
    def spherical_cosines(A,B,c):
        """Return the angle C in spherical geometry where ABC is a spherical 
        triangle, and c is the interior angle across from c.
        """
        return np.arccos( np.cos(A)*np.cos(B) + np.sin(A)*np.sin(B)*np.cos(c) )

    @staticmethod
    def cherenkov_threshold(delta):
        """Calculate the Cherenkov threshold for this atmosphere

        Parameters:
            delta: the index-of-refraction minus one (n-1)
        
        Returns:
            E_Ck: The Cherenkov threshold energy [MeV]
        """
        n     = 1 + delta
        beta  = 1/n
        gamma = 1/np.sqrt((1-beta**2))
        E_Ck = gamma*m_e
        return E_Ck
    
    @staticmethod
    def cherenkov_angle(E,delta):
        """Calculate the Cherenkov angle for a given log energy and atmosphere

        Parameters:
            E: The energy for the producing electron [MeV]
            delta: the index-of-refraction minus one (n-1)
        
        Returns:
            theta_g: The angle of the Cherenkov cone for this atmosphere
        """
        n = 1+delta
        gamma = E/m_e
        beta  = np.sqrt(1-1/gamma**2)
        rnbeta = 1/n/beta
        if type(rnbeta) is np.ndarray:
            theta_g = np.empty_like(rnbeta)
            theta_g[rnbeta<=1] = np.arccos(rnbeta[rnbeta<=1])
            theta_g[rnbeta>1] = 0
        else:
            theta_g = 0. if rnbeta>1 else np.arccos(rnbeta)
        return theta_g

    @staticmethod
    def cherenkov_yield(E,delta):
        """Calculate the relative Cherenkov efficiency at this electron
        energy

        Parameters:
            E: The energy for the producing electron [MeV]
            delta: the index-of-refraction minus one (n-1)
        
        Returns:
            Y_c: The relative Cherenkov efficiency
        """
        Y_c = 1 - (CherenkovPhoton.cherenkov_threshold(delta)/E)**2
        if type(Y_c) is np.ndarray:
            Y_c[Y_c<0] = 0
        else:
            Y_c = max(Y_c,0.)
        return Y_c
        
    @staticmethod
    def inner_integrand(phi_e,theta,theta_g,g_e):
        """The function returns the inner integrand of the Cherenkov photon
        angular distribution.

        Parameters:
            phi_e: the internal angle between the shower-photon plane and the 
              electron-photon plane (this is the integration vairable)
            theta: the angle between the shower axis and the Cherenkov photon
            theta_g: the angle between the electron and the photon (the Cherenkov
              cone angle)
            g_e: an AngularDistribution object (normalized for the energy 
              of theta_g!)
        Returns:
            the inner integrand
        """
        theta_e = CherenkovPhoton.spherical_cosines(theta,theta_g,phi_e)
        return g_e.n_t_lE_Omega(theta_e/spc.degree) /spc.degree

    @staticmethod
    def inner_integral(theta,theta_g,g_e):
        """The function returns the inner integral of the Cherenkov photon
        angular distribution.

        Parameters:
            theta: the angle between the shower axis and the Cherenkov photon
            theta_g: the angle between the electron and the photon (the Cherenkov
              cone angle)
            g_e: an AngularDistribution object (normalized for the energy 
              of theta_g!)
        Returns:
            the inner integral
        """
        return quad( CherenkovPhoton.inner_integrand, 0.,twopi,args=(theta,theta_g,g_e),
                     epsrel=CherenkovPhoton.inner_precision,
                     epsabs=CherenkovPhoton.inner_precision )[0]

    @staticmethod
    def outer_integrand(l_g,theta,f_e,delta):
        """The function returns the outer integrand of the Cherenkov photon
        angular distribution.

        Parameters:
            l_g: the logarithm of the energy for which the Cherenkov angle is
              whatever it is (this is the integration variable)
            theta: the angle between the shower axis and the Cherenkov photon
            f_e: an EnergyDistribution object (normalized!)
            delta: the index-of-refraction minus one (n-1)

        Returns:
            the outer integrand
        """
        E_g = np.exp(l_g)
        theta_g = CherenkovPhoton.cherenkov_angle(E_g,delta)
        cherenkov_yield = CherenkovPhoton.cherenkov_yield(E_g,delta)
        g_e = AngularDistribution(l_g)
        inner = CherenkovPhoton.inner_integral(theta,theta_g,g_e)
        return np.sin(theta_g) * cherenkov_yield * f_e.spectrum(l_g) * inner

    @staticmethod
    def outer_integral(theta,f_e,delta):
        """The function returns the outer integral of the Cherenkov photon
        angular distribution.

        Parameters:
            theta: the angle between the shower axis and the Cherenkov photon
            f_e: an EnergyDistribution object (normalized!)
            t: the shower stage
            delta: the index-of-refraction minus one (n-1) for the Cherenkov angle
        Returns:
            the outer integral
        """
        ll = np.log(CherenkovPhoton.cherenkov_threshold(delta))
        # ul = 13.8 # np.log(1.e6)
        ul = 10.309 # np.log(3.e4)
        return quad( CherenkovPhoton.outer_integrand,ll,ul,args=(theta,f_e,delta),
                     epsrel = CherenkovPhoton.outer_precision )[0]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.ion()
    
    A = 0.1; B = 0.1; c = np.pi/2
    print("A = %6.4f, B = %6.4f,  c = %6.4f; C = %6.4f"%(A,B,c,CherenkovPhoton.spherical_cosines(0.1,0.1,np.pi/2)))
    delta = 2.9e-4
    print("delta = %8.6f; threshold = %5.2f MeV"%(delta,CherenkovPhoton.cherenkov_threshold(delta)))
    E = 30.
    print("delta = %8.6f, E = %6.2f MeV; theta_g = %6.4f rad"%(delta,E,CherenkovPhoton.cherenkov_angle(E,delta)))
    E = 100.
    print("delta = %8.6f, E = %6.2f MeV; theta_g = %6.4f rad"%(delta,E,CherenkovPhoton.cherenkov_angle(E,delta)))
    E = 20.
    print("delta = %8.6f, E = %6.2f MeV; theta_g = %6.4f rad"%(delta,E,CherenkovPhoton.cherenkov_angle(E,delta)))
    E = 30.
    print("delta = %8.6f, E = %6.2f MeV; Y_Ck = %6.4f"%(delta,E,CherenkovPhoton.cherenkov_yield(E,delta)))
    E = 100.
    print("delta = %8.6f, E = %6.2f MeV; Y_Ck = %6.4f"%(delta,E,CherenkovPhoton.cherenkov_yield(E,delta)))
    E = 20.
    print("delta = %8.6f, E = %6.2f MeV; Y_Ck = %6.4f"%(delta,E,CherenkovPhoton.cherenkov_yield(E,delta)))

    print()
    print("inner_integrand samples:")
    E = 30.
    lE = np.log(E)
    theta = 0.01
    theta_g = CherenkovPhoton.cherenkov_angle(E,delta)
    g_e = AngularDistribution(lE)
    print("  CherenkovPhoton.inner_integrand(phi_e,theta,theta_g,g_e)")
    print("  E = %.1f, lE = %.1f, theta = %.4f, theta_g = %.4f"%(E,lE,theta,theta_g))
    print("  g_e: ",g_e)
    phi_e = np.linspace(0,2*np.pi,361)
    inint = CherenkovPhoton.inner_integrand(phi_e,theta,theta_g,g_e)
    plt.figure(1)
    plt.plot(phi_e,inint)
    plt.xlabel('phi_e [rad]')
    plt.ylabel('Inner Integrand')
    plt.title('Inner Integrand with E=%.1f, theta=%.4f, theta_g=%.4f'%(E,theta,theta_g))
    
    print("inner_integral:")
    print("  CherenkovPhoton.inner_integral(theta,theta_g,g_e):")
    print("    %.2e"%CherenkovPhoton.inner_integral(theta,theta_g,g_e))

    print()
    print("outer_integrand samples:")
    t = 0.
    f_e = EnergyDistribution('Tot',t)
    ll = np.log(CherenkovPhoton.cherenkov_threshold(delta))
    ul = 10.309
    print("  CherenkovPhoton.outer_integrand(l_g,theta,f_e,delta)")
    print("  theta = %.4f, delta = %.4f, t = %.1f"%(theta,delta,t))
    print("  f_e: ",f_e)
    l_g = np.linspace(ll,ul,100)
    E_g = np.exp(l_g)
    outint = np.array([CherenkovPhoton.outer_integrand(l,theta,f_e,delta) for l in l_g])
    plt.figure(2)
    plt.plot(E_g,outint)
    plt.loglog()
    plt.xlabel('E_g [MeV]')
    plt.ylabel('Outer Integrand')
    plt.title('Outer Integrand with theta=%.4f, delta=%.4f, t=%.1f'%(theta,delta,t))
    
    print("outer_integral:")
    print("  CherenkovPhoton.outer_integral(theta,f_e,delta):")
    print("    %.2e"%CherenkovPhoton.outer_integral(theta,f_e,delta))

    #log10(pi/2)=0.19612
    lgtheta,dlgtheta = np.linspace(-3,0.2,161,retstep=True)
    theta = 10**lgtheta
    dtheta = 10**(lgtheta+dlgtheta/2)-10**(lgtheta-dlgtheta/2)
    gg_array = np.empty((11,161),dtype=float)
    t_array = np.linspace(-10,10,11)
    plt.figure(3)
    for i,t in enumerate(t_array):
        print("Stage %.0f"%t)
        f_e.set_stage(t)
        gg_array[i] = np.array([CherenkovPhoton.outer_integral(q,f_e,delta) for q in theta])
        gg_array[i] /= (gg_array[i]*dtheta).sum()
        plt.plot(theta,gg_array[i],label='t=%.0f, delta=%.6f'%(t,delta))
    plt.loglog()
    plt.xlabel('Theta [rad]')
    plt.ylabel('Photon Angle Distribution')
    plt.title('Photon Angular Distributions with delta=%.6f'%delta)
    plt.legend()
    plt.grid()
    plt.xlim(theta[0],theta[-1])
