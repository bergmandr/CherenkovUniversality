import numpy as np
from scipy.constants import value,nano
from cherenkov_photon import CherenkovPhoton
from cherenkov_photon_array import CherenkovPhotonArray
import atmosphere as at
from Orbital_Counter_Class import OrbitalCounters

class UpwardShower():
    """A class for generating upward shower profiles and their Cherenkov
    outputs. The shower can either be a Gaisser Hillas shower or a Griessen
    Shower.

    Parameters:
    X_max: depth at shower max (g/cm^2)
    N_max: number of charged particles at X_max
    d0: height of first interaction (meters)
    earth_emergence: angle the primary particle makes with the tangent line of
    the Earth's surface as it emerges (radians)
    azimuth: azimuthal angle of Earth emergence (radians)
    n_stage: number of depth steps for calculating Cherenkov light
    ckarray: filename of the Cherenkov distribution table
    tel_area: surface area of the orbital telescopes (m^2)
    """
    earth_radius = 6.371e6
    Lambda = 70
    atm = at.Atmosphere()
    axis_h = np.linspace(0,atm.maximum_height,1000)
    axis_rho = atm.density(axis_h)
    axis_delta = atm.delta(axis_h)

    def __init__(self,X_max,N_max,d0,earth_emergence,azimuth=0,
                ckarray='gg_t_delta_theta_2020_normalized.npz',tel_area = 1):

        # self.n_stage = n_stage
        self.atmosphere = at.Atmosphere()
        self.gga = CherenkovPhotonArray(ckarray)
        self.tel_area = tel_area

        self.reset_shower(X_max,N_max,d0,earth_emergence,azimuth=0)

    def reset_shower(self,X_max,N_max,d0,earth_emergence,azimuth=0):
        '''Set necessary attributes and perform calculations
        '''
        self.X_max = X_max
        self.N_max = N_max
        self.d0   = d0
        self.earth_emergence = earth_emergence
        self.zenith = np.pi / 2 - earth_emergence
        self.azimuth = azimuth
        self.axis_cem = np.cos(np.pi - self.zenith)
        self.crunch_numbers()

    def crunch_numbers(self):
        '''This function performs all the calculations in order.
        '''
        self.set_shower_axis()
        self.select_shower_steps()
        self.calculate_gg()
        self.calculate_yield()

    def h_to_axis_R_LOC(self,h,cos_EM):
        '''Return the length along the shower axis from the point of Earth
        emergence to the height above the surface specified

        Parameters:
        h: array of heights in meters
        cos_EM: cosine of the Earth emergence angle plus 90 degrees

        returns: r (same size as h), an array of distances along the shower axis_sp.
        '''
        R = self.earth_radius
        r_CoE= h + R # distance from the center of the earth to the specified height
        r = R*cos_EM + np.sqrt(R**2*cos_EM**2-R**2+r_CoE**2)
        return r

    def flat_earth_difference(self,r,EM,CEM):
        R = self.earth_radius
        h_flat = r * np.sin(EM)
        h_true = np.sqrt(r**2 + R**2 -2*r*R*CEM) - R
        return h_true - h_flat

    def set_shower_axis(self):
        '''Create a table of 10000 distances and depths along the shower axis
        to interpolate into across its whole atmospheric path
        '''
        self.axis_r = self.h_to_axis_R_LOC(self.axis_h, self.axis_cem)
        self.axis_dr = self.axis_r[1:] - self.axis_r[:-1]
        axis_deltaX = np.sqrt(self.axis_rho[1:]*self.axis_rho[:-1])*self.axis_dr / 10# converting to g/cm^2
        self.axis_X = np.concatenate((np.array([0]),np.cumsum(axis_deltaX)))
        self.axis_dr = np.concatenate((np.array([0]),self.axis_dr))


    def select_shower_steps(self):
        '''Select the depth indices steps where the shower is producing light,
        i.e. where there are charged particles

        '''
        self.axis_start_r = self.h_to_axis_R_LOC(self.d0, self.axis_cem)
        self.X0 = np.interp(self.axis_start_r,self.axis_r,self.axis_X)
        self.X_max += self.X0
        self.axis_nch = self.size(self.axis_X)
        self.axis_nch[self.axis_nch<1.e-1] = 0
        self.axis_t = self.stage(self.axis_X)
        self.i_ch = np.nonzero(self.axis_nch)
        self.n_stage = np.size(self.i_ch)
        axis_r = self.axis_r[self.i_ch] - self.axis_start_r
        tel_r = self.h_to_axis_R_LOC(525.e3, self.axis_cem) - self.axis_start_r
        self.OC = OrbitalCounters(axis_r,100,100.e3,tel_r)


    def size(self,X):
        """Return the size of the shower at a slant-depth X

        Parameters:
            X: the slant depth at which to calculate the shower size [g/cm2]

        Returns:
            N: the shower size
        """
        x =         (X-self.X0)/self.Lambda
        g0 = x>0.
        m = (self.X_max-self.X0)/self.Lambda
        n = np.zeros_like(x)
        n[g0] = np.exp( m*(np.log(x[g0])-np.log(m)) - (x[g0]-m) )
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

    def calculate_gg(self):
        '''This function sets the class attribute gg which is the value of
        the normalized Cherenkov distribution for each stage going to each
        telescope.
        '''

        self.gg = np.empty_like(self.OC.travel_length)
        for i in range(self.OC.travel_length.shape[0]):
            for j in range(self.OC.travel_length.shape[1]):
                if self.axis_t[self.i_ch][j]>self.gga.t[0] and self.axis_t[self.i_ch][j]<self.gga.t[-1]:
                    self.gg[i,j] = self.gga.interpolate(self.axis_t[self.i_ch][j],self.axis_delta[self.i_ch][j],self.OC.tel_q[i,j])
                else:
                    self.gg[i,j] = 0

    def calculate_yield(self):
        '''This function calculates the number of Cherenkov photons at each
        telescope using the normalized angular distribution values interpolated
        from the table.
        '''
        alpha_over_hbarc = 370.e2 # per eV per m, from PDG
        tel_dE =1.377602193180103 # Energy interval calculated from Cherenkov wavelengths
        chq = CherenkovPhoton.cherenkov_angle(1.e12,self.axis_delta[self.i_ch])
        cy = alpha_over_hbarc*np.sin(chq)**2*tel_dE
        tel_factor = self.axis_nch[self.i_ch] * self.axis_dr[self.i_ch] * cy
        self.ng = self.gg * self.OC.tel_omega * tel_factor
        self.ng_sum = self.ng.sum(axis = 1)

    # def calculate_times(self):
    #     axis_vertical_delay = np.cumsum((axis_delta*axis_dh))/c/spc.nano
    #     axis_time = axis_r[:-1]/c/spc.nano




if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.cm import Spectral
    us = UpwardShower(500,1.e7,5000,np.radians(5))
    plt.ion()
    plt.figure()
    plt.plot(us.axis_r,us.axis_X,'bo')
    plt.plot(us.axis_r[us.i_ch],us.axis_X[us.i_ch],'ro')
    plt.xlabel("Distance Along Axis from Earth's Surface (m)")
    plt.ylabel('grammage')
    plt.figure()
    plt.plot(us.axis_X[us.i_ch],us.axis_nch[us.i_ch])

    plt.figure()
    heights = [0,2000,4000,6000,8000,10000]
    for h in heights:
        us.reset_shower(500,1.e7,h,np.radians(5))
        plt.plot(us.OC.tel_cart_vectors[:,1],us.ng_sum, label='Height = %.0f'%h)
    plt.xlabel('Counter Position [m from axis]')
    plt.ylabel('Number of Photons')
    plt.suptitle('Lateral Distribution at altitude 525 Km')
    plt.title('(5 degree Earth emergence angle)')
    plt.legend()
    plt.grid()
    plt.show()
