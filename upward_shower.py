import numpy as np
from scipy.constants import value,nano
from cherenkov_photon import CherenkovPhoton
from cherenkov_photon_array import CherenkovPhotonArray
import atmosphere as at
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import Spectral

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
    def __init__(self,X_max,N_max,d0,earth_emergence,azimuth=0,n_stage=1000,
                ckarray='gg_t_delta_theta_2020_normalized.npz',tel_area = 1):

        self.n_stage = n_stage
        self.atmosphere = at.Atmosphere()
        self.gga = CherenkovPhotonArray(ckarray)
        self.tel_area = tel_area
        self.earth_radius = 6.371e6 # Earth radius in meters

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

    def h_to_axis_R_LOC(self,h,cos_EM_plus_90):
        '''Return the length along the shower axis from the point of Earth
        emergence to the height above the surface specified

        Parameters:
        h: array of heights in meters
        cos_EM_plus_90: cosine of the Earth emergence angle plus 90 degrees

        returns: r (same size as h), an array of distances along the shower axis_sp.
        '''
        r_CoE= h + self.earth_radius # distance from the center of the earth to the specified height
        r = self.earth_radius*cos_EM_plus_90 + np.sqrt(
                self.earth_radius**2*cos_EM_plus_90**2-self.earth_radius**2+r_CoE**2)
        return r

    def set_shower_axis(self):
        '''Create a table of 10000 distances and depths along the shower axis
        to interpolate into across its whole atmospheric path
        '''
        atm = self.atmosphere
        axis_cq = np.cos(self.zenith)
        axis_sq = np.sin(self.zenith)
        axis_cp = np.cos(self.azimuth)
        axis_sp = np.sin(self.azimuth)
        self.axis_h = np.linspace(0,atm.maximum_height,10000)
        self.axis_r = self.h_to_axis_R_LOC(self.axis_h, self.axis_cem)
        axis_rho = atm.density(self.axis_h)
        self.axis_dr = self.axis_r[1:] - self.axis_r[:-1]
        axis_deltaX = np.sqrt(axis_rho[1:]*axis_rho[:-1])*self.axis_dr * 1000 / 10000# converting to g/cm^2
        self.axis_X = np.concatenate((np.array([0]),np.cumsum(axis_deltaX)))
        self.axis_delta = np.nan_to_num(atm.delta(self.axis_h),nan=0)

    def interpolate_shower_steps(self):
        '''Superimpose 2000 1 g/cm^2 depth steps onto axis X vs r curve starting
        at the r corresponding to the height of first interaction
        '''
        self.shower_X = np.linspace(0,2000,self.n_stage)
        self.shower_start_r = self.h_to_axis_R_LOC(self.d0, self.axis_cem)
        self.start_depth = np.interp(self.shower_start_r,self.axis_r,self.axis_X)
        self.shower_axis_X = self.shower_X + self.start_depth
        self.shower_r = np.interp(self.shower_axis_X,self.axis_X,self.axis_r)
        shower_dr = self.shower_r[1:] - self.shower_r[:-1]
        self.shower_dr =np.concatenate((shower_dr,np.array([shower_dr[-1]])))
        self.shower_delta = np.interp(self.shower_r,self.axis_r,self.axis_delta)

    def set_telescope_to_axis(self,n_tel,array_width,array_height):
        '''Set the coordinates of a line of counters in the sky
        perpendicular to the shower axis at a height set by array height.
        Then set the vectors from the axis to each telescope.

        Parameters:
        n_tel: number of counters
        array_width: distance from the axis to the furthest detectors.
        array height: height above the Earth's surface of the center of the array.

        Sets class attributes:
        (these coordinates now use the shower axis as the cartesian z axis)
        tel_cart_vectors: vectors from first interaction point to each telescope
        axis_cart_vectors: vectors to each point on the shower axis
        travel_vector: cartesian vectors from each point on the shower axis to
        each telescopes
        travel_length: the distance between every axis point to each telescope
        '''
        array_z = self.h_to_axis_R_LOC(array_height, self.axis_cem)
        self.tel_cart_vectors = np.empty([n_tel,3])
        self.tel_cart_vectors[:,0] = np.zeros(n_tel)
        self.tel_cart_vectors[:,1] = np.linspace(-array_width,array_width,n_tel)
        self.tel_cart_vectors[:,2] = np.full((1,n_tel),array_z)
        self.axis_cart_vectors = np.empty([self.n_stage,3])
        self.axis_cart_vectors[:,0] = np.zeros(self.n_stage)
        self.axis_cart_vectors[:,1] = np.zeros(self.n_stage)
        self.axis_cart_vectors[:,2] = self.shower_r - self.shower_start_r
        self.travel_vector = self.tel_cart_vectors.reshape(-1,1,3) - self.axis_cart_vectors
        self.travel_length =  np.sqrt( (self.travel_vector**2).sum(axis=2) )
        self.tel_n = self.travel_vector / self.travel_length[:,:,np.newaxis]
        self.tel_cq = self.tel_n[:,:,-1] # cosines of the angles between vertical and vector from vertical height to telescope
        self.tel_q = np.arccos(self.tel_cq) # angle between axis and vector
        self.tel_omega = self.tel_area / self.travel_length **2

    def size(self,X):
        """Return the size of the shower at a slant-depth X

        Parameters:
            X: the slant depth at which to calculate the shower size [g/cm2]

        Returns:
            N: the shower size
        """
        Lambda = 70
        X_0 = 0

        ln_axis_nch = np.log(self.N_max) + ((self.X_max-X_0)/Lambda) * (np.log(X-X_0)
                    - np.log(self.X_max-X_0)) + ((self.X_max-X)/Lambda)
        axis_nch = np.exp(ln_axis_nch)
        return np.nan_to_num(axis_nch,nan=0)

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

    def set_shower_GH_stage(self):
        self.shower_nch = self.size(self.shower_X)
        self.shower_t = self.stage(self.shower_X)

    def calculate_gg(self):
        '''This function sets the class attribute gg which is the value of
        the normalized Cherenkov distribution for each stage going to each
        telescope.
        '''
        self.gg = np.empty_like(self.travel_length)
        for i in range(self.travel_length.shape[0]):
            for j in range(self.travel_length.shape[1]):
                self.gg[i,j] = self.gga.interpolate(self.shower_t[j],self.shower_delta[j],self.tel_q[i,j])
        #cut values from stages earlier than the ones explicitly tabulated
        gg_bt = self.shower_t<self.gga.t[0]
        self.gg[:,gg_bt] = 0
        #cut values from stages later than the ones explicitly tabulated
        gg_bt = self.shower_t>self.gga.t[-1]
        self.gg[:,gg_bt] = 0

    def calculate_yield(self):
        '''This function calculates the number of Cherenkov photons at each
        telescope using the normalized angular distribution values interpolated
        from the table.
        '''
        alpha_over_hbarc = 370.e2 # per eV per m, from PDG
        tel_dE =1.377602193180103 # Energy interval calculated from Cherenkov wavelengths
        chq = CherenkovPhoton.cherenkov_angle(1.e12,self.shower_delta)
        cy = alpha_over_hbarc*np.sin(chq)**2*tel_dE
        tel_factor = self.shower_nch * self.shower_dr * cy
        self.ng = self.gg * self.tel_omega * tel_factor
        self.ng_sum = self.ng.sum(axis = 1)

    # def calculate_times(self):
    #     axis_vertical_delay = np.cumsum((axis_delta*axis_dh))/c/spc.nano
    #     axis_time = axis_r[:-1]/c/spc.nano

    def crunch_numbers(self):
        '''This function performs all the calculations in order.
        '''
        self.set_shower_axis()
        self.interpolate_shower_steps()
        self.set_telescope_to_axis(100,100.e3,525.e3)
        self.set_shower_GH_stage()
        self.calculate_gg()
        self.calculate_yield()


if __name__ == '__main__':
    us = UpwardShower(500,1.e7,1000,np.radians(5))
    plt.figure()
    plt.plot(us.axis_r,us.axis_X,'bo')
    plt.plot(us.shower_r,us.shower_axis_X,'ro')
    plt.xlabel("Distance Along Axis from Earth's Surface (m)")
    plt.ylabel('grammage')
    plt.figure()
    plt.plot(us.shower_X,us.shower_nch)
    plt.ion()
    plt.figure()
    heights = [0,2000,4000,6000,8000,10000]
    for h in heights:
        us.reset_shower(500,1.e7,h,np.radians(5))
        plt.plot(us.tel_cart_vectors[:,1],us.ng_sum, label='Height = %.0f'%h)
    plt.xlabel('Counter Position [m from axis]')
    plt.ylabel('Number of Photons')
    plt.suptitle('Lateral Distribution at altitude 525 Km')
    plt.title('(5 degree Earth emergence angle)')
    plt.grid()
    plt.show()
