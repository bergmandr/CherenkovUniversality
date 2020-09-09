import numpy as np

class OrbitalCounters():
    '''Class for calculating Cherenkov yield of upward going showers at a
    hypothetical orbital telescope array_z

    Parameters:
    shower_R: array of distances along shower axis (m)
    n_tel: number of telescopes
    tel_distance: how far they are along the axis from first interaction point (m)
    tel_area: surface area of telescopes (m^2)

    '''

    def __init__(self,shower_R,n_tel,array_width,tel_distance,tel_area = 1):
        self.n_tel = n_tel
        self.tel_distance = tel_distance
        self.tel_area = tel_area
        self.n_stage = np.shape(shower_R)[0]
        self.axis_cart_vectors = np.empty([self.n_stage,3])
        self.axis_cart_vectors[:,0] = np.zeros(self.n_stage)
        self.axis_cart_vectors[:,1] = np.zeros(self.n_stage)
        self.axis_cart_vectors[:,2] = shower_R
        self.tel_cart_vectors = np.empty([self.n_tel,3])
        self.tel_cart_vectors[:,0] = np.zeros(n_tel)
        self.tel_cart_vectors[:,1] = np.linspace(-array_width,array_width,n_tel)
        self.tel_cart_vectors[:,2] = np.full((1,n_tel),tel_distance)
        self.set_travel_length()

    def set_travel_length(self):
        self.travel_vector = self.tel_cart_vectors.reshape(-1,1,3) - self.axis_cart_vectors
        self.travel_length =  np.sqrt( (self.travel_vector**2).sum(axis=2) )
        self.tel_n = self.travel_vector / self.travel_length[:,:,np.newaxis]
        self.tel_cq = self.tel_n[:,:,-1] # cosines of the angles between vertical and vector from vertical height to telescope
        self.tel_q = np.arccos(self.tel_cq) # angle between axis and vector
        self.tel_omega = self.tel_area / self.travel_length **2
