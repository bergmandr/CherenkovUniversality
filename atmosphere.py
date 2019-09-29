import numpy as np
from scipy.integrate import quad

class atmosphere:
    """
    Class containing constants and methods for using the US Standard Atmosphere of 1976
    
    The temperature is assumed to change linearly with height above sea level. From this
    and the assumption of hybrdostatic equilibrium the pressure and density are calculated.

    While instantiated, the default parameters give the US Standard Atmosphere. Other
    atmospheres can be provided
    """

    # Class constants for 1976 US Standard Atmosphere
    temperature_sea_level = 288.15    # K
    pressure_sea_level    = 101325    # Pa
    density_sea_level     = 1.225     # kg/m3
    air_mol_weight        = 28.9644   # amu
    gravity               = 9.80665   # m/s2
    gas_constant          = 8.31432   # J/MolK
    gMR           = gravity * air_mol_weight / gas_constant

    def __init__(s,altitudes=None,rel_pressure=None,temperatures=None,temp_gradient=None):
        """
        Create and instance of an atmospheric model. 

        If no parameters are provided, the US Standard Atmosphere is
        used. It uses these values
            altitudes    = np.array([0.,11000.,20000.,32000.,47000.,51000.,71000.,84852.])
            rel_pressure = np.array([1.,
                                     2.23361105092158e-1,5.40329501078488e-2,8.56667835929167e-3,
                                     1.09456013377711e-3,6.60635313285837e-4,3.90468337334393e-5,
                                     3.68500952357479e-6])
            temperatures = np.array([288.15,216.65,216.65,228.65,270.65,270.65,214.65,186.946])
            temp_gradient = np.array([-0.0065,0.,0.001,0.0028,0.,-0.0028,-0.002,0.])

        If another atmosphere is to be created, each of the parameters should be an identical length
        ndarray. All the parameters must be specified or none of them. The altitudes array must
        be ordered.
        """
        if altitudes is None and rel_pressure is None and \
           temperatures is None and temp_gradient is None:
            s.altitudes    = np.array([0.,11000.,20000.,32000.,47000.,51000.,71000.,84852.])            # m above SL
            s.rel_pressure = np.array([1.,
                                       2.23361105092158e-1,5.40329501078488e-2,8.56667835929167e-3,
                                       1.09456013377711e-3,6.60635313285837e-4,3.90468337334393e-5,
                                       3.68500952357479e-6])
            s.temperatures = np.array([288.15,216.65,216.65,228.65,270.65,270.65,214.65,186.946]) # K
            s.temp_gradient = np.array([-0.0065,0.,0.001,0.0028,0.,-0.0028,-0.002,0.])    # K/m
        else:
            s.altitudes     = altitudes
            s.rel_pressure  = rel_pressure
            s.temperatures  = temperatures
            s.temp_gradient = temp_gradient
        s.maximum_height = s.altitudes[-1]
        s.minimum_height = s.altitudes[0]

    def atmosphere(s,h):
        """
        This function returns atmospheric temperature, pressure, and density as a function of height.
        
        Parameters:
            h - height in atmosphere. This can be an ndarray or a single value. [m]

        Returns:
            T   - temperature [K]
            P   - pressure [Pa]
            rho - density [kg/m3]
        """
        if not type(h) is np.ndarray:
            h = np.array([h],dtype=float)
            nin = 0
        else:
            nin = len(h)

        # Find the entry in the tables for each height
        too_low  = h < s.minimum_height
        too_high = h > s.maximum_height
        indx  = np.searchsorted(s.altitudes,h,side='right')
        idx = indx - 1

        # Find the temperature at height
        altitude        = s.altitudes[idx]
        base_temp       = s.temperatures[idx]
        temp_gradient   = s.temp_gradient[idx]
        delta_altitude  = h - altitude
        temperature     = base_temp + temp_gradient*delta_altitude

        # Find the relative pressure at height
        base_rel_pressure = s.rel_pressure[idx]
        flat = np.abs(temp_gradient) < 1.e-10
        rel_pressure = np.empty_like(h)
        rel_pressure[flat]  = base_rel_pressure[flat]  * \
                              np.exp(-s.gMR/1000*delta_altitude[flat]/base_temp[flat])
        rel_pressure[~flat] = base_rel_pressure[~flat] * \
                              (base_temp[~flat]/temperature[~flat])**(s.gMR/1000/temp_gradient[~flat])
        pressure = rel_pressure * s.pressure_sea_level
        density  = rel_pressure * s.density_sea_level * s.temperature_sea_level/temperature 

        temperature[too_low] = s.temperature_sea_level
        pressure[too_low]    = s.pressure_sea_level
        density[too_low]     = s.density_sea_level
        temperature[too_high] = 0.
        pressure[too_high]    = 0.
        density[too_high]     = 0.
        
        T = temperature
        P = pressure
        rho = density

        if nin == 0:
            return T[0],P[0],rho[0]
        else:
            return T,P,rho

    def temperature(s,h):
        """
        This function returns temperature as a function of height.
        
        Parameters:
            h - height in atmosphere. This can be an ndarray or a single value. [m]

        Returns:
            T - temperature [K]
        """
        T,_,_ = s.atmosphere(h)
        return(T)

    def pressure(s,h):
        """
        This function returns pressure as a function of height.
        
        Parameters:
            h - height in atmosphere. This can be an ndarray or a single value. [m]

        Returns:
            P - pressure [Pa]
        """
        _,P,_ = s.atmosphere(h)
        return(P)
    
    def density(s,h):
        """
        This function returns density as a function of height.
        
        Parameters:
            h - height in atmosphere. This can be an ndarray or a single value. [m]

        Returns:
            rho - density [kg/m3]
        """
        _,_,rho = s.atmosphere(h)
        return(rho)
    
    def delta(s,h):
        """
        This function returns the difference of the index-of-refraction from unity.
        
        Parameters:
            h - height in atmosphere. This can be an ndarray or a single value. [m]
        
        Returns:
            delta - equal to n - 1.
        """
        T,P,_ = s.atmosphere(h)
        P /= 1000.       # Pa -> kPa
        return 7.86e-4*P/T
        
    def depth(s,h1,h2=None):
        """
        This function returns atmospheric depth. It is the integral of atmospheric density between two heights.
        
        Parameters:
        These parameters can be ndarrays or single values.
        
        h1 - height 1 in atmosphere. This can be an ndarray or a single value. [m]
        h2 - height 2; Default is hMaxAtm. This can be an ndarray or a single value [m]
        
        If both h1 and h2 are ndarrays, they must be the same size (the length 
        of the shorter array is used).
        
        If h1 or h2 is greater than hMaxAtm, hMaxAtm is used.
        
        Returns:
        The integral of rho from h1 to h2. The result is converted into g/cm2.
        
        """
        if h2 is None:
            h2 = s.maximum_height*np.ones_like(h1)
            
        if not type(h1) is np.ndarray and not type(h2) is np.ndarray:
            h1 = np.array([h1],dtype=float)
            h2 = np.array([h2],dtype=float)
            nin = 0
        elif not type(h2) is np.ndarray:
            h2 = h2*np.ones_like(h1)
            nin = len(h1)
        elif not type(h1) is np.ndarray:
            h1 = h1*np.ones_like(h2)
            nin = len(h2)
        else:
            nin = min(len(h1),len(h2))

        A = h1.copy()
        B = h2.copy()
        A[A<s.minimum_height] = s.minimum_height
        B[B<s.minimum_height] = s.minimum_height
        A[A>s.maximum_height] = s.maximum_height
        B[B>s.maximum_height] = s.maximum_height

        depth = np.array([quad(s.density,a,b)[0] for a,b in zip(A,B)])
        depth /= 10. # 1 km/m2 == 1000/10,000 g/cm2

        if nin == 0:
            return depth[0]
        else:
            return depth

    def slant_depth(s,theta,d1,d2=None):
        """
        This function returns atmospheric depth as a function of the slant angle with respect to the vertical.
        
        Parameters:
            theta - slant angle with respect to the vertical.This can be an ndarray or a single value. [rad]
            d1 - Distance along slant trajectory. This can be an ndarray or a single value. [m]
            d2 - Distance along slant trajectory. This can be an ndarray or a single value. [m]
        
        If both theta, d1, and d2 are all ndarrays, they must be the same size (the length 
        of the shortest array is used).
        
        If d1 or d2 is are beyond the limits of the atmosphere, the limit of the atmosphere is used

        If d2 is not specified, the limit of the atmosphere is used.

        A flat-Earth model is assumed, so theta=pi/2 will give infinite results
        
        Returns:
            The slant depth from d2 to d1 at angle theta. [g/cm2]
        """
        if d2 is None:
            d2 = s.maximum_height/np.cos(theta)

        if not type(theta) is np.ndarray and \
           not type(d1) is np.ndarray and \
           not type(d2) is np.ndarray:
            theta = np.array([theta],dtype=float)
            d1 = np.array([d1],dtype=float)
            d2 = np.array([d2],dtype=float)
            nin = 0
        elif not type(d1) is np.ndarray and \
             not type(d2) is np.ndarray:
            d1 = d1*np.ones_like(theta)
            d2 = d2*np.ones_like(theta)
            nin = len(theta)
        elif not type(theta) is np.ndarray and \
             not type(d2) is np.ndarray:
            theta = theta*np.ones_like(d1)
            d2 = d2*np.ones_like(d1)
            nin = len(d1)
        elif not type(theta) is np.ndarray and \
             not type(d1) is np.ndarray:
            theta = theta*np.ones_like(d2)
            d1 = d1*np.ones_like(d2)
            nin = len(d2)
        elif not type(theta) is np.ndarray:
            theta = theta*np.ones_like(d1)
            nin = min(len(d1),len(d2))
        elif not type(d1) is np.ndarray:
            d1 = d1*np.ones_like(theta)
            nin = min(len(theta),len(d2))
        elif not type(d2) is np.ndarray:
            d2 = d2*np.ones_like(theta)
            nin = min(len(theta),len(d1))
        else:
            nin = min(len(theta),len(d1),len(d2))

        costheta = np.cos(theta)
        A = d1.copy()
        B = d2.copy()
        A[A<s.minimum_height] = s.minimum_height
        B[B<s.minimum_height] = s.minimum_height
        bigA = A>s.maximum_height/costheta
        A[bigA] = s.maximum_height/costheta[bigA]
        bigB = B>s.maximum_height/costheta
        B[bigB] = s.maximum_height/costheta[bigB]

        h1 = A*costheta
        h2 = B*costheta

        if nin == 0:
            return s.depth(h1,h2)/costheta[0]
        else:
            return s.depth(h1,h2)/costheta
        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.ion()
    us77 = atmosphere()
    h = np.linspace(0,100000,101)
    T,P,rho = us77.atmosphere(h)
    X = us77.depth(h)
    plt.plot(h,T,label='Temperature')
    plt.plot(h,P,label='Pressure')
    plt.plot(h,rho,label='Density')
    plt.plot(h,X,label='Depth')
    X30 = us77.slant_depth(30*np.pi/180,h)
    X60 = us77.slant_depth(60*np.pi/180,h)
    X75 = us77.slant_depth(75*np.pi/180,h)
    plt.plot(h,X30,label='Slant Depth, 30deg')
    plt.plot(h,X60,label='Slant Depth, 60deg')
    plt.plot(h,X75,label='Slant Depth, 75deg')
    plt.yscale('log')
    plt.xlim(h[0],h[-1])
    plt.grid()
    plt.legend()

    
