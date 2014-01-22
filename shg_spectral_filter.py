# shg_spectral_filter.py
# Daniel Wilcox
# this Python module evaluates the SHG spectral filter for frequency-doubling
# refer to my October 2013 lab notebook "Simulation of second-harmonic-generation in absolute units"

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

c = 299.792458 # in nm per femtosecond
T = 10000 # in nm
theta_deg = 35.9 # in degrees
# theta_deg = 35.0 # in degrees
chi2 = 40 # in  nm*sqrt(fs/nJ)
theta = np.pi*theta_deg/180.0 # in radians

# indices of refraction in nm
def n_O_BBO(lm): # lm in nm
    if (lm > 220) and (lm < 1060):
        return np.sqrt(2.7504 + 0.0184/( (0.001*lm)**2 - 0.0179 ) - 0.0155*(0.001*lm)**2 )
    else:
        return 1.0 # return something non-absurd for unimportant wavelengths
def n_E_BBO(lm): # lm in nm
    if (lm > 220) and (lm < 1060):
        return np.sqrt(2.3730 + 0.0128/( (0.001*lm)**2 - 0.0156 ) - 0.0044*(0.001*lm)**2 )
    else:
        return 1.0 # return something non-absurd for unimportant wavelengths

# a derivative
def dn_dlambda_O_BBO(lm): # lm in nm
    if (lm > 220) and (lm < 1060):
        return 0.5/np.sqrt(2.7504 + 0.0184/( (0.001*lm)**2 - 0.0179 ) - 0.0155*(0.001*lm)**2 ) * ( -0.0184*2*(0.001*lm)*0.001/( (0.001*lm)**2 - 0.0179 )**2 - 2*0.0155*(0.001*lm)*0.001 )
    else:
        return 1.0

# lambda_vac in terms of omega
def lambda_vac(omega):
    return 2*np.pi*c/omega # omega in fs^-1, lambda0 in nm
    
# a derivative
def dlambda_domega(omega):
    return -2*np.pi*c/omega**2
    
# indices of refraction in angular inverse femtoseconds
def n_O(omega):
    return n_O_BBO(lambda_vac(omega))
def n_E(omega):
    return n_E_BBO(lambda_vac(omega))

# a derivative
def dn_domega_O_BBO(omega): 
    return dn_dlambda_O_BBO(lambda_vac(omega)) * dlambda_domega(omega)

    
# index of refraction for e-wave along the particular angle
def n_e(omega):
    return 1 / np.sqrt( np.sin(theta)**2/n_E(omega)**2 + np.cos(theta)**2/n_O(omega)**2 )


# define the spectral filter
def spectral_filter_single(omega, omega0, sigma):
    chi2_tilde = chi2 / ( np.sqrt(2*np.pi) * sigma )
    n_omega0 = n_O(omega0)
    vg = c/(n_omega0 + omega0*dn_domega_O_BBO(omega0))
    k = omega*n_e(omega)/c
    constant = 2 * omega0**2 * dn_domega_O_BBO(omega0) / c
    sinc_like = lambda a: (np.exp(-1j*a*T) - 1)/(-1j*a)
    return chi2_tilde*omega/(4*c*n_e(omega))*sinc_like(-k+omega/vg-constant)

# expand the spectral filter to many values of omega
# omega should be in inverse femtoseconds
# sigma should be in nm
# the spectral filter should be applied to squared fields that were in sqrt(nJ/fs)
# before squaring to produce second-harmonic
def spectral_filter(omega, omega0, sigma):
    return np.array(map(lambda o: spectral_filter_single(o, omega0, sigma), omega))

