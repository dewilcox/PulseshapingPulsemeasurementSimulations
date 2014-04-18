# -*- coding: utf-8 -*-
# commonsimulation.py
# This Python script defines the field attributes common to
# the various  simulations in this folder

from __future__ import division
import numpy as np
#import matplotlib.pyplot as plt
import scipy.integrate
import os
import sys

# fundamental constants
c = 299.792458 # in nm per fs
h = 6.62606957e-10 # in nJ*fs


# laser characteristics
central_wavelength = 650.0 # in nm
bandwidth_nm = 110.0 
central_f = c/central_wavelength # in fs^-1
bandwidth_f = c*bandwidth_nm/central_wavelength**2

# frequency grid (relative to central_f)
num_points = 65536 #32768 #16384 #8192 #262144 #131072 
f, df = np.linspace(-20*bandwidth_f, 20*bandwidth_f, num_points, retstep=True)

# spectral intensity (not amplitude)
total_energy = 1e3 #20 # in nanojoules
# the first spectral intensity is an asymmetric Gaussian
spectral_intensity1 = np.exp( -(f-0.6*bandwidth_f)**2/(bandwidth_f**2) ) * np.exp( -np.power( np.abs(f)/bandwidth_f, 3 ) )
__intensity_integral1 = scipy.integrate.simps(spectral_intensity1, f)
spectral_intensity1[:] *= total_energy/__intensity_integral1
# the second spectral intensity is more complicated
spectral_intensity2 = (
    0.8 * np.exp( -(f+0.4*bandwidth_f)**2/(0.05*bandwidth_f**2) ) + 
    0.5 * np.exp( -(f+0.2*bandwidth_f)**2/(0.06*bandwidth_f**2) ) + 
    0.6 * np.exp( -(f-0.1*bandwidth_f)**2/(0.04*bandwidth_f**2) ) + 
    0.8 * np.exp( -(f-0.7*bandwidth_f)**2/(0.05*bandwidth_f**2) ) 
    )
__intensity_integral2 = scipy.integrate.simps(spectral_intensity2, f)
spectral_intensity2[:] *= total_energy/__intensity_integral2
# at this point in time spectral_intensity is in units of nanojoules per inverse femtosecond

# create the spectral group-delay dispersion; four different values
# the first one is simple
spectral_gdd1 = 20 + 50*2*np.pi*f # just second- and third-order dispersion
spectral_gd1 = scipy.integrate.cumtrapz(spectral_gdd1, 2*np.pi*f, initial=0.0)
spectral_gd1[:] -= spectral_gd1[np.argmin(np.abs(f-0.0))]
spectral_phase1 = scipy.integrate.cumtrapz(spectral_gd1, 2*np.pi*f, initial=0.0)
spectral_phase1[:] -= spectral_phase1[np.argmin(np.abs(f-0.0))]
# the second one is realistic but complex
spectral_gdd2 = ( 
    15*np.sin( 2*np.pi*(f-0.1*bandwidth_f)*0.3/bandwidth_f ) +
    18*np.sin( 2*np.pi*(f-0.3*bandwidth_f)/bandwidth_f ) + 
    22*np.sin( 2*np.pi*(f+0.2*bandwidth_f)*1.7/bandwidth_f ) +
    13*np.sin( 2*np.pi*(f-0.13*bandwidth_f)*2.6/bandwidth_f ) +
    9*np.sin( 2*np.pi*(f-0.21*bandwidth_f)*4.3/bandwidth_f ) +
    1)
spectral_gd2 = scipy.integrate.cumtrapz(spectral_gdd2, 2*np.pi*f, initial=0.0)
spectral_gd2[:] -= spectral_gd2[np.argmin(np.abs(f-0.0))]
spectral_phase2 = scipy.integrate.cumtrapz(spectral_gd2, 2*np.pi*f, initial=0.0)
spectral_phase2[:] -= spectral_phase2[np.argmin(np.abs(f-0.0))]
# the third and final one
spectral_gdd3 = (
    52*np.sin( 2*np.pi*(f+0.1*bandwidth_f)*0.3/bandwidth_f ) +
    48*np.sin( 2*np.pi*(f-0.1*bandwidth_f)/bandwidth_f ) + 
    61*np.sin( 2*np.pi*(f-0.14*bandwidth_f)*1.7/bandwidth_f ) +
    40*np.sin( 2*np.pi*(f-0.26*bandwidth_f)*2.6/bandwidth_f ) +
    26*np.sin( 2*np.pi*(f+0.35*bandwidth_f)*4.3/bandwidth_f ) +
    15*np.sin( 2*np.pi*(f+0.19*bandwidth_f)*7.8/bandwidth_f ) +
    20)
spectral_gd3 = scipy.integrate.cumtrapz(spectral_gdd3, 2*np.pi*f, initial=0.0)
spectral_gd3[:] -= spectral_gd3[np.argmin(np.abs(f-0.0))]
spectral_phase3 = scipy.integrate.cumtrapz(spectral_gd3, 2*np.pi*f, initial=0.0)
spectral_phase3[:] -= spectral_phase3[np.argmin(np.abs(f-0.0))]

    
# create the spectral intensity + group-delay combinations
spectral_combos = (
    (spectral_gd1, spectral_intensity1, spectral_phase1), # option 1
    (spectral_gd1, spectral_intensity2, spectral_phase1), # option 2
    (spectral_gd2, spectral_intensity1, spectral_phase2), # option 3
    (spectral_gd3, spectral_intensity2, spectral_phase3) # option 4
    )
    
# decide what the combination of spectral intensity and group-delay is
pulse_combination_number = int(sys.argv[1])
spectral_gd = spectral_combos[pulse_combination_number][0]
spectral_intensity = spectral_combos[pulse_combination_number][1]
spectral_phase = spectral_combos[pulse_combination_number][2]
spectral_amplitude = np.sqrt(spectral_intensity)


# create the spectral field
# compressed_spectral_field = spectral_amplitude
spectral_field = spectral_amplitude * np.exp(1j*spectral_phase)

# create a temporal array
t = np.fft.fftfreq(num_points, d=df)
t = np.fft.fftshift( t )
assert( np.all(np.diff(t) > 0) )
dt = t[1] - t[0]

# define transformations from the spectral to temporal domains
# and vice versa
spectral_to_temporal = lambda data: np.fft.fftshift( np.fft.ifft( np.fft.ifftshift(data) ) ) * data.size * df
temporal_to_spectral = lambda data: np.fft.fftshift( np.fft.fft( np.fft.ifftshift(data) ) ) * dt

# create the temporal field
# in units of sqrt(nanojoules per femtosecond)
# compressed_temporal_field = spectral_to_temporal(compressed_spectral_field)
temporal_field = spectral_to_temporal(spectral_field)




###########################
# now, define a conversion from spectral fields to CCD counts
def compute_ccd_mapping_function(in_central_f):
    sim_lambda = c / (f + in_central_f)
    d_lambda = np.zeros_like(sim_lambda)
    d_lambda[1:-1] = np.abs( 0.5 * (sim_lambda[2:] - sim_lambda[:-2]) )
    # here, the pixel size is 20 microns (20e-6 meters), and the grating spacing is (1e-3/grating) meters, and the focal length is 320mm for the iHR-320
    # the conversion from meters to nm is (1e9 nm/meter)
    grating = 600 # in grooves per mm
    nm_per_pixel = 1e9*(20e-6)*(1e-3/grating)/(0.320) # pixels are 20 microns, focal length is 0.320 meters
    num_pixels = 1340.0
    spec_lambda = c/(in_central_f) + nm_per_pixel * np.linspace( -(num_pixels-1.0)/2.0, (num_pixels-1.0)/2.0, num_pixels ) 
    spec_lambda_boundaries = c/(in_central_f) + nm_per_pixel * np.linspace( -num_pixels/2.0, num_pixels/2.0, num_pixels+1 )
    #spec_f_boundaries = c 
    # point_spread_const = 5*nm_per_pixel # this is a 100-micron slit since pixels are 20 microns
    point_spread_const = 2.5*nm_per_pixel # this is a 50-micron slit since pixels are 20 microns
    # point_spread_const = 1.5*nm_per_pixel # this is a 30-micron slit since pixels are 20 microns
    ccd_map = np.zeros( (num_pixels, num_points) )
    # now, each simulation frequency has a spot size and therefore contributes to each pixel
    detectable_lambda = np.logical_and(sim_lambda > np.amin(spec_lambda_boundaries), sim_lambda < np.amax(spec_lambda_boundaries))
    for i in np.flatnonzero(detectable_lambda):
        # I use a sech distribution for each simulation frequency; I integrate
        # that distribution into pixels below
        # Now, why sech? I chose it because the tails are heavier than Gaussian
        # and it seems like the point-spread of the spectrometer (including aberrations)
        # is heavier than Gaussian
        ccd_map[:, i] = (2.0/np.pi)*np.arctan(np.exp( 0.5*np.pi*(sim_lambda[i] - spec_lambda_boundaries[:-1])/point_spread_const )) - (2.0/np.pi)*np.arctan(np.exp( 0.5*np.pi*(sim_lambda[i] - spec_lambda_boundaries[1:])/point_spread_const ))
        # the above transformation keeps the units constant;
        # i.e. the input data is nJ per inverse femtosecond
        # but the output should be in photoelectrons
        energy_per_datapoint = df # that is, df*datapoint is the energy in the bin
        energy_per_photon = h*(f[i] + in_central_f)
        photons_per_datapoint = energy_per_datapoint / energy_per_photon
        ccd_counts_per_photon = 1 # one photon per CCD count (highest gain setting)
        quantum_efficiency = 0.35 # of the CCD detector; see http://www.princetoninstruments.com/Uploads/Princeton/Documents/Datasheets/PIXIS/Princeton_Instruments_PIXIS_100_eXcelon_rev_N5_8.21.2012.pdf
        grating_efficiency = 0.7 # of the spectrometer grating; see grating 51012 of http://www.horiba.com/us/en/scientific/products/diffraction-gratings/catalog/510/#51012xxx
        slit_efficiency = 0.6 # for a 50-micron slit this is reasonable
        other_efficiency = 0.6 # including various mirror reflectivities and so on
        efficiency = quantum_efficiency * grating_efficiency * other_efficiency * slit_efficiency
        # final discretization
        ccd_map[:, i] *= ccd_counts_per_photon * photons_per_datapoint * efficiency
    assert(np.all(np.isfinite(ccd_map)))
    # plt.figure()
    # plt.imshow(ccd_map, aspect='auto', interpolation='nearest')
    # plt.figure()
    # plt.plot( np.sum(ccd_map, axis=0) )
    # plt.ylim(-0.1, 1.1)
    return ccd_map, spec_lambda


    
def variance_level(in_data, num_shots_averaged):
    # read noise, as measured in photoelectrons
    read_noise = 11.0 # this comes from http://www.princetoninstruments.com/Uploads/Princeton/Documents/Datasheets/PIXIS/Princeton_Instruments_PIXIS_100_eXcelon_rev_N5_8.21.2012.pdf
    return (in_data+read_noise**2)/num_shots_averaged 
    
###########################
# now, add read noise and shot noise to the spectrometer-discretized light
def measure(in_data, prng, num_shots_averaged):
    noise = np.sqrt(variance_level(in_data, num_shots_averaged))
    assert( in_data.size == in_data.shape[0] * in_data.shape[1] )
    return in_data + noise * prng.randn( in_data.shape[0], in_data.shape[1] )
