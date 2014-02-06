# -*- coding: utf-8 -*-
# MasterScript.py
# Daniel Wilcox
# This Python script produces the figures for the "SPEAR comparison of S-N" paper
# It runs the whole process, thus called "MasterScript"

# The organization of this script is as follows: 
# The first section defines functions that create spectral filters
# The second section defines functions to create data from those spectral filters
# The third section defines various analysis functions
# The fourth section defines a figure-creating function
# The fifth and final section runs everything



from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('PDF') # this is so we don't require an X window session
import matplotlib.pyplot as plt
import commonsimulation as cs
import shg_spectral_filter
import scipy.integrate
import scipy.interpolate
import matplotlib
import multiprocessing
import scipy.optimize
import general_pulseshaping_analysis
import general_pulseshaping_analysis_Fourier
import general_pulseshaping_analysis_FourierII
import time




#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
# This is the section that creates spectral filters

def create_SPIDER_spectral_filters():
    # define the parameters of the phase-cycled SPIDER
    tau = 200 # in fs, much larger than pulse-duration
    delta_omega_list = np.array([0.7, 0.7, 0.2, 0.15])*cs.bandwidth_f*2*np.pi
    delta_omega = delta_omega_list[cs.pulse_combination_number] # this is the spectral resolution
    phi2 = tau/delta_omega # this is the chirp of the chirped pulse
    Delta_omega = 10*delta_omega # this is the spectral width of the chirped pulse
    
    # create spectral filters for the various components
    H1 = lambda omega: np.exp(-1j*omega*tau/2) # one time-delayed pulse
    H2 = lambda omega: np.exp(1j*omega*tau/2) # the other time-delayed pulse
    Hc = lambda omega: np.exp( -np.power(np.abs(omega/Delta_omega), 4) ) * np.exp(0.5j*phi2*omega**2) # the chirped pulse
    
    # create the final spectral filters
    Hpp = lambda omega: ( H1(omega) + H2(omega) + Hc(omega) ) / 3.0
    Hpm = lambda omega: ( H1(omega) - H2(omega) + Hc(omega) ) / 3.0
    Hmp = lambda omega: ( -H1(omega) + H2(omega) + Hc(omega) ) / 3.0
    Hmm = lambda omega: ( -H1(omega) - H2(omega) + Hc(omega) ) / 3.0
    
    # put them in a list
    spectral_filters = [Hpp, Hpm, Hmp, Hmm]
    
    # return the results: the filters themselves and some data to be used later on
    return (spectral_filters, tau, delta_omega)


def create_MIIPS_spectral_filters():
    # define the parameters of MIIPS
    num_filters = 64
    alpha_list = np.array([2.5, 2.0, 50.0, 70.0])*np.pi
    alpha = alpha_list[cs.pulse_combination_number] # in radians
    gamma = 5 # in fs
    deltas = np.linspace(0, 2*np.pi, num_filters, endpoint=False)
    
    # create the spectral filters in a list
    spectral_filters = [ 
        ( lambda omega, phase=deltas[i]: np.exp(1j*alpha*np.sin(gamma*omega - phase)) )
        for i in range(num_filters)
        ]
    
    # return the results: the filters and some extra information for retrieval
    return (spectral_filters, alpha, gamma, deltas)
    

def create_CRT_spectral_filters():
    # define the parameters of CRT
    alphas_list = [ [-300, 300], [-300, 300], [-300, 300], [-600, 600] ]
    alphas = np.array( alphas_list[cs.pulse_combination_number] )
    num_filters = alphas.size
    
    # create the spectral filters in a list
    spectral_filters = [ 
        ( lambda omega, cur_alpha=alphas[i]: np.exp(0.5j*cur_alpha*omega**2) )
        for i in range(num_filters)
        ]
    
    # return the results: the filters and some extra information for retrieval
    return (spectral_filters, alphas)
    

def create_SPEAR_spectral_filters():
    # define the parameters of SPEAR
    alphas_list = [
        [-400, -300, 300, 400],
        [-400, -300, 300, 400],
        [-400, -300, 300, 400],
        [-700, -600, 600, 700] ]
    alphas = np.array( alphas_list[cs.pulse_combination_number] )
    num_filters = alphas.size
    
    # create the spectral filters in a list
    spectral_filters = [ 
        ( lambda omega, cur_alpha=alphas[i]: np.exp(0.5j*cur_alpha*omega**2) )
        for i in range(num_filters)
        ]
    
    # return the results: the filters and some extra information for retrieval
    return (spectral_filters, alphas)
    

def create_FROG_spectral_filters():
    # define the parameters of the phase-cycled FROG
    num_big_T = 35
    last_T = 60 # in fs
    dT_near_zero = 1.0 # in fs
    b = scipy.optimize.ridder( lambda b: np.arcsinh( last_T / (b*dT_near_zero) ) - (num_big_T-1.)/b, 1e-5, 1e5 * num_big_T )
    #b = scipy.optimize.ridder( lambda b: last_T - b*dT_near_zero*np.sinh( (num_big_T-1.)/b ), 5e-2, 1e5 * num_big_T )
    a = b * dT_near_zero
    T = a * np.sinh( np.arange(num_big_T) / b )
    num_phases = 4
    
    # create the parameters of the filters
    num_filters = num_phases * num_big_T
    which_phase = np.mod( np.arange(num_filters) , num_phases )
    which_T = np.floor( np.arange(num_filters) / num_phases )
    
    # create the spectral filters in a list
    spectral_filters = [ 
        ( lambda omega, cur_T=T[which_T[i]], cur_phase=(np.pi/2)*which_phase[i]: 0.5 + 0.5*np.exp(1j*omega*cur_T + 1j*cur_phase) )
        for i in range(num_filters)
        ]
    
    # return the results: the filters themselves are all we need
    # but return as a tuple for format consistency
    return (spectral_filters,)

    
def create_fitMIIPS_spectral_filters():
    # define the parameters of MIIPS
    num_filters = 128
    alpha = 0.05*np.pi # in radians
    gamma = 10 # in fs
    deltas = np.linspace(0, 2*np.pi, num_filters, endpoint=False)
    
    # create the spectral filters in a list
    spectral_filters = [ 
        ( lambda omega, phase=deltas[i]: np.exp(1j*alpha*np.sin(gamma*omega - phase)) )
        for i in range(num_filters)
        ]
    
    # return the results: the filters and some extra information for retrieval
    return (spectral_filters, alpha, gamma, deltas)
    
    

def create_ChirpScan_spectral_filters():
    # define the parameters of ChirpScan
    num_filters = 10
    alphas = np.linspace(-200, 200, num_filters)
    
    # create the spectral filters in a list
    spectral_filters = [ 
        ( lambda omega, cur_alpha=alphas[i]: np.exp(0.5j*cur_alpha*omega**2) )
        for i in range(num_filters)
        ]
    
    # return the results: the filters and some extra information for retrieval
    return (spectral_filters, alphas)
    

    
    


#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
# This is the section that creates data from those spectral filters

  
# compute the mapping function from nJ per inverse femtosecond to CCD counts
ccd_map_function, spec_lambda = cs.compute_ccd_mapping_function(2*cs.central_f)

# compute the spectral filter due to BBO thickness
rms_spot_radius = 50000 # in nm; four times this is the D4sigma spot size
BBO_spectral_filter = shg_spectral_filter.spectral_filter(
                        2*np.pi*(2*cs.central_f+cs.f), 
                        2*np.pi*cs.central_f,
                        rms_spot_radius) 
# plt.plot(2*cs.central_f+cs.f, np.abs(BBO_spectral_filter))
# plt.show()


def create_data(spectral_filters_tuple, num_shots):
    # since I'm doing multi-processing, I want to initialize the seed from /dev/urandom independently for each process
    prng = np.random.RandomState() 

    # decide how many laser shots to spend on each spectral filter
    spectral_filters = spectral_filters_tuple[0]
    num_filters = len(spectral_filters)
    assert(np.mod(num_shots, num_filters) == 0)
    num_shots_per_spectral_filter = num_shots / num_filters
    omega = 2*np.pi*cs.f

    # show the user the spectral filters
    # plt.plot(cs.f, np.array([spectral_filters[i](omega) for i in range(num_filters)]).T)
    # plt.show()
    
    # create the SHG data
    SHG_data = np.zeros( (cs.num_points, num_filters) )
    for i in range(num_filters):
        # apply the spectral filter to the field
        f_field = cs.spectral_field * spectral_filters[i](omega)
        # convert it to the time domain
        t_field = cs.spectral_to_temporal(f_field)
        # produce some second harmonic
        shg_t_field = t_field**2
        # convert the second-harmonic to the frequency domain; also add in the 
        # the spectral filter that is the BBO
        shg_f_field = BBO_spectral_filter * cs.temporal_to_spectral(shg_t_field)
        # here's the SHG spectral intensity
        SHG_data[:, i] = np.abs(shg_f_field)**2 # in nJ per inverse femtosecond

    # re-discretize onto the spectrometer, and convert to CCD counts from nJ per inverse femtosecond
    SHG_spec_data = np.dot( ccd_map_function, SHG_data )
    num_pixels = spec_lambda.size

    # then add shot noise and read noise
    SHG_detected_data = cs.measure(SHG_spec_data, prng, num_shots_per_spectral_filter)
    
    # # show the user
    # plt.plot(spec_lambda, SHG_detected_data)
    # plt.show()

    # take into account varying spectral width of pixels
    inter_lambda = 0.5*(spec_lambda[1:] + spec_lambda[:-1])
    delta_f_poly = np.polyfit( cs.c/inter_lambda, np.abs(np.diff(cs.c/spec_lambda)), 3 )
    SHG_detected_data *= cs.df / np.polyval( delta_f_poly, cs.c/spec_lambda )[:, np.newaxis]

    # estimate the noise
    noise_estimate = np.sqrt(np.abs(cs.variance_level(SHG_detected_data,num_shots_per_spectral_filter)))
    noise_estimate *= cs.df / np.polyval( delta_f_poly, cs.c/spec_lambda )[:, np.newaxis]

    # fit a spline to resample back into the frequency domain
    N = 256
    dt_resampled = 2.5 # in fs
    f_resampled = np.fft.fftfreq(N, dt_resampled)
    # f_resampled = np.linspace(np.amin(cs.c/spec_lambda)-2*cs.central_f, np.amax(cs.c/spec_lambda)-2*cs.central_f, N)
    SHG_resampled = np.zeros( (num_filters, N) )
    noise_resampled = np.zeros( (num_filters, N) )
    for i in range(num_filters):
        SHG_spline = scipy.interpolate.UnivariateSpline((cs.c/spec_lambda)[::-1], SHG_detected_data[::-1, i], w=(1.0/noise_estimate)[::-1, i])
        SHG_resampled[i, :] = SHG_spline(f_resampled+2*cs.central_f)
        noise_spline = scipy.interpolate.interp1d((cs.c/spec_lambda)[::-1], noise_estimate[::-1, i], bounds_error=False)
        noise_resampled[i, :] = noise_spline(f_resampled+2*cs.central_f)
    should_be_zero = np.logical_or(
        f_resampled+2*cs.central_f > np.amax(cs.c/spec_lambda), 
        f_resampled+2*cs.central_f < np.amin(cs.c/spec_lambda))
    SHG_resampled[:, should_be_zero] = 0.0
    noise_resampled[:, should_be_zero] = 10 # just something small but nonzero
    
    # # show the user
    # plt.plot(np.fft.fftshift(f_resampled), np.fft.fftshift(SHG_resampled, axes=1).T)
    # plt.show()
    assert(np.all(np.isfinite(SHG_resampled)))
    assert(np.all(np.isfinite(f_resampled)))
    assert(np.all(np.isfinite(noise_resampled)))

    # all done! return the traces
    return SHG_resampled, f_resampled, noise_resampled


    
    
    

    
    
    
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
# This is the section that defines various analysis functions

def analyze_SPIDER(SPIDER_data_tuple, SPIDER_filters_tuple):
    # extract the various inputs
    SHG_resampled = SPIDER_data_tuple[0]
    f_resampled = SPIDER_data_tuple[1]
    f_straight = np.fft.fftshift(f_resampled)
    tau = SPIDER_filters_tuple[1]
    delta_omega = SPIDER_filters_tuple[2]
    
    # create a time-axis as well
    df_resampled = f_resampled[1] - f_resampled[0]
    N = f_resampled.size
    t_resampled = np.fft.fftfreq(N, df_resampled)

    # combine the data with the phase-cycling
    phase_cycle_sign = np.array( [1, -1, -1, 1] )
    Delta_I = np.dot(SHG_resampled.T, phase_cycle_sign)
    # plt.plot(np.fft.fftshift(f_resampled), np.fft.fftshift(Delta_I)/np.amax(Delta_I))
    # plt.show()

    # do the spectral interferometry to get the phase
    Delta_I_fft = np.fft.fft(Delta_I)
    window = np.exp( -np.power(np.abs((t_resampled - tau)/(0.5*tau)), 6) )
    # plt.plot(np.fft.fftshift(t_resampled), np.fft.fftshift(np.abs(Delta_I_fft))/np.amax(np.abs(Delta_I_fft)))
    # plt.plot(np.fft.fftshift(t_resampled), np.fft.fftshift(window))
    # plt.show()
    # go back to the frequency domain
    frequency_domain = np.fft.ifft(Delta_I_fft * window)
    # compute the phase
    measured_phase = np.unwrap(np.angle(np.fft.fftshift(frequency_domain)))
    # remove the linear term
    measured_phase[:] -= 2*np.pi*f_straight*tau
    # subtract off the constant term
    measured_phase[:] -= measured_phase[np.argmin(np.abs(f_straight))]
    
    # now we have a group delay
    measured_gd = -measured_phase / delta_omega
    
    # show the user
    # plt.plot(f_straight, measured_gd)
    # plt.plot(cs.f, cs.spectral_gd)
    # plt.ylim(-15, 15)
    # plt.xlim(-2.5*cs.bandwidth_f, 2.5*cs.bandwidth_f)
    # plt.show()
    
    # return it
    return (f_straight, measured_gd)




def create_MIIPS_compensation(prev_compensation, data_tuple, filter_tuple):
    # extract the various inputs
    SHG_resampled = data_tuple[0]
    f_resampled = data_tuple[1]
    df = f_resampled[1] - f_resampled[0]
    omega = 2*np.pi*f_resampled
    N = f_resampled.size
    alpha = filter_tuple[1]
    gamma = filter_tuple[2]
    deltas = filter_tuple[3]
    n_deltas = deltas.size
    d_delta = deltas[1] - deltas[0]
    
    # find the two maxima for each frequency
    all_maxima = np.zeros( (N,2) )
    for i in range(N):
        for j in range(2):
            # compute where we expect this peak for a TL pulse
            expected_peak = gamma*(0.5*omega[i]) + j*np.pi # the 0.5*omega is because we detect second-harmonic rather than first-harmonic
            # convert it to the range we have
            expected_peak = expected_peak - 2*np.pi * np.floor(expected_peak / (2*np.pi))
            # construct a window in which to look for a maximum
            distance = deltas-expected_peak
            distance = distance - 2*np.pi * np.round(distance / (2*np.pi))
            window = np.abs(distance) <= np.pi/2
            # find that maximum
            cur_maximum = np.argmax(window*SHG_resampled[:, i])
            # fit a parabola to the neighboring points
            r = SHG_resampled[(cur_maximum+1) % n_deltas, i]
            c = SHG_resampled[cur_maximum, i]
            l = SHG_resampled[cur_maximum-1, i]
            fitted_peak = cur_maximum + 0.5*(l-r)/(l+r-2*c)
            # save it
            all_maxima[i, j] = d_delta*fitted_peak
    
    # assure finiteness
    all_maxima[np.logical_not(np.isfinite(all_maxima))] = 0.0
    
    # # show the stuff to the user
    # plt.imshow(np.flipud(np.fft.fftshift(SHG_resampled, axes=1).T), aspect='auto', interpolation='nearest',
        # extent = (np.amin(deltas) - 0.5*d_delta, np.amax(deltas) + 0.5*d_delta,
                    # np.amin(f_resampled) - 0.5*df, np.amax(f_resampled) + 0.5*df))
    # plt.plot(all_maxima, f_resampled, '.')
    # plt.xlim(0, 2*np.pi)
    # plt.ylim(-0.15, 0.2)
    # plt.show()
    
    # compute the second-derivative-of-phase estimate
    phi_pp_estimate = np.fft.fftshift( 0.5 * alpha * gamma**2 * (np.sin(gamma*(0.5*omega) - all_maxima[:,0]) + np.sin(gamma*(0.5*omega) - all_maxima[:,1])) )
    omega_straight = np.fft.fftshift(0.5*omega)
    
    # # show the user the measured GDD
    # plt.plot(omega_straight/(2*np.pi), phi_pp_estimate)
    # plt.plot(cs.f, cs.spectral_gdd)
    # # plt.ylim(-15, 15)
    # plt.xlim(-2.5*cs.bandwidth_f, 2.5*cs.bandwidth_f)
    # plt.show()
    
    # compute the group-delay estimate
    phi_p_estimate = scipy.integrate.cumtrapz(phi_pp_estimate, omega_straight, initial=0.0)
    phi_p_estimate[:] -= phi_p_estimate[np.argmin(np.abs(omega_straight))]
    
    # # show the user the measured group-delay
    # plt.plot(omega_straight/(2*np.pi), phi_p_estimate)
    # plt.plot(cs.f, cs.spectral_gd)
    # # plt.ylim(-15, 15)
    # plt.xlim(-2.5*cs.bandwidth_f, 2.5*cs.bandwidth_f)
    # plt.show()
    
    # compute the phase estimate
    phi_estimate = scipy.integrate.cumtrapz(phi_p_estimate, omega_straight, initial=0.0)
    phi_estimate[:] -= phi_estimate[np.argmin(np.abs(omega_straight))]
    
    # create a phase-estimate spline
    phi_spline = scipy.interpolate.interp1d(omega_straight, phi_estimate, bounds_error=False, fill_value=0.0)
    
    # return the compensation function
    return lambda input_omega: np.exp(-1j*phi_spline(input_omega)) * prev_compensation(input_omega)
    
    


def analyze_MIIPS(in_compensation, data_tuple):
    # extract the various inputs
    f_resampled = data_tuple[1]
    f_straight = np.fft.fftshift(f_resampled)
    omega = 2*np.pi*f_resampled
    omega_straight = np.fft.fftshift(omega)
    d_omega = omega[1] - omega[0]
    N = f_resampled.size
    
    # get the current measured phase from the compensation
    phi_estimate = np.unwrap( -np.angle(in_compensation(omega_straight)) )
    
    # convert to a group-delay
    gd_estimate = np.diff(phi_estimate) / d_omega
    f_gd = 0.5*( f_straight[:-1] + f_straight[1:] )
    
    # # show the user the measured group-delay
    # plt.plot(f_gd, gd_estimate)
    # plt.plot(cs.f, cs.spectral_gd)
    # # plt.ylim(-15, 15)
    # plt.xlim(-2.5*cs.bandwidth_f, 2.5*cs.bandwidth_f)
    # plt.show()

    # return it
    return (f_gd, gd_estimate)

    
    

def analyze_CRT(CRT_data_tuple, CRT_filters_tuple):
    # since I'm doing multi-processing, I want to initialize the seed from /dev/urandom independently for each process
    prng = np.random.RandomState() 

    # extract the various inputs
    SHG_resampled = CRT_data_tuple[0]
    f_resampled = CRT_data_tuple[1]
    f_straight = np.fft.fftshift(f_resampled)
    omega = 2*np.pi*f_resampled
    omega_straight = np.fft.fftshift(omega)
    alphas = CRT_filters_tuple[1]
    
    # compute the measured GDD values
    measured_gdd = -(alphas[0]*SHG_resampled[0, :] + alphas[1]*SHG_resampled[1, :])/np.sum(SHG_resampled, axis=0)
    # assure finiteness
    unknown_indices = np.logical_not(np.isfinite(measured_gdd))
    measured_gdd[unknown_indices] = 100*prng.randn( np.sum(unknown_indices) ) # standard deviation of 100 fs^2
    # arrange the frequencies in straight order, not FFT order
    measured_gdd = np.fft.fftshift(measured_gdd)
    
    # # show the user the measured GDD
    # plt.plot(0.5*f_straight, measured_gdd)
    # plt.plot(cs.f, cs.spectral_gdd)
    # # plt.ylim(-15, 15)
    # plt.xlim(-2.5*cs.bandwidth_f, 2.5*cs.bandwidth_f)
    # plt.show()
    
    # compute the group delay
    measured_gd = scipy.integrate.cumtrapz(measured_gdd, 0.5*omega_straight, initial=0.0)
    measured_gd[:] -= measured_gd[np.argmin(np.abs(0.5*omega_straight))]
    
    # # show the user the measured group-delay
    # plt.plot(0.5*f_straight, measured_gd)
    # plt.plot(cs.f, cs.spectral_gd)
    # # plt.ylim(-15, 15)
    # plt.xlim(-2.5*cs.bandwidth_f, 2.5*cs.bandwidth_f)
    # plt.show()

    # return it
    return (0.5*f_straight, measured_gd)


    

    
    

# this is the SPEAR transformation from measured data to measured phase-second-derivative
def __compute_SPEAR(alpha_list, y_list):
    # create the analytic equation for b
    s_list = 2.0*(alpha_list>0).astype(int) - 1.0
    term1 = lambda b: np.sum(s_list*y_list/(b+alpha_list))
    term2 = lambda b: np.sum(1.0/(b+alpha_list)**3)
    term3 = lambda b: np.sum(1.0/(b+alpha_list)**2)
    term4 = lambda b: np.sum(s_list*y_list/(b+alpha_list)**2)
    equation = lambda b: term1(b) * term2(b) - term3(b) * term4(b)
    c_function = lambda b: term1(b) / term3(b)
    
    # # actually, it can make sense to define a one-dimensional objective function
    # obj_function = lambda b: np.sum( (y_list-s_list*c_function(b)/(b+alpha_list))**2 )
    
    # create limits for b
    alpha_least_positive = np.min(alpha_list[alpha_list > 0])
    alpha_least_negative = np.max(alpha_list[alpha_list < 0])
    b_limit1 = 0.001*alpha_least_positive + 0.999*alpha_least_negative
    b_limit2 = 0.999*alpha_least_positive + 0.001*alpha_least_negative
    
    if( equation(b_limit1)*equation(b_limit2) < 0 ): # this checks if there is a zero
        b_fitted = scipy.optimize.brentq(equation, b_limit1, b_limit2)
    else:
        b_fitted = np.nan
    
    # compute c, if interested
    c_fitted = c_function(b_fitted)
    if(not np.isfinite(c_fitted)):
        c_fitted = 0.0 # b can be NaN, but c must be finite
    
    # return the answers
    return b_fitted, c_fitted



def analyze_SPEAR(SPEAR_data_tuple, SPEAR_filters_tuple):
    # since I'm doing multi-processing, I want to initialize the seed from /dev/urandom independently for each process
    prng = np.random.RandomState() 

    # extract the various inputs
    SHG_resampled = SPEAR_data_tuple[0]
    f_resampled = SPEAR_data_tuple[1]
    f_straight = np.fft.fftshift(f_resampled)
    N = f_resampled.size
    omega = 2*np.pi*f_resampled
    omega_straight = np.fft.fftshift(omega)
    alphas = SPEAR_filters_tuple[1]
    
    # compute the measured GDD values
    measured_gdd = np.array([ __compute_SPEAR(alphas, SHG_resampled[:, i])[0] for i in range(N) ])
    # assure finiteness
    unknown_indices = np.logical_not(np.isfinite(measured_gdd))
    measured_gdd[unknown_indices] = 100*prng.randn( np.sum(unknown_indices) ) # standard deviation of 100 fs^2
    # arrange the frequencies in straight order, not FFT order
    measured_gdd = np.fft.fftshift(measured_gdd)
    
    # # show the user the measured GDD
    # plt.plot(0.5*f_straight, measured_gdd)
    # plt.plot(cs.f, cs.spectral_gdd)
    # # plt.ylim(-15, 15)
    # plt.xlim(-2.5*cs.bandwidth_f, 2.5*cs.bandwidth_f)
    # plt.show()
    
    # compute the group delay
    measured_gd = scipy.integrate.cumtrapz(measured_gdd, 0.5*omega_straight, initial=0.0)
    measured_gd[:] -= measured_gd[np.argmin(np.abs(0.5*omega_straight))]
    
    # # show the user the measured group-delay
    # plt.plot(0.5*f_straight, measured_gd)
    # plt.plot(cs.f, cs.spectral_gd)
    # # plt.ylim(-15, 15)
    # plt.xlim(-2.5*cs.bandwidth_f, 2.5*cs.bandwidth_f)
    # plt.show()

    # return it
    return (0.5*f_straight, measured_gd)


    
    
    
    
def analyze_general(data_tuple, filters_tuple, in_M1, in_M2, smart_start):
    # extract the various inputs
    SHG_resampled = data_tuple[0]
    f_resampled = data_tuple[1]
    noise_resampled = data_tuple[2]
    filters = filters_tuple[0]
    
    # compute a good initial guess, if wanted
    if(smart_start):
        chosen_M1 = 5
        chosen_M2 = 5
        best_a1, a1_bf, best_b1, p1_bf = (
            general_pulseshaping_analysis.analyze(SHG_resampled, f_resampled, filters, chosen_M1, chosen_M2, noise_resampled, 500, None)
            )
        amp1_estimate = np.dot(best_a1, np.array([ a1_bf[i](2*np.pi*f_resampled) for i in range(chosen_M1) ]))
        phi1_estimate = np.dot(best_b1, np.array([ p1_bf[i](2*np.pi*f_resampled) for i in range(chosen_M2) ]))
        Ef1_estimate = amp1_estimate * np.exp(1j*phi1_estimate)
        
        # do it again
        chosen_M1 = 10
        chosen_M2 = 10
        best_a2, a2_bf, best_b2, p2_bf = (
            general_pulseshaping_analysis.analyze(SHG_resampled, f_resampled, filters, chosen_M1, chosen_M2, noise_resampled, 100, Ef1_estimate)
            )
        amp2_estimate = np.dot(best_a2, np.array([ a2_bf[i](2*np.pi*f_resampled) for i in range(chosen_M1) ]))
        phi2_estimate = np.dot(best_b2, np.array([ p2_bf[i](2*np.pi*f_resampled) for i in range(chosen_M2) ]))
        Ef_estimate = amp2_estimate * np.exp(1j*phi2_estimate)
    else:
        Ef_estimate = None
    
    # compute the results
    best_a, amplitude_basis_functions, best_b, phase_basis_functions = general_pulseshaping_analysis.analyze(SHG_resampled, f_resampled, filters, in_M1, in_M2, noise_resampled, 50, Ef_estimate)
    # M = len(basis_functions)
    
    # convert to frequency-domain, finely spaced
    f_final, df_final = np.linspace(np.amin(f_resampled), np.amax(f_resampled), 4096, retstep=True)
    phi_estimate = np.dot(best_b, np.array([ phase_basis_functions[i](2*np.pi*f_final) for i in range(in_M2) ]))
    # a = np.array(best_a[:M]) + 1j*np.array(best_a[M:])
    # field_f = np.dot(a, np.array([ basis_functions[i](2*np.pi*f_final) for i in range(M) ]))
    
    # # get the current measured phase from the compensation
    # phi_estimate = np.unwrap( np.angle(field_f) )
    # phi_estimate[:] -= phi_estimate[np.argmin(np.abs(f_final))]
    
    # convert to a group-delay
    d_omega = 2*np.pi*df_final
    gd_estimate = np.diff(phi_estimate) / d_omega
    gd_estimate[:] -= gd_estimate[np.argmin(np.abs(f_final))]
    f_gd = 0.5*( f_final[:-1] + f_final[1:] )
    
    # some of the spectral filters are time-ambiguous, requiring a sign flip
    if( cs.pulse_combination_number == 2 ):
        if(gd_estimate[np.argmin(np.abs(f_gd-0.025))] > 0):
            gd_estimate[:] *= -1
    else:
        if(gd_estimate[np.argmin(np.abs(f_gd-0.06))] < 0):
            gd_estimate[:] *= -1
    
    # # show the user the measured group-delay
    # plt.plot(f_gd, gd_estimate)
    # plt.plot(cs.f, cs.spectral_gd)
    # # plt.ylim(-15, 15)
    # plt.xlim(-2.5*cs.bandwidth_f, 2.5*cs.bandwidth_f)
    # plt.show()
    
    # return it
    return (f_gd, gd_estimate)


    
    
    
def analyze_general_Fourier(data_tuple, filters_tuple, num_basin_hops, smart_start):
    # extract the various inputs
    SHG_resampled = data_tuple[0]
    f_resampled = data_tuple[1]
    df_resampled = f_resampled[1] - f_resampled[0]
    noise_resampled = data_tuple[2]
    filters = filters_tuple[0]
    
    # compute a good initial guess, if wanted
    if(smart_start):
        #chosen_M1 = 5
        #chosen_M2 = 5
        #best_a, amplitude_basis_functions, best_b, phase_basis_functions = (
            #general_pulseshaping_analysis.analyze(SHG_resampled, f_resampled, filters, chosen_M1, chosen_M2, noise_resampled, 700, None)
            #)
        #amp_estimate = np.dot(best_a, np.array([ amplitude_basis_functions[i](2*np.pi*f_resampled) for i in range(chosen_M1) ]))
        #phi_estimate = np.dot(best_b, np.array([ phase_basis_functions[i](2*np.pi*f_resampled) for i in range(chosen_M2) ]))
        #Ef_estimate = amp_estimate * np.exp(1j*phi_estimate)
        Ef_estimate = general_pulseshaping_analysis_FourierII.analyze(SHG_resampled, f_resampled, filters, noise_resampled)
    else:
        Ef_estimate = None
    
    # compute the results
    best_E = general_pulseshaping_analysis_Fourier.analyze(SHG_resampled, f_resampled, filters, noise_resampled, num_basin_hops, Ef_estimate)
    #best_E = general_pulseshaping_analysis_FourierII.analyze(SHG_resampled, f_resampled, filters, noise_resampled, num_basin_hops, Ef_estimate)
    
    # center best_E in the time-domain; this is multiplying by a complex exponential in the frequency domain
    t_resampled = np.fft.fftfreq(f_resampled.size, df_resampled)
    def centered_ness(tau):
        E_t = np.fft.ifft(best_E * np.exp(2*np.pi*1j*f_resampled*tau))
        center_t = np.sum(t_resampled * np.abs(E_t)**2) / np.sum(np.abs(E_t)**2)
        return center_t**2
    best_tau_result = scipy.optimize.minimize_scalar(centered_ness)
    best_E[:] *= np.exp(2*np.pi*1j*f_resampled*best_tau_result.x)
    
    # smooth best_E in the frequency-domain; this is multiplying by a filter in the time domain
    fft_filter = np.exp( -np.power(np.abs(t_resampled / 70), 4) )
    best_E = np.fft.fft( fft_filter * np.fft.ifft(best_E) )
    
    # fftshift the results
    f_final = np.fft.fftshift(f_resampled)
    E_final = np.fft.fftshift(best_E)
    
    # get the current measured phase from the compensation
    phi_estimate = np.unwrap( np.angle(E_final) )
    phi_estimate[:] -= phi_estimate[np.argmin(np.abs(f_final))]
    
    # convert to a group-delay
    d_omega = 2*np.pi*df_resampled
    gd_estimate = np.diff(phi_estimate) / d_omega
    center_pixel = np.argmin(np.abs(f_final))
    gd_estimate[:] -= np.mean(gd_estimate[(center_pixel-3):(center_pixel+3)])
    f_gd = 0.5*( f_final[:-1] + f_final[1:] )
    
    # some of the spectral filters are time-ambiguous, requiring a sign flip
    if( cs.pulse_combination_number == 2 ):
        if(gd_estimate[np.argmin(np.abs(f_gd-0.025))] > 0):
            gd_estimate[:] *= -1
    else:
        if(gd_estimate[np.argmin(np.abs(f_gd-0.06))] < 0):
            gd_estimate[:] *= -1
    
    # # show the user the measured group-delay
    # plt.plot(f_gd, gd_estimate)
    # plt.plot(cs.f, cs.spectral_gd)
    # # plt.ylim(-15, 15)
    # plt.xlim(-2.5*cs.bandwidth_f, 2.5*cs.bandwidth_f)
    # plt.show()
    
    # return it
    return (f_gd, gd_estimate)


    

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
# This is the section for creating figures

def create_figure(f, group_delays, file_name):
    # find the percentiles
    percentiles_9010 = np.percentile(group_delays, (90, 10), axis=0)
    percentiles_7030 = np.percentile(group_delays, (70, 30), axis=0)
        
    # here's the colors for plotting
    color_9010 = (0.8, 0.9, 1.0)
    color_7030 = (0.6, 0.8, 1.0)
    color_true = 'r'

    # create the figure
    plt.figure(figsize=(3.25, 2.5))
    my_font_size = 7
    matplotlib.rc('xtick', labelsize=my_font_size)
    matplotlib.rc('ytick', labelsize=my_font_size)
    # some plotting ranges for the different pulse-combination numbers
    view_minimum = [ -40, -40, -10, -40] # in fs
    view_maximum = [ 40, 40, 10, 50] # in fs
    # view_minimum = -10 # in fs
    # view_maximum = 10 # in fs

    # shade a couple of areas
    plt.fill_between(f+cs.central_f, percentiles_9010[0], percentiles_9010[1], lw=0.0, edgecolor=color_9010, facecolor=color_9010)
    plt.fill_between(f+cs.central_f, percentiles_7030[0], percentiles_7030[1], lw=0.0, edgecolor=color_7030, facecolor=color_7030)
    
    # plot the true group-delay
    plt.plot(cs.f+cs.central_f, cs.spectral_gd, color_true)
    
    # a few more details
    # plt.ylim(view_minimum, view_maximum) 
    plt.ylim(view_minimum[cs.pulse_combination_number], view_maximum[cs.pulse_combination_number]) 
    plt.ylabel('Spectral group-delay (fs)', fontsize=my_font_size)
    plt.xlim(-1.1*cs.bandwidth_f+cs.central_f, 1.3*cs.bandwidth_f+cs.central_f)
    plt.xlabel('Frequency (PHz)', fontsize=my_font_size)

    # add the legend
    # create not-drawn proxies compatible with "legend"
    p1 = plt.Rectangle((0, 0), 1, 1, linewidth=0.0, edgecolor=color_9010, facecolor=color_9010)
    p2 = plt.Rectangle((0, 0), 1, 1, linewidth=0.0, edgecolor=color_7030, facecolor=color_7030)
    # another proxy (the one fill_between call corrupts everything)
    p3 = plt.Line2D((0,), (1,), color=color_true)
    plt.legend(
        [p3, p2, p1], 
        ['true spectral group-delay', '30$^{\mathrm{\mathsf{th}}}$ to 70$^{\mathrm{\mathsf{th}}}$ percentiles', '10$^{\mathrm{\mathsf{th}}}$ to 90$^{\mathrm{\mathsf{th}}}$ percentiles'], 
        fontsize=my_font_size,
        loc='lower right')

    # rearrange the plot to be nicely spaced
    plt.tight_layout()
    
    # save the plot
    plt.savefig(file_name, dpi=600)
    # plt.show()

    
    
def create_tiled_figure(list_f, list_group_delays, list_method_names, file_name):
        
    # here's the colors for plotting
    color_9010 = (0.8, 0.9, 1.0)
    color_7030 = (0.6, 0.8, 1.0)
    color_true = 'r'
    
    # some plotting ranges for the different pulse-combination numbers
    view_minimum = [ -40, -40, -15, -50] # in fs
    view_maximum = [ 30, 30, 10, 50] # in fs

    # create the figure
    plt.figure(figsize=(5.75, 5.5))
    my_font_size = 7
    matplotlib.rc('xtick', labelsize=my_font_size)
    matplotlib.rc('ytick', labelsize=my_font_size)

    for which_plot in range(6):
        # go to the current subplot
        cur_subplot = plt.subplot(3, 2, which_plot+1)
        
        # find the percentiles
        percentiles_9010 = np.percentile(list_group_delays[which_plot], (90, 10), axis=0)
        percentiles_7030 = np.percentile(list_group_delays[which_plot], (70, 30), axis=0)

        # shade a couple of areas
        plt.fill_between(list_f[which_plot]+cs.central_f, percentiles_9010[0], percentiles_9010[1], lw=0.0, edgecolor=color_9010, facecolor=color_9010)
        plt.fill_between(list_f[which_plot]+cs.central_f, percentiles_7030[0], percentiles_7030[1], lw=0.0, edgecolor=color_7030, facecolor=color_7030)
        
        # plot the true group-delay
        plt.plot(cs.f+cs.central_f, cs.spectral_gd, color_true)
        
        # a few more details
        plt.ylim(view_minimum[cs.pulse_combination_number], view_maximum[cs.pulse_combination_number]) 
        if(np.mod(which_plot, 2) == 0):
            plt.ylabel('Spectral group-delay (fs)', fontsize=my_font_size)
        plt.xlim(-1.05*cs.bandwidth_f+cs.central_f, 1.3*cs.bandwidth_f+cs.central_f)
        if(which_plot >= 4):
            plt.xlabel('Frequency (PHz)', fontsize=my_font_size)

        # add the legend
        # create not-drawn proxies compatible with "legend"
        p1 = plt.Rectangle((0, 0), 1, 1, linewidth=0.0, edgecolor=color_9010, facecolor=color_9010)
        p2 = plt.Rectangle((0, 0), 1, 1, linewidth=0.0, edgecolor=color_7030, facecolor=color_7030)
        # another proxy (the one fill_between call corrupts everything)
        p3 = plt.Line2D((0,), (1,), color=color_true)
        plt.legend(
            [p3, p2, p1], 
            ['true spectral group-delay', '30$^{\mathrm{\mathsf{th}}}$ to 70$^{\mathrm{\mathsf{th}}}$ percentiles', '10$^{\mathrm{\mathsf{th}}}$ to 90$^{\mathrm{\mathsf{th}}}$ percentiles'], 
            fontsize=my_font_size,
            loc='lower right')
        
        # add a label
        plt.text(x=0.03, y=0.94, s=list_method_names[which_plot], horizontalalignment='left', verticalalignment='top', fontsize=my_font_size, transform = cur_subplot.transAxes )

    # rearrange the plot to be nicely spaced
    plt.tight_layout()
    
    # save the plot
    plt.savefig(file_name, dpi=600)
    # plt.show()


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
# This is the section that runs the simulations

# number of times to run every method
num_iterations = 120




## SPIDER first
#SPIDER_filters = create_SPIDER_spectral_filters()
#SPIDER_shots_list = [20000, 40000, 400000, 400000]
#num_SPIDER_shots = SPIDER_shots_list[cs.pulse_combination_number]
#def single_SPIDER_iteration(iteration_number):
    #SPIDER_data = create_data(SPIDER_filters, num_SPIDER_shots)
    #SPIDER_results = analyze_SPIDER(SPIDER_data, SPIDER_filters)
    #print 'finished #' + str(iteration_number) + ' of the SPIDER simulations.'
    #return SPIDER_results
## do many SPIDER iterations
#SPIDER_pool = multiprocessing.Pool(processes=12)
#all_SPIDER_results = SPIDER_pool.map(single_SPIDER_iteration, range(num_iterations))
#SPIDER_f = all_SPIDER_results[0][0]
#SPIDER_gd = np.array([ all_SPIDER_results[i][1] for i in range(num_iterations) ])
##print 'Creating the SPIDER figure...'
### create the SPIDER figures
##create_figure(SPIDER_f, SPIDER_gd, 'SPIDER.pdf')


## now MIIPS
#MIIPS_filters = create_MIIPS_spectral_filters()
#MIIPS_shots_per_iteration_list = [5120, 5120, 40960, 40960]
#num_MIIPS_shots_per_iteration = MIIPS_shots_per_iteration_list[cs.pulse_combination_number]
#num_MIIPS_iterations = 4
#def single_MIIPS_solution(iteration_number):
    #MIIPS_compensation = lambda omega: np.ones_like(omega) # to start, the pulse-shaper doesn't do any compensation
    #for i in range(num_MIIPS_iterations):
        ## figure out what the current filters are
        #cur_filters = [
            #( lambda omega, cur_filter=MIIPS_filters[0][i]: cur_filter(omega) * MIIPS_compensation(omega) )
            #for i in range(len(MIIPS_filters[0]))]
        ## take some MIIPS data
        #cur_MIIPS_data = create_data( (cur_filters,) , num_MIIPS_shots_per_iteration)
        ## set the pulse-shaper to compensate
        #MIIPS_compensation = create_MIIPS_compensation(MIIPS_compensation, cur_MIIPS_data, MIIPS_filters)
    ## now look at the last version of the compensation to get the final results
    #MIIPS_results = analyze_MIIPS(MIIPS_compensation, cur_MIIPS_data)
    #print 'finished #' + str(iteration_number) + ' of the MIIPS simulations.'
    #return MIIPS_results
## do many MIIPS solutions
#MIIPS_pool = multiprocessing.Pool(processes=12)
#all_MIIPS_results = MIIPS_pool.map(single_MIIPS_solution, range(num_iterations))
#MIIPS_f = all_MIIPS_results[0][0]
#MIIPS_gd = np.array([ all_MIIPS_results[i][1] for i in range(num_iterations) ])
## print 'Creating the MIIPS figure...'
## # create the MIIPS figures
## create_figures(MIIPS_f, MIIPS_gd, 'MIIPS.pdf')


## CRT next
#CRT_filters = create_CRT_spectral_filters()
#num_CRT_shots = 500
#def single_CRT_iteration(iteration_number):
    #CRT_data = create_data(CRT_filters, num_CRT_shots)
    #CRT_results = analyze_CRT(CRT_data, CRT_filters)
    #print 'finished #' + str(iteration_number) + ' of the CRT simulations.'
    #return CRT_results
## do many CRT iterations
#CRT_pool = multiprocessing.Pool(processes=12)
#all_CRT_results = CRT_pool.map(single_CRT_iteration, range(num_iterations))
#CRT_f = all_CRT_results[0][0]
#CRT_gd = np.array([ all_CRT_results[i][1] for i in range(num_iterations) ])
## print 'Creating the CRT figure...'
## # create the CRT figure
## create_figures(CRT_f, CRT_gd, 'CRT.pdf')


## now SPEAR
#SPEAR_filters = create_SPEAR_spectral_filters()
#num_SPEAR_shots = 500
#def single_SPEAR_iteration(iteration_number):
    #SPEAR_data = create_data(SPEAR_filters, num_SPEAR_shots)
    #SPEAR_results = analyze_SPEAR(SPEAR_data, SPEAR_filters)
    #print 'finished #' + str(iteration_number) + ' of the SPEAR simulations.'
    #return SPEAR_results
## do many SPEAR iterations
#SPEAR_pool = multiprocessing.Pool(processes=12)
#all_SPEAR_results = SPEAR_pool.map(single_SPEAR_iteration, range(num_iterations))
#SPEAR_f = all_SPEAR_results[0][0]
#SPEAR_gd = np.array([ all_SPEAR_results[i][1] for i in range(num_iterations) ])
## print 'Creating the SPEAR figure...'
## # create the SPEAR figure
## create_figures(SPEAR_f, SPEAR_gd, 'SPEAR.pdf')


# and ChirpScan 
ChirpScan_filters = create_ChirpScan_spectral_filters()
num_ChirpScan_shots = 50
def single_ChirpScan_iteration(iteration_number):
    ChirpScan_data = create_data(ChirpScan_filters, num_ChirpScan_shots)
    # ChirpScan_results = analyze_general(ChirpScan_data, ChirpScan_filters, 30, 30)
    ChirpScan_results = analyze_general_Fourier(ChirpScan_data, ChirpScan_filters, num_basin_hops=0, smart_start=True)
    print 'finished #' + str(iteration_number) + ' of the ChirpScan simulations.'
    return ChirpScan_results
# do many ChirpScan iterations
ChirpScan_start_fitting = time.time()
processors = 12
ChirpScan_pool = multiprocessing.Pool(processes=12)
all_ChirpScan_results = ChirpScan_pool.map(single_ChirpScan_iteration, range(num_iterations))
# processors = 1
# all_ChirpScan_results = map(single_ChirpScan_iteration, range(num_iterations))
ChirpScan_end_fitting = time.time()
print 'total ChirpScan time per iteration: ' + str( (ChirpScan_end_fitting - ChirpScan_start_fitting)/num_iterations )
print 'total ChirpScan time * processors per iteration: ' + str( (ChirpScan_end_fitting - ChirpScan_start_fitting)*processors/num_iterations )
ChirpScan_f = all_ChirpScan_results[0][0]
ChirpScan_gd = np.array([ all_ChirpScan_results[i][1] for i in range(num_iterations) ])
# print 'Creating the ChirpScan figure...'
# create the ChirpScan figure
create_figure(ChirpScan_f, ChirpScan_gd, 'ChirpScan.pdf')


## FROG next
#FROG_filters = create_FROG_spectral_filters()
#num_FROG_shots = 3500
#def single_FROG_iteration(iteration_number):
    #FROG_data = create_data(FROG_filters, num_FROG_shots)
    ## FROG_results = analyze_general(FROG_data, FROG_filters, 15, 15, smart_start=True)
    #FROG_results = analyze_general_Fourier(FROG_data, FROG_filters, num_basin_hops=0, smart_start=True)
    #print 'finished #' + str(iteration_number) + ' of the FROG simulations.'
    #return FROG_results
## do many FROG iterations
#FROG_start_fitting = time.time()
#processors = 12
#FROG_pool = multiprocessing.Pool(processes=processors)
#all_FROG_results = FROG_pool.map(single_FROG_iteration, range(num_iterations))
##processors = 1
##all_FROG_results = map(single_FROG_iteration, range(num_iterations))
#FROG_end_fitting = time.time()
#print 'total FROG time per iteration: ' + str( (FROG_end_fitting - FROG_start_fitting)/num_iterations )
#print 'total FROG time * processors per iteration: ' + str( (FROG_end_fitting - FROG_start_fitting)*processors/num_iterations )
#FROG_f = all_FROG_results[0][0]
#FROG_gd = np.array([ all_FROG_results[i][1] for i in range(num_iterations) ])
## print 'Creating the FROG figure...'
## create the FROG figure
#create_figure(FROG_f, FROG_gd, 'FROG.pdf')


# # direct-fitted MIIPS 
# fitMIIPS_filters = create_fitMIIPS_spectral_filters()
# num_fitMIIPS_shots = 128000 #10000
# def single_fitMIIPS_iteration(iteration_number):
    # fitMIIPS_data = create_data(fitMIIPS_filters, num_fitMIIPS_shots)
    # # fitMIIPS_results = analyze_general(fitMIIPS_data, fitMIIPS_filters, 30, 30)
    # print 'finished #' + str(iteration_number) + ' of the fitMIIPS simulations.'
    # return fitMIIPS_results
# # do many fitMIIPS iterations
# fitMIIPS_pool = multiprocessing.Pool(processes=12)
# all_fitMIIPS_results = fitMIIPS_pool.map(single_fitMIIPS_iteration, range(num_iterations))
# fitMIIPS_f = all_fitMIIPS_results[0][0]
# fitMIIPS_gd = np.array([ all_fitMIIPS_results[i][1] for i in range(num_iterations) ])
# print 'Creating the fitMIIPS figure...'
# # create the fitMIIPS figure
# create_figures(fitMIIPS_f, fitMIIPS_gd, 'fitMIIPS.pdf')




# plot the results
print 'Creating the tiled figure...'
create_tiled_figure(
    [SPIDER_f, MIIPS_f, FROG_f, CRT_f, SPEAR_f, ChirpScan_f],
    [SPIDER_gd, MIIPS_gd, FROG_gd, CRT_gd, SPEAR_gd, ChirpScan_gd],
    ['SPIDER', 'MIIPS', 'FROG', 'CRT', 'SPEAR', 'ChirpScan'],
    'TiledFigure' + str(cs.pulse_combination_number) + '.pdf')












    
    
    
    
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
# This is the section that creates the test-pulse-information figures

    
print 'Creating the pulse-information figure'
# plot the temporal and spectral intensity and phase
fig = plt.figure(figsize=(5.75, 5.0))
my_font_size = 7
matplotlib.rc('xtick', labelsize=my_font_size)
matplotlib.rc('ytick', labelsize=my_font_size)

#################################################################
#################################################################
# add the spectral intensity and group-delay of the first pulse
ax1 = fig.add_subplot(4, 2, 1)
ax1.plot(cs.f+cs.central_f, cs.spectral_combos[0][1]/np.amax(cs.spectral_combos[0][1]), 'b:')
# ax1.set_xlabel('Frequency (PHz)', fontsize=my_font_size)
ax1.set_ylabel('Intensity (a.u.)', fontsize=my_font_size)
ax1.set_ylim(0.0, 1.15)
plt.yticks([0.0, 1.0])
ax2 = ax1.twinx()
ax2.plot(cs.f+cs.central_f, cs.spectral_combos[0][0], 'r-')
ax2.set_xlim(-1.5*cs.bandwidth_f+cs.central_f, 1.7*cs.bandwidth_f+cs.central_f)
ax2.set_ylim(-10, 45)
plt.yticks([0, 20, 40])
ax2.set_ylabel('Group-delay (fs)', fontsize=my_font_size)
plt.text(x=0.04, y=0.9, s='case 1', horizontalalignment='left', verticalalignment='top', fontsize=my_font_size, transform = ax1.transAxes )

# now the temporal intensity and phase of the first pulse
ax3 = fig.add_subplot(4, 2, 2)
temporal_field1 = cs.spectral_to_temporal(np.sqrt(cs.spectral_combos[0][1])*np.exp(1j*cs.spectral_combos[0][2]))
temporal_intensity1 = np.abs(temporal_field1)**2
ax3.plot(cs.t, temporal_intensity1/np.amax(temporal_intensity1), 'b:')
# ax3.set_xlabel('Time (fs)', fontsize=my_font_size)
ax3.set_ylabel('Intensity (a.u.)', fontsize=my_font_size)
ax3.set_ylim(0.0, 1.15)
plt.yticks([0.0, 1.0])
ax4 = ax3.twinx()
temporal_phase = np.unwrap(np.angle(temporal_field1))
temporal_phase[:] -= temporal_phase[np.argmin(np.abs(cs.t-0.0))]
temporal_phase[:] -= 0.125*cs.t
ax4.plot(cs.t, temporal_phase, 'r-')
ax4.set_xlim(-40, 20)
ax4.set_ylim(-12, 2)
plt.yticks([-10, 0])
# plt.yticks([-3*np.pi, -2*np.pi, -np.pi, 0], ('-3$\mathsf{\pi}$', '-2$\mathsf{\pi}$', '-$\mathsf{\pi}$', '0'))
ax4.set_ylabel('Phase (rad.)', fontsize=my_font_size)
plt.text(x=0.04, y=0.9, s='case 1', horizontalalignment='left', verticalalignment='top', fontsize=my_font_size, transform = ax3.transAxes )

#################################################################
#################################################################
# add the spectral intensity and group-delay of the second pulse
ax1 = fig.add_subplot(4, 2, 3)
ax1.plot(cs.f+cs.central_f, cs.spectral_combos[1][1]/np.amax(cs.spectral_combos[1][1]), 'b:')
# ax1.set_xlabel('Frequency (PHz)', fontsize=my_font_size)
ax1.set_ylabel('Intensity (a.u.)', fontsize=my_font_size)
ax1.set_ylim(0.0, 1.15)
plt.yticks([0.0, 1.0])
ax2 = ax1.twinx()
ax2.plot(cs.f+cs.central_f, cs.spectral_combos[1][0], 'r-')
ax2.set_xlim(-1.5*cs.bandwidth_f+cs.central_f, 1.7*cs.bandwidth_f+cs.central_f)
ax2.set_ylim(-10, 45)
plt.yticks([0, 20, 40])
ax2.set_ylabel('Group-delay (fs)', fontsize=my_font_size)
plt.text(x=0.04, y=0.9, s='case 2', horizontalalignment='left', verticalalignment='top', fontsize=my_font_size, transform = ax1.transAxes )

# now the temporal intensity and phase of the second pulse
ax3 = fig.add_subplot(4, 2, 4)
temporal_field2 = cs.spectral_to_temporal(np.sqrt(cs.spectral_combos[1][1])*np.exp(1j*cs.spectral_combos[1][2]))
temporal_intensity2 = np.abs(temporal_field2)**2
ax3.plot(cs.t, temporal_intensity2/np.amax(temporal_intensity2), 'b:')
# ax3.set_xlabel('Time (fs)', fontsize=my_font_size)
ax3.set_ylabel('Intensity (a.u.)', fontsize=my_font_size)
ax3.set_ylim(0.0, 1.15)
plt.yticks([0.0, 1.0])
ax4 = ax3.twinx()
temporal_phase = np.unwrap(np.angle(temporal_field2))
temporal_phase[:] -= temporal_phase[np.argmin(np.abs(cs.t-0.0))]
temporal_phase[:] -= 0.125*cs.t
ax4.plot(cs.t, temporal_phase, 'r-')
ax4.set_xlim(-40, 20)
ax4.set_ylim(-12, 2)
plt.yticks([-10, 0])
ax4.set_ylabel('Phase (rad.)', fontsize=my_font_size)
plt.text(x=0.04, y=0.9, s='case 2', horizontalalignment='left', verticalalignment='top', fontsize=my_font_size, transform = ax3.transAxes )

#################################################################
#################################################################
# add the spectral intensity and group-delay of the third pulse
ax1 = fig.add_subplot(4, 2, 5)
ax1.plot(cs.f+cs.central_f, cs.spectral_combos[2][1]/np.amax(cs.spectral_combos[2][1]), 'b:')
# ax1.set_xlabel('Frequency (PHz)', fontsize=my_font_size)
ax1.set_ylabel('Intensity (a.u.)', fontsize=my_font_size)
ax1.set_ylim(0.0, 1.15)
plt.yticks([0.0, 1.0])
ax2 = ax1.twinx()
ax2.plot(cs.f+cs.central_f, cs.spectral_combos[2][0], 'r-')
ax2.set_xlim(-1.5*cs.bandwidth_f+cs.central_f, 1.7*cs.bandwidth_f+cs.central_f)
ax2.set_ylim(-5, 12)
plt.yticks([0, 10])
ax2.set_ylabel('Group-delay (fs)', fontsize=my_font_size)
plt.text(x=0.04, y=0.9, s='case 3', horizontalalignment='left', verticalalignment='top', fontsize=my_font_size, transform = ax1.transAxes )

# now the temporal intensity and phase of the third pulse
ax3 = fig.add_subplot(4, 2, 6)
temporal_field3 = cs.spectral_to_temporal(np.sqrt(cs.spectral_combos[2][1])*np.exp(1j*cs.spectral_combos[2][2]))
temporal_intensity3 = np.abs(temporal_field3)**2
ax3.plot(cs.t, temporal_intensity3/np.amax(temporal_intensity3), 'b:')
# ax3.set_xlabel('Time (fs)', fontsize=my_font_size)
ax3.set_ylabel('Intensity (a.u.)', fontsize=my_font_size)
ax3.set_ylim(0.0, 1.15)
plt.yticks([0.0, 1.0])
ax4 = ax3.twinx()
temporal_phase = np.unwrap(np.angle(temporal_field3))
temporal_phase[:] -= temporal_phase[np.argmin(np.abs(cs.t-0.0))]
temporal_phase[:] -= 0.125*cs.t
ax4.plot(cs.t, temporal_phase, 'r-')
ax4.set_xlim(-40, 20)
ax4.set_ylim(-5, 1)
plt.yticks([-4, 0])
ax4.set_ylabel('Phase (rad.)', fontsize=my_font_size)
plt.text(x=0.04, y=0.9, s='case 3', horizontalalignment='left', verticalalignment='top', fontsize=my_font_size, transform = ax3.transAxes )

#################################################################
#################################################################
# add the spectral intensity and group-delay of the fourth pulse
ax1 = fig.add_subplot(4, 2, 7)
ax1.plot(cs.f+cs.central_f, cs.spectral_combos[3][1]/np.amax(cs.spectral_combos[3][1]), 'b:')
ax1.set_xlabel('Frequency (PHz)', fontsize=my_font_size)
ax1.set_ylabel('Intensity (a.u.)', fontsize=my_font_size)
ax1.set_ylim(0.0, 1.15)
plt.yticks([0.0, 1.0])
ax2 = ax1.twinx()
ax2.plot(cs.f+cs.central_f, cs.spectral_combos[3][0], 'r-')
ax2.set_xlim(-1.5*cs.bandwidth_f+cs.central_f, 1.7*cs.bandwidth_f+cs.central_f)
ax2.set_ylim(-10, 45)
plt.yticks([0, 20, 40])
ax2.set_ylabel('Group-delay (fs)', fontsize=my_font_size)
plt.text(x=0.04, y=0.9, s='case 4', horizontalalignment='left', verticalalignment='top', fontsize=my_font_size, transform = ax1.transAxes )

# now the temporal intensity and phase of the fourth pulse
ax3 = fig.add_subplot(4, 2, 8)
temporal_field4 = cs.spectral_to_temporal(np.sqrt(cs.spectral_combos[3][1])*np.exp(1j*cs.spectral_combos[3][2]))
temporal_intensity4 = np.abs(temporal_field4)**2
ax3.plot(cs.t, temporal_intensity4/np.amax(temporal_intensity4), 'b:')
ax3.set_xlabel('Time (fs)', fontsize=my_font_size)
ax3.set_ylabel('Intensity (a.u.)', fontsize=my_font_size)
ax3.set_ylim(0.0, 1.15)
plt.yticks([0.0, 1.0])
ax4 = ax3.twinx()
temporal_phase = np.unwrap(np.angle(temporal_field4))
temporal_phase[:] -= temporal_phase[np.argmin(np.abs(cs.t-0.0))]
temporal_phase[:] -= 0.125*cs.t
ax4.plot(cs.t, temporal_phase, 'r-')
ax4.set_xlim(-40, 20)
ax4.set_ylim(-5, 5)
plt.yticks([-4, 0, 4])
ax4.set_ylabel('Phase (rad.)', fontsize=my_font_size)
plt.text(x=0.04, y=0.9, s='case 4', horizontalalignment='left', verticalalignment='top', fontsize=my_font_size, transform = ax3.transAxes )

##################################################################
##################################################################
# save the plot
fig.tight_layout()
fig.savefig('PulseCharacteristics.pdf', dpi=600)

    
    
    
    
    
