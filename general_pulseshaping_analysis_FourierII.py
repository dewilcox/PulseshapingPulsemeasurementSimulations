# -*- coding: utf-8 -*-
# general_pulseshaping_analysis_FourierII.py
# Daniel E Wilcox
# This script takes measurements of shaped pulses and converts
# them to fitted fundamental field
# for many of the definitions refer to my LyX lab notebook 
# "Gradient of pulse-shaping (Fourier basis) with filter" (February 2014)


from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import commonsimulation as cs
import scipy.interpolate
import scipy.linalg
import scipy.optimize
import scipy.stats
from multiprocessing import Pool


    
    
    


################################################
# Here is the main analysis function
# it takes measured data 'in_data' of size N by P
# it takes an FFT-compatible frequency axis of size P
# it takes a list of spectral filters (which are themselves functions) of size N
def analyze(in_data, in_f, filters, in_noise_estimate):
    # since I'm doing multi-processing, I want to initialize the seed from /dev/urandom independently for each process
    prng = np.random.RandomState() 

    # parse the input a bit
    P = in_f.size
    N = len(filters)
    df = in_f[1] - in_f[0]
    omega = 2*np.pi*in_f
    t = np.fft.fftfreq(P, df)
    
    # create the T functions referred to in the lab notebook
    create_T = lambda S, E: np.fft.ifft(S * E[np.newaxis, :], axis=1)
    
    # create the U functions
    create_U = lambda T: np.fft.fft(T**2, axis=1)
    
    # define an objective function
    def objective(x, I_measured, S):
        assert(np.all(np.isfinite(x)))
        E = x[:P] + 1j*x[P:]
        # preliminaries
        T = create_T(S, E)
        U = create_U(T)
        V = np.abs(U)**2
        inv_noise = 1.0 / in_noise_estimate
        #inv_noise_squared = inv_noise**2
        #N = np.sum(I_measured * V * inv_noise_squared)
        #inv_D = 1.0 / (1e9 + np.sum(V**2 * inv_noise_squared, axis=0))
        #h = N * inv_D
        h = 1.0
        # the residual
        pre_residual = I_measured - h*V
        cur_residual = pre_residual * inv_noise
        # some assertions
        assert(np.all(np.isfinite(cur_residual)))
        assert(np.all(np.isreal(cur_residual)))
        # the objective function itself
        WRSS = np.sum(cur_residual**2)
        #print('current WRSS = ' + str(WRSS))
        return WRSS
    # define an objective-gradient function
    def objective_gradient(x, I_measured, S):
        assert(np.all(np.isfinite(x)))
        E = x[:P] + 1j*x[P:]
        # preliminaries
        T = create_T(S, E)
        U = create_U(T)
        V = np.abs(U)**2
        inv_noise = 1.0 / in_noise_estimate
        inv_noise_squared = inv_noise**2
        #N = np.sum(I_measured * V * inv_noise_squared, axis=0)
        #inv_D = 1.0 / (1e9 + np.sum(V**2 * inv_noise_squared, axis=0))
        #h = N * inv_D
        h = 1.0
        # the residual
        pre_residual = I_measured - h*V
        cur_residual = pre_residual * inv_noise
        # some assertions
        assert(np.all(np.isfinite(cur_residual)))
        assert(np.all(np.isreal(cur_residual)))
        # the objective function itself
        WRSS = np.sum(cur_residual**2)
        
        # now do the gradient
        #L_over_D = np.sum(cur_residual * V * inv_noise, axis=0) * inv_D
        #inside = pre_residual * h   +   I_measured * L_over_D   -   V * (2*h*L_over_D) 
        inside = pre_residual * h
        Wirtinger_grad = -4*np.sum( 
            S * np.fft.ifft(
                T * np.fft.fft(
                    np.conj(U) * inv_noise_squared * inside, 
                    axis=1), 
                axis=1), 
            axis=0)
        # convert to regular gradient
        result = np.zeros( (x.size,) )
        result[:P] = 2*np.real(Wirtinger_grad)
        result[P:] = -2*np.imag(Wirtinger_grad)
        ## give some feedback to the user
        #if(prng.random_integers(0, 10) == 0):
            #print('current h-less WRSS = ' + str( np.sum(((I_measured - V)*inv_noise)**2) ))
            #print('current WRSS = ' + str(WRSS))
            #D_diff = np.zeros( (2*P,) )
            #for i in range(2*P):
                #h = 1e-7
                #cur_direction = np.zeros( (2*P,) )
                #cur_direction[i] = 1
                #delta = h*cur_direction
                #D_diff[i] = (final_objective(x + delta) - final_objective(x - delta))/(2*h)
            #assert(np.all(np.isfinite(D_diff)))
            #assert(np.all(np.isreal(D_diff)))
            #relative_error = lambda J1, J2: np.linalg.norm(J1-J2) / (0.5*(np.linalg.norm(J1)+np.linalg.norm(J2)))
            #print( 'relative gradient error = ' + str(relative_error(result, D_diff)) )
            #plt.figure()
            #plt.plot( result )
            #plt.plot( D_diff )
            #plt.figure()
            #plt.plot( (result - D_diff)/(0.5*(result+D_diff)) )
            #plt.show()
        return result

        
    # create a matrix of spectral filters
    S = np.array([filters[i](omega) for i in range(N)])
    assert(S.shape[0] == N)
    assert(S.shape[1] == P)
    assert(len(S.shape) == 2)
    
    # create the final objective function
    # final_objective = lambda a_real: objective_real(a_real, in_data, S, partial_T)
    final_objective = lambda x: objective(x, in_data, S)
    final_gradient = lambda x: objective_gradient(x, in_data, S)

    # create a guessing filter
    guess_filter = np.exp( -20 * df**2 * t**2 )
    
    # # compute the chi2 cutoff; since we're doing direct fitting and since the input
    # # is a smoothed spline we expect this cutoff to be rather less than the regular chi-2 cutoff
    # chi2cutoff = error_ratio*N*P
    
    # create a starting guess; sample a few places
    n_samples = 2
    x_samples = np.zeros( (2*P, n_samples) )
    y_samples = np.zeros( (n_samples,) )
    for i in range(n_samples):
        #print('trying guess #' + str(i))
        E_guess_t_raw = 1.0 + 0.5*prng.randn(P) + 0.5j*prng.randn(P)
        E_guess_f = np.fft.fft(E_guess_t_raw * guess_filter) * guess_filter
        x_guess = np.zeros( (2*P) )
        x_guess[:P] = np.real(E_guess_f)
        x_guess[P:] = np.imag(E_guess_f)
        # how good is this one?
        result = scipy.optimize.minimize(final_objective, x_guess, method='L-BFGS-B', jac=final_gradient, options={'maxiter': 3*P, 'maxcor': 50})
        x_samples[:, i] = result.x
        y_samples[i] = final_objective(result.x)
    print('found best guess = ' + str(np.amin(y_samples)))
    x_guess = x_samples[:, np.argmin(y_samples)]
    
    
    # do a final polishing step
    result = scipy.optimize.minimize(final_objective, x_guess, method='L-BFGS-B', jac=final_gradient, options={'maxcor': 0.5*P})
    x_final = result.x
    print('final error = ' + str(final_objective(x_final)))

    # all done!
    return x_final[:P] + 1j*x_final[P:]


