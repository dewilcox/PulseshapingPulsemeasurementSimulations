# general_pulseshaping_analysis_Fourier.py
# Daniel E Wilcox
# This script takes measurements of shaped pulses and converts
# them to fitted fundamental field
# for many of the definitions refer to my LyX lab notebook 
# "Gradient of pulse-shaping (Fourier basis) with BBO" (December 2013)


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
def analyze(in_data, in_f, filters, in_noise_estimate, error_ratio, Ef_estimate):
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
    def objective(x, I_measured, S, L0, L1):
        assert(np.all(np.isfinite(x)))
        E = x[:P] + 1j*x[P:-2]
        E_t = np.fft.ifft(E)
        # preliminaries
        T = create_T(S, E)
        U = create_U(T)
        V = np.abs(U)**2
        h = np.exp( -0.5 * x[-2]**2 * (omega + x[-1])**2 )
        # the residual
        cur_residual = I_measured - h*V
        # plt.figure()
        # plt.plot(h)
        # plt.figure()
        # plt.imshow(I_measured, aspect='auto', interpolation='nearest')
        # plt.colorbar()
        # plt.figure()
        # plt.imshow(V, aspect='auto', interpolation='nearest')
        # plt.colorbar()
        # plt.show()
        cur_residual_reweighted = cur_residual / in_noise_estimate
        # the regularizers
        R0 = L0 * np.imag(np.sum(E))**2
        R1 = L1 * np.sum( t * np.abs(E_t)**2 )**2
        # some assertions
        assert(np.all(np.isfinite(cur_residual_reweighted)))
        assert(np.all(np.isreal(cur_residual_reweighted)))
        assert(np.isfinite(R0))
        assert(np.isreal(R0))
        assert(np.isfinite(R1))
        assert(np.isreal(R1))
        # the objective function itself
        WRSS = np.sum(cur_residual_reweighted**2)
        # print('current WRSS = ' + str(WRSS))
        # print('  current fraction = ' + str(WRSS/chi2cutoff))
        # print('current R0 = ' + str(R0))
        # print('current R1 = ' + str(R1))
        return WRSS + R0 + R1
    # define an objective-gradient function
    def objective_gradient(x, I_measured, S, L0, L1):
        assert(np.all(np.isfinite(x)))
        E = x[:P] + 1j*x[P:-2]
        E_t = np.fft.ifft(E)
        # preliminaries
        T = create_T(S, E)
        U = create_U(T)
        V = np.abs(U)**2
        h = np.exp( -0.5 * x[-2]**2 * (omega + x[-1])**2 )
        # the residual
        cur_residual = I_measured - h*V
        cur_residual_reweighted = cur_residual / in_noise_estimate
        # the regularizers
        R0 = L0 * np.imag(np.sum(E))**2
        R1 = L1 * np.sum( t * np.abs(E_t)**2 )**2
        # the central part of the gradient
        main_gradient = -4 * np.sum( S * np.fft.ifft( T * 
            np.fft.fft( cur_residual_reweighted * h * np.conj(U) / in_noise_estimate
            , axis=1 ), axis=1 ), axis=0 )
        # the gradient of h
        h_grad = np.zeros( (2, P) )
        h_grad[-1, :] = (-x[-2]**2 * (omega + x[-1])) * h
        h_grad[-2, :] = (-x[-2] * (omega + x[-1])**2) * h
        # the gradient of the regularizers
        R0_grad = -1j * E * L0 * np.imag(np.sum(E))
        R1_grad = 2 * L1 * np.sum( t * np.abs(E_t)**2 ) * np.fft.ifft( t * np.conj(E_t) )
        # create the objective gradient
        result = np.zeros( (x.size,) )
        result[:P] = 2*np.real(main_gradient + R0_grad + R1_grad)
        result[P:-2] = -2*np.imag(main_gradient + R0_grad + R1_grad)
        result[-2] = 2*np.sum(-cur_residual_reweighted * V * h_grad[-2, :] / in_noise_estimate)
        result[-1] = 2*np.sum(-cur_residual_reweighted * V * h_grad[-1, :] / in_noise_estimate)
        # # give some feedback to the user
        # if(prng.random_integers(0, 10) == 0):
            # print('current WRSS = ' + str(np.sum(cur_residual_reweighted**2)))
            # D_diff = np.zeros( (2*P+2,) )
            # for i in range(2*P+2):
                # h = 1e-4
                # if (i >= 2*P):
                    # h = 1e-6
                # cur_direction = np.zeros( (2*P+2,) )
                # cur_direction[i] = 1
                # delta = h*cur_direction
                # D_diff[i] = (final_objective(x + delta) - final_objective(x - delta))/(2*h)
            # assert(np.all(np.isfinite(D_diff)))
            # assert(np.all(np.isreal(D_diff)))
            # relative_error = lambda J1, J2: np.linalg.norm(J1-J2) / (0.5*(np.linalg.norm(J1)+np.linalg.norm(J2)))
            # print( 'relative gradient error = ' + str(relative_error(result, D_diff)) )
            # plt.figure()
            # plt.plot( result )
            # plt.plot( D_diff )
            # plt.figure()
            # plt.plot( (result - D_diff)/(0.5*(result+D_diff)) )
            # plt.show()
        return result

        
    # create a matrix of spectral filters
    S = np.array([filters[i](omega) for i in range(N)])
    assert(S.shape[0] == N)
    assert(S.shape[1] == P)
    assert(len(S.shape) == 2)
    
    # create the amplitudes of the regularizers
    L0 = 0.0 #0.01*np.sum(in_data**2)
    L1 = 0.0 #0.01*np.sum(in_data**2)
    
    # create the final objective function
    # final_objective = lambda a_real: objective_real(a_real, in_data, S, partial_T)
    final_objective = lambda x: objective(x, in_data, S, L0, L1)
    final_gradient = lambda x: objective_gradient(x, in_data, S, L0, L1)

    # create a guessing filter
    guess_filter = np.exp( -20 * df**2 * t**2 )
    
    # compute the chi2 cutoff; since we're doing direct fitting and since the input
    # is a smoothed spline we expect this cutoff to be rather less than the regular chi-2 cutoff
    chi2cutoff = error_ratio*N*P
    
    # do the minimization
    num_minima = 5
    best_errors = [0]*num_minima
    best_answers = [0]*num_minima
    for which_minimum in range(num_minima):
        fractional_error = 1e99 # just a big number
        while fractional_error > 1: 
            # create a new guess
            if(Ef_estimate is not None):
                E_guess_f = Ef_estimate + 0.05*prng.randn(P) + 0.05j*prng.randn(P)
            else:
                E_guess_t_raw = 1.0 + 0.3*prng.randn(P) + 0.3j*prng.randn(P)
                E_guess_f = np.fft.fft(E_guess_t_raw * guess_filter) * guess_filter
            x_guess = np.zeros( (2*P + 2) )
            x_guess[:P] = np.real(E_guess_f)
            x_guess[P:-2] = np.imag(E_guess_f)
            x_guess[-2] = 1 # a measure of inverse width in fs^2
            x_guess[-1] = 0 # a measure of center-shift in fs^-1
            # get the scaling right
            def scaling_objective(scale):
                x_guess_new = np.copy(x_guess)
                x_guess_new[:(2*P)] *= scale
                return final_objective(x_guess_new)
            best_scale_result = scipy.optimize.minimize_scalar(scaling_objective, bracket=(0.001, 1000))
            x_guess[:(2*P)] *= best_scale_result.x
            # print(' best scale is ' + str(best_scale_result.x))
            # do the minimization
            # result = scipy.optimize.minimize(final_objective, x_guess, method='BFGS', jac=final_gradient, options={'maxiter': P})
            result = scipy.optimize.minimize(final_objective, x_guess, method='L-BFGS-B', jac=final_gradient, options={'maxiter': 2*P, 'maxcor': 50})
            x_solved = result.x
            partial_fractional_error = final_objective(x_solved) / chi2cutoff
            if(partial_fractional_error < 10):
                result = scipy.optimize.minimize(final_objective, x_solved, method='L-BFGS-B', jac=final_gradient, options={'maxiter': 2*P, 'maxcor': 100})
                x_solved = result.x
                partial_fractional_error = final_objective(x_solved) / chi2cutoff
                if(partial_fractional_error < 1.1):
                    # result = scipy.optimize.minimize(final_objective, x_solved, method='BFGS', jac=final_gradient, options={'maxiter': 2*P})
                    result = scipy.optimize.minimize(final_objective, x_solved, method='L-BFGS-B', jac=final_gradient, options={'maxiter': 2*P, 'maxcor': 200})
                    # result = scipy.optimize.minimize(final_objective, x_solved, method='Newton-CG', jac=final_gradient)
                    x_solved = result.x
            fractional_error = final_objective(x_solved) / chi2cutoff
            print( '    current fractional_error=' + str(fractional_error) )
            print( '      which is old-fractional = ' + str(final_objective(x_solved) / np.sum( (in_data/in_noise_estimate)**2 )) )
        # done the current minimization; do another one
        best_errors[which_minimum] = fractional_error
        best_answers[which_minimum] = x_solved
        
  
    # # check the output
    # plt.figure()
    # plt.imshow(np.fft.fftshift(np.sqrt(np.abs(in_data)), axes=1), aspect='auto', interpolation='nearest')
    # plt.colorbar()
    # plt.figure()
    # a = np.array(result[0][:M]) + 1j*np.array(result[0][M:])
    # simulation = simulatedSHG(a, in_data, S)
    # plt.imshow(np.fft.fftshift(np.sqrt(np.abs(simulation)), axes=1), aspect='auto', interpolation='nearest')
    # plt.colorbar()
    # plt.show()
    
    # look at the results
    best_errors = np.array(best_errors)
    print(best_errors)
    print('best error = ' + str(np.amin(best_errors)))
    x_best = best_answers[np.argmin(best_errors)]
    
    # do a final polishing step
    # result = scipy.optimize.minimize(final_objective, x_best, method='Newton-CG', jac=final_gradient)
    result = scipy.optimize.minimize(final_objective, x_best, method='L-BFGS-B', jac=final_gradient, options={'maxcor': 0.5*P})
    x_final = result.x
    print('final error = ' + str(final_objective(x_final) / chi2cutoff) + ' from ' + str(np.amin(best_errors)))

    # all done!
    return x_final[:P] + 1j*x_final[P:-2]


