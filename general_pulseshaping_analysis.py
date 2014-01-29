# -*- coding: utf-8 -*-
# general_pulseshaping_analysis.py
# Daniel E Wilcox
# This script takes measurements of shaped pulses and converts
# them to fitted fundamental field
# for many of the definitions refer to my LyX lab notebook 
# "Gradient of pulse-shaping (separated amplitude and phase) with BBO" (December 2013)


from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import commonsimulation as cs
import scipy.interpolate
import scipy.linalg
import scipy.optimize
from multiprocessing import Pool



# define the first several cardinal (non-centered) B-splines; for their form see
# http://en.wikipedia.org/wiki/Irwin%E2%80%93Hall_distribution#Special_cases
def cardinal_bspline_1(x):
    window_before = x < 0
    window_center = np.logical_and(x >= 0, x < 1)
    window_after = x >= 1
    result = np.zeros_like( x )
    result[window_before] = 0
    result[window_center] = 1
    result[window_after] = 0
    return result
def cardinal_bspline_2(x):
    window_before = x < 0
    window_center1 = np.logical_and(x >= 0, x < 1)
    window_center2 = np.logical_and(x >= 1, x < 2)
    window_after = x >= 2
    result = np.zeros_like( x )
    result[window_before] = 0
    result[window_center1] = x[window_center1]
    result[window_center2] = 2-x[window_center2]
    result[window_after] = 0
    return result
def cardinal_bspline_3(x):
    window_before = x < 0
    window_center1 = np.logical_and(x >= 0, x < 1)
    window_center2 = np.logical_and(x >= 1, x < 2)
    window_center3 = np.logical_and(x >= 2, x < 3)
    window_after = x >= 3
    result = np.zeros_like( x )
    result[window_before] = 0
    result[window_center1] = 0.5*x[window_center1]**2
    result[window_center2] = -x[window_center2]**2 + 3*x[window_center2] - 1.5
    result[window_center3] = 0.5*x[window_center3]**2 - 3*x[window_center3] + 4.5
    result[window_after] = 0
    return result
def cardinal_bspline_4(x):
    window_before = x < 0
    window_center1 = np.logical_and(x >= 0, x < 1)
    window_center2 = np.logical_and(x >= 1, x < 2)
    window_center3 = np.logical_and(x >= 2, x < 3)
    window_center4 = np.logical_and(x >= 3, x < 4)
    window_after = x >= 4
    result = np.zeros_like( x )
    result[window_before] = 0
    result[window_center1] = (1/6.)*x[window_center1]**3
    result[window_center2] = (1/6.)*( -3*x[window_center2]**3 + 12*x[window_center2]**2 - 12*x[window_center2] + 4 )
    result[window_center3] = (1/6.)*( 3*x[window_center3]**3 - 24*x[window_center3]**2 + 60*x[window_center3] - 44 )
    result[window_center4] = (1/6.)*( -x[window_center4]**3 + 12*x[window_center4]**2 - 48*x[window_center4] + 64 )
    result[window_after] = 0
    return result


###############################################
# define an elementary basis function
def basic_basis_function(x):
    # return np.exp( -x**2 / 2 )
    # return cardinal_bspline_3(x + 1.5)
    return cardinal_bspline_4(x + 2)


    
    
    


################################################
# Here is the main analysis function
# it takes measured data 'in_data' of size N by P
# it takes an FFT-compatible frequency axis of size P
# it takes a list of spectral filters (which are themselves functions) of size N
def analyze(in_data, in_f, filters, in_M1, in_M2, in_noise_estimate, in_num_evals, Ef_estimate):
    # since I'm doing multi-processing, I want to initialize the seed from /dev/urandom independently for each process
    prng = np.random.RandomState() 

    # parse the input a bit
    P = in_f.size
    N = len(filters)
    omega = 2*np.pi*in_f
    
    # define a range over which to fit the field
    left_range = -0.09
    right_range = 0.12
    total_range = right_range - left_range
    center_range = 0.5*(right_range + left_range)
    
    # define a set of amplitude basis functions
    amplitude_basis_centers_f, amplitude_basis_df = np.linspace( left_range, right_range, in_M1, retstep=True )
    amplitude_basis_functions = [
        (lambda omega, center=2*np.pi*amplitude_basis_centers_f[i], width=2*np.pi*amplitude_basis_df:
            basic_basis_function( (omega-center)/width ))
        for i in range(in_M1)]
    
    # define a set of phase basis functions
    # I use the Legendre polynomials, excluding the 0 and 1 orders because they don't matter
    phase_basis_functions = [
        (lambda omega, n=2+i, center=2*np.pi*center_range, scale=np.pi*total_range:
            (omega > 2*np.pi*left_range) * (omega < 2*np.pi*right_range) * scipy.special.eval_legendre(n, (omega-center)/scale))
        for i in range(in_M2)]

    # evaluate the basis functions at the omegas under current consideration
    Be1 = np.array([ amplitude_basis_functions[i](omega) for i in range(in_M1) ])
    Be2 = np.array([ phase_basis_functions[i](omega) for i in range(in_M2) ])
    # plt.figure()
    # plt.plot(Be1.T)
    # plt.figure()
    # plt.plot(Be2.T)
    # plt.show()
    
    # create the E functions referred to in the lab notebook
    create_E = lambda a, b: np.dot(a, Be1) * np.exp(1j * np.dot(b, Be2))
    # create_E = lambda a, b: np.dot(a**2, Be1) * np.exp(1j * np.dot(b, Be2))
    
    # create the T functions referred to in the lab notebook
    create_T = lambda S, E: np.fft.ifft(S * E, axis=1)
    
    # create the U functions
    create_U = lambda T: np.fft.fft(T**2, axis=1)
    
    # create the partial-E functions
    create_partial_E_a = lambda b: Be1 * np.exp(1j * np.dot(b, Be2))
    create_partial_E_b = lambda E: 1j*Be2 * E
    # this is alpha-by-omega in size
    create_partial_E = lambda E, b: np.vstack( (create_partial_E_a(b), create_partial_E_b(E)) )
    
    # create the partial-T function
    # this is alpha-by-filter-by-time in size
    create_partial_T = lambda S, partial_E: np.fft.ifft(S * partial_E[:, np.newaxis, :], axis=2)
    
    # create the partial-U function
    # this is alpha-by-filter-by-omega in size
    create_partial_U = lambda T, partial_T: np.fft.fft( 2*T * partial_T, axis=2 )
    
    # create the Hessian-E functions
    # this is a-by-b-by-omega
    create_hessian_E_ab = lambda b: 1j * Be1[:, np.newaxis, :] * Be2[np.newaxis, :, :] * np.exp(1j * np.dot(b, Be2))
    # this is b-by-b-by-omega
    create_hessian_E_bb = lambda E: -Be2[:, np.newaxis, :] * Be2[np.newaxis, :, :] * E[np.newaxis, np.newaxis, :]
    # this is beta-by-alpha-by-time
    def create_hessian_E(E, b):
        h_E_ab = create_hessian_E_ab(b)
        h_E_bb = create_hessian_E_bb(E)
        hessian_E = np.zeros( (in_M1+in_M2, in_M1+in_M2, P), dtype=np.complex128 )
        hessian_E[:in_M1, :in_M1, :] = 0.0
        hessian_E[:in_M1, in_M1:, :] = h_E_ab
        hessian_E[in_M1:, :in_M1, :] = np.transpose(h_E_ab, axes=(1, 0, 2))
        hessian_E[in_M1:, in_M1:, :] = h_E_bb
        return hessian_E
    
    # create the Hessian-T functions
    # this is beta-by-alpha-by-filter-by-time
    create_hessian_T = lambda hessian_E: np.fft.ifft(S * hessian_E[:, :, np.newaxis, :], axis=3)
    
    # create the Hessian-U function
    # this is beta-by-alpha-by-filter-by-omega
    create_hessian_U = lambda T, partial_T, hessian_T: 2 * np.fft.fft( partial_T[:, np.newaxis, :, :] * partial_T[np.newaxis, :, :, :] + T * hessian_T, axis=3 )
    
    # create the Hessian-V function
    create_hessian_V = lambda U, partial_U, hessian_U: 2 * np.real(hessian_U * np.conj(U) + partial_U[np.newaxis, :, :, :] * np.conj(partial_U[:, np.newaxis, :, :]))
    
    
    # create the Hessian-T-p functions
    # this is beta-by-filter-by-time
    create_hessian_T_p = lambda hessian_E, p: np.fft.ifft(S * 
            np.tensordot(hessian_E, p, axes=([1], [0]))[:, np.newaxis, :]
        , axis=2)
    
    # create the Hessian-U-p function
    # this is beta-by-filter-by-omega
    create_hessian_U_p = lambda T, partial_T, hessian_T_p, p: 2 * np.fft.fft( 
        partial_T * np.tensordot(partial_T, p, axes=([0], [0]))
            + T * hessian_T_p
        , axis=2 )
    
    # create the Hessian-V-p function
    create_hessian_V_p = lambda U, partial_U, hessian_U_p, p: 2 * np.real(
        np.conj(U) * hessian_U_p
        + np.conj(partial_U) * np.tensordot(partial_U, p, axes=([0], [0]))
        )
    
    # define a residual function
    def residual(x, I_measured, S):
        assert(np.all(np.isfinite(x)))
        a = x[:in_M1]
        b = x[in_M1:-2]
        # preliminaries
        E = create_E(a, b)
        T = create_T(S, E)
        U = create_U(T)
        V = np.abs(U)**2
        h = np.exp( -0.5 * x[-2]**2 * (omega + x[-1])**2 )
        # the residual
        cur_residual = I_measured - h*V
        cur_residual_reweighted = cur_residual / in_noise_estimate
        assert(np.all(np.isfinite(cur_residual_reweighted)))
        assert(np.all(np.isreal(cur_residual_reweighted)))
        # print('current WRSS = ' + str(np.sum(cur_residual_reweighted**2)))
        return cur_residual_reweighted.flatten()
    # define a residual-gradient function
    def residual_Jacobian(x, I_measured, S):
        assert(np.all(np.isfinite(x)))
        a = x[:in_M1]
        b = x[in_M1:-2]
        # preliminaries
        E = create_E(a, b)
        T = create_T(S, E)
        U = create_U(T)
        V = np.abs(U)**2
        h = np.exp( -0.5 * x[-2]**2 * (omega + x[-1])**2 )
        # start to work on the gradients
        partial_E = create_partial_E(E, b)
        partial_T = create_partial_T(S, partial_E)
        partial_U = create_partial_U(T, partial_T)
        # I expand partial_V to include the h_vars as well
        partial_V = np.zeros( (x.size, partial_U.shape[1], partial_U.shape[2]) )
        # convert from Wirtinger derivative to regular derivative
        partial_V[:-2, :, :] = 2 * np.real(partial_U * np.conj(U))
        partial_h = np.zeros( (x.size, partial_U.shape[1], partial_U.shape[2]) )
        partial_h[-1, :, :] = (-x[-2]**2 * (omega + x[-1])) * h
        partial_h[-2, :, :] = (-x[-2] * (omega + x[-1])**2) * h
        # the partial_R
        partial_R = -partial_h*V - h*partial_V
        partial_R_reweighted = partial_R / in_noise_estimate
        assert(np.all(np.isfinite(partial_R_reweighted)))
        assert(np.all(np.isreal(partial_R_reweighted)))
        # reshape the Jacobian
        Jacobian_reshaped = np.zeros( (V.size, in_M1+in_M2+2) )
        for i in range(in_M1+in_M2+2):
            Jacobian_reshaped[:, i] = (partial_R_reweighted[i, :, :]).flatten()
        # # give some feedback to the user
        # if(prng.random_integers(0, 10) == 0):
            # print('current RSS = ' + str(np.sum((I_measured - h*V)**2)))
            # J_diff = np.zeros( (V.size, in_M1+in_M2+2) )
            # for i in range(in_M1+in_M2+2):
                # h = 1e-5
                # cur_direction = np.zeros( (in_M1+in_M2+2,) )
                # cur_direction[i] = 1
                # delta = h*cur_direction
                # J_diff[:, i] = (final_residual(x + delta) - final_residual(x - delta))/(2*h)
            # # plt.imshow(J_diff - Jacobian_reshaped, aspect='auto', interpolation='nearest')
            # # plt.colorbar()
            # assert(np.all(np.isfinite(J_diff)))
            # assert(np.all(np.isreal(J_diff)))
            # relative_error = lambda J1, J2: np.linalg.norm(J1-J2) / (0.5*(np.linalg.norm(J1)+np.linalg.norm(J2)))
            # print( 'relative Jacobian error = ' + str(relative_error(Jacobian_reshaped, J_diff)) )
            # plt.figure()
            # plt.plot( np.array([relative_error(Jacobian_reshaped[:, i], J_diff[:, i]) for i in range(in_M1+in_M2+3)]) )
            # # plt.figure()
            # # plt.plot( J_diff[:, -1] )
            # # plt.plot( Jacobian_reshaped[:, -1] )
            # # plt.figure()
            # # plt.plot( J_diff[:, -2] )
            # # plt.plot( Jacobian_reshaped[:, -2] )
            # plt.show()
        return Jacobian_reshaped
    # define a residual-Hessian function
    def residual_Hessian(x, I_measured, S):
        assert(np.all(np.isfinite(x)))
        a = x[:in_M1]
        b = x[in_M1:-2]
        # preliminaries
        E = create_E(a, b)
        T = create_T(S, E)
        U = create_U(T)
        V = np.abs(U)**2
        h = np.exp( -0.5 * x[-2]**2 * (omega + x[-1])**2 )
        # start to work on the gradients
        partial_E = create_partial_E(E, b)
        partial_T = create_partial_T(S, partial_E)
        partial_U = create_partial_U(T, partial_T)
        partial_V = np.zeros( (x.size, partial_U.shape[1], partial_U.shape[2]) )
        partial_V[:-2, :, :] = 2 * np.real(partial_U * np.conj(U))
        partial_h = np.zeros( (x.size, partial_U.shape[1], partial_U.shape[2]) )
        partial_h[-1, :, :] = (-x[-2]**2 * (omega + x[-1])) * h
        partial_h[-2, :, :] = (-x[-2] * (omega + x[-1])**2) * h
        # start to work on the Hessians
        hessian_E = create_hessian_E(E, b)
        hessian_T = create_hessian_T(hessian_E)
        hessian_U = create_hessian_U(T, partial_T, hessian_T)
        hessian_V = np.zeros( (in_M1+in_M2+2, in_M1+in_M2+2, partial_U.shape[1], partial_U.shape[2]) )
        hessian_V[:-2, :-2, :, :] = create_hessian_V(U, partial_U, hessian_U)
        hessian_h = np.zeros( (in_M1+in_M2+2, in_M1+in_M2+2, partial_U.shape[1], partial_U.shape[2]) )
        hessian_h[-1, -1, :, :] = -x[-2]**2 * h   +   x[-2]**4 * (omega + x[-1])**2 * h
        hessian_h[-1, -2, :, :] = -2 * x[-2] * (omega + x[-1]) * h   +   (-x[-2]**2 * (omega + x[-1])) * (-x[-2] * (omega + x[-1])**2) * h
        hessian_h[-2, -1, :, :] = hessian_h[-1, -2, :, :]
        hessian_h[-2, -2, :, :] = -(omega + x[-1])**2 * h   +   x[-2]**2 * (omega + x[-1])**4 * h
        # the hessian_R
        hessian_R = -hessian_h*V - partial_h[np.newaxis, :, :, :]*partial_V[:, np.newaxis, :, :] - partial_h[:, np.newaxis, :, :]*partial_V[np.newaxis, :, :, :] - h*hessian_V
        hessian_R_reweighted = hessian_R / in_noise_estimate
        assert(np.all(np.isfinite(hessian_R_reweighted)))
        assert(np.all(np.isreal(hessian_R_reweighted)))
        # reshape the Hessian
        Hessian_reshaped = np.zeros( (V.size, in_M1+in_M2+2, in_M1+in_M2+2) )
        for i in range(in_M1+in_M2+2):
            for j in range(in_M1+in_M2+2):
                Hessian_reshaped[:, i, j] = (hessian_R_reweighted[i, j, :, :]).flatten()
        # # give some feedback to the user
        # if(prng.random_integers(0, 2) == 0):
            # # print('current WRSS = ' + str(np.sum((I_measured - h*V)**2)))
            # H_diff = np.zeros( (V.size, in_M1+in_M2+2, in_M1+in_M2+2) )
            # for i in range(in_M1+in_M2+2):
                # for j in range(in_M1+in_M2+2):
                    # h = 5e-4
                    # cur_direction1 = np.zeros( (in_M1+in_M2+2,) )
                    # cur_direction1[i] = 1
                    # delta1 = h*cur_direction1
                    # cur_direction2 = np.zeros( (in_M1+in_M2+2,) )
                    # cur_direction2[j] = 1
                    # delta2 = h*cur_direction2
                    # if(i == j):
                        # H_diff[:, i, i] = (final_residual(x + delta1) 
                                            # - 2*final_residual(x) 
                                            # + final_residual(x - delta1)
                                            # )/(h**2)
                    # else:
                        # H_diff[:, i, j] = (final_residual(x + delta1 + delta2) 
                                            # - final_residual(x + delta1 - delta2) 
                                            # - final_residual(x - delta1 + delta2) 
                                            # + final_residual(x - delta1 - delta2)
                                            # )/(4*h**2)
            # assert(np.all(np.isfinite(H_diff)))
            # assert(np.all(np.isreal(H_diff)))
            # relative_error = lambda H1, H2: np.linalg.norm(H1-H2) / (0.5*(np.linalg.norm(H1)+np.linalg.norm(H2)))
            # print( 'relative Hessian error = ' + str(relative_error(Hessian_reshaped, H_diff)) )
            # relative_Hessian_error = np.zeros( (in_M1+in_M2+2, in_M1+in_M2+2) )
            # Hessian_norm = np.zeros( (in_M1+in_M2+2, in_M1+in_M2+2) )
            # Hdiff_norm = np.zeros( (in_M1+in_M2+2, in_M1+in_M2+2) )
            # for i in range(in_M1+in_M2+2):
                # for j in range(in_M1+in_M2+2):
                    # relative_Hessian_error[i, j] = relative_error(Hessian_reshaped[:, i, j], H_diff[:, i, j])
                    # Hessian_norm[i, j] = np.linalg.norm(Hessian_reshaped[:, i, j])
                    # Hdiff_norm[i, j] = np.linalg.norm(H_diff[:, i, j])
            # plt.figure()
            # plt.imshow(relative_Hessian_error, aspect='auto', interpolation='nearest')
            # plt.colorbar()
            # plt.figure()
            # # plt.imshow(Hessian_norm, aspect='auto', interpolation='nearest')
            # # plt.colorbar()
            # # plt.figure()
            # # plt.imshow(Hdiff_norm, aspect='auto', interpolation='nearest')
            # # plt.colorbar()
            # # plt.figure()
            # # plt.plot(Hessian_reshaped[:, 0, 0])
            # # plt.plot(H_diff[:, 0, 0])
            # # plt.figure()
            # # plt.plot(Hessian_reshaped[:, 1, 1])
            # # plt.plot(H_diff[:, 1, 1])
            # # plt.figure()
            # # plt.plot(Hessian_reshaped[:, 0, 1])
            # # plt.plot(H_diff[:, 0, 1])
            # plt.show()
        return Hessian_reshaped
    # define a residual-Hessian-product function
    def residual_Hessian_P(x, I_measured, S, p):
        assert(np.all(np.isfinite(x)))
        a = x[:in_M1]
        b = x[in_M1:-2]
        # preliminaries
        E = create_E(a, b)
        T = create_T(S, E)
        U = create_U(T)
        V = np.abs(U)**2
        h = np.exp( -0.5 * x[-2]**2 * (omega + x[-1])**2 )
        # start to work on the gradients
        partial_E = create_partial_E(E, b)
        partial_T = create_partial_T(S, partial_E)
        partial_U = create_partial_U(T, partial_T)
        partial_V = np.zeros( (x.size, partial_U.shape[1], partial_U.shape[2]) )
        partial_V[:-2, :, :] = 2 * np.real(partial_U * np.conj(U))
        partial_h = np.zeros( (x.size, partial_U.shape[1], partial_U.shape[2]) )
        partial_h[-1, :, :] = (-x[-2]**2 * (omega + x[-1])) * h
        partial_h[-2, :, :] = (-x[-2] * (omega + x[-1])**2) * h
        # start to work on the Hessian-products
        hessian_E = create_hessian_E(E, b)
        hessian_T_p = create_hessian_T_p(hessian_E, p[:-2])
        hessian_U_p = create_hessian_U_p(T, partial_T, hessian_T_p, p[:-2])
        hessian_V_p = np.zeros( (in_M1+in_M2+2, partial_U.shape[1], partial_U.shape[2]) )
        hessian_V_p[:-2, :, :] = create_hessian_V_p(U, partial_U, hessian_U_p, p[:-2])
        hessian_h = np.zeros( (2, 2, partial_U.shape[1], partial_U.shape[2]) )
        hessian_h[-1, -1, :, :] = -x[-2]**2 * h   +   x[-2]**4 * (omega + x[-1])**2 * h
        hessian_h[-1, -2, :, :] = -2 * x[-2] * (omega + x[-1]) * h   +   (-x[-2]**2 * (omega + x[-1])) * (-x[-2] * (omega + x[-1])**2) * h
        hessian_h[-2, -1, :, :] = hessian_h[-1, -2, :, :]
        hessian_h[-2, -2, :, :] = -(omega + x[-1])**2 * h   +   x[-2]**2 * (omega + x[-1])**4 * h
        hessian_h_p = np.zeros( (in_M1+in_M2+2, partial_U.shape[1], partial_U.shape[2]) )
        hessian_h_p[-2:, :, :] = np.tensordot(hessian_h, p[-2:], axes=([1], [0]))
        # the hessian_R_p
        hessian_R_p = (
            -V * hessian_h_p 
            -partial_h * np.tensordot(partial_V, p, axes=([0], [0]))
            -partial_V * np.tensordot(partial_h, p, axes=([0], [0]))
            -h * hessian_V_p )
        hessian_R_p_reweighted = hessian_R_p / in_noise_estimate
        assert(np.all(np.isfinite(hessian_R_p_reweighted)))
        assert(np.all(np.isreal(hessian_R_p_reweighted)))
        # reshape the Hessian
        Hessian_p_reshaped = np.zeros( (V.size, in_M1+in_M2+2) )
        for i in range(in_M1+in_M2+2):
            Hessian_p_reshaped[:, i] = (hessian_R_p_reweighted[i, :, :]).flatten()
        # # give some feedback to the user
        # if(prng.random_integers(0, 2) == 0):
            # # print('current WRSS = ' + str(np.sum((I_measured - h*V)**2)))
            # H_diff = np.zeros( (V.size, in_M1+in_M2+2, in_M1+in_M2+2) )
            # for i in range(in_M1+in_M2+2):
                # for j in range(in_M1+in_M2+2):
                    # h = 5e-4
                    # cur_direction1 = np.zeros( (in_M1+in_M2+2,) )
                    # cur_direction1[i] = 1
                    # delta1 = h*cur_direction1
                    # cur_direction2 = np.zeros( (in_M1+in_M2+2,) )
                    # cur_direction2[j] = 1
                    # delta2 = h*cur_direction2
                    # if(i == j):
                        # H_diff[:, i, i] = (final_residual(x + delta1) 
                                            # - 2*final_residual(x) 
                                            # + final_residual(x - delta1)
                                            # )/(h**2)
                    # else:
                        # H_diff[:, i, j] = (final_residual(x + delta1 + delta2) 
                                            # - final_residual(x + delta1 - delta2) 
                                            # - final_residual(x - delta1 + delta2) 
                                            # + final_residual(x - delta1 - delta2)
                                            # )/(4*h**2)
            # assert(np.all(np.isfinite(H_diff)))
            # assert(np.all(np.isreal(H_diff)))
            # H_diff_p = np.dot(H_diff, p)
            # relative_error = lambda H1, H2: np.linalg.norm(H1-H2) / (0.5*(np.linalg.norm(H1)+np.linalg.norm(H2)))
            # print( 'relative Hessian-p error = ' + str(relative_error(Hessian_p_reshaped, H_diff_p)) )
            # relative_Hessian_p_error = np.zeros( (in_M1+in_M2+2) )
            # Hessian_p_norm = np.zeros( (in_M1+in_M2+2) )
            # Hdiff_p_norm = np.zeros( (in_M1+in_M2+2) )
            # for i in range(in_M1+in_M2+2):
                # relative_Hessian_p_error[i] = relative_error(Hessian_p_reshaped[:, i], H_diff_p[:, i])
                # Hessian_p_norm[i] = np.linalg.norm(Hessian_p_reshaped[:, i])
                # Hdiff_p_norm[i] = np.linalg.norm(H_diff_p[:, i])
            # plt.figure()
            # plt.plot(relative_Hessian_p_error)
            # plt.show()
        return Hessian_p_reshaped
    
    
    # create a matrix of spectral filters
    S = np.array([filters[i](omega) for i in range(N)])
    assert(S.shape[0] == N)
    assert(S.shape[1] == P)
    assert(len(S.shape) == 2)
    
    # # compute the partial_T
    # partial_T = create_partial_T(S)
    
    # # compute a weight function
    # weight_func = np.sum(in_data, axis=0) / np.sum(in_data)
    
    # create the final objective function
    # final_objective = lambda a_real: objective_real(a_real, in_data, S, partial_T)
    final_residual = lambda x: residual(x, in_data, S)
    final_Jacobian = lambda x: residual_Jacobian(x, in_data, S)
    final_residualHessian = lambda x: residual_Hessian(x, in_data, S)
    final_residualHessian_P = lambda x, p: residual_Hessian_P(x, in_data, S, p)
    def objective(x):
        R = final_residual(x)
        return np.sum(R**2)
    def gradient(x):
        R = final_residual(x)
        J = final_Jacobian(x)
        return 2*np.dot(J.T, R)
    def hessian(x):
        R = final_residual(x)
        J = final_Jacobian(x)
        H = final_residualHessian(x)
        return 2*np.dot(J.T, J) + 2*np.tensordot(R, H, axes=([0], [0]))
    def hessian_p(x, p):
        R = final_residual(x)
        J = final_Jacobian(x)
        Hp = final_residualHessian_P(x, p)
        return 2*np.dot(J.T, np.dot(J, p)) + 2*np.tensordot(R, Hp, axes=([0], [0]))
    def hessian_approx(x):
        J = final_Jacobian(x)
        return 2*np.dot(J.T, J)
    
    # create an initial guess based on Ef_estimate
    if(Ef_estimate is not None):
        amplitude_estimate = np.abs(Ef_estimate)
        a_estimate = np.linalg.lstsq(Be1.T, amplitude_estimate)[0]
        phase_estimate = np.fft.ifftshift(np.unwrap(np.angle(np.fft.fftshift(Ef_estimate))))
        b_estimate = np.linalg.lstsq(Be2.T, phase_estimate)[0]

    # do the minimization
    if(in_num_evals == 0):
        fractional_error = 1e99 # just a big number
        while fractional_error > 1e-4: # this depends on 'n'
            # create a new guess
            if(Ef_estimate is None):
                a_guess = 5*(1 + 0.2*prng.randn(in_M1))
                b_guess = 200*prng.randn(in_M2)
                b_guess[:] *= np.array([1/(i+1)**3 for i in range(in_M2)])
                b_guess_shift = 5*prng.randn(in_M2)
                b_guess_shift[:] *= np.array([1/(i+1)**3 for i in range(in_M2)])
                b_guess[:] += b_guess_shift
            else:
                a_guess = a_estimate * np.exp(0.01*prng.randn(in_M1))
                b_guess = b_estimate * np.exp(0.01*prng.randn(in_M2))
            x_guess = np.zeros( (in_M1+in_M2+2,) )
            x_guess[:in_M1] = a_guess
            x_guess[in_M1:-2] = b_guess
            x_guess[-2] = 1 # a measure of inverse width in fs^2
            x_guess[-1] = 0 # a measure of center-shift in fs^-1

            result = scipy.optimize.leastsq(final_residual, x_guess, Dfun=final_Jacobian, full_output=True, maxfev=(in_M1+in_M2), ftol=1e-5)
            x_solved = result[0]
            # nfev1 = result[2]['nfev']
            partial_fractional_error = objective(x_solved) / np.sum( (in_data/in_noise_estimate)**2 )
            if(partial_fractional_error < 2e-3):
                # # print('Allowing further optimization; nfev so far is ' + str(nfev1))
                result = scipy.optimize.leastsq(final_residual, x_solved, Dfun=final_Jacobian, full_output=True, maxfev=3*(in_M1+in_M2), ftol=1e-5)
                x_solved = result[0]
                partial_fractional_error = objective(x_solved) / np.sum( (in_data/in_noise_estimate)**2 )
                if (partial_fractional_error < 2e-4):
                    result = scipy.optimize.leastsq(final_residual, x_solved, Dfun=final_Jacobian, full_output=True, maxfev=10*(in_M1+in_M2), ftol=1e-9)
                    x_solved = result[0]
            fractional_error = objective(x_solved) / np.sum( (in_data/in_noise_estimate)**2 )
            # fractional_error = objective(result[0]) / np.sum( in_data**2 )
            print( '   current fractional_error=' + str(fractional_error) )
            # print( 'nfev = ' + str(nfev1 + nfev2) )
    else:
        solved_xs = np.zeros( (in_M1+in_M2+2, in_num_evals) )
        objectives = np.zeros( (in_num_evals,) )
        for which_opt in range(in_num_evals):
            # create a new guess
            # create a new guess
            if(Ef_estimate is None):
                a_guess = 5*(1 + 0.2*prng.randn(in_M1))
                b_guess = 200*prng.randn(in_M2)
                b_guess[:] *= np.array([1/(i+1)**3 for i in range(in_M2)])
            else:
                a_guess = a_estimate * np.exp(0.01*prng.randn(in_M1))
                b_guess = b_estimate * np.exp(0.01*prng.randn(in_M2)) 
                b_guess_shift = prng.randn(in_M2)
                b_guess_shift[:] *= np.array([1/(i+1)**4 for i in range(in_M2)])
                b_guess[:] += b_guess_shift
            x_guess = np.zeros( (in_M1+in_M2+2,) )
            x_guess[:in_M1] = a_guess
            x_guess[in_M1:-2] = b_guess
            x_guess[-2] = 1 # a measure of inverse width in fs^2
            x_guess[-1] = 0 # a measure of center-shift in fs^-1

            result = scipy.optimize.leastsq(final_residual, x_guess, Dfun=final_Jacobian, full_output=True, maxfev=2*(in_M1+in_M2), ftol=1e-5)
            solved_xs[:, which_opt] = result[0]
            objectives[which_opt] = objective(solved_xs[:, which_opt]) / np.sum( (in_data/in_noise_estimate)**2 )
            #print( '   current fractional_error=' + str(objectives[which_opt]) )
        # which optimization was best?
        fractional_error = np.amin(objectives)
        x_solved = solved_xs[:, np.argmin(objectives)]
        # polish it a bit
        result_polish = scipy.optimize.leastsq(final_residual, x_solved, Dfun=final_Jacobian, full_output=True, maxfev=10*(in_M1+in_M2), ftol=1e-9)
        #result_polish = scipy.optimize.minimize(objective, x_solved, jac=gradient, method='L-BFGS-B', options={'maxcor': in_M1+in_M2+2})
        x_solved = result_polish[0] #.x
        fractional_error = objective(x_solved) / np.sum( (in_data/in_noise_estimate)**2 )
  
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
    
    print( '   final fractional_error=' + str(fractional_error) )
    print( '   which is absolute=' + str(objective(x_solved)) )

    # all done!
    return x_solved[:in_M1], amplitude_basis_functions, x_solved[in_M1:-2], phase_basis_functions # just the final numbers (in real form) is adequate


