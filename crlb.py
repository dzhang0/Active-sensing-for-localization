from cgi import test
from re import X
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random as random
from mpl_toolkits import mplot3d
from numpy import exp, abs, angle
from util_func import random_beamforming
import math
from wsr_bcd.generate_channel import generate_location_old,path_loss_r,generate_channel_fullRician, channel_complex2real
import time
import datetime as dt
import os.path
from os import path



location_irs=np.array([-40, 40, 0])
noiseSTD_per_dim = np.sqrt(0.5) #STD of the Gaussian noise per real dim.
Rician_factor = 13




def calculate_crlb(crlb_component):
    (j_11, j_12, j_21, j_22) = crlb_component
    numerat = (j_11 + j_22) 
    denom =  (j_11*j_22 - j_12*j_21)
    crlb = numerat / denom
    return numerat, denom, crlb

def calcProcessTime(starttime, cur_iter, max_iter):

    telapsed = time.time() - starttime
    testimated = (telapsed/cur_iter)*(max_iter)

    finishtime = starttime + testimated
    finishtime = dt.datetime.fromtimestamp(finishtime).strftime("%H:%M:%S")  # in time

    lefttime = testimated-telapsed  # in seconds

    return (int(telapsed), int(lefttime), finishtime)

def get_ris_element_position(params_system, d, irs_Nh):
    '''based on antenna separation (d), calculate the position of individual ris element
        starting from top left corner
    '''
    (Nt, N, Nr) =params_system

    offset = np.array([0, -np.floor(irs_Nh/2)*d, np.floor(N/irs_Nh) * d - d])
    position = []
    # start at top left corner
    for nn in range(N):
        col_num = np.mod(nn,irs_Nh)
        row_num = np.floor(nn/irs_Nh)
        position_nn = location_irs + offset + np.array([0, d*col_num , - row_num*d ])
        position.append(position_nn)
    return position


def get_Aod_AoA(params_system, ris_position, location_user, location_bs):
    (Nt, N, Nr) =params_system

    # ========bs-irs==============
    d0 = np.linalg.norm(location_bs - location_irs)
    pathloss_irs_bs = path_loss_r(d0)
    aod_azi_set = []
    aod_ele_set = []
    for nn in range(N):
        position_nn = ris_position[nn]
        aod_bs_ris_y = (location_bs[1] - position_nn[1] ) / d0
        aod_bs_ris_z = (location_bs[2] - position_nn[2] )/ d0

        ele_angle = math.asin( aod_bs_ris_z )
        azi_angle = math.asin(  aod_bs_ris_y/math.cos(ele_angle) )
    # aoa_bs = ( location_irs[0] - location_bs[0]) / d0
    # aod_irs_y = (location_bs[1]-location_irs[1]) / d0
    # aod_irs_z = (location_bs[2]-location_irs[2]) / d0
        aod_azi_set.append(azi_angle)
        aod_ele_set.append(ele_angle)
    
    aod_azi_set = np.array(aod_azi_set)
    aod_ele_set = np.array(aod_ele_set)

    # =========irs-user=============
    pathloss_irs_user = []
    aoa_azi_set = []
    aoa_ele_set = []

    d_k = np.linalg.norm(location_user - location_irs)
    pathloss_irs_user.append(path_loss_r(d_k))

    for nn in range(N):
        position_nn = ris_position[nn]
        aoa_irs_y_k = (location_user[1] - position_nn[1]) / d_k
        aoa_irs_z_k = (location_user[2] - position_nn[2]) / d_k
        
        ele_angle = math.asin( aoa_irs_z_k )
        azi_angle = math.asin(  aoa_irs_y_k/math.cos(ele_angle) )

        aoa_azi_set.append(azi_angle)
        aoa_ele_set.append(ele_angle)

    aoa_azi_set = np.array(aoa_azi_set)
    aoa_ele_set = np.array(aoa_ele_set)

    pathloss = (pathloss_irs_bs , np.array(pathloss_irs_user))
    aoa_aod = ( aod_azi_set, aod_ele_set, aoa_azi_set, aoa_ele_set)
    return pathloss, aoa_aod

def get_Aod_AoA_2(params_system, ris_position, location_user_set, location_bs):
    (Nt, N, Nr) =params_system
    aoa_aod_set = []
    for test_i in range(location_user_set.shape[0]):
        # =========bs-irs=============
        aod_azi_set = []
        aod_ele_set = []

        for nn in range(N):
            position_nn = ris_position[nn]

            tmp = np.linalg.norm(location_bs[0:2] - position_nn[0:2])
            ele_angle = np.arctan(tmp/position_nn[2])
            azi_angle = np.arccos(-(np.abs(location_bs[0]-position_nn[0]))/tmp)

            aod_azi_set.append(azi_angle)
            aod_ele_set.append(ele_angle)

        aod_azi_set = np.array(aod_azi_set)
        aod_ele_set = np.array(aod_ele_set)
        # =========irs-user=============
        aoa_azi_set = [] 
        aoa_ele_set = []

        for nn in range(N):
            position_nn = ris_position[nn]

            tmp = np.linalg.norm(location_user_set[test_i][0:2] - position_nn[0:2])
            ele_angle = np.arctan(tmp/position_nn[2])
            azi_angle = np.arccos(-(np.abs(location_user_set[test_i][0]-position_nn[0]))/tmp)

            aoa_azi_set.append(azi_angle)
            aoa_ele_set.append(ele_angle)

        aoa_azi_set = np.array(aoa_azi_set)
        aoa_ele_set = np.array(aoa_ele_set)

        aoa_aod = (aod_azi_set, aod_ele_set, aoa_azi_set, aoa_ele_set)
        aoa_aod_set.append(aoa_aod)
    return aoa_aod_set

def generate_channel(params_system, aoa_set):
    (Nt, N, Nr) =params_system
    H_set = []
    A_t_set = []
    A_r_set = []
    for test_i in range(len(aoa_set)):
        aoa = aoa_set[test_i]

        A_t = np.zeros([Nt, N], dtype=complex)  # array of 1,  1 x N
        A_r = np.zeros([Nr, N], dtype=complex) 

        (aod_azi_angle,aod_ele_angle,aoa_azi_angle, aoa_ele_angle) = aoa

        i1 = np.arange(Nt)  # 0,1,2,...
        for nn in range(N):
            a_t_nn = np.exp(1j * np.pi * i1 * np.sin(aod_azi_angle[nn])* np.sin(aod_ele_angle[nn]))
            A_t[:, nn] = a_t_nn

        i1 = np.arange(Nr)  # 0,1,2,...
        for nn in range(N):
            a_r_nn = np.exp(1j * np.pi * i1 * np.sin(aoa_azi_angle[nn])* np.sin(aoa_ele_angle[nn]))
            A_r[:, nn] = a_r_nn

        # generate h
        h = np.ones(N, dtype=complex)                                                                    
        H = np.diag(h)
        H_set.append(H)
        A_t_set.append(A_t)
        A_r_set.append(A_r)

    return H_set, A_t_set, A_r_set

def combine_channel(H,a_t,a_r,ris_diag):
    tmp = np.matmul(H,ris_diag)
    tmp = np.matmul(a_r,tmp)
    return np.matmul(tmp, np.asmatrix(a_t).getH())

def get_received_pilot(params_system,H_tilde, L, P_temp):
    (Nt, N, Nr) =params_system
    y  = np.zeros( [L * Nr,1], dtype=complex) 
    x = np.ones( [Nt,1], dtype=complex) 
    
    for ll in range(L):
        noise = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[Nr, 1]) \
              + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[Nr, 1])
        y[ll*Nr:(ll+1)*Nr, :] = np.complex(np.sqrt(P_temp),0.0) * np.matmul(H_tilde,x) + noise

    return y

def complex_to_polar(ris_config):
    return angle(ris_config)

def polar_to_complex(ris_phase):
    return 1 * exp( 1j * ris_phase)

def get_htilde_derivative(params_system, h_tilde, aod_aoa, ris_cmplx, P_temp):
    (Nt, N, Nr) =params_system
    ( aod_azi_set, aod_ele_set, aoa_azi_set, aoa_ele_set) = aod_aoa
    row, col = h_tilde.shape[0], h_tilde.shape[1]   # Nt x Nr
    xi_dot_mtx = np.zeros((N, row, col), dtype=complex)   #
    xi_dotdot_mtx = np.zeros((N, row, col), dtype=complex)

    xi_dot_p_mtx = np.zeros((N, row, col), dtype=complex)   #
    xi_dotdot_p_mtx = np.zeros((N, row, col), dtype=complex)

    deriv_wrt_v = np.zeros((N, row, col), dtype=complex)   #
    deriv_wrt_phi = np.zeros((N, row, col), dtype=complex)

    ris_phase = complex_to_polar(ris_cmplx)

    for ii in range(N):
        for mm in range(row):
            for nn in range(col):
                tmp = np.exp(1j * ((mm - 1)* np.pi*math.sin(aoa_azi_set[ii])*math.sin(aoa_ele_set[ii]) \
                                    + (1-nn)* np.pi*math.sin(aod_azi_set[ii])*math.sin(aod_ele_set[ii])))

                xi_dot_mtx[ii, mm, nn] = 1j * (mm - 1)* np.pi * math.sin(aoa_azi_set[ii]) * math.cos(aoa_ele_set[ii]) * tmp
                xi_dotdot_mtx[ii, mm, nn] = 1j * (mm - 1)* np.pi * math.sin(aoa_ele_set[ii]) * math.cos(aoa_azi_set[ii]) * tmp

        xi_dot_p_mtx[ii, :, :] = np.complex(np.sqrt(P_temp),0.0) * xi_dot_mtx[ii, :, :]
        xi_dotdot_p_mtx[ii, :, :] = np.complex(np.sqrt(P_temp),0.0) * xi_dotdot_mtx[ii, :, :]

        deriv_wrt_v[ii, :, :] = np.exp(1j* ris_phase[ii]) * xi_dot_p_mtx[ii, :, :]
        deriv_wrt_phi[ii, :,:] = np.exp(1j* ris_phase[ii]) * xi_dotdot_p_mtx[ii, :, :]

    derivs = (deriv_wrt_v, deriv_wrt_phi, xi_dot_mtx, xi_dotdot_mtx, xi_dot_p_mtx, xi_dotdot_p_mtx)

    return derivs

def get_T_mtx(params_system, location_user, ris_position, aod_aoa):     # verify matri E
    (Nt, N, Nr) =params_system
    ( aod_azi_set, aod_ele_set, aoa_azi_set, aoa_ele_set) = aod_aoa
    
    mathcalE = np.zeros((2,N), dtype=complex)
    mathcalF = np.zeros((2,N), dtype=complex)

    for nn in range(N):
        position_nn = ris_position[nn]
        tmp = np.linalg.norm(location_user[0:2] - position_nn[0:2])

        mathcalE[0,nn] = position_nn[2]/(tmp ** 2 + (position_nn[2])**2) * -1*np.cos(aoa_azi_set[nn])
        mathcalE[1,nn] = position_nn[2]/(tmp ** 2 + (position_nn[2])**2) *np.sin(aoa_azi_set[nn])

        mathcalF[0,nn] = 1/tmp * np.sin(aoa_azi_set[nn])
        mathcalF[1,nn] = 1/tmp * np.cos(aoa_azi_set[nn])

    return mathcalE, mathcalF

def get_crlb(params_system, mathcalE, mathcalF, derivs, ris_cmplx):
    (Nt, N, Nr) =params_system
    (_, _, _, _, xi_dot_p_mtx, xi_dotdot_p_mtx) = derivs 
    ris_phase = complex_to_polar(ris_cmplx)

    kappa_11_mtx = np.zeros((N,N), dtype=complex)
    kappa_12_mtx = np.zeros((N,N), dtype=complex)
    kappa_21_mtx = np.zeros((N,N), dtype=complex)
    kappa_22_mtx = np.zeros((N,N), dtype=complex)

    for mm in range(N):
        for nn in range(N):
            kappa_11_mtx[mm,nn] = mathcalE[0,mm]*mathcalE[0,nn]*np.matmul((np.asmatrix(xi_dot_p_mtx[mm]).getH()), xi_dot_p_mtx[nn]) \
                                + mathcalF[0,mm]*mathcalE[0,nn]*np.matmul((np.asmatrix(xi_dotdot_p_mtx[mm]).getH()), xi_dot_p_mtx[nn]) \
                                + mathcalE[0,mm]*mathcalF[0,nn]*np.matmul((np.asmatrix(xi_dot_p_mtx[mm]).getH()), xi_dotdot_p_mtx[nn]) \
                                + mathcalF[0,mm]*mathcalF[0,nn]*np.matmul((np.asmatrix(xi_dotdot_p_mtx[mm]).getH()), xi_dotdot_p_mtx[nn])

            kappa_12_mtx[mm,nn] = mathcalE[0,mm]*mathcalE[1,nn]*np.matmul((np.asmatrix(xi_dot_p_mtx[mm]).getH()), xi_dot_p_mtx[nn]) \
                                + mathcalF[0,mm]*mathcalE[1,nn]*np.matmul((np.asmatrix(xi_dotdot_p_mtx[mm]).getH()), xi_dot_p_mtx[nn]) \
                                + mathcalE[0,mm]*mathcalF[1,nn]*np.matmul((np.asmatrix(xi_dot_p_mtx[mm]).getH()), xi_dotdot_p_mtx[nn]) \
                                + mathcalF[0,mm]*mathcalF[1,nn]*np.matmul((np.asmatrix(xi_dotdot_p_mtx[mm]).getH()), xi_dotdot_p_mtx[nn])
            
            kappa_21_mtx[mm,nn] = mathcalE[1,mm]*mathcalE[0,nn]*np.matmul((np.asmatrix(xi_dot_p_mtx[mm]).getH()), xi_dot_p_mtx[nn]) \
                                + mathcalF[1,mm]*mathcalE[0,nn]*np.matmul((np.asmatrix(xi_dotdot_p_mtx[mm]).getH()), xi_dot_p_mtx[nn]) \
                                + mathcalE[1,mm]*mathcalF[0,nn]*np.matmul((np.asmatrix(xi_dot_p_mtx[mm]).getH()), xi_dotdot_p_mtx[nn]) \
                                + mathcalF[1,mm]*mathcalF[0,nn]*np.matmul((np.asmatrix(xi_dotdot_p_mtx[mm]).getH()), xi_dotdot_p_mtx[nn])

            kappa_22_mtx[mm,nn] = mathcalE[1,mm]*mathcalE[1,nn]*np.matmul((np.asmatrix(xi_dot_p_mtx[mm]).getH()), xi_dot_p_mtx[nn]) \
                                + mathcalF[1,mm]*mathcalE[1,nn]*np.matmul((np.asmatrix(xi_dotdot_p_mtx[mm]).getH()), xi_dot_p_mtx[nn]) \
                                + mathcalE[1,mm]*mathcalF[1,nn]*np.matmul((np.asmatrix(xi_dot_p_mtx[mm]).getH()), xi_dotdot_p_mtx[nn]) \
                                + mathcalF[1,mm]*mathcalF[1,nn]*np.matmul((np.asmatrix(xi_dotdot_p_mtx[mm]).getH()), xi_dotdot_p_mtx[nn])
    j_11, j_12, j_21, j_22 = 0, 0, 0, 0
    for mm in range(N):
        for nn in range(N):
            j_11 += np.real(np.exp(1j*(ris_phase[nn] - ris_phase[mm]))*kappa_11_mtx[mm,nn])
            j_12 += np.real(np.exp(1j*(ris_phase[nn] - ris_phase[mm]))*kappa_12_mtx[mm,nn])
            j_21 += np.real(np.exp(1j*(ris_phase[nn] - ris_phase[mm]))*kappa_21_mtx[mm,nn])
            j_22 += np.real(np.exp(1j*(ris_phase[nn] - ris_phase[mm]))*kappa_22_mtx[mm,nn])
    j_11 = j_11 *2 / np.sqrt(0.5)
    j_12 = j_12 *2 / np.sqrt(0.5)
    j_21 = j_21 *2 / np.sqrt(0.5)
    j_22 = j_22 *2 / np.sqrt(0.5)

    crlb_component = (j_11, j_12, j_21, j_22)
    kappas = (kappa_11_mtx , kappa_12_mtx, kappa_21_mtx, kappa_22_mtx)
    return crlb_component ,kappas

def get_crlb_fixedkappa(params_system, ris_cmplx,kappas):
    (Nt, N, Nr) =params_system
    ris_phase = complex_to_polar(ris_cmplx)
    (kappa_11_mtx,kappa_12_mtx,kappa_21_mtx,kappa_22_mtx) = kappas

    j_11, j_12, j_21, j_22 = 0, 0, 0, 0
    for mm in range(N):
        for nn in range(N):
            j_11 += np.real(np.exp(1j*(ris_phase[nn] - ris_phase[mm]))*kappa_11_mtx[mm,nn])
            j_12 += np.real(np.exp(1j*(ris_phase[nn] - ris_phase[mm]))*kappa_12_mtx[mm,nn])
            j_21 += np.real(np.exp(1j*(ris_phase[nn] - ris_phase[mm]))*kappa_21_mtx[mm,nn])
            j_22 += np.real(np.exp(1j*(ris_phase[nn] - ris_phase[mm]))*kappa_22_mtx[mm,nn])
    j_11 = j_11 *2 / np.sqrt(0.5)
    j_12 = j_12 *2 / np.sqrt(0.5)
    j_21 = j_21 *2 / np.sqrt(0.5)
    j_22 = j_22 *2 / np.sqrt(0.5)

    crlb_component = (j_11, j_12, j_21, j_22)
    return crlb_component

def crlb_deriv(params_system, ris_phase, kappas):
    L = 1
    (Nt, N, Nr) =params_system
    (kappa_11_mtx , kappa_12_mtx, kappa_21_mtx, kappa_22_mtx) = kappas

    j11_deriv = np.zeros(N, dtype=complex)
    j12_deriv = np.zeros(N, dtype=complex)
    j21_deriv = np.zeros(N, dtype=complex)
    j22_deriv = np.zeros(N, dtype=complex)

    const_coef = 4/0.5

    for ii in range(N):
        j11_deriv_ii, j12_deriv_ii, j21_deriv_ii, j22_deriv_ii = 0,0,0,0
        for ll in range(L):
            for nn in range(N):
                if nn != ii:
                    tmp = 1j * np.exp(1j*(ris_phase[ii]-ris_phase[nn]))
                    j11_deriv_ii += np.real(tmp*kappa_11_mtx[nn,ii])
                    j12_deriv_ii += np.real(tmp*kappa_12_mtx[nn,ii])
                    j21_deriv_ii += np.real(tmp*kappa_21_mtx[nn,ii])
                    j22_deriv_ii += np.real(tmp*kappa_22_mtx[nn,ii])
        j11_deriv[ii], j12_deriv[ii] = const_coef*j11_deriv_ii,const_coef*j12_deriv_ii
        j21_deriv[ii], j22_deriv[ii] = const_coef*j21_deriv_ii,const_coef*j22_deriv_ii
    j_derivs = (j11_deriv, j12_deriv, j21_deriv ,j22_deriv)
    return j_derivs

def find_gradient(params_system, j_derivs, crlb_component):
    (Nt, N, Nr) =params_system
    (j_11, j_12, j_21, j_22) = crlb_component
    (j11_deriv, j12_deriv, j21_deriv ,j22_deriv) = j_derivs

    Ru, De, _ = calculate_crlb(crlb_component)
    directions = np.zeros(N, dtype=complex)

    for ii in range(N):
        tmp = Ru* (j11_deriv[ii]*j_22 + j_11*j22_deriv[ii]-j12_deriv[ii]*j_21- j_12*j21_deriv[ii])
        direc_ii = ((j11_deriv[ii] + j22_deriv[ii])*De - tmp)/(De ** 2)
        directions[ii] = direc_ii
    return directions

def find_function_diff(params_system,H_tilde,aod_aoa,ris_vec,P_temp,location_user,position, crlb_component_old,kappas):
    #derivs = get_htilde_derivative(params_system, H_tilde, aod_aoa, ris_vec,P_temp)
    #mathcalE, mathcalF = get_T_mtx(params_system, location_user, position, aod_aoa)
    crlb_component_new = get_crlb_fixedkappa(params_system, ris_vec,kappas)
    _,_, crlb_old = calculate_crlb(crlb_component_old)
    _,_, crlb_new = calculate_crlb(crlb_component_new)
    return crlb_new, crlb_component_new, np.abs(crlb_new - crlb_old)

def move_in_the_direction(params_system, ris_phase, direct_gradient, crlb_component, 
                            H_tilde,aod_aoa, step_size,P_temp,location_user,position,kappas):
    (Nt, N, Nr) =params_system
    epsilon = 0.05
    func_diff = 100
    crlb_component_old = crlb_component
    iicount = 0
    while func_diff > epsilon and iicount < 50:
        for ii in range(N):
            tmp = step_size*direct_gradient[ii]
            ris_phase[ii] = ris_phase[ii] - tmp
        iicount += 1
        crlb_old, crlb_component_old, func_diff = find_function_diff(params_system,H_tilde,aod_aoa,\
                        polar_to_complex(ris_phase),P_temp,location_user,position, crlb_component_old,kappas)
    return ris_phase
    
def estimate_AOA(params_system, pilot_y, steering_vec_codebook, H,A_t,ris_profile):
    (Nt, N, Nr) =params_system
    resolution = steering_vec_codebook.shape[0]
    min = -math.inf
    
    for rr1 in range(resolution):
        for rr2 in range(resolution):
            try_A_r = np.zeros([Nr, N], dtype=complex) 
            try_A_r[:, 0] = steering_vec_codebook[rr1,rr2,:]

            ris_diag = np.diag(ris_profile)
            try_H_tilde = combine_channel(H,A_t,try_A_r,ris_diag)

            tmp = np.matmul(np.asmatrix(try_H_tilde).getH(), pilot_y)
            tmp = np.abs(tmp) **2
            if tmp > min:
                min = tmp
                found_rr1 = rr1
                found_rr2 = rr2
                
    main_ang_el =np.linspace(-np.pi, np.pi, resolution)
    main_ang_az =np.linspace(-np.pi, np.pi, resolution)
    


def design_next_ris(params_system,ris_cmplx,crlb_component,kappas, H_tilde,aod_aoa, 
                    step_size,P_temp,location_user,position, pilot_y, steering_vec_codebook,H,A_t,ris_vec):
    
    ris_phase = complex_to_polar(ris_cmplx)

    ############## estiamte AOA, need to recalculate kappa and mathcalEF ##############
    #( aod_azi_set, aod_ele_set, aoa_azi_set, aoa_ele_set) = aod_aoa
    #estimate_AOA(params_system, pilot_y, steering_vec_codebook, H,A_t,ris_vec)
    #derivs = get_htilde_derivative(params_system, H_tilde, aod_aoa, ris_vec,P_temp)
    #mathcalE, mathcalF = get_T_mtx(params_system, location_user, position, aod_aoa)
    #crlb_component, kappa get_crlb(params_system, mathcalE, mathcalF, derivs, ris_cmplx)

    j_derivs = crlb_deriv(params_system, ris_phase, kappas)
    
    direct_gradient = find_gradient(params_system, j_derivs, crlb_component)

    ris_phase = move_in_the_direction(params_system, ris_phase, direct_gradient, crlb_component, \
                                        H_tilde,aod_aoa, step_size,P_temp,location_user,position,kappas)

    return polar_to_complex(ris_phase)


def get_position_cb(params_system, ris_position):
    (Nt, N, Nr) =params_system
    x_lowerlimit, x_upperlimit = 5, 35
    y_lowerlimit, y_upperlimit = -35, 35
    z_fixed = -20

    x_range = x_upperlimit - x_lowerlimit + 1
    y_range = y_upperlimit - y_lowerlimit + 1

    position_codebook = np.zeros([x_range, y_range, 2*N], dtype=complex) 

    azi_list = []
    ele_list = []

    for x_i in range(x_range):
        for y_i in range(y_range):
            coordinate_k = np.array([x_i + x_lowerlimit, y_i + y_lowerlimit, z_fixed])
            # generage channel/fingerprint based on location
            location_user = np.empty(3)
            location_user[:] = coordinate_k
            
            aoa_azi_set = [] 
            aoa_ele_set = []

            for nn in range(N):
                position_nn = ris_position[nn]

                tmp = np.linalg.norm(location_user[0:2] - position_nn[0:2])
                ele_angle = np.arctan(tmp/position_nn[2])
                azi_angle = np.arccos(-(np.abs(location_user[0]-position_nn[0]))/tmp)

                aoa_azi_set.append(azi_angle)
                aoa_ele_set.append(ele_angle)
                azi_list.append(azi_angle)
                ele_list.append(ele_angle)
            
            position_codebook[x_i,y_i,0:N] = aoa_azi_set
            position_codebook[x_i,y_i,N:2*N] = aoa_ele_set
    angle_limit = (max(azi_list), min(azi_list), max(ele_list), min(ele_list))
    return position_codebook,angle_limit




def save_channel_info(params_system,num_test,position,location_bs,initial_run,L,resolution):
    (Nt, N, Nr) =params_system
    save_channel_path = './loc/CRLB/channel'+'_numtest_'+str(num_test)+'_'+str(params_system)+'.mat'
    if path.exists(save_channel_path) and initial_run==0:
        data_loadout = sio.loadmat(save_channel_path)
        location_user_set, aod_aoa_set = data_loadout['location_user_set'], data_loadout['aod_aoa_set']
        H_set, A_t_set, A_r_set = data_loadout['H_set'],data_loadout['A_t_set'],data_loadout['A_r_set']
        steering_vec_codebook, position_codebook = data_loadout['steering_vec_codebook'], data_loadout['position_codebook']
        print("Channel data exisits !")
    else:
        print("New Channel data  !")
        location_user_set = generate_location_old(num_test) # just one user
        ############## fix a user s location

        location_user_target = np.empty([1, 3])
        coordinate_k = np.array([10 , -15, -20])

        location_user_target[0, :] = coordinate_k
        location_user_set = location_user_target

        ###############
        aod_aoa_set =  get_Aod_AoA_2(params_system, position, location_user_set, location_bs)
        H_set, A_t_set, A_r_set  = generate_channel(params_system, aod_aoa_set)
        position_codebook, angle_limit = get_position_cb(params_system, position)
        #steering_vec_codebook = get_steering_vector_cb(params_system, position, location_user_set, location_bs, resolution,angle_limit)
        steering_vec_codebook = 1
        sio.savemat(save_channel_path, {'location_user_set': location_user_set, 'aod_aoa_set':aod_aoa_set,
                                        'H_set':H_set,'A_t_set':A_t_set, 'A_r_set':A_r_set, 
                                        'steering_vec_codebook': steering_vec_codebook,'position_codebook':position_codebook })
        print("Creating new Channel data !")
    save_ris_path = './loc/CRLB/ris'+'_numtest_'+str(num_test)+'_size_'+str(N)+'_tau_'+str(L)+'.mat'
    if path.exists(save_ris_path):
        data_loadout = sio.loadmat(save_ris_path)
        ris_vec_rnd = data_loadout['ris_vec_rnd']
        print("RIS Profile data exisits !")
    else:
        _,ris_vec_rnd = random_beamforming(num_test*L, Nt, N, Nr)
        sio.savemat(save_ris_path, {'ris_vec_rnd': ris_vec_rnd})
        print("Creating new RIS profile !")

    return location_user_set,aod_aoa_set,H_set,A_t_set,A_r_set,ris_vec_rnd,steering_vec_codebook, position_codebook

def main():

    Nt, N, Nr = 1, 64, 4
    params_system = (Nt, N, Nr)
    antenna_separation = 0.3
    irs_Nh = 16
    location_bs = np.array([40,-40,0])
    num_test = 1
    initial_run = 1
    resolution = 8

    L = 20
    random_ris = False
    step_size = 0.01

    snr_const = 35 #The SNR
    snr_const = np.array([snr_const])
    P_temp = 10**(snr_const[0]/10)

    # Rician_factor = 10
    # location_user = None
    start = time.time()
    crlb_vs_l = np.zeros(L)
    position = get_ris_element_position(params_system, antenna_separation,irs_Nh) # list of N positions

    location_user_set,aod_aoa_set,H_set,A_t_set,A_r_set,ris_vec_rnd,steering_vec_codebook, position_codebook= \
                        save_channel_info(params_system,num_test,position,location_bs,initial_run,L,resolution)
    dist_error_l = []
    ris_set = []
    for test_i in range(num_test):

        location_user = location_user_set[test_i]
        aod_aoa =  aod_aoa_set[test_i]
        H, A_t, A_r = H_set[test_i], A_t_set[test_i], A_r_set[test_i]

        # generate RIS
        j_11_sum, j_12_sum, j_21_sum, j_22_sum = 0,0,0,0
        for ll in range(L):
            
            if random_ris or ll == 0:
                ris_vec = ris_vec_rnd[test_i*L + ll]
                
            else:
                ris_vec = design_next_ris(params_system,ris_vec,crlb_component,kappas, 
                                H_tilde,aod_aoa, step_size,P_temp,location_user,position, 
                                    pilot_y,steering_vec_codebook, H,A_t,ris_vec)    
            ris_diag = np.diag(ris_vec)
            
            ris_set.append(ris_vec)

            # get combined channel

            H_tilde = combine_channel(H,A_t,A_r,ris_diag)

            pilot_y = get_received_pilot(params_system, H_tilde, 1, P_temp) # return  Nr*L x  1

            derivs = get_htilde_derivative(params_system, H_tilde, aod_aoa, ris_vec,P_temp)

            mathcalE, mathcalF = get_T_mtx(params_system, location_user, position, aod_aoa)

            crlb_component, kappas = get_crlb(params_system, mathcalE, mathcalF, derivs, ris_vec)
            (j_11, j_12, j_21, j_22) = crlb_component
            j_11_sum += j_11
            j_12_sum += j_12
            j_21_sum += j_21
            j_22_sum += j_22
            _, _, crlb_ll = calculate_crlb((j_11_sum,j_12_sum,j_21_sum,j_22_sum))

            
            crlb_vs_l[ll] += crlb_ll
        
        prstime = calcProcessTime(start,test_i+1,num_test)
        print("","time elapsed: %s(s), time left: %s(s), estimated finish time: %s"%prstime)

    ris_set_np = np.array(ris_set)
    print(ris_set_np.shape)


    crlb_vs_l = crlb_vs_l/num_test


    plt.semilogy(np.arange(L)+1, crlb_vs_l)    
    plt.grid()
    plt.xlabel('L')
    plt.ylabel('Average MSE')
    plt.show()
        
    
if __name__ == '__main__':
    main()




    