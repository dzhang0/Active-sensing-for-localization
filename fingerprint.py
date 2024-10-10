import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
import matplotlib.pyplot as plt
from func_codedesign_cont import func_codedesign_cont
import scipy.io as sio
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense
from wsr_bcd.generate_channel import generate_channel_fullRician, channel_complex2real, generate_location
from util_func import random_beamforming
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error

#from wsr.bcd.generate_received_pilots import generate_received_pilots_batch

'System Information'
N = 1   #Number of BS's antennas
delta_inv = 128 #Number of posterior intervals inputed to DNN 
delta = 1/delta_inv 
S = np.log2(delta_inv) 
OS_rate = 20 #Over sampling rate in each AoA interval (N_s in the paper)
delta_inv_OS = OS_rate*delta_inv #Total number of AoAs for posterior computation
delta_OS = 1/delta_inv_OS 
'Channel Information'

location_bs_new = np.array([40,-40,0])

tau = 14
#snr_const = [0,10, 25, 35] #The SNR
#SNR_CONST = [20]
SNR_CONST = [20]
snr_const = np.array([SNR_CONST])

Pvec = 10**(snr_const/10) #Set of considered TX powers

save_result = True

mean_true_alpha = 0.0 + 0.0j #Mean of the fading coefficient
std_per_dim_alpha = np.sqrt(0.5) #STD of the Gaussian fading coefficient per real dim.
noiseSTD_per_dim = np.sqrt(0.5) #STD of the Gaussian noise per real dim.
#####################################################
'RIS'
N_ris = 64
num_users = 1
params_system = (N,N_ris,num_users)
Rician_factor_array = [5,10,15,20,40,80,1000000]


location_user = None

Num_test = 1500
num_neighbors = 6

#####################################################
'Learning Parameters'
initial_run = 1 #0: Continue training; 1: Starts from the scratch
n_epochs = 1500 #Num of epochs
learning_rate = 0.0005 #Learning rate
batch_per_epoch = 100 #Number of mini batches per epoch
batch_size_order = 8 #Mini_batch_size = batch_size_order*delta_inv
val_size_order = 782 #Validation_set_size = val_size_order*delta_inv
scale_factor = 1 #Scaling the number of tests
test_size_order = 782 #Test_set_size = test_size_order*delta_inv*scale_factor
######################################################
tf.reset_default_graph() #Reseting the graph
he_init = tf.variance_scaling_initializer() #Define initialization method
######################################## Place Holders
loc_input = tf.placeholder(tf.float32, shape=(None,1,3), name="loc_input")
channel_bs_irs_user = tf.placeholder(tf.float32, shape=(None, 2 * N_ris, 2 * N, num_users), name="channel_bs_irs_user")
channel_bs_user = tf.placeholder(tf.float32, shape=(None, 2 * N, num_users), name="channel_bs_user")
theta_T = tf.placeholder(tf.float32, shape=(None, tau, 2 * N_ris), name="theta_T")
######################################################
path_pilots = './loc/DNN_fixSNR/theta_training_tau_'+ str(tau) +'_'+'SNR'+str(snr_const[0])+'.mat'
if initial_run == 0: # continue training
    data_loadout = sio.loadmat(path_pilots)
    the_theta = data_loadout['the_theta']
else:   # training from scratch
    _, the_theta = random_beamforming(tau, N , N_ris, num_users)    # num test x num ris
    sio.savemat(path_pilots, {'the_theta': the_theta})

# first receive and store fingerprint
x_lowerlimit, x_upperlimit = 5, 35
y_lowerlimit, y_upperlimit = -35, 35
z_fixed = -20

x_range = x_upperlimit - x_lowerlimit + 1
y_range = y_upperlimit - y_lowerlimit + 1

radio_map = np.zeros([x_range,y_range,tau])

def generate_RSS(A_T_real, Hd_real , P_temp):
    RSS_list = np.zeros((tau), dtype=float)
    for tau_i in range(tau):
        theta_i = the_theta[[tau_i], :]
        theta = np.concatenate([theta_i.real, theta_i.imag], axis=1)
        theta_T = np.reshape(theta, [-1, 1, 2 * N_ris])
        A_T_k = A_T_real[0, :, :, 0]
        theta_A_k_T = np.matmul(theta_T, A_T_k) # 1, 1, 2

        h_d = Hd_real[0,:,0]            # 1,2,1 - > 2
        h_d_T = np.reshape(h_d, [-1, 1, 2 * N])  # 1, 1, 2
        h_d_plus_h_cas = h_d_T + theta_A_k_T
        h_d_plus_h_cas_re = h_d_plus_h_cas[:,:,0]
        h_d_plus_h_cas_im = h_d_plus_h_cas[:,:,1]
        noise =  np.random.normal(size = np.shape(h_d_plus_h_cas_re), loc = 0.0, scale = noiseSTD_per_dim) + \
                    1j*np.random.normal(size = np.shape(h_d_plus_h_cas_re), loc = 0.0, scale = noiseSTD_per_dim)
        RSS_i =  abs( (np.sqrt(P_temp)+ 1j*0.0) *(h_d_plus_h_cas_re + 1j*h_d_plus_h_cas_im)+ noise) ** 2 
        RSS_list[tau_i] = RSS_i
    return RSS_list


mse_avg_list = []
for Rician_factor in Rician_factor_array:
#for pow_i in Pvec[0]:
    pow_i = Pvec[0][0]
    fingerprint_dict = {}
    fingerprint_list = []
    loc_index_dict = {}
    loc_index = 0
    for x_i in range(x_range):
        for y_i in range(y_range):
            coordinate_k = np.array([x_i + x_lowerlimit, y_i + y_lowerlimit, z_fixed])
            # generage channel/fingerprint based on location
            location_user = np.empty([num_users, 3])
            location_user[0, :] = coordinate_k
            location_user = np.expand_dims(location_user, axis=0)
            channel_true, set_location_user = generate_channel_fullRician(params_system, location_bs = location_bs_new, num_samples=1,
                                            location_user_initial=location_user, Rician_factor=Rician_factor)
            A_T_real, Hd_real, channel_bs_irs_user = channel_complex2real(channel_true)
            
            RSS_offline = generate_RSS(A_T_real, Hd_real,pow_i)
            loc_i = (x_i + x_lowerlimit , y_i + y_lowerlimit, z_fixed)
            fingerprint_dict[loc_i] = RSS_offline
            fingerprint_list.append(RSS_offline)
            loc_index_dict[loc_index] = loc_i
            loc_index += 1

            radio_map[x_i, y_i, :] = RSS_offline

    # sio.savemat('./loc/DNN_fixSNR/radiomap'+ str(tau) +'_Ricianfactor' + str(Rician_factor)+ '_SNR'+ str(snr_const[0][0]) +'.mat',\
    #                                 dict(radio_map= radio_map,\
    #                                    snr_const=snr_const,N=N,N_ris = N_ris,\
    #                                    mean_true_alpha=mean_true_alpha,\
    #                                     Rician_factor = Rician_factor,\
    #                                    std_per_dim_alpha=std_per_dim_alpha,\
    #                                    noiseSTD_per_dim=noiseSTD_per_dim, tau=tau))

    # Test and Map to database
    mse_list = []
    for test_i in range(Num_test):

        loc_test = generate_location(num_users)
        loc_test = np.expand_dims(loc_test, axis=0)
        channel_true_test, _ = generate_channel_fullRician(params_system,location_bs = location_bs_new, num_samples=1,
                                                location_user_initial=loc_test, Rician_factor=Rician_factor)
        A_T_real_test, Hd_real_test, _ = channel_complex2real(channel_true_test)
        RSS_test = generate_RSS(A_T_real_test,Hd_real_test, pow_i)

        fingerprint_list_w_test_data = fingerprint_list.copy() 
        fingerprint_list_w_test_data.append(RSS_test)
        nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='auto').fit(fingerprint_list_w_test_data)
        distances, indices = nbrs.kneighbors(fingerprint_list_w_test_data)

        index_online = x_range * y_range 
        weight_total = sum(1/distances[index_online][1:num_neighbors])
        for n_i in range(num_neighbors):
            if n_i == 0:
                pass
            elif n_i == 1:
                est_loc_i = loc_index_dict[indices[index_online][n_i]]
                est_loc_avg = [np.array(est_loc_i)]
                weighted_est_loc_avg = np.multiply((1/distances[index_online][n_i])/weight_total, [np.array(est_loc_i)])
            elif n_i > 1:
                est_loc_i = loc_index_dict[indices[index_online][n_i]]
                est_loc_avg = np.concatenate([est_loc_avg,[est_loc_i]], axis = 0)   # num_neighbor x 3
                
                weighted_est_loc_i = np.multiply((1/distances[index_online][n_i])/weight_total , loc_index_dict[indices[index_online][n_i]])
                weighted_est_loc_avg = np.concatenate([weighted_est_loc_avg,[weighted_est_loc_i]], axis = 0)

        est_loc_avg = np.sum(weighted_est_loc_avg, axis = 0)
        
        #est_loc_avg = np.average(est_loc_avg, axis=0)

        # print(indices[index_online])
        # print(indices[index_online][1])
        # print(loc_index_dict[indices[index_online][1]])print(loc_index_dict[indices[index_online][2]])print(loc_index_dict[indices[index_online][3]])

        dist_diff = np.array(loc_test) - est_loc_avg
        mse = np.sum(np.square(dist_diff[0]))
        mse_list.append(mse)

    mse_avg = sum(mse_list) / Num_test
    print('Current MSE: ',mse_avg) 
    mse_avg_list.append(mse_avg)

if save_result:
    sio.savemat('./loc/DNN_fixSNR/data_fingerprint_wKNN_loc_3D_tau_'+ str(tau) + '_snr_' + str(SNR_CONST)\
                    +'_Ricianfactor' + str(Rician_factor_array) +'.mat',dict(performance= mse_avg_list,\
                                       snr_const=snr_const,N=N,N_ris = N_ris,\
                                       mean_true_alpha=mean_true_alpha,\
                                        Rician_factor = Rician_factor,\
                                       std_per_dim_alpha=std_per_dim_alpha,\
                                       noiseSTD_per_dim=noiseSTD_per_dim, tau=tau))

