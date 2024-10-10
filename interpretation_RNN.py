import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense
from wsr_bcd.generate_channel import generate_channel_fullRician, channel_complex2real, generate_location
from util_func import random_beamforming
import time

drive_save_path = './loc/journal/RNN'

'System Information'
N = 1   #Number of BS's antennas
delta_inv = 128 #Number of posterior intervals inputed to DNN 
delta = 1/delta_inv 
S = np.log2(delta_inv) 
OS_rate = 20 #Over sampling rate in each AoA interval (N_s in the paper)
delta_inv_OS = OS_rate*delta_inv #Total number of AoAs for posterior computation
delta_OS = 1/delta_inv_OS 
'Channel Information'
phi_min = -60*(np.pi/180) #Lower-bound of AoAs
phi_max = 60*(np.pi/180) #Upper-bound of AoAs
num_SNR = 8 #Number of considered SNRs


tau = 6 #Pilot length

snr_const = 25 #The SNR
snr_const = np.array([snr_const])
Pvec = 10**(snr_const/10) #Set of considered TX powers

location_bs_new = np.array([0, 0, 0])
location_ris_1 = np.array([-40, 40, 10])
#location_ris_2 = np.array([-40, 60, 5])
num_ris = 1

mean_true_alpha = 0.0 + 0.0j #Mean of the fading coefficient
std_per_dim_alpha = np.sqrt(0.5) #STD of the Gaussian fading coefficient per real dim.
noiseSTD_per_dim = np.sqrt(0.5) #STD of the Gaussian noise per real dim.
#####################################################
'RIS'
N_ris = 64
num_users = 1
params_system = (N,N_ris,num_users)
Rician_factor = 10
location_user = None

#####################################################
'Learning Parameters'
initial_run = 0    #0: Continue training; 1: Starts from the scratch
if initial_run == 1:
  print('!!!!! training from scratch !!!!!')
n_epochs = 1 #Num of epochs
learning_rate = 0.00005 #Learning rate
batch_per_epoch = 100 #Number of mini batches per epoch
batch_size_order = 8 #Mini_batch_size = batch_size_order*delta_inv
val_size_order = 782 #Validation_set_size = val_size_order*delta_inv
scale_factor = 1 #Scaling the number of tests
test_size_order = 782 #Test_set_size = test_size_order*delta_inv*scale_factor
######################################################
tf.reset_default_graph() #Reseting the graph
he_init = tf.variance_scaling_initializer() #Define initialization method
######################################## Place Holders
#alpha_input = tf.placeholder(tf.float32, shape=(None,1), name="alpha_input")
loc_input = tf.placeholder(tf.float32, shape=(None,1,3), name="loc_input")
channel_bs_irs_user = tf.placeholder(tf.float32, shape=(None, 2 * N_ris, 2 * N, num_users), name="channel_bs_irs_user")
channel_bs_user = tf.placeholder(tf.float32, shape=(None, 2 * N, num_users), name="channel_bs_irs_user")
######################################################

def generate_RSS_adaptive(A_T_real, the_theta, P_temp):
    RSS_list = np.zeros((tau), dtype=float)
    for tau_i in range(tau):
        theta_i = the_theta[[tau_i], :]
        theta = np.concatenate([theta_i.real, theta_i.imag], axis=1)
        theta_T = np.reshape(theta, [-1, 1, 2 * N_ris])
        A_T_k = A_T_real[0, :, :, 0]
        theta_A_k_T = np.matmul(theta_T, A_T_k)
        theta_A_k_T_re = theta_A_k_T[:,:,0]
        theta_A_k_T_im = theta_A_k_T[:,:,1]
        RSS_i =  abs( (np.sqrt(P_temp)+ 1j*0.0) *(theta_A_k_T_re + 1j*theta_A_k_T_im)) ** 2 

        RSS_list[tau_i] = RSS_i
        
    return RSS_list

def generate_radio_map(theta_test):
    #x_lowerlimit, x_upperlimit = 5, 35
    #y_lowerlimit, y_upperlimit = -35, 35
    x_lowerlimit, x_upperlimit = -35, -5
    y_lowerlimit, y_upperlimit = 5, 75
    z_fixed = -20

    x_range = x_upperlimit - x_lowerlimit + 1
    y_range = y_upperlimit - y_lowerlimit + 1


    radio_map = np.zeros([x_range,y_range,tau])

    for x_i in range(x_range):
        for y_i in range(y_range):
            coordinate_k = np.array([x_i + x_lowerlimit, y_i + y_lowerlimit, z_fixed])
            # generage channel/fingerprint based on location
            location_user = np.empty([num_users, 3])
            location_user[0, :] = coordinate_k
            # channel_true, set_location_user = generate_channel(params_system, num_samples=1,
            #                                 location_user_initial=location_user, Rician_factor=Rician_factor)
            # A_T_real, Hd_real, channel_bs_irs_user = channel_complex2real(channel_true)
            channel_true, set_location_user_train = generate_channel_fullRician(params_system, location_bs_new,location_ris_1,
                                                        num_samples= 1 ,
                                                       location_user_initial=location_user, Rician_factor=Rician_factor)
            A_T_real, Hd_real_train , _ = channel_complex2real(channel_true)
            RSS_offline = generate_RSS_adaptive(A_T_real, theta_test ,Pvec[0])

            radio_map[x_i, y_i, :] = RSS_offline
    return radio_map, theta_test


##################### NETWORK
with tf.name_scope("array_response_construction"):
    lay = {}
    lay['P'] = tf.constant(1.0)
    ###############
    from0toN = tf.cast(tf.range(0, N, 1),tf.float32)
    #### Actual Channel
    # phi = tf.reshape(phi_input,[-1,1])
    # h_act = {0: 0}
    # hR_act = {0: 0}
    # hI_act = {0: 0}   
    # phi_expanded = tf.tile(phi,(1,N))
    # a_phi = (tf.exp(1j*np.pi*tf.cast(tf.multiply(tf.sin(phi_expanded),from0toN),tf.complex64)))

with tf.name_scope("channel_sensing"):
    hidden_size = 512
    A1 = tf.get_variable("A1",  shape=[hidden_size,1024], dtype=tf.float32, initializer= he_init)
    A2 = tf.get_variable("A2",  shape=[1024,1024], dtype=tf.float32, initializer= he_init)
    A3 = tf.get_variable("A3",  shape=[1024,1024], dtype=tf.float32, initializer= he_init)
    A4 = tf.get_variable("A4",  shape=[1024,2*N_ris], dtype=tf.float32, initializer= he_init)
    
    b1 = tf.get_variable("b1",  shape=[1024], dtype=tf.float32, initializer= he_init)
    b2 = tf.get_variable("b2",  shape=[1024], dtype=tf.float32, initializer= he_init)
    b3 = tf.get_variable("b3",  shape=[1024], dtype=tf.float32, initializer= he_init)
    b4 = tf.get_variable("b4",  shape=[2*N_ris], dtype=tf.float32, initializer= he_init)
        
    w_dict = []
    posterior_dict = []
    idx_est_dict = []
    layer_Ui = Dense(units=hidden_size, activation='linear')
    layer_Wi = Dense(units=hidden_size, activation='linear')
    layer_Uf = Dense(units=hidden_size, activation='linear')
    layer_Wf = Dense(units=hidden_size, activation='linear')
    layer_Uo = Dense(units=hidden_size, activation='linear')
    layer_Wo = Dense(units=hidden_size, activation='linear')
    layer_Uc = Dense(units=hidden_size, activation='linear')
    layer_Wc = Dense(units=hidden_size, activation='linear')
    def RNN(input_x, h_old, c_old):
        i_t = tf.sigmoid(layer_Ui(input_x) + layer_Wi(h_old))
        f_t = tf.sigmoid(layer_Uf(input_x) + layer_Wf(h_old))
        o_t = tf.sigmoid(layer_Uo(input_x) + layer_Wo(h_old))
        c_t = tf.tanh(layer_Uc(input_x) + layer_Wc(h_old))
        c = i_t * c_t + f_t * c_old     # cell state
        h_new = o_t * tf.tanh(c)        # hidden state
        return h_new, c
    
    snr = lay['P']*tf.ones(shape=[tf.shape(loc_input)[0],1],dtype=tf.float32)
    snr_dB = tf.log(snr)/np.log(10)
    snr_normal = (snr_dB-1)/np.sqrt(1.6666) #Normalizing for the range -10dB to 30dB
    
    theta_list = []

    for t in range(tau):      
        'DNN designs the next sensing direction'
        if t == 0:
            y_real = tf.ones([tf.shape(loc_input)[0],2])
            h_old = tf.zeros([tf.shape(loc_input)[0],hidden_size])
            c_old = tf.zeros([tf.shape(loc_input)[0],hidden_size])
        h_old, c_old = RNN(tf.concat([y_real,snr_normal],axis=1), h_old, c_old)

        x1 = tf.nn.relu(h_old@A1+b1)
        x1 = BatchNormalization()(x1)
        x2 = tf.nn.relu(x1@A2+b2)
        x2 = BatchNormalization()(x2)
        x3 = tf.nn.relu(x2@A3+b3)
        x3 = BatchNormalization()(x3)
        '''
            RIS implementation
        '''
        ris_her_unnorm = x3 @ A4 + b4
        ris_her_r = ris_her_unnorm[:, 0:N_ris]
        ris_her_i = ris_her_unnorm[:, N_ris:2 * N_ris]                      # (? , N_ris)
        theta_tmp = tf.sqrt(tf.square(ris_her_r) + tf.square(ris_her_i))    # (? , N_ris)
        theta_real = ris_her_r / theta_tmp                                  # (? , N_ris)
        theta_imag = ris_her_i / theta_tmp                                  # (? , N_ris)
        theta = tf.concat([theta_real, theta_imag], axis=1)                 # (? , 2*N_ris)      hmmmmm
        #print('theta:', theta.shape)
        theta_T = tf.reshape(theta, [-1, 1, 2 * N_ris])                     # (? , 1 , 2 * N_ris)
        #print('theta_T:', theta_T.shape)                                   #  (?, 1, 128)
        theta_list.append(theta_T[:,0,:])
        'BS observes the next measurement'
        
        A_T_k = channel_bs_irs_user[:, :, :, 0] # since 1 user
        theta_A_k_T = tf.matmul(theta_T, A_T_k)                             # (? , 1 , 2 * N ) 

        h_d = channel_bs_user[:,:,0]
        h_d_T = tf.reshape(h_d, [-1, 1, 2 * N])

        h_d_plus_h_cas = h_d_T + theta_A_k_T
        h_d_plus_h_cas_re = h_d_plus_h_cas[:,:,0]
        h_d_plus_h_cas_im = h_d_plus_h_cas[:,:,1]
        noise =  tf.complex(tf.random_normal(tf.shape(h_d_plus_h_cas_re), mean = 0.0, stddev = noiseSTD_per_dim),\
                    tf.random_normal(tf.shape(h_d_plus_h_cas_re), mean = 0.0, stddev = noiseSTD_per_dim))
        y_complex = tf.complex(tf.sqrt(lay['P']),0.0)*tf.complex(h_d_plus_h_cas_re,h_d_plus_h_cas_im) + noise
        y_real = tf.concat([tf.real(y_complex),tf.imag(y_complex)],axis=1)/tf.sqrt(lay['P'])

    h_old, c_old = RNN(tf.concat([y_real,snr_normal],axis=1), h_old, c_old)
    c_old = Dense(units=200, activation='linear')(c_old)
    c_old = Dense(units=200, activation='linear')(c_old)
    loc_hat = Dense(units=3, activation='linear')(c_old)

####################################################################################
####### Loss Function
a = tf.math.reduce_euclidean_norm(loc_input[:,0,:]-loc_input[:,0,:], 1)
b = tf.math.reduce_euclidean_norm(loc_hat-loc_input[:,0,:], 1)
loss = tf.keras.losses.mean_squared_error(a, b)
print(loss)
####### Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss, name="training_op")
init = tf.global_variables_initializer()
saver = tf.train.Saver()
#########################################################################
###########  Validation Set

channel_true_val, set_location_user_val = generate_channel_fullRician(params_system, location_bs_new,location_ris_1,
                                                        num_samples=val_size_order*delta_inv, location_user_initial=location_user, Rician_factor=Rician_factor)
A_T_1_real_val, Hd_real_val , _ = channel_complex2real(channel_true_val)
feed_dict_val = {loc_input: np.array(set_location_user_val),
                    channel_bs_irs_user: A_T_1_real_val,
                    channel_bs_user : Hd_real_val,
                    lay['P']: Pvec[0]}
###########  Training
with tf.Session() as sess:
    if initial_run == 1:
        init.run()
    else:
        saver.restore(sess, drive_save_path + '/params_closeBS_fullRician_3D_1RIS_newcoordinateSISO_N_1_tau_6_snr_25')
    best_loss, pp = sess.run([loss,posterior_dict], feed_dict=feed_dict_val)
    print(best_loss)
    print(tf.test.is_gpu_available()) #Prints whether or not GPU is on
    for epoch in range(n_epochs):
        batch_iter = 0
        for rnd_indices in range(batch_per_epoch):

            snr_temp = snr_const[0]
            P_temp = 10**(snr_temp/10)
            
            '''
                RIS implementation
            '''

            channel_true_train, set_location_user_train = generate_channel_fullRician(params_system, location_bs_new,location_ris_1,
                                                        num_samples=batch_size_order*delta_inv,
                                                       location_user_initial=location_user, Rician_factor=Rician_factor)
            A_T_1_real, Hd_real_train , _ = channel_complex2real(channel_true_train)
            #print(A_T_real.shape)
            #print(channel_bs_irs_user_train.shape)
            #print(channel_bs_user_train.shape) #(num_samples,1,1)
            #print(channel_irs_user_train.shape) # (num_samples,64,1)
            #print(channel_bs_irs_train.shape) # (num_samples,1,64)
            #print(aoa_irs_y_set_train.shape) # this is what we wanna estimate, shape: (num_sample,1)
            #print(pathloss_irs_user_set_train.shape)  # this is known  shape  (num_sample, )
            feed_dict_batch = {loc_input: np.array(set_location_user_train),
                              channel_bs_irs_user: A_T_1_real,
                               channel_bs_user : Hd_real_train,
                              lay['P']: P_temp}
            sess.run(training_op, feed_dict=feed_dict_batch)
            batch_iter += 1

        
        loss_val = sess.run(loss, feed_dict=feed_dict_val)
        print('epoch',epoch,'  loss_test:%2.5f'%loss_val,'  best_test:%2.5f'%best_loss) 
        #print('epoch',epoch,'  loss_test:',loss_val, 'best_test:',best_loss) 
        if epoch%2 == 1: #Every 10 iterations it checks if the validation performace is improved, then saves parameters
            if loss_val < best_loss:
                save_path = saver.save(sess, drive_save_path+'/params_closeBS_fullRician_3D_1RIS_newcoordinateSISO_N_1_tau_6_snr_25')
                best_loss = loss_val


#    x_lowerlimit, x_upperlimit = -35, -5
#    y_lowerlimit, y_upperlimit = 5, 75
###########  Final Test    
    performance = np.zeros([1,scale_factor])
    for j in range(scale_factor):
        print(j)
        location_user_target = np.empty([num_users, 3])
        coordinate_k = np.array([-10 , 40, -20])

        #coordinate_k = np.array([-10 , 20, -20])
        location_user_target[0, :] = coordinate_k
        
        channel_true_test, set_location_user_test = generate_channel_fullRician(params_system, location_bs_new,location_ris_1,
                                                        num_samples=1,
                                                       location_user_initial=location_user_target, Rician_factor=Rician_factor)
        A_T_1_real_test, Hd_real_test , channel_bs_irs_user_test = channel_complex2real(channel_true_test)
 
        feed_dict_test = {loc_input: np.array(set_location_user_test),
                            channel_bs_irs_user: A_T_1_real_test,
                            channel_bs_user : Hd_real_test,
                            lay['P']: Pvec[0]}
        mse_loss,phi_hat_test,theta_test= sess.run([loss,loc_hat,theta_list],feed_dict=feed_dict_test)
        performance[0,j] = mse_loss
        # test location
        # location_user_target = generate_location(num_users)
        # channel_true, set_location_user_test = generate_channel(params_system, num_samples=1,
        #                                         location_user_initial=location_user_target, Rician_factor=Rician_factor)
        # A_T_real, Hd_real, channel_bs_irs_user = channel_complex2real(channel_true)
        # # print(type(np.array(set_location_user_test)))
        # # print(type(A_T_real))
        # # print(type(Hd_real))
        # # print(type(Pvec[0]))
        # feed_dict_test = {loc_input: np.array(set_location_user_test),
        #                             channel_bs_irs_user: A_T_real,
        #                             channel_bs_user : Hd_real,
        #                             lay['P']: Pvec[0]}
        # mse_loss,phi_hat_test,theta_test= sess.run([loss,loc_hat,theta_list],feed_dict=feed_dict_test)
        theta_test = np.array(theta_test)
        print(theta_test.shape)
        theta_test = theta_test[:,0,0:N_ris] + 1j* theta_test[:,0,N_ris:2*N_ris]
        print(theta_test.shape)
        ## get radio map
        radio_map, theta_test = generate_radio_map(theta_test)
        

    #performance = np.mean(performance,axis=1)       

    

######### Plot the test result /params_unknownalpha_RIS_snr_'+ str(snr_const[0]))
plt.semilogy(snr_const, performance)        
plt.grid()
plt.xlabel('SNR (dB)')
plt.ylabel('Average MSE')
plt.show()
sio.savemat('./plot/interpret/interpret_RNN_1RIS_newcoordinateSISO_N_1_tau_6_snr_25.mat',dict(performance= performance,\
                                       snr_const=snr_const,N=N,N_ris = N_ris,epoch = n_epochs,delta_inv=delta_inv,\
                                       mean_true_alpha=mean_true_alpha,\
                                        theta_test = theta_test, \
                                        radio_map = radio_map, \
                                        loc_true = location_user_target,\
                                       std_per_dim_alpha=std_per_dim_alpha,\
                                       noiseSTD_per_dim=noiseSTD_per_dim, tau=tau))

