from operator import gt
import numpy as np
import matplotlib.pyplot as plt
import transformations as tr
import yaml
import math
from esekf import *
import pandas as pd

adj_vobose = 0
adj_add_pos_rot_noise = False
adj_show_plt = True

'''
IMU.frequency: 200
IMU.start_time_s: 0
IMU.end_time_s: 60

IMU.acc_noise_sigma: 0.019    # m/sqrt(s^3)     (continuous noise)
IMU.gyro_noise_sigma: 0.015   # rad/sqrt(s)     (continuous noise)

IMU.acc_bias_sigma: 0.0001    # m/sqrt(s^5)     (continuous bias)
IMU.gyro_bias_sigma: 2.0e-5   # rad/sqrt(s^3)   (continuous bias)

Gravity: [0, 0, -9.81]
'''

def load_imu_parameters():
    f = open('./data/params.yaml', 'r')
    yml = yaml.load(f.read())
    params = ImuParameters()
    params.frequency = yml['IMU.frequency']
    params.sigma_a_n = yml['IMU.acc_noise_sigma']  # m/sqrt(s^3)
    params.sigma_w_n = yml['IMU.gyro_noise_sigma']  # rad/sqrt(s)
    params.sigma_a_b = yml['IMU.acc_bias_sigma']     # m/sqrt(s^5)
    params.sigma_w_b = yml['IMU.gyro_bias_sigma']    # rad/sqrt(s^3)
    f.close()
    return params


def _load_dataset_org():
    imu_data = np.loadtxt('./data/imu_noise.txt')
    gt_data = np.loadtxt('./data/traj_gt.txt')
    #gt_data = np.delete(gt_data, (7,8,9), axis=1)

    # w: gyro a: acce
    pd_imu_data = pd.DataFrame(imu_data, columns=["ts", "w_x", "w_y", "w_z", "a_x", "a_y", "a_z"])
    print("\nPD_IMU_DATA")
    print(pd_imu_data)
    print(pd_imu_data.describe())
    if adj_show_plt:
        pd_imu_data[["w_x", "w_y", "w_z"]][0:100].plot()
        pd_imu_data[["a_x", "a_y", "a_z"]][0:100].plot()
        plt.show()

    pd_gt_data = pd.DataFrame(gt_data, columns=["ts", "pos_x", "pos_y", "pos_z", "qr_w", "qr_x", "qr_y", "qr_z", "v_x", "v_y", "v_z"])
    print("\nPD_GT_DATA")
    print(pd_gt_data)
    print(pd_gt_data.describe())

    return imu_data, gt_data    


def load_dataset():
    return _load_dataset_org()


def load_ekf_init_state_by_gt(gt_data):

    #if gt_data.shape[1] == 11: # ts, pos, quat, vel  1+3+4+3
    #    init_pqv = gt_data[0, 1:]  #pos, quat, vel
    #else:
    gt_data0 = gt_data[0]
    pos = gt_data0[1:4].tolist()
    quat = gt_data0[4:8].tolist()
    if gt_data.shape[1] == 11:
        vel = gt_data0[8:12].tolist()
    else:
        vel = [0., 0., 0.]
    pqv = [*pos, *quat, *vel]
    init_pqv = np.array(pqv)

    init_nominal_state = np.zeros((19,))
    init_nominal_state[:10] = init_pqv                      # init p, q, v
    init_nominal_state[10:13] = 0                           # init ba
    init_nominal_state[13:16] = 0                           # init bg
    init_nominal_state[16:19] = np.array([0, 0, -9.81])     # init g

    init_traj0 = gt_data[0, :8]
    return init_nominal_state, init_traj0    

def load_ekf_init_state_by_imu(imu_data):

    init_pqv = np.zeros((10))
    #init_pqv = gt_data[0, 1:]  #pos, quat, vel
    #pos
    init_pqv[0] = 0
    init_pqv[1] = 0
    init_pqv[2] = 0
    #qaut
    init_pqv[3] = 1
    init_pqv[4] = 0
    init_pqv[5] = 0
    init_pqv[6] = 0
    #vel
    init_pqv[7] = 0
    init_pqv[8] = 0
    init_pqv[9] = 0

    init_nominal_state = np.zeros((19,))
    init_nominal_state[:10] = init_pqv                      # init p, q, v
    init_nominal_state[10:13] = 0                           # init ba
    init_nominal_state[13:16] = 0                           # init bg
    init_nominal_state[16:19] = np.array([0, 0, -9.81])     # init g

    init_traj0 = np.zeros((8))
    init_traj0 = imu_data[0][0]
    init_traj0[0] = 0
    init_traj0[1] = 0
    init_traj0[2] = 0
    #qaut
    init_traj0[3] = 1
    init_traj0[4] = 0
    init_traj0[5] = 0
    init_traj0[6] = 0

    return init_nominal_state, init_traj0    

def save_traj(traj_gt, traj_est):
    pd_traj_gt = pd.DataFrame(traj_gt, columns=["ts", "pos_x", "pos_y", "pos_z", "r_w", "r_x", "r_y", "r_z"])
    print("\nPD_TRAJ_GT")
    print(pd_traj_gt)
    print(pd_traj_gt.describe())
    np.savetxt('./data/traj_gt_out.txt', traj_gt)

    pd_traj_est = pd.DataFrame(traj_est, columns=["ts", "pos_x", "pos_y", "pos_z", "r_w", "r_x", "r_y", "r_z"])
    print("\nPD_TRAJ_EST")
    print(pd_traj_est)
    print(pd_traj_est.describe())
    np.savetxt('./data/traj_esekf_out.txt', traj_est)

def get_data_in_duration(imu_data, gt_data, test_duration_s=None):
    if test_duration_s is None:
        test_duration_s = [0., 61.]
    start_time = imu_data[0, 0]
    mask_imu = np.logical_and(imu_data[:, 0] <= start_time + test_duration_s[1],
                              imu_data[:, 0] >= start_time + test_duration_s[0])
    mask_gt = np.logical_and(gt_data[:, 0] <= start_time + test_duration_s[1],
                             gt_data[:, 0] >= start_time + test_duration_s[0])

    imu_data = imu_data[mask_imu, :]
    gt_data = gt_data[mask_gt, :]
    return imu_data, gt_data

def align_estimator_by_gt(estimator, timestamp, i, gt_data, sigma_measurement, sigma_measurement_p, sigma_measurement_q):
    # we assume the timestamps are aligned.
    assert math.isclose(gt_data[i, 0], timestamp)
    gt_pose = gt_data[i, 1:8].copy()  # gt_pose = [p, q]

    if adj_add_pos_rot_noise:
        # add position noise
        gt_pose[:3] += np.random.randn(3,) * sigma_measurement_p
        
        # add rotation noise, u = [1, 0.5 * noise_angle_axis]
        # u = 0.5 * np.random.randn(4,) * sigma_measurement_q
        # u[0] = 1
        u = np.random.randn(3, ) * sigma_measurement_q
        qn = tr.quaternion_about_axis(la.norm(u), u / la.norm(u))
        gt_pose[3:] = tr.quaternion_multiply(gt_pose[3:], qn)

    # Ethan: this is tough for our case, as the only measurement we have is the IMU, 
    # no any other ground truth data for Kalman filter to realize there are error to correct
    # update filter by measurement.
    estimator.update(gt_pose, sigma_measurement)

def get_new_pose_from_estimator_state(estimator, timestamp):
    if adj_vobose >= 1:
        print('[%f]:' % timestamp, estimator.nominal_state)    

    frame_pose = np.zeros(8,)
    frame_pose[0] = timestamp
    frame_pose[1:] = estimator.nominal_state[:7]
    return frame_pose        

def get_sigma_parameters():
    sigma_measurement_p = 0.02   # in meters
    sigma_measurement_q = 0.015  # in rad
    sigma_measurement = np.eye(6)
    sigma_measurement[0:3, 0:3] *= sigma_measurement_p**2
    sigma_measurement[3:6, 3:6] *= sigma_measurement_q**2
    return sigma_measurement_p, sigma_measurement_q, sigma_measurement    

def main():

    imu_data, gt_data = load_dataset()

    imu_parameters = load_imu_parameters()
    init_nominal_state, init_traj0 = load_ekf_init_state_by_gt(gt_data)
    #init_nominal_state, init_traj0 = load_ekf_init_state_by_imu(imu_data)

    estimator = ESEKF(init_nominal_state, imu_parameters)

    # select data from duration 
    imu_data, gt_data = get_data_in_duration(imu_data, gt_data, test_duration_s=[0., 61.])
    print(imu_data.shape)
    print(gt_data.shape)

    traj_est = [init_traj0]  

    update_ratio = int(imu_data.shape[0]/20)    # control the frequency of ekf updating.
    #update_ratio = imu_data.shape[0] + 1
    print("update_ratio", update_ratio)

    sigma_measurement_p, sigma_measurement_q, sigma_measurement = get_sigma_parameters()
    for i in range(1, imu_data.shape[0]):
        timestamp = imu_data[i, 0]
        estimator.predict(imu_data[i, :])

        if i % update_ratio == 1:
            align_estimator_by_gt(estimator, timestamp, i, gt_data, sigma_measurement, sigma_measurement_p, sigma_measurement_q)

        traj_est.append(get_new_pose_from_estimator_state(estimator, timestamp))

    # save trajectory to TUM format
    traj_est = np.array(traj_est)
    traj_gt = gt_data[:, :8]
    save_traj(traj_gt, traj_est)


if __name__ == '__main__':
    main()
