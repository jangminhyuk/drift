'''
    ESKF class
'''
import numpy as np
from scipy import linalg
import math
from pyquaternion import Quaternion

# ------------------ parameters ------------------ #
# Process noise
w_accxyz = 2.0;      w_gyro_rpy = 0.1    # rad/sec
w_vel = 0;           w_pos = 0;          w_att = 0;        
# Constants
GRAVITY_MAGNITUDE = 9.81
DEG_TO_RAD  = math.pi/180.0
e3 = np.array([0, 0, 1]).reshape(-1,1)   

class ESKF:
    '''initialization'''
    def __init__(self, X0, q0, P0, K, mu_w=None, mu_v=0.0):
        # Standard devirations of UWB meas. (tuning parameter)
        self.std_uwb_tdoa = np.sqrt(0.05)
        # external calibration: translation vector from the quadcopter to UWB tag
        self.t_uv = np.array([-0.01245, 0.00127, 0.0908]).reshape(-1,1)

        # Nominal noise means (default: zero-mean)
        # mu_w: process noise mean [acc_bias(3), gyro_bias(3)] in body frame
        #       acc_bias in m/s^2, gyro_bias in rad/s
        # mu_v: measurement noise mean (scalar, e.g. NLOS bias in meters)
        self.mu_w = np.zeros(6) if mu_w is None else np.asarray(mu_w, dtype=float)
        self.mu_v = float(mu_v)

        self.nis_log = []  # list of (timestep_index, nis_value, dof)
        self._last_dr_diagnostics = None  # populated by DR subclasses; adapter reads this
        self._last_innov_diagnostics = None  # nominal innovation stats; classical adapters read this

        self.f = np.zeros((K, 3))
        self.omega = np.zeros((K,3))
        self.q_list = np.zeros((K,4))    # quaternion list
        self.R_list = np.zeros((K,3,3))  # Rotation matrix list (from body frame to inertial frame) 

        # nominal-state X = [x, y, z, vx, vy, vz]
        self.Xpr = np.zeros((K,6))
        self.Xpo = np.zeros((K,6))
        self.Ppr = np.zeros((K, 9, 9))
        self.Ppo = np.zeros((K, 9, 9))

        self.Ppr[0] = P0
        self.Ppo[0] = P0
        self.Xpr[0] = X0.T
        self.Xpo[0] = X0.T
        self.q_list[0,:] = np.array([q0.w, q0.x, q0.y, q0.z])
        # current rotation matrix list (from body frame to inertial frame) 
        self.R = q0.rotation_matrix

        # Optional Q_base override: if set by MUSE adapter, replaces default Qi
        # Must be a (6,6) numpy array. When None, uses default w_accxyz/w_gyro_rpy.
        self.Q_base = None

        # # Process noise
        self.Fi = np.block([
            [np.zeros((3,3)),   np.zeros((3,3))],
            [np.eye(3),         np.zeros((3,3))],
            [np.zeros((3,3)),   np.eye(3)      ]
        ])

    '''ESKF prediction using IMU'''
    def predict(self, imu, dt, imu_check, k):
        # construct noise
        if self.Q_base is not None:
            Qi = self.Q_base * dt**2
        else:
            Vi = (w_accxyz**2)*(dt**2)*np.eye(3)
            Thetai = (w_gyro_rpy**2)*(dt**2)*np.eye(3)
            Qi = np.block([
                [Vi,               np.zeros((3,3)) ],
                [np.zeros((3,3)),  Thetai          ]
            ])
        
        if imu_check:
            # We have a new IMU measurement
            # update the prior Xpr based on accelerometer and gyroscope data
            omega_k = imu[3:] * DEG_TO_RAD + self.mu_w[3:6]  # gyro + bias
            self.omega[k] = omega_k
            Vpo = self.Xpo[k-1,3:6]
            # Acc: G --> m/s^2
            f_k = imu[0:3] * GRAVITY_MAGNITUDE + self.mu_w[0:3]  # acc + bias
            self.f[k] = f_k
            dw = omega_k * dt                      # Attitude error
            # nominal state motion model
            # position prediction 
            self.Xpr[k,0:3] = self.Xpo[k-1, 0:3] + Vpo.T*dt + 0.5 * np.squeeze(self.R.dot(f_k.reshape(-1,1)) - GRAVITY_MAGNITUDE*e3) * dt**2
            # velocity prediction
            self.Xpr[k,3:6] = self.Xpo[k-1, 3:6] + np.squeeze(self.R.dot(f_k.reshape(-1,1)) - GRAVITY_MAGNITUDE*e3) * dt
            # if CF is on the ground
            if self.Xpr[k, 2] < 0:  
                self.Xpr[k, 2:6] = np.zeros((1,4))    
            # quaternion update
            qk_1 = Quaternion(self.q_list[k-1,:])
            dqk  = Quaternion(self.zeta(dw))           # convert incremental rotation vector to quaternion
            q_pr = qk_1 * dqk                          # compute quaternion multiplication with package
            self.q_list[k,:] = np.array([q_pr.w, q_pr.x, q_pr.y, q_pr.z])  # save quaternion in q_list
            self.R_list[k]   = q_pr.rotation_matrix                        # save rotation prediction to R_list
            # error state covariance matrix 
            # use the rotation matrix from timestep k-1
            self.R = qk_1.rotation_matrix          
            # Jacobian matrix
            Fx = np.block([
                [np.eye(3),         dt*np.eye(3),      -0.5*dt**2*self.R.dot(self.cross(f_k))],
                [np.zeros((3,3)),   np.eye(3),         -dt*self.R.dot(self.cross(f_k))       ],
                [np.zeros((3,3)),   np.zeros((3,3)),   linalg.expm(self.cross(dw)).T    ]            
            ])
            # Process noise matrix Fi, Qi are defined above
            self.Ppr[k] = Fx.dot(self.Ppo[k-1]).dot(Fx.T) + self.Fi.dot(Qi).dot(self.Fi.T) 
            # Enforce symmetry
            self.Ppr[k] = 0.5*(self.Ppr[k] + self.Ppr[k].T)  

        else:
            # if we don't have IMU data
            self.Ppr[k] = self.Ppo[k-1] + self.Fi.dot(Qi).dot(self.Fi.T)
            # Enforce symmetry
            self.Ppr[k] = 0.5*(self.Ppr[k] + self.Ppr[k].T)  
            
            self.omega[k] = self.omega[k-1]
            self.f[k] = self.f[k-1]
            dw = self.omega[k] * dt                      # Attitude error
            # nominal state motion model
            # position prediction 
            Vpo = self.Xpo[k-1,3:6]
            self.Xpr[k,0:3] = self.Xpo[k-1, 0:3] + Vpo.T*dt + 0.5 * np.squeeze(self.R.dot(self.f[k].reshape(-1,1)) - GRAVITY_MAGNITUDE*e3) * dt**2
            # velocity prediction
            self.Xpr[k,3:6] = self.Xpo[k-1, 3:6] + np.squeeze(self.R.dot(self.f[k].reshape(-1,1)) - GRAVITY_MAGNITUDE*e3) * dt
            # if CF is on the ground
            # if Xpr[k, 2] < 0:  
            #     Xpr[k, 2:6] = np.zeros((1,4))    
            # quaternion update
            qk_1 = Quaternion(self.q_list[k-1,:])
            dqk  = Quaternion(self.zeta(dw))       # convert incremental rotation vector to quaternion
            q_pr = qk_1 * dqk                 # compute quaternion multiplication with package
            self.q_list[k] = np.array([q_pr.w, q_pr.x, q_pr.y, q_pr.z])    # save quaternion in q_list
            self.R_list[k]   = q_pr.rotation_matrix                        # save rotation prediction to R_list
        
        # End of Prediction

        # Initially take our posterior estimates as the prior estimates
        # These are updated if we have sensor measurements (UWB)
        self.Xpo[k] = self.Xpr[k]
        self.Ppo[k] = self.Ppr[k]

    '''ESKF correction using UWB'''
    def UWB_correct(self, uwb, anchor_position, k):
        an_A = anchor_position[int(uwb[0]),:]   # idA
        an_B = anchor_position[int(uwb[1]),:]   # idB

        qk_pr = Quaternion(self.q_list[k])
        C_iv = qk_pr.rotation_matrix

        # measurement model: L2 norm between anchor and uwb tag position 
        p_uwb = C_iv.dot(self.t_uv) + self.Xpr[k,0:3].reshape(-1,1)    # uwb tag position            
        d_A = linalg.norm(an_A - np.squeeze(p_uwb)) 
        d_B = linalg.norm(an_B - np.squeeze(p_uwb))
        predicted = d_B - d_A
        err_uwb = uwb[2] - predicted - self.mu_v

        # compute the gradient of measurement model
        # G is 1 x 9
        G = self.computeG_grad(an_A, an_B, self.t_uv, self.Xpr[k,0:3], self.q_list[k])

        # uwb covariance
        Q = self.std_uwb_tdoa**2
        M = np.squeeze(G.dot(self.Ppr[k]).dot(G.T) + Q)     # scalar
        d_m = math.sqrt(err_uwb**2/M)

        # Nominal innovation diagnostics for classical adaptive estimators
        # (kept separate from _last_dr_diagnostics, which MUSE consumes)
        self._last_innov_diagnostics = {
            'err_uwb': float(err_uwb),   # innovation after mu_v removal
            'S_nom': float(M),           # nominal innovation variance
            'R_used': float(Q),          # injected measurement variance
            'accepted': bool(d_m < 5),
        }

        # -------------------- Statistical Validation -------------------- #
        if d_m < 5:
            # NIS = err^2 / S  where S = M (innovation covariance), dof = 1 (scalar)
            nis_value = float(err_uwb**2 / M)
            self.nis_log.append((k, nis_value, 1))
            # Kk is 9 x 1
            Kk = (self.Ppr[k].dot(G.T) / M).reshape(-1,1)           # in scalar case
            # update the posterios covariance matrix for error states
            self.Ppo[k]= (np.eye(9) - Kk.dot(G.reshape(1,-1))).dot(self.Ppr[k])
            # enforce symmetry
            self.Ppo[k] = 0.5 * (self.Ppo[k] + self.Ppo[k].T)
            derror = Kk.dot(err_uwb)             
            # update nominal states 
            self.Xpo[k] = self.Xpr[k] +  np.squeeze(derror[0:6])
            dq_k = Quaternion(self.zeta(np.squeeze(derror[6:])))
            #update quaternion: q_list
            qk_po = qk_pr * dq_k
            self.q_list[k] = np.array([qk_po.w, qk_po.x, qk_po.y, qk_po.z])
        else:
            # keep the previous state
            self.Xpo[k] = self.Xpr[k]
            self.Ppo[k] = self.Ppr[k]
            # keep the previous quaterion  q_list[k]

    '''compute gradient of meas.model'''
    def computeG_grad(self, an_A, an_B, t_uv, Xpr, q_k):
        # compute the gradient considering the lever-arm effect
        qk_pr = Quaternion(q_k)
        C_iv = qk_pr.rotation_matrix
        # uwb tag position
        p_uwb = C_iv.dot(t_uv) + Xpr.reshape(-1,1)
        d_A = linalg.norm(np.squeeze(p_uwb) - an_A)
        d_B = linalg.norm(np.squeeze(p_uwb) - an_B)
        g_p = ((np.squeeze(p_uwb) - an_B)/d_B).reshape(1,-1) - ((np.squeeze(p_uwb) - an_A)/d_A).reshape(1,-1)
        g_v = np.zeros((1,3))

        # q_k = [q_w, q_x, q_y, q_z] = [q_w, q_v]
        q_w = q_k[0];  q_v = np.array([ q_k[1], q_k[2], q_k[3] ])
        
        # d_RVq = 2[q_w t_uv + q_v x t_uv, q_v^T t_uv I(3) + q_v t_uv^T - t_uv q_v - q_w[t_uv]x]
        d_vec = q_w*t_uv + self.cross(q_v).dot(t_uv).reshape(-1,1)   # 3 x 1 vector
        d_mat = q_v.reshape(1,-1).dot(t_uv) * np.eye(3) + q_v.reshape(-1,1).dot(np.transpose(t_uv)) - t_uv.dot(q_v.reshape(1,-1)) - q_w * self.cross(t_uv)

        d_RVq = 2*np.concatenate((d_vec, d_mat), axis=1)

        g_q = ((np.squeeze(p_uwb) - an_B)/d_B).reshape(1,-1).dot(d_RVq) - ((np.squeeze(p_uwb) - an_A)/d_A).reshape(1,-1).dot(d_RVq)
        G_x = np.concatenate((g_p, g_v, g_q), axis=1)

        Q_dtheta = 0.5*np.array([
            [-q_k[1], -q_k[2], -q_k[3]],
            [ q_w,    -q_k[3],  q_k[2]],
            [ q_k[3],  q_w,    -q_k[1]],
            [-q_k[2],  q_k[1],  q_w]
        ])
        G_dx = linalg.block_diag(np.eye(6), Q_dtheta)
        G = G_x.dot(G_dx)
        return G
    
    '''help function'''
    def cross(self, v):    # input: 3x1 vector, output: 3x3 matrix
        v = np.squeeze(v)
        vx = np.array([
            [ 0,    -v[2], v[1]],
            [ v[2],  0,   -v[0]],
            [-v[1],  v[0], 0 ] 
        ])
        return vx
        
    '''help function'''
    def zeta(self, phi):
        phi_norm = np.linalg.norm(phi)
        if phi_norm == 0:
            dq = np.array([1, 0, 0, 0])
        else:
            dq_xyz = (phi*(math.sin(0.5*phi_norm)))/phi_norm
            dq = np.array([math.cos(0.5*phi_norm), dq_xyz[0], dq_xyz[1], dq_xyz[2]])
        return dq