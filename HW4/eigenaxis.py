import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

savefig = True

combined_torque = np.load("torque.npy")

# Quaternion functions

def hat(v):
    return np.array([
        [0,    -v[2],  v[1]],
        [v[2],  0,    -v[0]],
        [-v[1], v[0],  0]
    ])

def unhat(S):
    return 0.5 * np.array([
        S[2, 1] - S[1, 2],
        S[0, 2] - S[2, 0],
        S[1, 0] - S[0, 1]
    ])

H = np.vstack((np.zeros((1, 3)), np.eye(3)))

T = np.block([
    [np.array([[1.0]]), np.zeros((1, 3))],
    [np.zeros((3, 1)), -np.eye(3)]
])

def L(q):
    q0 = q[0]
    qv = q[1:4]
    return np.block([
        [np.array([[q0]]), -qv.reshape(1, 3)],
        [qv.reshape(3, 1), q0 * np.eye(3) + hat(qv)]
    ])

def R(q):
    q0 = q[0]
    qv = q[1:4]
    return np.block([
        [np.array([[q0]]), -qv.reshape(1, 3)],
        [qv.reshape(3, 1), q0 * np.eye(3) - hat(qv)]
    ])

def G(q): # Relates qdot to angular velocity omega
    return L(q) @ H

def Q(q):
    return H.T @ R(q).T @ L(q) @ H

def expq(phi):
    # Maps a rotation vector phi in R^3 to a unit quaternion
    theta = np.linalg.norm(phi)

    if theta < 1e-8:
        return np.concatenate(([1.0], phi))

    return np.concatenate(([np.cos(theta)], phi * (np.sin(theta) / theta)))

def logq(q):
    c = q[0]
    s = np.linalg.norm(q[1:])
    theta = np.atan2(s, c)
    return q[1:] / np.sinc(theta/np.pi)

# Dynamics

J = np.diag([0.0306, 0.0086, 0.022])

def dynamics(x, u, tau): # Gets rate of change of state at some state
    q = x[0:4]
    q = q / np.linalg.norm(q)

    omega = x[4:7]
    b = x[7:10]

    qdot = 0.5 * G(q) @ omega
    omegadot = -np.linalg.inv(J) @ u + np.linalg.inv(J) @ tau
    bdot = np.zeros(3)

    xdot = np.concatenate((qdot, omegadot, bdot))
    return xdot

def state_prediction(x, u, h):
    """
    x = [q; beta]  (7,)
    u = gyro measurement (3,)
    returns predicted [q_next; beta_next]
    """
    q = x[0:4]

    omega_hat = u
    dq = expq(0.5 * h * omega_hat)

    q_next = L(q) @ dq
    q_next = q_next / np.linalg.norm(q_next)

    return q_next

def state_prediction_deriv(x, u, h):
    """
    Returns 6x6 linearized error-state transition matrix A
    for error state [delta_theta; delta_beta]
    """
    q = x[0:4]

    omega_hat = u
    dq = expq(0.5 * h * omega_hat)
    qn = L(q) @ dq
    qn = qn / np.linalg.norm(qn)

    Aqq = G(qn).T @ R(dq) @ G(q)

    Aqb = -0.5 * G(qn).T @ G(q)

    Abq = np.zeros((3, 3))
    Abb = np.eye(3)

    A = np.block([
        [Aqq, Aqb],
        [Abq, Abb]
    ])

    return Aqq

def rkstep(x, u, tau, h):
    f1 = dynamics(x, u, tau)
    f2 = dynamics(x + 0.5 * h * f1, u, tau)
    f3 = dynamics(x + 0.5 * h * f2, u, tau)
    f4 = dynamics(x + h * f3, u, tau)

    xn = x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
    xn[:4] = xn[:4] / np.linalg.norm(xn[:4])
    # xn[7:10] += np.random.multivariate_normal(np.zeros(3), W_gyro)
    return xn

t_sim = np.linspace(0, 45, 1000)

phi_initial = np.zeros(3)
qk = expq(phi_initial)
omegak = np.zeros(3)
bk = np.zeros(3)
xk = np.concatenate((qk, omegak, bk))
x_array = [xk]
r_act_array = []

## initialize filter
# q_filt = [0.99,0.01,np.sqrt(1-0.99**2-0.01**2),0]
q_filt = expq(phi_initial)
# x_filt = np.concatenate((q_filt, b_filt))
x_filt = np.array(q_filt)
x_filt_array = [x_filt]
P_filt = np.eye(3)
P_array = [P_filt]

### initialize controller
theta_final = np.pi
alpha = np.pi/40 # 40 seconds
def versine(t):
    return theta_final*1/2*(1-np.cos(alpha*t))

def get_qref(k):
    t = t_sim[k]
    theta = versine(t)
    phi_ref = theta * r_ref
    return expq(phi_ref)

r_ref = np.array([0,1,0])
w0_ref = [0,0,0]
def get_xref(k):
    return np.concatenate((get_qref(k), w0_ref))
x_ref = get_xref(0)
x_ref_array = [x_ref]
P_controller = 0.01
D_controller = 0.01
last_error = 0
u_array = []

### initialize sensors
m = 4  # number of inertial reference vectors
W_all = [np.diag([1.78e-6, 1.78e-6, 1.78e-6]), np.diag([3.38e-5, 3.38e-5, 3.38e-5]), np.diag([3.05e-6, 3.05e-6, 3.05e-6]), np.diag([3402.8, 136.1, 136.1]) / 60 / 60 * np.pi / 180]
rN = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1]), np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0])]
V_gyro = 2.35e-9 * np.eye(3) # Gyro measurement white noise covariance
W_gyro = 2.12e-10 * np.eye(3) # gyro random walk driving covariance
V_filter = np.block([[V_gyro, np.zeros((3,3))],
                     [np.zeros((3,3)), V_gyro]]) # use gyro white noise as estimate of process noise

def get_gyro(xk):
    measurement = xk[4:7] + np.random.multivariate_normal(np.zeros(3), V_gyro)
    return measurement


def get_control(xfilt, xref, gyro_meas, dt):
    q_ref = xref[0:4]
    q_filt = xfilt[0:4]
    q_error = L(q_ref).T @ q_filt
    phi = logq(q_error)
    u = P_controller * phi + D_controller * gyro_meas
    return u

for i in range(len(t_sim)-1):
    x_ref = get_xref(i)
    x_ref_array.append(x_ref)
    dt = t_sim[i+1] - t_sim[i]

    ### take gyro measurement
    gyro_meas = get_gyro(xk)

    ### control
    u = get_control(x_filt, x_ref, gyro_meas, dt)
    u_array.append(u)

    ### predict
    x_filt = state_prediction(x_filt, gyro_meas, dt)
    Apred = state_prediction_deriv(x_filt, gyro_meas, dt)
    P_filt = Apred @ P_filt @ Apred.T + V_gyro

    ### measure
    for j in range(m):
        ### generate measurement
        W = W_all[j]
        qk = xk[0:4]
        meas = Q(qk) @ rN[j] + np.random.multivariate_normal(np.zeros(3), W)

        ### predict
        q_filt = x_filt[0:4]
        meas_pred = Q(q_filt) @ rN[j]
        z_filt = meas - meas_pred
        # C = -H.T @ (L(q_filt).T @ L(H@rN[j]) + R(q_filt) @ R(H@rN[j]) @ T) @ G(q_filt)
        C = -2 * Q(q_filt) @ hat(rN[j]) # equivalent C matrix
        S = C @ P_filt @ C.T + W
        K = P_filt @ C.T @ np.linalg.inv(S)
        phi_filt = K @ z_filt
        phi = phi_filt[0:3]
        x_filt = L(x_filt[0:4]) @ np.array([np.sqrt(1-np.dot(phi, phi)), *phi])

        P_filt = (np.eye(3)-K@C) @ P_filt @ (np.eye(3)-K@C).T + K@W@K.T
    x_filt_array.append(x_filt)
    P_array.append(P_filt)

    ### propagate dynamics
    tau = combined_torque[i]
    xk = rkstep(xk, u, tau, dt)
    x_array.append(xk)
    
    ### compute pointing error
    r_act_array.append(Q(xk[0:4]) @ np.array([1,0,0]))

x_array = np.array(x_array)
x_filt_array = np.array(x_filt_array)
P_array = np.array(P_array)
u_array = np.array(u_array)
x_ref_array = np.array(x_ref_array)

### compute RMS pointing error
err_array = []
for i, r_act in enumerate(r_act_array):
    r_des = Q(x_ref_array[i,0:4]) @ np.array([1,0,0])
    err = np.acos(np.dot(r_des, r_act))
    err_array.append(err)
err_array = np.array(err_array)

theta_rms_rad = np.sqrt(np.mean(err_array**2))
theta_rms_deg = np.rad2deg(theta_rms_rad)

print(f"RMS pointing error: {theta_rms_rad:.6e} rad")
print(f"RMS pointing error: {theta_rms_deg:.6f} deg")


plt.figure()
plt.plot(t_sim, x_array[:, 0], label="q0")
plt.plot(t_sim, x_array[:, 1], label="q1")
plt.plot(t_sim, x_array[:, 2], label="q2")
plt.plot(t_sim, x_array[:, 3], label="q3")
plt.plot(t_sim, x_ref_array[:,0], linestyle="--", c="C0")
plt.plot(t_sim, x_ref_array[:,1], linestyle="--", c="C1")
plt.plot(t_sim, x_ref_array[:,2], linestyle="--", c="C2")
plt.plot(t_sim, x_ref_array[:,3], linestyle="--", c="C3")
plt.legend()
plt.title("Quaternion simulated components")
plt.xlabel("Time (s)")
if savefig:
    plt.savefig(f"figs/eigen-quat-comp.png", dpi=400)

plt.figure()
plt.plot(t_sim, np.rad2deg(x_array[:,4]), label="w1")
plt.plot(t_sim, np.rad2deg(x_array[:,5]), label="w2")
plt.plot(t_sim, np.rad2deg(x_array[:,6]), label="w3")
plt.legend()
plt.ylabel("Angular velocity (deg/s)")
plt.title("Angular velocity")
plt.xlabel("Time (s)")
if savefig:
    plt.savefig(f"figs/eigen-angular-velocity.png", dpi=400)

plt.figure()
plt.plot(t_sim[:-1], u_array[:, 0]*1e3, label="tau1")
plt.plot(t_sim[:-1], u_array[:, 1]*1e3, label="tau2")
plt.plot(t_sim[:-1], u_array[:, 2]*1e3, label="tau3")
plt.title("Control history")
plt.ylabel("Torque (mNm)")
plt.xlabel("Time (s)")
plt.legend()
plt.tight_layout()
if savefig:
    plt.savefig(f"figs/eigen-control-history.png", dpi=400)
plt.show()