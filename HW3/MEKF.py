import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt


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


# Dynamics

J = np.diag([1.0, 1.25, 1.5])

def dynamics(x): # Gets rate of change of state at some state
    q = x[0:4]
    q = q / np.linalg.norm(q)

    omega = x[4:7]

    qdot = 0.5 * G(q) @ omega
    omegadot = -np.linalg.solve(J, hat(omega) @ (J @ omega))

    xdot = np.concatenate((qdot, omegadot))
    return xdot

def rkstep(x, h):
    f1 = dynamics(x)
    f2 = dynamics(x + 0.5 * h * f1)
    f3 = dynamics(x + 0.5 * h * f2)
    f4 = dynamics(x + h * f3)

    xn = x + (h / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
    xn[:4] = xn[:4] / np.linalg.norm(xn[:4])
    return xn


# Setup sim

h = 0.1
n = 600
tf = n * h

# Random initial attitude
q0 = np.random.randn(4)
q0 = q0 / np.linalg.norm(q0)

# Random angular velocity
omega0 = 0.1 * np.random.randn(3)

# Truth state = [q; omega]
x0 = np.concatenate((q0, omega0))

xtraj = np.zeros((7, n)) # This will be populated as you go through the sim
xtraj[:, 0] = x0 # First column is the initial state

for k in range(n - 1):
    xtraj[:, k + 1] = rkstep(xtraj[:, k], h) # Populating the true state trajectory using RK4; basically just numerically integrating


# Noise & measurement setup

m = 3  # number of inertial reference vectors

# W = 0.01 * np.eye(3 * m) # Measurement noise
W = np.diag([1.78e-6, 1.78e-6, 1.78e-6, 3.38e-5, 3.38e-5, 3.38e-5, 3.05e-6, 3.05e-6, 3.05e-6])
V_gyro = 2.35e-9 * np.eye(3) # Gyro measurement noise covariance; you choose this. In reality you want this to be close to your sensor covariance
V_filter = 0.01 * np.eye(6) # Uncertainty in how your true state evolves. Block matrix; top left is V_theta, bottom right is V_bias

# Gyro bias
beta0 = np.random.randn(3)

# Generate gyro measurements (truth omega + constant bias + noise)
gyro = np.zeros((3, n))
Lgyro = np.linalg.cholesky(V_gyro)

for k in range(n):
    gyro[:, k] = xtraj[4:7, k] + beta0 + Lgyro @ np.random.randn(3) # Add gyro bias

# Random inertial vectors (known directions in inertial frame). They're random for the sim because you just need something to anchor to, but IRL they are something like the direction of magnetic north.
r_N = np.zeros((3, m)) # All entries are zero
for ell in range(m):
    v = np.random.randn(3) # Generate random inertial vector
    r_N[:, ell] = v / np.linalg.norm(v) # Populate each row of a certain column at ell with the norm of the random inertial vector

# Generate noisy vector measurements
ytraj = np.zeros((3 * m, n)) # Each column is all the inertial vectors expressed in the body frame; as the spacecraft rotates over the time steps, these values will change
Lw = np.linalg.cholesky(W)

for k in range(n):
    Qk = Q(xtraj[0:4, k]) # State quaternion
    yk = np.zeros((3, m)) # Each inertial vector is noisily measured per timestep 

    w = (Lw @ np.random.randn(3 * m)).reshape(3, m) # Generate noise with covariance Lw

    for ell in range(m):
        Qw = expm(hat(w[:, ell])) # Turn noise into small rotation matrix
        yk[:, ell] = Qw.T @ Qk.T @ r_N[:, ell] # Perturb true direction

    ytraj[:, k] = yk.reshape(3 * m, order='F') # Stores all the noisy measured vectors at time step k into column k of ytraj


# MEKF with Bias

def state_prediction(x, u, h):
    """
    x = [q; beta]  (7,)
    u = gyro measurement (3,)
    returns predicted [q_next; beta_next]
    """
    q = x[0:4]
    beta = x[4:7]

    omega_hat = u - beta
    dq = expq(0.5 * h * omega_hat)

    q_next = L(q) @ dq
    q_next = q_next / np.linalg.norm(q_next)

    beta_next = beta.copy()

    return np.concatenate((q_next, beta_next))


def state_prediction_deriv(x, u, h):
    """
    Returns 6x6 linearized error-state transition matrix A
    for error state [delta_theta; delta_beta]
    """
    q = x[0:4]
    beta = x[4:7]

    omega_hat = u - beta
    dq = expq(0.5 * h * omega_hat)
    qn = L(q) @ dq
    qn = qn / np.linalg.norm(qn)

    Aqq = G(qn).T @ R(dq) @ G(q)

    Aqb = -0.5 * h * np.eye(3)

    Abq = np.zeros((3, 3))
    Abb = np.eye(3)

    A = np.block([
        [Aqq, Aqb],
        [Abq, Abb]
    ])

    return A


def measurement_prediction(x, r_N):
    """
    x = [q; beta]
    returns stacked vector measurements of shape (3m,)
    """
    q = x[0:4]

    Qk = Q(q)
    y = Qk.T @ r_N

    return y.reshape(3 * r_N.shape[1], order='F')


def measurement_prediction_deriv(x, r_N):
    """
    Returns C with shape (3m, 6)
    for error state [delta_theta; delta_beta]
    """
    q = x[0:4]
    m_local = r_N.shape[1]

    C = np.zeros((3 * m_local, 6))

    for k in range(m_local):
        rk_quat = H @ r_N[:, k]

        C[3*k:3*k+3, 0:3] = (
            H.T
            @ (
                L(q).T @ L(rk_quat)
                + R(q) @ R(rk_quat) @ T
            )
            @ G(q)
        )

    return C

# Initialize MEKF 

xfilt = np.zeros((7, n))
xfilt[0:4, 0] = q0 + 0.1 * np.random.randn(4)
xfilt[0:4, 0] = xfilt[0:4, 0] / np.linalg.norm(xfilt[0:4, 0])

xfilt[4:7, 0] = np.zeros(3)

P = np.zeros((6, 6, n))
P[:, :, 0] = 0.5 * np.eye(6)

for k in range(n - 1):
    # Prediction
    xpred = state_prediction(xfilt[:, k], gyro[:, k], h)
    A = state_prediction_deriv(xfilt[:, k], gyro[:, k], h)
    Ppred = A @ P[:, :, k] @ A.T + V_filter

    # Innovation
    z = ytraj[:, k + 1] - measurement_prediction(xpred, r_N)
    C = measurement_prediction_deriv(xpred, r_N)
    S = C @ Ppred @ C.T + W


    K = Ppred @ C.T @ np.linalg.inv(S)

    dx = K @ z


    phi = dx[0:3]
    dbeta = dx[3:6]

    phi_norm_sq = phi.T @ phi
    dq = np.concatenate(([np.sqrt(max(0.0, 1.0 - phi_norm_sq))], phi))

    q_upd = L(xpred[0:4]) @ dq
    q_upd = q_upd / np.linalg.norm(q_upd)

    beta_upd = xpred[4:7] + dbeta

    xfilt[:, k + 1] = np.concatenate((q_upd, beta_upd))

    # Covariance update
    I6 = np.eye(6)
    P[:, :, k + 1] = (I6 - K @ C) @ Ppred @ (I6 - K @ C).T + K @ W @ K.T


# Plot

for i in range(4):
    plt.figure()
    plt.plot(xfilt[i, :], label='xfilt')
    plt.plot(xtraj[i, :], label='xtraj')
    plt.title(f'Quaternion component q{i}')
    plt.xlabel('Time step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

for i in range(3):
    plt.figure()
    plt.plot(xfilt[4 + i, :], label='estimated bias')
    plt.plot(beta0[i] * np.ones(n), label='true bias')
    plt.title(f'Gyro bias component beta{i}')
    plt.xlabel('Time step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()