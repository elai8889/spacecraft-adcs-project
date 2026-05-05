import numpy as np
from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from HW3 import MEKF
from HW4 import environmental_torques as env

# J matrix from HW2

J11 = 0.0086
J22 = 0.022
J33 = 0.0306
J = np.array([[J33, 0, 0],
              [0, J11, 0],
              [0, 0, J22]])
J_inv = np.linalg.inv(J)

# Hat matrix: whatv = w x v

def hat(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def unhat(S):
    return 0.5 * np.array([
        S[2, 1] - S[1, 2],
        S[0, 2] - S[2, 0],
        S[1, 0] - S[0, 1]
    ])

def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_mult(q, p):
    return L(q) @ p

# H: maps a rotation/angular velocity into a quaternion change in R4

def H():
    return np.vstack((np.zeros((1, 3)), np.eye(3)))
# T is the quaternion conjugation matrix

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

def G_matrix(q): # Relates qdot to angular velocity omega
    return L(q) @ H()

def Q(q):
    return H.T @ R(q).T @ L(q) @ H()

def quat_normalize(q):
    return q / np.linalg.norm(q)

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

# Need to define an E matrix, which maps from 7D quat matrix to 6D matrix with phi = attitude error

def E_matrix(q):
    G = G_matrix(q)

    return np.block([
        [G, np.zeros((4, 3))],
        [np.zeros((3, 3)), np.eye(3)]
    ])

def A_full(q, omega, J):

    omega_quat = np.hstack((0.0, omega))

    A_full = np.zeros((7, 7))

    # dqdot/dq
    A_full[0:4, 0:4] = 0.5 * R(omega_quat)

    # dqdot/domega
    A_full[0:4, 4:7] = 0.5 * L(q) @ H()

    # domega_dot/domega
    h = J @ omega
    A_full[4:7, 4:7] = J_inv @ (hat(h) - hat(omega) @ J)

    return A_full

def B_full(J):
   
    B_full = np.zeros((7, 3))

    # torque directly affects angular acceleration only
    B_full[4:7, :] = J_inv

    return B_full

def reduced_AB(q, omega, J, dt):
    """
    Reduces the full quat matrices into LQR matrices
    Reduced state has an attitude error instead of a quaternion
    Returns LQR matrices in discrete time
    """

    # Call E matrix
    E_mat = E_matrix(q)

    # Call full state matrices
    A_full_mat = A_full(q, omega, J)
    B_full_mat = B_full(J)

    # Reduce A_full and B_full by solving for dx_full and dxdot_full in terms of dxdot_reduced and dx_reduced and plugging into full state dynamics
    A_c = E_mat.T @ A_full_mat @ E_mat
    B_c = E_mat.T @ B_full_mat

    # Forward Euler/discretization: x_k+1 = x_k + xdot_k * dt; substitute reduced dynamics for xdot_k and solve for A_d and B_d
    A_d = np.eye(6) + A_c * dt
    B_d = B_c * dt

    return A_d, B_d

def finite_horizon_lqr(A_list, B_list, Q, Rmat, Qf):
    """
    Computes finite-horizon discrete LQR gains.

    Returns:
        K_list: list of feedback gains K_k
        S_list: list of Riccati matrices S_k

    You don't know how bad your final state will be, but you can define a benchmark of "bad" that you want to 
    stay below (Qf), and work backwards to compute the gains necessary to ensure you don't go above this
    """

    N = len(A_list) # Number of time steps

    S_list = [None] * (N + 1) # Stores Riccati cost to go matrices
    K_list = [None] * N # Stores all the gains

    S_list[N] = Qf

    for k in reversed(range(N)):
        A = A_list[k] # Discrete reduced A at current time step
        B = B_list[k] # Discrete reduced B at current time step
        S_next = S_list[k + 1] # Get next cost go matrix, which determines how expensive future state will be.
        # Cost to go = how much error we expect to accumulate if control proceeds optimally until the end.

        K = np.linalg.solve( # Compute K using formula from notes (in .solve form)
            Rmat + B.T @ S_next @ B,
            B.T @ S_next @ A
        )

        S = (
            Q
            + K.T @ Rmat @ K
            + (A - B @ K).T @ S_next @ (A - B @ K)
        )

        K_list[k] = K # Store gain from this time step
        S_list[k] = S # Store Riccati matrix from this time step

    return K_list, S_list

def attitude_error_phi(q_des, q_hat):
    """
    Returns attitude error phi
    q_hat = estimated orientation
    q_des = desired orientation
    """
    q_des = quat_normalize(q_des)
    q_hat = quat_normalize(q_hat)

    dq = quat_mult(quat_conj(q_des), q_hat) # Quaternion error

    # Choose shortest rotation (negative scalar component corresponds to larger rotation than 360 - this larger rotation)
    if dq[0] < 0:
        dq = -dq 

    phi = 2.0 * dq[1:4] # Take definition of quaternion and phi = theta * uhat to solve for vector part of quat (dq[1:4])

    return phi

def lqr_control(q_des, q_hat, omega_hat, K):
    phi = attitude_error_phi(q_des, q_hat)
    x_err = np.hstack((phi, omega_hat))
    u = -K @ x_err
    return u, x_err