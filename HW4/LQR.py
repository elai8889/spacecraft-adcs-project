import numpy as np
from scipy.linalg import solve_discrete_are

# J matrix from HW2

J11 = 0.0086
J22 = 0.022
J33 = 0.0306
J = np.array([[J33, 0, 0],
              [0, J11, 0],
              [0, 0, J22]])
J_inv = np.linalg.inv(J)

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

# H: maps a rotation/angular velocity into a quaternion change in R4

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

# Need to define an E matrix, which maps from 7D quat matrix to 6D matrix with phi = attitude error

def E(q):
    return np.block([
            [G, np.zeros((4, 3))],
            [np.zeros((3, 3)), np.eye(3)]
            ])

def A_full(q, omega, J):
    omega_quat = np.hstack((0.0, omega)) # Omega quaternion

    A_full = np.zeros((7, 7))

    A_full[0:4, 0:4] = 0.5 * R(omega_quat)

    A_full[0:4, 4:7] = 0.5 * L(q) @ H()

    h = J @ omega

    A_full[4:7, 4:7] = J_inv @ (hat(h) - hat(omega) @ J)

    return A_full

def B_full(J):
   
    B_full = np.zeros((7, 3))
    B_full[4:7, :] = J_inv

    return B_full

def reduced_AB(q, omega, J, dt):
    """
    Reduces the full quat matrices into LQR matrices
    Reduced state is has an attitude error instead of a quaternion
    Returns LQR matrices in discrete time
    """

    A_c = E.T @ A_full @ E # Evolve state using full state dynamics and then project it back into 6D
    B_c = E.T @ B_full

def dlqr(A, B, Q, R):
    S = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(B.T @ S @ B + R) @ (B.T @ S @ A)

    return K



# def reduced_linearization(q, omega, J, dt):


if __name__ == "__main__":

    # Desired regulation point:
    # identity attitude, zero angular velocity

    q_des = np.array([1.0, 0.0, 0.0, 0.0])
    omega_des = np.array([0.0, 0.0, 0.0])

    dt = 0.1

    A, B = reduced_linearization(q_des, omega_des, J, dt)

    Q = np.diag([
        100, 100, 100,   # attitude error phi
        10, 10, 10       # angular velocity error
    ])

    R = np.diag([
        1, 1, 1          # control torque
    ])

    K = dlqr(A, B, Q, R)

    print("A reduced shape:", A.shape)
    print("B reduced shape:", B.shape)
    print("K shape:", K.shape)
    print("K =\n", K)
# A and B matrices; n = number of states, m = number of control inputs

# State has 6 inputs (attitude error + angular velocity)
# Control has 3 inputs (torque)
