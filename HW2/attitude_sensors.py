import numpy as np
import matplotlib.pyplot as plt
import scipy
from time import time

savefig = False

rng = np.random.default_rng()
t = np.linspace(0, 1, 1000)

### gyro ###
def get_gyro():
    return rng.multivariate_normal(np.zeros(3), 0.762*np.eye(3))

y_gyro = np.array([get_gyro() for _ in t])

plt.plot(t, y_gyro[:,0])
plt.plot(t, y_gyro[:,1])
plt.plot(t, y_gyro[:,2])
plt.ylabel("mrad/s")
plt.title("Gyroscope noise")
if savefig:
    plt.savefig("figs/gyro_noise.png", dpi=400)

print("Gyroscope Covariance")
print(np.cov(y_gyro.T))

### magnetometer ###
def get_mag():
    return rng.multivariate_normal(np.zeros(3), 277.8*np.eye(3))

y_mag = np.array([get_mag() for _ in t])

plt.figure()
plt.plot(t, y_mag[:,0])
plt.plot(t, y_mag[:,1])
plt.plot(t, y_mag[:,2])
plt.ylabel("nT")
plt.title("Magnetometer noise")
if savefig:
    plt.savefig("figs/mag_noise.png", dpi=400)

print("Magnetometer Covariance")
print(np.cov(y_mag.T))

### sun sensor ###
def get_sun():
    return rng.multivariate_normal(np.zeros(2), 3.05*np.eye(2))

y_sun = np.array([get_sun() for _ in t])

plt.figure()
plt.plot(t, y_sun[:,0])
plt.plot(t, y_sun[:,1])
plt.ylabel("mrad")
plt.title("Sun Sensor noise")
if savefig:
    plt.savefig("figs/sun_noise.png", dpi=400)

print("Sun Sensor Covariance")
print(np.cov(y_sun.T))

### earth horizon sensor ###
def get_earth():
    return rng.multivariate_normal(np.zeros(2), 33.8*np.eye(2))

y_earth = np.array([get_earth() for _ in t])

plt.figure()
plt.plot(t, y_earth[:,0])
plt.plot(t, y_earth[:,1])
plt.ylabel("mrad")
plt.title("Earth Horizon Sensor noise")
if savefig:
    plt.savefig("figs/earth_noise.png", dpi=400)

print("Earth Horizon Sensor Covariance")
print(np.cov(y_earth.T))


#### STATIC ATTITUDE ESTIMATION ####

def hat(v):
    v = v.reshape(3,)
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def L(q):
    s = q[0]
    v = q[1:].reshape(3,1)
    return np.block([[s, -v.T],
                    [v, s*np.eye(3)+hat(v)]])

def R(q):
    s = q[0]
    v = q[1:].reshape(3,1)
    return np.block([[s, -v.T],
                     [v, s*np.eye(3)-hat(v)]])

H = np.block([[np.zeros(3)],
              [np.eye(3)]])

def makeQ(q):
    return H.T @ L(q) @ R(q).T @ H

svd_time_arr = []
dav_time_arr = []
svd_error_arr = []
dav_error_arr = []

for _ in range(1000):

    w = np.array([3, 2, 1]) # weights
    M_true_B = np.array([30000, 0, 0]) # nT
    M_measured = M_true_B + get_mag()

    M_true_B = M_true_B / np.linalg.norm(M_true_B)
    M_measured = M_measured / np.linalg.norm(M_measured)

    S_true_B = np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0])
    S_error = get_sun()/1000
    S_measured = S_true_B + np.array([np.sin(S_error[0]), np.sin(S_error[1]), 0])
    S_measured = S_measured / np.linalg.norm(S_measured)

    E_true_B = np.array([0, 0, 1])
    E_error = get_earth()/1000
    E_measured = E_true_B + np.array([np.sin(E_error[0]), np.sin(E_error[1]), 0])
    E_measured = E_measured / np.linalg.norm(E_measured)
    
    phi = np.random.randn(3)
    Q = scipy.linalg.expm(hat(phi))

    M_true_N = Q @ M_true_B
    S_true_N = Q @ S_true_B
    E_true_N = Q @ E_true_B

    ### SVD Method ###
    svd_start = time()
    B = w[0]*np.outer(M_measured, M_true_N) + w[1]*np.outer(S_measured, S_true_N) + w[2]*np.outer(E_measured, E_true_N)
    U, S, Vh = np.linalg.svd(B)
    Q_est_svd = Vh.T @ U.T
    svd_time = time()-svd_start

    ### Davenport q Method ###
    dav_start = time()
    D = w[0]*L(H@M_true_N).T@R(H@M_measured) + w[1]*L(H@S_true_N).T@R(H@S_measured) + w[2]*L(H@E_true_N).T@R(H@E_measured)
    eigval, eigvec = np.linalg.eig(D)
    idx = np.argmax(eigval)
    q_est = eigvec[:, idx]
    dav_time = time()-dav_start
    Q_est_dav = makeQ(q_est)

    test_vec = np.array([1, 0, 0])
    inertial_est_svd = Q_est_svd @ test_vec
    inertial_est_dav = Q_est_dav @ test_vec
    inertial_true = Q @ test_vec

    svd_error = np.arccos(np.dot(inertial_est_svd, inertial_true) / (np.linalg.norm(inertial_est_svd)*np.linalg.norm(inertial_true)))
    dav_error = np.arccos(np.dot(inertial_est_dav, inertial_true) / (np.linalg.norm(inertial_est_dav)*np.linalg.norm(inertial_true)))

    svd_time_arr.append(svd_time)
    dav_time_arr.append(dav_time)
    svd_error_arr.append(np.rad2deg(svd_error))
    dav_error_arr.append(np.rad2deg(dav_error))

print("SVD METHOD")
print("Time")
print(f"  Mean: {np.mean(svd_time_arr)}")
print(f"  Variance: {np.var(svd_time_arr)}")
print("Error")
print(f"  Mean: {np.mean(svd_error_arr)}")
print(f"  Variance: {np.var(svd_error_arr)}")

print("DAVENPORT Q METHOD")
print("Time")
print(f"  Mean: {np.mean(dav_time_arr)}")
print(f"  Variance: {np.var(dav_time_arr)}")
print("Error")
print(f"  Mean: {np.mean(dav_error_arr)}")
print(f"  Variance: {np.var(dav_error_arr)}")

plt.figure()
plt.plot([i for i in range(len(svd_error_arr))], svd_error_arr)
plt.show()
