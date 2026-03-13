import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

rng = np.random.default_rng(4)

### 1 ###
J11 = 0.0086
J22 = 0.022
J33 = 0.0306
J = np.array([[J11, 0, 0],
              [0, J22, 0],
              [0, 0, J33]])

# eigendecomposition
D = J
V = np.eye(3)

# perturb eigendecomposition
mean = np.zeros(3)
d_cov = np.array([[J11*0.01, 0, 0],
                  [0, J22*0.01, 0],
                  [0, 0, J33*0.01]])
d = rng.multivariate_normal(mean, d_cov).reshape(3,)
v = J11*0.1*rng.random((3,))

def hat(v):
    v = v.reshape(3,)
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

D_tilde = D + np.diag(d)
V_tilde = V @ expm(hat(v))

J_tilde = V_tilde @ D_tilde @ V_tilde.T
print("Perturbed inertia matrix:")
print(J_tilde)

### 2 ###
w = np.array([0, 0, 10*2*np.pi/60]).reshape(3,1) # rad/s

### 3 ###
ws = np.linalg.norm(w)
Js = (w/ws).T @ J_tilde @ (w/ws)
rhos = (1.2*J33-Js) * ws

rho = np.linalg.inv(np.hstack((w, hat(w).T)) @ np.vstack((w.T, hat(w)))) @ np.hstack((w, hat(w).T)) @ np.vstack((np.array([ws * rhos]).reshape(1,1), -(np.cross(w.reshape(3,), (J_tilde@w).reshape(3,))).reshape(3,1)))
rho = rho.reshape(3,)
print("Rotor momentum:")
print(rho)

### 4 ###
# no torque, no rho dot
def attitude_dynamics(w, t, J=J_tilde, rho=rho):
    w = w.reshape(3,)
    wdot = np.linalg.inv(J) @ (-np.cross(w, J @ w + rho)).reshape(3,1)
    return wdot

### 5 ###
def RK4_step(f, state, t, dt):
    k1 = f(state, t)
    k2 = f(state + dt/2*k1, t + dt/2)
    k3 = f(state + dt/2*k2, t + dt/2)
    k4 = f(state + dt*k3, t + dt)
    
    state_next = state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    return state_next

def RK4_integrate(f, initial_state, t0, tf, dt):
    t_values = [t0]
    state_values = [initial_state]
    t = t0
    state = initial_state

    while t<tf:
        state = RK4_step(f, state, t, dt)
        t += dt
        state_values.append(state)
        t_values.append(t)

    return np.array(state_values), np.array(t_values)

tf = 100
dt = 0.01

perturb = rng.normal(0, ws/10, 3).reshape(3,1)
w_perturbed = w + perturb

state_values, t_values = RK4_integrate(attitude_dynamics, w_perturbed, 0, tf, dt)

plt.plot(t_values, state_values[:,0], label="x")
plt.plot(t_values, state_values[:,1], label="y")
plt.plot(t_values, state_values[:,2], label="z")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Spin rates (rad/s)")
plt.title("Spin rates with gyros")
plt.savefig("figs/spin_rate_gyros.png", dpi=400)
plt.show()

