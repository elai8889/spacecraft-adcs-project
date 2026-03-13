import numpy as np
import matplotlib.pyplot as plt
import scipy

mu = 398600.4418 # [km/s]
Re = 6371 # km
h = 360 # km
J = np.array([[ 2.40287398e-02, 1.49601815e-06, -2.52300907e-06],
              [1.49601815e-06, 1.94085242e-02, 1.38145802e-08],
              [-2.52300907e-06, 1.38145802e-08, 1.91983127e-02]])

def keplerian2ECI(a, e, i, RAAN, AOP, theta):
    # luckily i had this conversion in my back pocket from an old project
    A11 = np.cos(RAAN)*np.cos(AOP) - np.sin(RAAN)*np.sin(AOP)*np.cos(i)
    A12 = np.sin(RAAN)*np.cos(AOP) + np.cos(RAAN)*np.sin(AOP)*np.cos(i)
    A13 = np.sin(AOP)*np.sin(i)
    A21 = -np.cos(RAAN)*np.sin(AOP)-np.sin(RAAN)*np.cos(AOP)*np.cos(i)
    A22 = -np.sin(RAAN)*np.sin(AOP)+np.cos(RAAN)*np.cos(AOP)*np.cos(i)
    A23 = np.cos(AOP)*np.sin(i)
    A31 = np.sin(RAAN)*np.sin(i)
    A32 = -np.cos(RAAN)*np.sin(i)
    A33 = np.cos(i)

    A = np.array([[A11, A12, A13],
                    [A21, A22, A23],
                    [A31, A32, A33]])

    E = np.arctan2(np.sqrt(1-e**2)*np.sin(theta), e+np.cos(theta))
    x_p = a*(np.cos(E)-e)
    y_p = a*np.sqrt(1-e**2)*np.sin(E)

    r_vec = A.T @ np.array([x_p, y_p, 0])
    r = np.linalg.norm(r_vec)
    x = r_vec[0]
    y = r_vec[1]
    z = r_vec[2]

    vx_p = -np.sqrt(mu*a)/r * np.sin(E)
    vy_p = np.sqrt(mu*a*(1-e**2))/r * np.cos(E)
    v_vec = A.T @ np.array([vx_p, vy_p, 0])
    vx = v_vec[0]
    vy = v_vec[1]
    vz = v_vec[2]
    return x, y, z, vx, vy, vz

### Keplerian OEs ###
a = Re + h
e = 0.01
i = np.deg2rad(51.6)
RAAN = np.deg2rad(125.6)
AOP = np.deg2rad(137.2)
theta0 = 0

### initial quaternion ###
q0, q1, q2, q3 = [np.sqrt(2)/2, 0, -np.sqrt(2)/2, 0]
q_init = np.array([q0, q1, q2, q3])

### initial angular velocity
wx, wy, wz = [-0.06528897, 0.01556466, 0.87878852] # from previous part

### rotor momentum
rhox, rhoy, rhoz = [2.64208892e-06, -1.44665945e-08, 1.83486680e-02]

### get initial state ###
x, y, z, vx, vy, vz = keplerian2ECI(a, e, i, RAAN, AOP, theta0)
initial_state = np.array([x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz, rhox, rhoy, rhoz])

### quaternion helpers ###
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

def Q(q):
    return H.T @ L(q) @ R(q).T @ H

def unhat(S):
    return 0.5 * np.array([S[2,1] - S[1,2],
                           S[0,2] - S[2,0],
                           S[1,0] - S[0,1]])


### define ODE ###
def dynamics(state, t):
    r = state[:3]
    v = state[3:6]
    q = state[6:10]
    w = state[10:13]
    rho = state[13:16]

    rdot = v
    vdot = -mu * r / (np.linalg.norm(r))**3
    
    rhodot = np.zeros(3)
    wdot = - np.linalg.inv(J) @ (np.cross(w, J @ w + rho))
    qdot = 1/2 * L(q) @ H @ w

    return np.concatenate((rdot, vdot, qdot, wdot, rhodot))

### solve IVP with RK4 ###
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

# T = 2*np.pi * np.sqrt(a**3 / mu) # one period
# tf = 4*T # simulate for 4 periods
tf = 40 # s
dt = 0.01
state_values, t_values = RK4_integrate(dynamics, initial_state, 0, tf, dt)

### plotting ###
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(state_values[:,0], state_values[:,1], state_values[:,2])
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('Orbital Trajectory')

fig = plt.figure()
plt.plot(t_values, state_values[:,10], label="x")
plt.plot(t_values, state_values[:,11], label="y")
plt.plot(t_values, state_values[:,12], label="z")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Spin rates (rad/s)")

fig = plt.figure()
[plt.plot(t_values, state_values[:,i]) for i in range(6, 10)]
plt.xlabel("Time (s)")
plt.title("Quaternion components")
plt.savefig("figs/quaternion_components.png", dpi=400)

fig = plt.figure()
plt.plot(t_values, np.sqrt(sum([state_values[:,i]**2 for i in range(6,10)])))
plt.xlabel("Time (s)")
plt.title("Quaternion norm")
plt.savefig("figs/quaternion_norm.png", dpi=400)

## want negative z to point towards positive x
Q_desired = np.array([[0, 0, -1],
                      [0, 1, 0],
                      [1, 0, 0]])


q_values = state_values[:,6:10]
pointing_error = []
for q in q_values:
    bore_vec = Q(q) @ np.array([0, 0, -1])
    des_vec = np.array([1, 0, 0])
    coserror = np.dot(bore_vec, des_vec) / (np.linalg.norm(bore_vec) * np.linalg.norm(des_vec)) 
    pointing_error.append(np.deg2rad(np.acos(coserror)))

pointing_error = np.array(pointing_error)
plt.figure()
plt.plot(t_values, pointing_error)
plt.xlabel("Time (s)")
plt.ylabel("Pointing error (deg)")
plt.ylim([-0.1, 0.1])
plt.title("Solar panel to sun vector pointing error")
plt.tight_layout()
plt.savefig("figs/pointing_error.png", dpi=400)
plt.show()