import numpy as np
import matplotlib.pyplot as plt

mu = 398600.4418 # [km/s]
Re = 6371 # km
h = 360 # km

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

### get state ###
x, y, z, vx, vy, vz = keplerian2ECI(a, e, i, RAAN, AOP, theta0)
initial_state = np.array([x, y, z, vx, vy, vz])

### define ODE ###
def dynamics(state, t):
    r = state[:3]
    v = state[3:]

    rdot = v
    vdot = -mu * r / (np.linalg.norm(r))**3
    return np.concatenate((rdot, vdot))

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

T = 2*np.pi * np.sqrt(a**3 / mu) # one period
tf = 4*T # simulate for 4 periods
state_values, t_values = RK4_integrate(dynamics, initial_state, 0, tf, 1)

### plotting ###
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot(state_values[:,0], state_values[:,1], state_values[:,2])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Orbital Trajectory')
plt.show()