import numpy as np
import matplotlib.pyplot as plt
import json

savefig = False

### REUSE CODE FROM HW 1 ###
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

### get initial state ###
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
tf = 75*T # simulate for 4 periods
state_values, t_values = RK4_integrate(dynamics, initial_state, 0, tf, 1)
### REUSE CODE FROM HW 1 ###

x = state_values[:,0]
y = state_values[:,1]
z = state_values[:,2]

J = np.diag([0.0306, 0.0086, 0.022])

gg_torque = []
for k in range(len(x)):
    r0 = np.array([x[k], y[k], z[k]])

    tau = 3*mu/(np.dot(r0, r0))**(5/2) * np.cross(r0, J@r0)
    gg_torque.append(tau)

gg_torque = np.array(gg_torque)

### aero
CD = 2.6
T = -131.21+0.00299 * h*1000
p = 2.488 * ((T+273.1)/216.6)**(-11.388)
rho = p / (0.2869*(T+273.1))
print(f"Atmospheric density: {rho} kg/m^3")

vx = state_values[:,3]*1000 # m/s
vy = state_values[:,4]*1000 # m/s
vz = state_values[:,5]*1000 # m/s

with open("faces.json", "r") as f:
    faces = json.load(f)

aero_torque = []
for i in range(len(vx)):
    tau = np.zeros(3)
    v = np.array([vx[i], vy[i], vz[i]])
    v_unit = v / np.linalg.norm(v)
    for f in faces:
        ni = np.array(f["n"])
        ri = np.array(f["r"])
        Ai = f["A"] * np.dot(v_unit, ni)
        if Ai >= 0:
            F = 1/2*CD*rho*np.linalg.norm(v)**2 * Ai
            F_vec = -F*v_unit
            tau = tau + np.cross(ri, F_vec)
    aero_torque.append(tau)

aero_torque = np.array(aero_torque)

print(f"Max gg torque magnitude: {max(np.linalg.norm(gg_torque, axis=1))} Nm")
print(f"Max aero torque magnitude: {max(np.linalg.norm(aero_torque, axis=1))} Nm")
plt.plot(t_values/3600, np.linalg.norm(gg_torque, axis=1)*1e3, label="Gravity gradient")
plt.plot(t_values/3600, np.linalg.norm(aero_torque, axis=1)*1e3, label="Atmospheric drag")
plt.plot(t_values/3600, np.linalg.norm(gg_torque, axis=1)*1e3 + np.linalg.norm(aero_torque, axis=1)*1e3, label="Combined")
plt.xlabel("Time (hr)")
plt.ylabel("Torque (mNm)")
plt.legend(loc="upper right")
plt.title("Torque magnitude over 4 orbital periods")
if savefig:
    plt.savefig("figs/torque_mag.png", dpi=400)
plt.show()

plt.figure()
plt.plot(t_values/3600, aero_torque[:, 0]*1e3, label="x")
plt.plot(t_values/3600, aero_torque[:, 1]*1e3, label="y")
plt.plot(t_values/3600, aero_torque[:, 2]*1e3, label="z")
plt.xlabel("Time (hr)")
plt.ylabel("Torque (mNm)")
plt.legend(loc="upper right")
plt.title("Components of atmospheric drag torque")
plt.tight_layout()
if savefig:
    plt.savefig("figs/aero-components.png", dpi=400)
plt.show()

combined_torque = gg_torque + aero_torque

gg_momentum = np.cumsum(gg_torque, axis=0)
aero_momentum = np.cumsum(aero_torque, axis=0)
combined_momentum = np.cumsum(combined_torque, axis=0)

plt.figure()
plt.plot(t_values/3600, gg_momentum[:,0]*1e3, label="x")
plt.plot(t_values/3600, gg_momentum[:,1]*1e3, label="y")
plt.plot(t_values/3600, gg_momentum[:,2]*1e3, label="z")
plt.xlabel("Time (hr)")
plt.ylabel("Momentum (mNms)")
plt.legend(loc="upper right")
plt.title("Gravity gradient momentum acculmulation")
if savefig:
    plt.savefig("figs/gg-momentum.png", dpi=400)
plt.show()

plt.figure()
plt.plot(t_values/3600, aero_momentum[:,0]*1e3, label="x")
plt.plot(t_values/3600, aero_momentum[:,1]*1e3, label="y")
plt.plot(t_values/3600, aero_momentum[:,2]*1e3, label="z")
plt.xlabel("Time (hr)")
plt.ylabel("Momentum (mNms)")
plt.legend(loc="upper right")
plt.title("Atmospheric drag momentum acculmulation")
if savefig:
    plt.savefig("figs/aero-momentum.png", dpi=400)
plt.show()

plt.figure()
plt.plot(t_values/3600, combined_momentum[:,0]*1e3, label="x")
plt.plot(t_values/3600, combined_momentum[:,1]*1e3, label="y")
plt.plot(t_values/3600, combined_momentum[:,2]*1e3, label="z")
plt.xlabel("Time (hr)")
plt.ylabel("Momentum (mNms)")
plt.legend(loc="upper right")
plt.title("Combined momentum accumulation")
if savefig:
    plt.savefig("figs/comb-momentum.png", dpi=400)
plt.show()

c = np.cos(np.deg2rad(26.57))
s = np.sin(np.deg2rad(26.57))

Bw = np.array([[c, 0, -c, 0],
               [s, s, s, s],
               [0, c, 0, -c]])

Bw_pinv = Bw.T @ np.linalg.inv(Bw@Bw.T)
wheel_momenta = combined_momentum @ Bw_pinv.T

plt.figure()
plt.plot(t_values/3600, wheel_momenta[:, 0]*1e3, label="w1")
plt.plot(t_values/3600, wheel_momenta[:, 1]*1e3, label="w2")
plt.plot(t_values/3600, wheel_momenta[:, 2]*1e3, label="w3")
plt.plot(t_values/3600, wheel_momenta[:, 3]*1e3, label="w4")
plt.xlabel("Time (hr)")
plt.ylabel("Momentum (mNms)")
plt.title("Wheel momenta")
plt.legend(loc="upper right")
plt.axhline(16.2, c="r")
plt.axhline(-16.2, c="r")
plt.savefig("figs/wheel-sat.png", dpi=400)
plt.show()

np.save("torque.npy", combined_torque)