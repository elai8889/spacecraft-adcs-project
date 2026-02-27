import numpy as np
import matplotlib.pyplot as plt

savefig = False
np.random.seed(6)

J11 = 0.0086
J22 = 0.022
J33 = 0.0306
J = np.array([J11, J22, J33])

def euler(omega, t):
    # zero torque euler equations
    w1, w2, w3 = omega
    w1dot = -(J33-J22)*w2*w3 / J11
    w2dot = -(J11-J22)*w1*w3 / J22
    w3dot = -(J22-J11)*w1*w2 / J33
    return np.array([w1dot, w2dot, w3dot])

### get initial state options ###
major_spin_rate = 10 * 2*np.pi/60 # rad/s
h = major_spin_rate * J33
inter_spin_rate = h / J22
minor_spin_rate = h / J11

perturb = np.random.normal(0, 0.5, 3)

state_major_spin = np.array([0, 0, major_spin_rate]) + perturb
state_inter_spin = np.array([0, inter_spin_rate, 0]) + perturb
state_minor_spin = np.array([minor_spin_rate, 0, 0]) + perturb

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

tf = 100
major, t_major = RK4_integrate(euler, state_major_spin, 0, tf, 0.01)
inter, t_inter = RK4_integrate(euler, state_inter_spin, 0, tf, 0.01)
minor, t_minor = RK4_integrate(euler, state_minor_spin, 0, tf, 0.01)

### plotting ###
fig = plt.figure()
plt.plot(t_major, major[:,0], label="w1")
plt.plot(t_major, major[:,1], label="w2")
plt.plot(t_major, major[:,2], label="w3")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Spin rate (rad/s)")
plt.title("Spin rates (major axis)")
plt.grid(True)
if savefig:
    plt.savefig("figs/spin_rates_major.png", dpi=400)

fig = plt.figure()
plt.plot(t_inter, inter[:,0], label="w1")
plt.plot(t_inter, inter[:,1], label="w2")
plt.plot(t_inter, inter[:,2], label="w3")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Spin rate (rad/s)")
plt.title("Spin rates (intermediate axis)")
plt.grid(True)
if savefig:
    plt.savefig("figs/spin_rates_inter.png", dpi=400)

fig = plt.figure()
plt.plot(t_minor, minor[:,0], label="w1")
plt.plot(t_minor, minor[:,1], label="w2")
plt.plot(t_minor, minor[:,2], label="w3")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Spin rate (rad/s)")
plt.title("Spin rates (minor axis)")
plt.grid(True)
if savefig:
    plt.savefig("figs/spin_rates_minor.png", dpi=400)

### generate momentum sphere ###
# this was done with some help from ChatGPT

h_major = J * major
h_inter = J * inter
h_minor = J * minor

# normalize everything
h_major_hat = (h_major.T / np.linalg.norm(h_major, axis=1)).T
h_inter_hat = (h_inter.T / np.linalg.norm(h_inter, axis=1)).T
h_minor_hat = (h_minor.T / np.linalg.norm(h_minor, axis=1)).T

u = np.linspace(0, 2*np.pi, 80)
v = np.linspace(0, np.pi, 80)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))

equilibrium_points = np.array([[ 1, 0, 0], 
                               [-1, 0, 0],
                               [ 0, 1, 0], 
                               [ 0,-1, 0],
                               [ 0, 0, 1], 
                               [ 0, 0,-1],])

### plot momentum sphere
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(xs, ys, zs, alpha=0.5, color="gray")

ax.scatter(equilibrium_points[:,0], equilibrium_points[:,1], equilibrium_points[:,2], s=60)

ax.plot(h_major_hat[:,0], h_major_hat[:,1], h_major_hat[:,2], linewidth=2, label="major")
ax.plot(h_inter_hat[:,0], h_inter_hat[:,1], h_inter_hat[:,2], linewidth=2, label="intermediate")
ax.plot(h_minor_hat[:,0], h_minor_hat[:,1], h_minor_hat[:,2], linewidth=2, label="minor")

ax.set_xlabel("h1")
ax.set_ylabel("h2")
ax.set_zlabel("h3")
ax.set_title("Momentum Sphere")
ax.legend()

ax.set_box_aspect([1, 1, 1])
if savefig:
    plt.savefig("figs/momentum_sphere.png", dpi=400)

plt.show()