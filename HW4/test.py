import numpy as np

c = np.cos(np.deg2rad(26.57))
s = np.sin(np.deg2rad(26.57))

Bw = np.array([[c, 0, -c, 0],
               [s, s, s, s],
               [0, c, 0, -c]])

print(Bw)
# print(Bw.T @ np.linalg.inv(Bw@Bw.T))

# m = 2
# me = np.array([np.cos(np.deg2rad(11)),np.sin(np.deg2rad(11)),0]).reshape(3,1)
# r = np.array([0, 6731, 0]).reshape(3,1)
# RE = 6371
# B0 = 32e-6

# B = -RE**3*B0/(np.linalg.norm(r)**3) * (3*r@r.T / (r.T @ r) - np.eye(3)) @ me
# print(np.linalg.norm(B)*1e6)

# tau = m*np.linalg.norm(B)
# print(np.linalg.norm(tau)*1e3)

# J = np.eye(3)
# r = np.array([2, 1, 0])

# print((J@r).shape)

w = np.array([7, 7, 7, 7])
print(Bw@w)