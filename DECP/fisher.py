import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D


def fisher(a, b):
    mean_a = np.mean(a, axis=0)
    mean_b = np.mean(b, axis=0)
    x, y = np.shape(a)
    sw = np.zeros((y, y))

    for i in a:
        t = i - mean_a
        sw += t * t.reshape(3, 1)

    for i in b:
        t = i - mean_b
        sw += t * t.reshape(3, 1)

    u, s, v = np.linalg.svd(sw)
    sw_inv = np.dot(np.dot(v.T, np.linalg.inv(np.diag(s))), u.T)

    return np.dot(sw_inv, mean_a-mean_b)


t = np.array(Image.open("sub_t.jpg"), "f")
t = np.divide(t, 255)
x = t.shape[0]
y = t.shape[1]
z = t.shape[2]
t.resize(x*y, z)

g = np.array(Image.open("sub_g.jpg"), "f")
g = np.divide(g, 255)
x = g.shape[0]
y = g.shape[1]
z = g.shape[2]
g.resize(x*y, z)

w = fisher(t, g)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(g[:, 0], g[:, 1], g[:, 2], c='blue')
ax.scatter(t[:, 0], t[:, 1], t[:, 2], c='red')
ax.set_zlabel('B')
ax.set_ylabel('G')
ax.set_xlabel('R')
wx = np.arange(0, 1, 0.0001)
wy = np.arange(0, 1, 0.0001)
wx, wy = np.meshgrid(wx, wy)
wz = (-w[1] * wy-w[0]*wx) / w[2]

ax.plot_surface(wx, wy, wz, color='black')
