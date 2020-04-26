import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 读入森林
st = np.array(Image.open("sub_t.jpg"), "f")
st = np.divide(st, 255)
x = st.shape[0]
y = st.shape[1]
z = st.shape[2]
st.resize(x*y, z)

# 读入楼房
sg = np.array(Image.open("sub_g.jpg"), "f")
sg = np.divide(sg, 255)
x = sg.shape[0]
y = sg.shape[1]
z = sg.shape[2]
sg.resize(x*y, z)

# 构造向量
vec = {}
vec[1] = sg
vec[0] = st

t_cov = np.cov(st[:, 0:3].T)
g_cov = np.cov(sg[:, 0:3].T)
cov_avg = (t_cov+g_cov)/2.0
t_u = np.transpose(st[:, 0:3].mean(axis=0))
g_u = np.transpose(sg[:, 0:3].mean(axis=0))
p = np.log(0.5)


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(sg[:, 0], sg[:, 1], sg[:, 2], c='blue')
ax.scatter(st[:, 0], st[:, 1], st[:, 2], c='red')
ax.set_zlabel('B')
ax.set_ylabel('G')
ax.set_xlabel('R')
# 不相等的
w_g = np.linalg.inv(t_cov) * (-0.5)
w_t = np.linalg.inv(g_cov) * (-0.5)
W = np.subtract(w_g, w_t)
w_g_0 = np.dot(w_g * (-2.0), g_u.T)
w_t_0 = np.dot(w_t * (-2.0), t_u.T)
w = np.subtract(w_g_0, w_t_0)
w_g_1 = -0.5 * np.dot(np.dot(g_u, np.linalg.inv(t_cov)), g_u.T) - 0.5 * np.log(
    np.linalg.det(t_cov)) + np.log(0.5)
w_t_1 = -0.5 * np.dot(np.dot(t_u, np.linalg.inv(g_cov)), t_u.T) - 0.5 * np.log(
    np.linalg.det(g_cov)) + np.log(0.5)
wi_0 = w_g_1 - w_t_1

x1 = np.arange(0, 1, 0.001)
y1 = np.arange(0, 1, 0.001)
x1, y1 = np.meshgrid(x1, y1)

a = W[2, 2]
b = ((W[0, 2] + W[2, 0]) * y1 + (W[1, 2] + W[2, 1]) * x1 + w[2])
c = (W[0, 0] * x1 * x1 + W[1, 1] * y1 * y1 + (W[0, 1] + W[1, 0])
     * x1 * y1 + w[0] * x1 + w[1] * y1) + wi_0


z1 = ((-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a))
z2 = ((-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a))


ax.plot_surface(x1, y1, z1, color='black')
ax.plot_surface(x1, y1, z2, color='black')
