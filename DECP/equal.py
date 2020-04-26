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
# 相等的，以g_cov为例
w = np.dot(np.linalg.inv(g_cov), (t_u - g_u).T)
w_0 = (-0.5 * np.dot(np.dot((g_u + t_u), g_cov), (-g_u + t_u).T)) + p
x1 = np.arange(0, 1, 0.001)
y1 = np.arange(0, 1, 0.001)
x1, y1 = np.meshgrid(x1, y1)
z1 = (-w_0 - w[1] * y1 - w[0] * x1) / w[2]
ax.plot_surface(x1, y1, z1, color='black')
