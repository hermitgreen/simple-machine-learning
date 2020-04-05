import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

t = np.array(Image.open("sub_t.jpg"))
g = np.array(Image.open("sub_g.jpg"))


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(g[:, 0], g[:, 1], g[:, 2], c='blue')
ax.scatter(t[:, 0], t[:, 1], t[:, 2], c='red')
ax.set_zlabel('B')
ax.set_ylabel('G')
ax.set_xlabel('R')
