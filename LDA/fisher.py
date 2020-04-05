import numpy as np
import matplotlib.pyplot as plt


def fisher(a, b):
    mean_a = np.mean(a, axis=0)
    mean_b = np.mean(b, axis=0)
    x, y = np.shape(a)
    sw = np.zeros((y, y))

    for i in a:
        t = i - mean_a
        sw += t * t.reshape(2, 1)

    for i in b:
        t = i - mean_b
        sw += t * t.reshape(2, 1)
    print(sw)

    u, s, v = np.linalg.svd(sw)
    sw_inv = np.dot(np.dot(v.T, np.linalg.inv(np.diag(s))), u.T)

    return np.dot(sw_inv, mean_a-mean_b)


x1 = np.array([[0.697, 0.460],
               [0.774, 0.376],
               [0.634, 0.264],
               [0.608, 0.318],
               [0.556, 0.215],
               [0.403, 0.237],
               [0.481, 0.149],
               [0.437, 0.211]])

x2 = np.array([[0.666, 0.091],
               [0.243, 0.267],
               [0.245, 0.057],
               [0.343, 0.099],
               [0.639, 0.161],
               [0.657, 0.198],
               [0.360, 0.370],
               [0.593, 0.042],
               [0.719, 0.103]])


w = fisher(x1, x2)
plt.scatter(x1[:, 0], x1[:, 1], c='red')
plt.scatter(x2[:, 0], x2[:, 1], c='blue')
wx = np.arange(0, 1, 0.01)
wy = -(w[0] * wx) / w[1]
plt.plot(wx, wy, c='black')
plt.title("LDA projection")
