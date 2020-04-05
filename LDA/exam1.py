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


x1 = np.array([[4, 1], [2, 4], [2, 3], [3, 6], [4, 4]])
x2 = np.array([[9, 10], [6, 8], [9, 5], [8, 7], [10, 8]])

w = fisher(x1, x2)
plt.scatter(x1[:, 0], x1[:, 1], c='red')
plt.scatter(x2[:, 0], x2[:, 1], c='blue')
wx = np.arange(0, 10, 0.1)
wy = (w[0] * wx) / w[1]
plt.plot(wx, wy, c='black')
plt.title("LDA projection")
