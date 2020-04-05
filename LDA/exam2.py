import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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

test = np.array(Image.open("image.jpg"), "f")
test = np.divide(test, 255)
x = test.shape[0]
y = test.shape[1]
z = test.shape[2]
test.resize(x*y, z)

truth = np.array(Image.open("ground_truth.jpg"), "f")
truth = np.divide(truth, 255)
x = truth.shape[0]
y = truth.shape[1]
z = truth.shape[2]
truth.resize(x*y, z)

mean1 = np.mean(t, axis=0)
mean2 = np.mean(g, axis=0)
c1 = np.dot(w.T, mean1)
c2 = np.dot(w.T, mean2)


# 分类
ans = np.zeros((450*350, 1))
for i in range(1, 450*350):
    b = np.dot(w.T, test[i, :])
    if abs(b-c1) < abs(b-c2):
        ans[i] = 1


# 计算混淆矩阵
TP = 0
FN = 0
FP = 0
TN = 0

for i in range(1, 340*450):
    if truth[i][0] > 0.5:
        if ans[i] != 1:
            TP = TP+1
        else:
            FN = FN+1
    else:
        if ans[i] != 1:
            FP = FP+1
        else:
            TN = TN+1

cf = [[TP, FN], [FP, TN]]
print(cf)
