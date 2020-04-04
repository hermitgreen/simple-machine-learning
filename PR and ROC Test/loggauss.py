import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math

# read image1
img1 = np.array(Image.open("fish_a.jpg").convert('L'), 'f')
p1mean = img1.mean()
p1var = img1.var()

# read image2
img2 = np.array(Image.open("fish_b.jpg").convert('L'), 'f')
p2mean = img2.mean()
p2var = img2.var()

# prefix
mu1 = math.log(p1mean)-math.log(p1var/(p1mean*p1mean)+1)/2
sigma1 = math.sqrt(math.log(p1var/(p1mean*p1mean)+1))

mu2 = math.log(p2mean)-math.log(p2var/(p2mean*p2mean)+1)/2
sigma2 = math.sqrt(math.log(p2var/(p2mean*p2mean)+1))

# general model
np.random.seed(0)
n1 = np.random.normal(mu1, sigma1, 5000)
n2 = np.random.normal(mu2, sigma2, 1000)
plt.hist(n1, bins=200, color='blue')
plt.hist(n2, bins=200, color='red')
plt.title("Log Gaussian distribution")
plt.show()

p = []
r = []
tpr = []
fpr = []
bep = 0
t = 0
# calculate

for i in np.arange(4.0, 6.0, 0.01):
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    for c in n1:
        if(c < i):
            fn += 1
        else:
            tp += 1
    for c in n2:
        if(c < i):
            tn += 1
        else:
            fp += 1
    p.append(tp/(tp+fp))
    r.append(tp/(tp+fn))
    if abs(p[t]-r[t]) < 0.01:
        bep = p[t]
    tpr.append(tp/(tp+fn))
    fpr.append(fp/(fp+tn))
    t += 1

plt.plot(p, r)
st = "PR BEP="+str(bep)
plt.title(st)
plt.show()

plt.plot(fpr, tpr)
plt.title("ROC")
plt.show()
