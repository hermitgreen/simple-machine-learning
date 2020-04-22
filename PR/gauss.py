import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# read image1
img1 = np.array(Image.open("fish_a.jpg").convert('L'), 'f')
p1mean = img1.mean()
p1std = img1.std()

# read image2
img2 = np.array(Image.open("fish_b.jpg").convert('L'), 'f')
p2mean = img2.mean()
p2std = img2.std()

# general model
np.random.seed(0)
n1 = np.random.normal(p1mean, p1std, 5000)
n2 = np.random.normal(p2mean, p2std, 1000)
plt.hist(n1, bins=200, color='blue')
plt.hist(n2, bins=200, color='red')
plt.title("Gaussian distribution")
plt.show()

p = []
r = []
tpr = []
fpr = []
bep = 0
# calculate
for i in range(0, 255):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
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
    if abs(p[i]-r[i]) < 0.01:
        bep = p[i]
    tpr.append(tp/(tp+fn))
    fpr.append(fp/(fp+tn))

plt.plot(p, r)
st = "PR BEP="+str(bep)
plt.title(st)
plt.show()

plt.plot(fpr, tpr)
plt.title("ROC")
plt.show()
