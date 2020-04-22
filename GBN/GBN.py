import numpy as np
from PIL import Image
import math

# 读入原图
raw = np.array(Image.open("image.jpg"), "f")
raw = np.divide(raw, 255)
x = raw.shape[0]
y = raw.shape[1]
z = raw.shape[2]
raw.resize(x*y, z)

# 读入truth
truth = np.array(Image.open("ground_truth.jpg"), "f")
truth = np.divide(truth, 255)
x = truth.shape[0]
y = truth.shape[1]
z = truth.shape[2]
truth.resize(x*y, z)

# 正规化truth矩阵
label = truth[:, 0]
for i in range(0, len(label)):
    label[i] = 0 if label[i] < 0.5 else 1

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


# 均值
def mean(t):
    return sum(t)/float(len(t))


# 方差
def stdev(t):
    avg = mean(t)
    var = sum([pow(x-avg, 2) for x in t])/float(len(t)-1)
    return math.sqrt(var)


# 提取图像特征
def SMRF(dataset):
    SMR = [(mean(attr), stdev(attr))
           for attr in zip(*dataset)]
    return SMR


# 按rgb提取图像特征
def SMRFByClass(train):
    SMR = {}
    SMR[0] = SMRF(train[0])
    SMR[1] = SMRF(train[1])
    return SMR


# 高斯分布
def gaussd(x, mean, stdev):
    exp = math.exp(-(math.pow(x-mean, 2)/(2*math.pow(stdev, 2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exp


# 计算属类概率
def calClass(SMR, testcase):
    prob = {}
    for classValue, classSMR in SMR.items():
        prob[classValue] = 1
        for i in range(len(classSMR)):
            mean, stdev = classSMR[i]
            x = testcase[i]
            prob[classValue] *= gaussd(x, mean, stdev)
    return prob


# 计算预测
def predict(SMR, testcase):
    prob = calClass(SMR, testcase)
    bestLabel, bestProb = None, -1
    for classValue, prot in prob.items():
        if bestLabel is None or prot > bestProb:
            bestProb = prot
            bestLabel = classValue
    return bestLabel


# 分类
def getp(SMR, testSet):
    pred = []
    for i in range(len(testSet)):
        result = predict(SMR, testSet[i])
        pred.append(result)
    return np.array(pred)


SMR = SMRFByClass(vec)
p = getp(SMR, raw)

# 计算混淆矩阵
tp = 0
tn = 0
fp = 0
fn = 0

for i in range(len(label)):
    if p[i] == 1 and label[i] == 1:
        tp = tp+1
    if p[i] == 1 and label[i] == 0:
        fp = fp+1
    if p[i] == 0 and label[i] == 0:
        tn = tn+1
    if p[i] == 0 and label[i] == 1:
        fn = fn+1

Accuracy = (tp+tn)/(tp+tn+fp+fn)
CM = [[tp, fn], [fp, tn]]
