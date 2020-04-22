import sklearn
import numpy as np
from PIL import Image


# 读入
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

# 正规化truth矩阵
label = truth[:, 0]
for i in range(0, len(label)):
    label[i] = 0 if label[i] < 0.5 else 1

# 分开训练集和测试集
train_data, test_data, train_label, test_label = sklearn.model_selection.train_test_split(
    test, label, random_state=1, train_size=0.8, test_size=0.2)
# 训练高斯核svm
classifier = sklearn.svm.SVC(
    C=2, kernel='rbf', gamma=10, decision_function_shape='ovr')
classifier.fit(train_data, train_label.ravel())
# 输出结果
print("train：", classifier.score(train_data, train_label))
print("test：", classifier.score(test_data, test_label))
