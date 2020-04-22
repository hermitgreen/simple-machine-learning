import sklearn
import numpy as np

# 读入
data = np.loadtxt("data.txt", dtype=float, delimiter=',')

# x是训练数据,y是label
x, y = np.split(data, indices_or_sections=(2,), axis=1)
print(x)
print(y)
# 分开训练集和测试集
train_data, test_data, train_label, test_label = sklearn.model_selection.train_test_split(
    x, y, random_state=1, train_size=0.8, test_size=0.2)
# 训练线性核svm
classifier = sklearn.svm.SVC(
    C=2, kernel='linear', gamma=10, decision_function_shape='ovr')
classifier.fit(train_data, train_label.ravel())
# 输出结果
print("train：", classifier.score(train_data, train_label))
print("test：", classifier.score(test_data, test_label))
