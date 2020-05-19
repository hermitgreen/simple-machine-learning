import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# 读取数据
data = pd.read_csv("data1.csv")
Y = data['price']
X = data.drop(['price'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)


def count_mse(y_test, pred):
    cost = 0
    a = y_test.values
    for i in range(len(pred)):
        cost += (pred[i]-a[i])**2
    return cost/len(pred)


model = RandomForestClassifier(n_estimators=150)
model.fit(x_train, y_train)
pred = model.predict(x_test)

score = count_mse(y_test, pred)
print(score)

x = np.arange(1, len(pred)+1)
plt.plot(x, pred, label="predict")
plt.plot(x, y_test, label="truth")
plt.legend(loc='upper right')
plt.show()
