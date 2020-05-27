import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
# 读取数据
data = pd.read_csv("data2_training.csv")
label = data['status']
data = data.drop(['status'], axis=1)
# 计算infomation value


def cal_iv(x, y, n_bins=6, null_value=np.nan,):
    if len(x.unique()) == 1 or len(x) != len(y):
        return 0
    if x.dtype == np.number:
        if x.nunique() > n_bins:
            x = pd.qcut(x, q=n_bins, duplicates='drop')
    groups = x.groupby([x, list(y)]).size().unstack().fillna(0)
    t0, t1 = y.value_counts().index
    groups = groups / groups.sum()
    groups['iv_i'] = (groups[t0] - groups[t1]) * \
        np.log(groups[t0] / groups[t1])
    iv = sum(groups['iv_i'])
    return iv


fea_iv = data.apply(lambda x: cal_iv(x, label),
                    axis=0).sort_values(ascending=False)
imp_fea_iv = fea_iv[fea_iv > 0.05].index
imp_fea_iv
# 随机森林计算value，发现710-730都有可能出现最佳值
param = {'n_estimators': list(range(710, 730, 5))}
g = GridSearchCV(estimator=RandomForestClassifier(
    random_state=0), param_grid=param, cv=5)
g.fit(data, label)
g.best_estimator_

rf = g.best_estimator_
rf_impc = pd.Series(rf.feature_importances_,
                    index=data.columns).sort_values(ascending=False)
imp_fea_rf = rf_impc.index[:25]
# 合并两个
imp_fea = list(set(imp_fea_iv) | set(imp_fea_rf))
data = data[imp_fea]
# 取测试集
test = pd.read_csv("data2_test.csv")
test = test[imp_fea]
# 归一化
scaler = StandardScaler()

data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns)

X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
    data_scaled, label, test_size=0.2, random_state=0)

sm = SMOTE(random_state=0)
# 跑最终测试的时候把整个训练集都加进去，不要划分了
X_train_scaled, y_train_scaled = sm.fit_sample(data_scaled, label)

# 逻辑回归
log = LogisticRegression(random_state=0)
log.fit(X_train_scaled, y_train_scaled)
pred = log.predict(X_test_scaled)

cm = confusion_matrix(y_test_scaled, pred)
tn, fp, fn, tp = cm.ravel()

pre = tp/(tp+fp)
print("pre"+str(pre))
rec = tp/(tp+fn)
print("rec"+str(rec))
f1 = 2*tp/(2*tp+fp+fn)
print("f1"+str(f1))
acc = (tp+tn)/(tp+fn+fp+tn)
print("acc"+str(acc))
score = 2*f1+acc
print("score"+str(score))

p = log.predict(test_scaled)
p = pd.DataFrame(p)
p.columns = ["status"]
p.to_csv('res.csv', index=0)
