import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 假设文件路径是 'path_to_your_excel.xlsx'
file_path = 'ces2.xlsx'

# 读取Excel文件中的数据，指定使用'Sheet1'，并且数据从第二行第5列开始读取
data = pd.read_excel(file_path, sheet_name='Sheet1', usecols='E:N', skiprows=1)

# print(data)
# 确保没有空行，删除含有空值的行
data.dropna(inplace=True)

# 选取特征列E-K和目标列N
features = data.iloc[:, :-3]  # 除最后一列外的所有列作为特征
# print(features)
target = data.iloc[:, -1]     # 最后一列作为目标变量
print(target)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.15, random_state=40,shuffle=True)

# 定义参数集合
params = {
    'n_estimators': 300,
    'criterion': 'gini',
    'max_depth': 100,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'min_weight_fraction_leaf': 0.0,
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0,
    'bootstrap': True,
    'oob_score': False,
    'class_weight': None,
    'random_state': 42,
    'max_features': None,
}

# 创建随机森林分类器，并传入参数集合
rf_classifier = RandomForestClassifier(**params)

# 使用交叉验证评估模型性能
cv_scores = cross_val_score(rf_classifier, features, target, cv=2) # 2交叉
print("交叉验证准确率:", cv_scores)
print("平均交叉验证准确率:", cv_scores.mean())

# 训练模型
rf_classifier.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = rf_classifier.predict(X_test)

# 计算准确率和生成分类报告
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("准确率:", accuracy)
print("分类报告:\n", classification_rep)
