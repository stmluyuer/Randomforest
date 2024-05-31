import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import sklearn.datasets as datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

# 导入数据，路径中要么用\\或/或者在路径前加r
# 假设文件路径是 'path_to_your_excel.xlsx'
file_path = 'ces2.xlsx'

# 读取Excel文件中的数据，指定使用'Sheet1'，并且数据从第二行第5列开始读取
data = pd.read_excel(file_path, sheet_name='Sheet1', usecols='E:N', skiprows=1)

# 输出数据预览
print(data.head())


# 选取特征列E-K和目标列N
features = data.iloc[:, :-3]  # 除最后一列外的所有列作为特征
# print(features)
target = data.iloc[:, -1]     # 最后一列作为目标变量
print(target)
# 将数据分为训练集和测试集

X_train, X_test, y_train, y_test = train_test_split(features,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=0)
regr = RandomForestRegressor()
# regr = RandomForestRegressor(random_state=100,
#                              bootstrap=True,
#                              max_depth=2,
#                              max_features=2,
#                              min_samples_leaf=3,
#                              min_samples_split=5,
#                              n_estimators=3)
pipe = Pipeline([('scaler', StandardScaler()), ('reduce_dim', PCA()),
                 ('regressor', regr)])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# 计算准确率和生成分类报告
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)



from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
import os

# 执行一次
# os.environ['PATH'] = os.environ['PATH']+';'+r"D:\CLibrary\Graphviz2.44.1\bin\graphviz"
dot_data = StringIO()
export_graphviz(pipe.named_steps['regressor'].estimators_[0],
                out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('tree.png')
# Image(graph.create_png())


