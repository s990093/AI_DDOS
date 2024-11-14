import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, fbeta_score, confusion_matrix, precision_recall_curve
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
pd.options.mode.use_inf_as_na = True
plt.style.use('dark_background')

from dask import dataframe as dd
ddf = dd.read_csv("data/raw/iot23_combined.csv",blocksize='64MB')
print(ddf)

#只留下正常流量與DDoS流量
ddf = ddf[(ddf['Label'] == 'Benign') | (ddf['Label'] == 'DDoS')]
df = ddf.compute()

#新增編號欄位
df.insert(0,column = 'number', value = list(range(0,len(df))))
df.set_index(["number"], inplace=True)

df.columns = df.columns.str.replace(' ', '')

#所有欄位名稱（特徵值）
cloumn_name = df.columns
print(cloumn_name)

df.head(3) #查看前三筆的資料

(df['Label'].value_counts()) / len(df) * 100    #檢查Label欄位中出現的個數並進行排列

# df.drop(columns=['FlowID', 'SourceIP', 'DestinationIP', 'Timestamp', 'SimillarHTTP', 'SourcePort', 'DestinationPort'], inplace=True)
#要刪除的行列名稱，inplace代表刪除動作要不要改動原本的數據

df.dropna(inplace=True) #刪除為空值的行或列

cols = df.drop(columns=['Label']).columns.tolist()
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
#將參數轉成數值，coerce表示將不能轉換的數值轉會成NULL

labels = pd.get_dummies(df['Label'])
#以Label作為one hot encoding

X = df.drop(columns=["Label"], axis=1)
y = labels.Benign
#drop默認刪除axis=0，指刪除行，axis=1指刪除欄

import joblib
from dask.distributed import Client

client = Client(processes=False)

"""先找出重要特徵的排序"""

#分類器的訓練與測試資料，比例為80%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, stratify=y, random_state=10)

from sklearn.preprocessing import StandardScaler
Important_std = StandardScaler()
Important_X_train_scaled = pd.DataFrame(
    Important_std.fit_transform(X_train),
    columns = X_train.columns)

#執行訓練資料標準化

Important_X_test_scaled = pd.DataFrame(
    Important_std.transform(X_test),
    columns = X_test.columns)
#執行測試資料標準化

#初始化基於隨機森林樹的分類器
#max_depth預設為 None，表示深度，會持續到無法分割與最大深度為止
#n_estimators為決策樹數量
#random_state為每次隨機抽取的數量，此數值若固定，則每次執行結果都會相同，方便重現，若無設定，則採用特徵的平方根
#max_samples 參數來設定每棵決策樹使用的資料樣本數量，預設為1，表示使用全部資料，若設定0.8，則為80%
#n_jobs 使用的CPU核心數量 -1是使用剩餘全部
Important_rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

with joblib.parallel_backend("dask", wait_for_workers_timeout=20):
    Important_rf.fit(Important_X_train_scaled, y_train)

#所有特徵的重要性排序，feature_importances_存放重要特徵的數值
Important_f_i = list(zip(X,Important_rf.feature_importances_))
Important_f_i.sort(key = lambda x : x[1])

# Limit to top 20 features for plotting
top_n = 20
Important_f_i.sort(key=lambda x: x[1], reverse=True)
top_features = Important_f_i[:top_n]

plt.barh([x[0] for x in top_features], [x[1] for x in top_features], height=0.8, align='center')
plt.tick_params(axis='y', labelsize=4)
plt.tight_layout()
plt.show()

#預設選擇15個重要特徵值

D_selected_columns = ['orig_pkts',
                      'orig_ip_bytes',
                      'conn_state_OTH',
                      'duration',
                      'conn_state_S0',
                      'missed_bytes',
                      'orig_bytes',
                      'proto_tcp',
                      'proto_udp',
                      'resp_ip_bytes',
                      'proto_icmp']

X_D = X[D_selected_columns]
# print(X_D)

#分類器的訓練與測試資料，比例為75%
D_X_train, D_X_test, D_y_train, D_y_test = train_test_split(X_D, y, test_size = .25, stratify=y, random_state=10)

D_std = StandardScaler()
D_X_train_scaled = pd.DataFrame(
    D_std.fit_transform(D_X_train),
    columns = D_X_train.columns)

#執行訓練資料標準化

D_X_test_scaled = pd.DataFrame(
    D_std.transform(D_X_test),
    columns = D_X_test.columns)
#執行測試資料標準化

#初始化基於隨機森林樹的分類器
#max_depth預設為 None，表示深度，會持續到無法分割與最大深度為止
#n_estimators為決策樹數量
#random_state為每次隨機抽取的數量，此數值若固定，則每次執行結果都會相同，方便重現，若無設定，則採用特徵的平方根
#max_samples 參數來設定每棵決策樹使用的資料樣本數量，預設為1，表示使用全部資料，若設定0.8，則為80%
from sklearn.ensemble import RandomForestClassifier
# D_rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=3,max_depth=100)
D_rf = RandomForestClassifier(n_jobs=-1, random_state=3)

with joblib.parallel_backend("dask"):
    D_rf.fit(D_X_train_scaled, D_y_train)

#對D樹做預測
print("預設RF的預測結果\n")
D_y_val_preds = D_rf.predict(D_X_test_scaled)
print("Precision: {}, Recall: {}, F2 Score: {}".format(precision_score(D_y_test, D_y_val_preds), recall_score(D_y_test, D_y_val_preds), fbeta_score(D_y_test, D_y_val_preds, beta=2.0)))

#隨機森林樹，準確度預測
test_score = D_rf.score(D_X_test_scaled, D_y_test) * 100
print(f'Default RF Tree Accuracy: {test_score:.5f}%')

# # 提取'False Neg'的索引
# D_false_neg_indices = np.where((D_y_test == 1) & (D_y_val_preds == 0))[0]

# # 提取'False Neg'對應的紀錄
# D_false_neg_records = df.iloc[D_X_test.index[D_false_neg_indices]]

# # 提取'False Pos'的索引
# D_false_pos_indices = np.where((D_y_test == 0) & (D_y_val_preds == 1))[0]

# # 提取'False Pos'對應的紀錄
# D_false_pos_records = df.iloc[D_X_test.index[D_false_pos_indices]]

# AFN = pd.DataFrame(D_false_neg_records)
# AFN.to_csv("D_Flase_Neg.csv")
# AFP = pd.DataFrame(D_false_pos_records)
# AFP.to_csv("D_Flase_Pos.csv")

# test_scores = []
# for depth in range(1,6):
#     D_rf = RandomForestClassifier(n_jobs=-1, random_state=3,max_depth=depth*3)
#     D_rf.fit(X_train, y_train)
#     test_scores.append(D_rf.score(X_test, y_test))

# plt.style.use('classic')
# depths = np.linspace(3, 15, 5)
# plt.figure(figsize=(8, 6))
# plt.plot(depths, test_scores, marker='o', label='Train Score', color='blue')
# plt.xlabel('Training Depth')
# plt.ylabel('Accuracy')
# plt.title('IOT-23: Different Depths')
# plt.legend(loc = 'lower right')
# plt.grid(True)
# # plt.ticklabel_format(style='plain', axis='y')
# plt.show()

# test_scores = []
# for n_estimators_num in range(1,6):
#     D_rf = RandomForestClassifier(n_jobs=-1, random_state=3,n_estimators = n_estimators_num * 100)
#     D_rf.fit(X_train, y_train)
#     test_scores.append(D_rf.score(X_test, y_test))

# plt.style.use('classic')
# train_sizes = np.arange(100, 600, 100)
# print(test_scores)
# plt.figure(figsize=(8, 6))
# plt.plot(train_sizes, test_scores, marker='o', label='Train Score', color='blue')
# plt.xlabel('Number of Estimators')
# plt.ylabel('Accuracy')
# plt.title('IOT-23: Different Estimators')
# plt.legend(loc = 'lower right')
# plt.grid(True)
# plt.show()

# test_scores = []
# train_sizes = np.linspace(0.1, 1, 10)
# D_rf = RandomForestClassifier(n_jobs=-1, random_state=3,max_depth=15,n_estimators=200)
# for train_size in train_sizes:
#     #設定樣本數量
#     current_train_size = int(len(X_train) * train_size)
#     X_subset = X_train[:current_train_size]
#     y_subset = y_train[:current_train_size]

#     # 訓練模型並取得訓練準確率
#     D_rf.fit(X_subset, y_subset)
#     test_scores.append(D_rf.score(X_test, y_test))

# plt.style.use('classic')
# plt.figure(figsize=(8, 6))
# plt.plot(train_sizes*100, test_scores, marker='o', label='Train Score', color='blue')
# plt.xlabel('Training Set Size (%)')
# plt.ylabel('Accuracy')
# plt.title('IOT-23: Different Set Size')
# plt.legend(loc = 'lower right')
# plt.grid(True)
# plt.show()

#混淆矩陣的輸出結果
D_rf_confusion = confusion_matrix(D_y_test, D_y_val_preds)

D_group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
D_group_counts = ["{0:0.0f}".format(value) for value in D_rf_confusion.flatten()]
D_group_percentages = ["{0:0.4%}".format(value) for value in D_rf_confusion.flatten()/np.sum(D_rf_confusion)]
D_labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(D_group_names,D_group_counts,D_group_percentages)]
D_labels = np.asarray(D_labels).reshape(2,2)

#產生混淆矩陣結果
sns.heatmap(D_rf_confusion,cmap='rocket_r', annot=D_labels, fmt='', square=True, xticklabels=['DDoS', 'Benign'], yticklabels=['DDoS', 'Benign'])
plt.xlabel('Predicted Traffic Type')
plt.ylabel('Actual Traffic Type')
plt.title('Random Forest Confusion Matrix')

"""預設分類器資訊到這

分類器Ａ、B的設定（根據特徵重要性）
"""

#A預設選擇9個重要特徵值

A_selected_columns = ['orig_ip_bytes',
                      'orig_pkts',
                      'conn_state_S0',
                      'conn_state_OTH',
                      'duration',
                      'orig_bytes',
                      'resp_bytes',
                      'proto_tcp',
                      'proto_udp', ]

#B預設選擇5個重要特徵值
B_selected_columns = ['orig_ip_bytes',
                      'orig_pkts',
                      'conn_state_S0',
                      'conn_state_OTH',
                      'duration',
                      ]

X_A = X[A_selected_columns]
X_B = X[B_selected_columns]

#分類器的訓練與測試資料，比例為75%
A_X_train, A_X_test, A_y_train, A_y_test = train_test_split(X_A, y, test_size = .25, stratify=y, random_state=3)

std = StandardScaler()

A_X_train_scaled = pd.DataFrame(
    std.fit_transform(A_X_train),
    columns = A_X_train.columns)

#執行訓練資料標準化

A_X_test_scaled = pd.DataFrame(
    std.transform(A_X_test),
    columns = A_X_test.columns)
#執行測試資料標準化

#初始化基於隨機森林樹的分類器
#max_depth預設為 None，表示深度，會持續到無法分割與最大深度為止
#n_estimators為決策樹數量
#random_state為每次隨機抽取的數量，此數值若固定，則每次執行結果都會相同，方便重現，若無設定，則採用特徵的平方根
#max_samples 參數來設定每棵決策樹使用的資料樣本數量，預設為1，表示使用全部資料，若設定0.8，則為80%
A_rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=3)

with joblib.parallel_backend("dask"):
    A_rf.fit(A_X_train_scaled, A_y_train)

#A_model 的特徵重要性排序，feature_importances_存放重要特徵的數值
A_f_i = list(zip(A_selected_columns,A_rf.feature_importances_))
A_f_i.sort(key = lambda x : x[1])
plt.barh([x[0] for x in A_f_i],[x[1] for x in A_f_i])
plt.figure(figsize=(1080, 920))
plt.show()
print(len(A_rf.feature_importances_))

#對A樹做預測
A_y_val_preds = A_rf.predict(A_X_test_scaled)
print("Precision: {}, Recall: {}, F2 Score: {}".format(precision_score(A_y_test, A_y_val_preds), recall_score(A_y_test, A_y_val_preds), fbeta_score(A_y_test, A_y_val_preds, beta=2.0)))

#隨機森林樹，準確度預測
test_score = A_rf.score(A_X_test_scaled, A_y_test) * 100
print(f'Accuracy: {test_score:.5f}%')

#產生混淆矩陣結果
A_rf_confusion = confusion_matrix(A_y_test, A_y_val_preds)

# # 提取'False Neg'的索引
# A_false_neg_indices = np.where((A_y_test == 1) & (A_y_val_preds == 0))[0]
# # 提取'False Neg'對應的紀錄
# A_false_neg_records = df.iloc[A_X_test.index[A_false_neg_indices]]

# # 提取'False Pos'的索引
# A_false_pos_indices = np.where((A_y_test == 0) & (A_y_val_preds == 1))[0]
# # 提取'False Pos'對應的紀錄
# A_false_pos_records = df.iloc[A_X_test.index[A_false_pos_indices]]


# AFN = pd.DataFrame(A_false_neg_records)
# AFN.to_csv("A_Flase_Neg.csv")
# AFP = pd.DataFrame(A_false_pos_records)
# AFP.to_csv("A_Flase_Pos.csv")
# # (AFN['Label'].value_counts()) / len(AFN) * 100

group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
group_counts = ["{0:0.0f}".format(value) for value in A_rf_confusion.flatten()]
group_percentages = ["{0:0.4%}".format(value) for value in A_rf_confusion.flatten()/np.sum(A_rf_confusion)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)

sns.heatmap(A_rf_confusion,cmap='rocket_r', annot=labels, fmt='', square=True, xticklabels=['DDoS', 'Benign'], yticklabels=['DDoS', 'Benign'])
plt.xlabel('Predicted Traffic Type')
plt.ylabel('Actual Traffic Type')
plt.title('A Random Forest Confusion Matrix')

"""接下來分類樹B"""

B_X_train, B_X_test, B_y_train, B_y_test = train_test_split(X_B, y, test_size = .25, stratify=y, random_state=3)

B_std = StandardScaler()

B_X_train_scaled = pd.DataFrame(
    B_std.fit_transform(B_X_train),
    columns = B_X_train.columns)
#執行標準化

B_X_test_scaled = pd.DataFrame(
    B_std.transform(B_X_test),
    columns = B_X_test.columns)
#執行標準化

#初始化基於隨機森林樹的分類器
B_rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=3)

with joblib.parallel_backend("dask"):
    B_rf.fit(B_X_train_scaled, B_y_train)

# 查看每棵樹的深度，無指定的狀況可以查看，相關樹的設定都在estimators_中
trees = B_rf.estimators_

for i, tree in enumerate(trees):
    depth = tree.get_depth()
    print(f"Tree {i + 1} Depth: {depth}")

#B_model 的特徵重要性排序，feature_importances_存放重要特徵的數值
B_f_i = list(zip(B_selected_columns,B_rf.feature_importances_))
B_f_i.sort(key = lambda x : x[1])
plt.barh([x[0] for x in B_f_i],[x[1] for x in B_f_i])
plt.figure(figsize=(1080, 920))
plt.show()
print(len(B_rf.feature_importances_))

B_y_val_preds = B_rf.predict(B_X_test_scaled)

print("Precision: {}\nRecall: {}\nF1 Score: {}\nF2 Score: {}".format(precision_score(B_y_test, B_y_val_preds), recall_score(B_y_test, B_y_val_preds),f1_score(B_y_test, B_y_val_preds), fbeta_score(B_y_test, B_y_val_preds, beta=2.0)))

#隨機森林樹，準確度預測
B_test_score = B_rf.score(B_X_test_scaled, B_y_test) * 100
print(f'Accuracy: {B_test_score:.5f}%')

B_rf_confusion = confusion_matrix(B_y_test, B_y_val_preds)

# # 取得'False Neg'的索引
# B_false_neg_indices = np.where((B_y_test == 1) & (B_y_val_preds == 0))
# # 取得'False Neg'紀錄
# B_false_neg_records = df.iloc[B_X_test.index[B_false_neg_indices]]

# # 取得'False Pos'的索引
# B_false_pos_indices = np.where((B_y_test == 0) & (B_y_val_preds == 1))[0]
# # 取得'False Pos'紀錄
# B_false_pos_records = df.iloc[B_X_test.index[B_false_pos_indices]]



# BFN = pd.DataFrame(B_false_neg_records)
# BFN.to_csv("B_Flase_Neg.csv")
# BFP = pd.DataFrame(B_false_pos_records)
# BFP.to_csv("B_Flase_Pos.csv")
# (BFN['Label'].value_counts()) / len(BFN) * 100

# sameSampleFN = [x for x in B_false_neg_records.index if x in A_false_neg_records.index]
# resultFN = df.iloc[df.index[list(sameSampleFN)]]

# sameSampleFP = [x for x in B_false_pos_records.index if x in A_false_pos_records.index]
# resultFP = df.iloc[df.index[list(sameSampleFP)]]

# pd.DataFrame(resultFN).to_csv("Flase_Neg.csv")
# pd.DataFrame(resultFP).to_csv("Flase_Pos.csv")

B_group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
B_group_counts = ["{0:0.0f}".format(value) for value in B_rf_confusion.flatten()]
B_group_percentages = ["{0:0.4%}".format(value) for value in B_rf_confusion.flatten()/np.sum(B_rf_confusion)]
B_labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(B_group_names,B_group_counts,B_group_percentages)]
B_labels = np.asarray(B_labels).reshape(2,2)

sns.heatmap(B_rf_confusion,cmap='rocket_r', annot=B_labels, fmt='', square=True, xticklabels=['DDoS', 'Benign'], yticklabels=['DDoS', 'Benign'])
plt.xlabel('Predicted Traffic Type')
plt.ylabel('Actual Traffic Type')
plt.title('B Random Forest Confusion Matrix')