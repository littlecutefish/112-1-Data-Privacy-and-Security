import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize

# 修改此變數就好
k = 2

# 讀取資料，指定分隔符號為分號
original_data = pd.read_csv('data/informs.csv', delimiter=';')
anon_data = pd.read_csv('results/informs_mondrian_ldiv_' + str(k) + '.csv', delimiter=';')

# 將資料進行 One-Hot Encoding
X_original = pd.get_dummies(original_data.drop(columns=['poverty']), drop_first=True)
y_original = original_data['poverty']

X_anon = pd.get_dummies(anon_data.drop(columns=['poverty']), drop_first=True)
y_anon = anon_data['poverty']

# 確保訓練和測試資料具有相同的欄位
X_anon = X_anon.reindex(columns=X_original.columns, fill_value=0)

# 訓練和測試隨機森林模型
X_train, X_test, y_train, y_test = train_test_split(X_original, y_original, test_size=0.3, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 預測
y_pred_original = rf_model.predict(X_test)
y_pred_anon = rf_model.predict(X_anon)

# 檢查預測結果
print("Predicted labels (original data):", np.unique(y_pred_original, return_counts=True))
print("Predicted labels (anonymized data):", np.unique(y_pred_anon, return_counts=True))

# 對 y 進行二值化處理
y_test_bin = label_binarize(y_test, classes=[1, 2, 3, 4, 5])
y_anon_bin = label_binarize(y_anon, classes=[1, 2, 3, 4, 5])

# 計算指標
# 'macro': 分別計算每個類別的指標，然後取其未加權的平均值
metrics_original = {
    'Accuracy': accuracy_score(y_test, y_pred_original),
    'Precision': precision_score(y_test, y_pred_original, average='macro', zero_division=0),
    'Recall': recall_score(y_test, y_pred_original, average='macro', zero_division=0),
    'AUC': roc_auc_score(y_test_bin, rf_model.predict_proba(X_test), multi_class='ovr'),
    'Misclassification Error': 1 - accuracy_score(y_test, y_pred_original)
}

metrics_anon = {
    'Accuracy': accuracy_score(y_anon, y_pred_anon),
    'Precision': precision_score(y_anon, y_pred_anon, average='macro', zero_division=0),
    'Recall': recall_score(y_anon, y_pred_anon, average='macro', zero_division=0),
    'AUC': roc_auc_score(y_anon_bin, rf_model.predict_proba(X_anon), multi_class='ovr'),
    'Misclassification Error': 1 - accuracy_score(y_anon, y_pred_anon)
}

# 印出指標
print("Original Data Metrics:")
for metric, value in metrics_original.items():
    print(f"{metric}: {value:.4f}")

print("\nAnonymized Data Metrics with k = " + str(k) + ":")
for metric, value in metrics_anon.items():
    print(f"{metric}: {value:.4f}")

# 查看混淆矩陣
# print("\nConfusion Matrix (original data):")
# print(confusion_matrix(y_test, y_pred_original))
#
# print("\nConfusion Matrix (anonymized data):")
# print(confusion_matrix(y_anon, y_pred_anon))
