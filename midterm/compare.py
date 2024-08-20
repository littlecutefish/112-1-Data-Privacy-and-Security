import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize

# 讀取原始數據和處理後數據
data_original = pd.read_csv('data/adult.csv', sep=';')
data_anon = pd.read_csv('anon_k=2_l=2_t=0.2.csv', sep=',')

# 分割特徵和標籤
X_original = data_original.drop('salary-class', axis=1)
y_original = data_original['salary-class']

X_anon = data_anon.drop('salary-class', axis=1)
y_anon = data_anon['salary-class']

# 分割訓練集和測試集
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_original, y_original, test_size=0.2, random_state=42)
X_train_anon, X_test_anon, y_train_anon, y_test_anon = train_test_split(X_anon, y_anon, test_size=0.2, random_state=42)

# 預處理管道
numeric_features = X_original.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_original.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)])

# SVM 模型
svm_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', SVC(probability=True))])

# 訓練和評估 SVM 模型 (原始數據)
svm_pipeline.fit(X_train_orig, y_train_orig)
y_pred_orig = svm_pipeline.predict(X_test_orig)
y_prob_orig = svm_pipeline.predict_proba(X_test_orig)[:, 1]

# 評估指標 (原始數據)
accuracy_orig = accuracy_score(y_test_orig, y_pred_orig)
precision_orig = precision_score(y_test_orig, y_pred_orig, pos_label='>50K')
recall_orig = recall_score(y_test_orig, y_pred_orig, pos_label='>50K')
auc_orig = roc_auc_score(y_test_orig, y_prob_orig)
cm_orig = confusion_matrix(y_test_orig, y_pred_orig)

print("Original Data - SVM")
print(f"Accuracy: {accuracy_orig}")
print(f"Precision: {precision_orig}")
print(f"Recall: {recall_orig}")
print(f"AUC: {auc_orig}")
print(f"Confusion Matrix:\n {cm_orig}\n")


def convert_intervals_to_midpoints(data):
    for column in data.columns:
        if data[column].dtype == object:
            # 只處理包含區間的列
            if data[column].str.startswith('[').any():
                # 將區間轉換為中點
                data[column] = data[column].apply(lambda x: (float(x[1:-1].split('-')[0]) + float(x[1:-1].split('-')[1])) / 2)
    return data

# 將匿名化數據中的區間轉換為中點
X_train_anon = convert_intervals_to_midpoints(X_train_anon)
X_test_anon = convert_intervals_to_midpoints(X_test_anon)

# 重新訓練和評估模型
svm_pipeline.fit(X_train_anon, y_train_anon)
y_pred_anon = svm_pipeline.predict(X_test_anon)
y_prob_anon = svm_pipeline.predict_proba(X_test_anon)[:, 1]

# 評估指標
accuracy_anon = accuracy_score(y_test_anon, y_pred_anon)
precision_anon = precision_score(y_test_anon, y_pred_anon, pos_label='>50K')
recall_anon = recall_score(y_test_anon, y_pred_anon, pos_label='>50K')
auc_anon = roc_auc_score(y_test_anon, y_prob_anon)
cm_anon = confusion_matrix(y_test_anon, y_pred_anon)

print("Anonymized Data - SVM")
print(f"Accuracy: {accuracy_anon}")
print(f"Precision: {precision_anon}")
print(f"Recall: {recall_anon}")
print(f"AUC: {auc_anon}")
print(f"Confusion Matrix:\n {cm_anon}\n")


# 可以用類似的方式訓練和評估深度學習模型（MLP）
mlp_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', MLPClassifier(max_iter=300))])

# 訓練和評估 MLP 模型 (原始數據)
mlp_pipeline.fit(X_train_orig, y_train_orig)
y_pred_orig_mlp = mlp_pipeline.predict(X_test_orig)
y_prob_orig_mlp = mlp_pipeline.predict_proba(X_test_orig)[:, 1]

# 評估指標 (原始數據)
accuracy_orig_mlp = accuracy_score(y_test_orig, y_pred_orig_mlp)
precision_orig_mlp = precision_score(y_test_orig, y_pred_orig_mlp, pos_label='>50K')
recall_orig_mlp = recall_score(y_test_orig, y_pred_orig_mlp, pos_label='>50K')
auc_orig_mlp = roc_auc_score(y_test_orig, y_prob_orig_mlp)
cm_orig_mlp = confusion_matrix(y_test_orig, y_pred_orig_mlp)

print("Original Data - MLP")
print(f"Accuracy: {accuracy_orig_mlp}")
print(f"Precision: {precision_orig_mlp}")
print(f"Recall: {recall_orig_mlp}")
print(f"AUC: {auc_orig_mlp}")
print(f"Confusion Matrix:\n {cm_orig_mlp}\n")

# 訓練和評估 MLP 模型 (匿名化數據)
mlp_pipeline.fit(X_train_anon, y_train_anon)
y_pred_anon_mlp = mlp_pipeline.predict(X_test_anon)
y_prob_anon_mlp = mlp_pipeline.predict_proba(X_test_anon)[:, 1]

# 評估指標 (匿名化數據)
accuracy_anon_mlp = accuracy_score(y_test_anon, y_pred_anon_mlp)
precision_anon_mlp = precision_score(y_test_anon, y_pred_anon_mlp, pos_label='>50K')
recall_anon_mlp = recall_score(y_test_anon, y_pred_anon_mlp, pos_label='>50K')
auc_anon_mlp = roc_auc_score(y_test_anon, y_prob_anon_mlp)
cm_anon_mlp = confusion_matrix(y_test_anon, y_pred_anon_mlp)

print("Anonymized Data - MLP")
print(f"Accuracy: {accuracy_anon_mlp}")
print(f"Precision: {precision_anon_mlp}")
print(f"Recall: {recall_anon_mlp}")
print(f"AUC: {auc_anon_mlp}")
print(f"Confusion Matrix:\n {cm_anon_mlp}\n")
