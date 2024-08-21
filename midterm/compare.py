import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 修改此變數就好
k = 100
l = 2
t = 0.5


def encoder_and_scaler(table):
    table_x = table.iloc[:, :-1]
    table_y = convert_salary_class(table.iloc[:, -1].values)
    table_x = pd.get_dummies(table_x)
    sc = StandardScaler()
    table_x = sc.fit_transform(table_x.values)
    return table_x, table_y


def convert_salary_class(salary_list):
    return [1 if salary == '>50K' else 0 for salary in salary_list]


def print_metrics(y_true, preds):
    accuracy = accuracy_score(y_true, preds)
    precision = precision_score(y_true, preds)
    recall = recall_score(y_true, preds)
    auc = roc_auc_score(y_true, preds)
    misclassification_error = 1 - accuracy

    print(f'Accuracy score: {accuracy:.4f}')
    print(f'Precision score: {precision:.4f}')
    print(f'Recall score: {recall:.4f}')
    print(f'AUC score: {auc:.4f}')
    print(f'Misclassification Error: {misclassification_error:.4f}')
    print("======================================")


def train_and_predict(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

    # Use RandomForest
    rf_mod = RandomForestClassifier()

    # Fit the models
    rf_mod.fit(X_train, y_train)

    # Predict
    rf_y_pred = rf_mod.predict(X_test)

    # Print metrics
    print_metrics(y_test, rf_y_pred)


# 讀取資料，指定分隔符號為分號
original_data = pd.read_csv('data/adult.csv', delimiter=';')
anon_data = pd.read_csv('anon_k=' + str(k) + '_l=' + str(l) + '_t=' + str(t) + '.csv')

# 將資料進行 One-Hot Encoding
X_original, y_original = encoder_and_scaler(original_data)
X_anon, y_anon = encoder_and_scaler(anon_data)

# 訓練和測試
print("- original data -\n")
train_and_predict(X_original, y_original)

print("- anonymize data -\n")
train_and_predict(X_anon, y_anon)
