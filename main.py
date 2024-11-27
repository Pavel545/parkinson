import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
url = 'data/parkinsons.data'
data = pd.read_csv(url)

# Удаление ненужного столбца
data.drop(columns=['name'], inplace=True)

# Определение признаков и меток
X = data.drop(columns=['status'])
y = data['status']

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Нормализация данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Создание модели XGBoost
model = xgb.XGBClassifier(eval_metric='logloss')

# Обучение модели
model.fit(X_train_scaled, y_train)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test_scaled)

# Оценка точности
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Визуализация важности признаков
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, importance_type='weight', title='Feature Importance', xlabel='F оценка', ylabel='Функции')
plt.show()

# Построение матрицы путаницы
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Sick'], yticklabels=['Healthy', 'Sick'])
plt.title('Матрица')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
