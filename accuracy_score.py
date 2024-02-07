import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Загрузка и предобработка данных
def load_data(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return pd.DataFrame(data)

# Загрузка данных для обучения
train_data = load_data('train_data_1.json')

# Загрузка данных для тестирования
test_data = load_data('test_data_1.json')

# Подготовка данных для обучения
X_train = train_data['ingredients'].apply(lambda x: ' '.join(x))
y_train = train_data['cuisine']

# Подготовка данных для тестирования
X_test = test_data['ingredients'].apply(lambda x: ' '.join(x))
y_test = test_data['cuisine']

# Векторизация текста
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Обучение модели
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Предсказание на тестовых данных
y_pred = model.predict(X_test_vectorized)

# Оценка точности
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность: {accuracy}')

# Создание DataFrame с результатами предсказаний
df_results = pd.DataFrame({'True': y_test, 'Predicted': y_pred})
df_results['Correct'] = df_results['True'] == df_results['Predicted']

# Столбчатая диаграмма для сравнения предсказанных и фактических значений
plt.figure(figsize=(16, 8))
sns.countplot(x='True', hue='Correct', data=df_results)
plt.title('Количество правильных и неправильных предсказаний для каждой кухни')
plt.xlabel('Кухня')
plt.ylabel('Количество')
plt.xticks(rotation=45, ha='right')  # Поворот текста на оси x
plt.show()

# Создание второго графика
# Подсчет количества блюд для каждой кухни
counts = df_results['True'].value_counts()
df_counts = pd.DataFrame({'Cuisine': counts.index, 'Total': counts.values})

# Объединение DataFrame
df_combined = pd.merge(df_results, df_counts, left_on='True', right_on='Cuisine')

# Построение графика с отдельной линией для каждой кухни
plt.figure(figsize=(16, 8))
for cuisine in df_combined['Cuisine'].unique():
    subset = df_combined[df_combined['Cuisine'] == cuisine]
    sns.lineplot(x='Total', y='Correct', data=subset, marker='o', label=cuisine)

# Форматирование меток делений на вертикальной оси в процентах
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))

# Установка шага оси x в 100
plt.xticks(range(0, max(df_combined['Total'])+100, 100), rotation=45)

plt.title('Зависимость % верно найденных блюд от общего количества блюд кухни')
plt.xlabel('Общее количество блюд')
plt.ylabel('% верно найденных блюд')
plt.legend(title='Кухня', loc='upper right')
plt.show()
