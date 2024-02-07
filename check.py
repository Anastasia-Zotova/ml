import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

def hyperparameter_tuning(X_train_vectorized, y_train, X_test_vectorized, y_test, alphas):
    best_accuracy = 0
    best_alpha = None

    for alpha in alphas:
        model = MultinomialNB(alpha=alpha)
        model.fit(X_train_vectorized, y_train)
        y_pred = model.predict(X_test_vectorized)
        accuracy = metrics.accuracy_score(y_test, y_pred)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_alpha = alpha

    return best_alpha, best_accuracy


# Загрузка данных для первой модели
with open('train_data_1.json') as f:
    train_data_1 = json.load(f)

with open('test_data_1.json') as f:
    test_data_1 = json.load(f)

# Загрузка данных для второй модели
with open('train_data_2.json') as f:
    train_data_2 = json.load(f)

with open('test_data_2.json') as f:
    test_data_2 = json.load(f)

# Подготовка данных для первой модели
X_train_1 = [' '.join(entry['ingredients']) for entry in train_data_1]
y_train_1 = [entry['cuisine'] for entry in train_data_1]

X_test_1 = [' '.join(entry['ingredients']) for entry in test_data_1]
y_test_1 = [entry['cuisine'] for entry in test_data_1]

# Векторизация текста для первой модели
vectorizer_1 = CountVectorizer()
X_train_vectorized_1 = vectorizer_1.fit_transform(X_train_1)
X_test_vectorized_1 = vectorizer_1.transform(X_test_1)

# Пример перебора значений alpha для первой модели
alphas_1 = [0.1, 0.5, 1.0, 2.0]
best_alpha_1, best_accuracy_1 = hyperparameter_tuning(X_train_vectorized_1, y_train_1, X_test_vectorized_1, y_test_1, alphas_1)

# Обучение первой модели с настройкой параметра alpha
model_1 = MultinomialNB(alpha=best_alpha_1)
model_1.fit(X_train_vectorized_1, y_train_1)

# Предсказание на тестовых данных для первой модели
y_pred_1 = model_1.predict(X_test_vectorized_1)

# Вывод метрик для первой модели
print("Метрики для Модели 1:")
accuracy_1 = metrics.accuracy_score(y_test_1, y_pred_1)
precision_1 = metrics.precision_score(y_test_1, y_pred_1, average='weighted')
recall_1 = metrics.recall_score(y_test_1, y_pred_1, average='weighted')
print("Best alpha for Model 1:", best_alpha_1)
print("Accuracy (доля правильных ответов алгоритма):", accuracy_1)
print("Precision (точность):", precision_1)
print("Recall (полнота):", recall_1)

# Создание DataFrame с результатами предсказаний для первой модели
df_results_1 = pd.DataFrame({'True': y_test_1, 'Predicted': y_pred_1})
df_results_1['Correct'] = df_results_1['True'] == df_results_1['Predicted']

# Создание словаря для первой модели
cuisine_ingredients_1 = {}
for entry in test_data_1:
    cuisine = entry['cuisine']
    ingredients = entry['ingredients']
    if cuisine not in cuisine_ingredients_1:
        cuisine_ingredients_1[cuisine] = []
    cuisine_ingredients_1[cuisine].extend(ingredients)

# Определение максимальной частоты для первой модели
max_frequency = 200  # Измените это значение на нужное вам

# Создание холста для графиков для первой модели
fig1, axs1 = plt.subplots(5, 2, figsize=(12, 15))
fig2, axs2 = plt.subplots(5, 2, figsize=(12, 15))

# Создание графиков для первой модели с наиболее популярными ингредиентами для каждой кухни
for i, (cuisine, ingredients_list) in enumerate(cuisine_ingredients_1.items()):
    counter = Counter(ingredients_list)
    most_common = counter.most_common(5)  # Выбираем 5 наиболее популярных ингредиентов
    labels, values = zip(*most_common)

    if i < 10:
        row, col = divmod(i, 2)
        bars = axs1[row, col].bar(labels, values, color='skyblue')
        axs1[row, col].set_title(f"Топ 5 ингридиентов в {cuisine} кухне (Модель 1)", fontsize=8)
        axs1[row, col].set_ylabel("Частота", fontsize=8)
        axs1[row, col].tick_params(axis='x', rotation=45, labelsize=6)  # Поворот текста
        axs1[row, col].set_ylim(0, max_frequency)
    else:
        row, col = divmod(i - 10, 2)
        bars = axs2[row, col].bar(labels, values, color='skyblue')
        axs2[row, col].set_title(f"Топ 5 ингридиентов в {cuisine} кухне (Модель 1)", fontsize=8)
        axs2[row, col].set_ylabel("Частота", fontsize=8)
        axs2[row, col].tick_params(axis='x', rotation=45, labelsize=6)  # Поворот текста
        axs2[row, col].set_ylim(0, max_frequency)

fig1.tight_layout()
fig2.tight_layout()

# Вывод графиков для первой модели
plt.show()

# Подготовка данных для второй модели
X_train_2 = [' '.join(entry['ingredients']) for entry in train_data_2]
y_train_2 = [entry['cuisine'] for entry in train_data_2]

X_test_2 = [' '.join(entry['ingredients']) for entry in test_data_2]
y_test_2 = [entry['cuisine'] for entry in test_data_2]

# Векторизация текста для второй модели
vectorizer_2 = CountVectorizer()
X_train_vectorized_2 = vectorizer_2.fit_transform(X_train_2)
X_test_vectorized_2 = vectorizer_2.transform(X_test_2)

# Пример перебора значений alpha для второй модели
alphas_2 = [0.1, 0.5, 1.0, 2.0]
best_alpha_2, best_accuracy_2 = hyperparameter_tuning(X_train_vectorized_2, y_train_2, X_test_vectorized_2, y_test_2, alphas_2)

# Обучение второй модели с настройкой параметра alpha
model_2 = MultinomialNB(alpha=best_alpha_2)
model_2.fit(X_train_vectorized_2, y_train_2)

# Предсказание на тестовых данных для второй модели
y_pred_2 = model_2.predict(X_test_vectorized_2)

# Вывод метрик для второй модели
print("\nМетрики для Модели 2:")
accuracy_2 = metrics.accuracy_score(y_test_2, y_pred_2)
precision_2 = metrics.precision_score(y_test_2, y_pred_2, average='weighted')
recall_2 = metrics.recall_score(y_test_2, y_pred_2, average='weighted')
print("Best alpha for Model 2:", best_alpha_2)
print("Accuracy (доля правильных ответов алгоритма):", accuracy_2)
print("Precision (точность):", precision_2)
print("Recall (полнота):", recall_2)

# Создание DataFrame с результатами предсказаний для второй модели
df_results_2 = pd.DataFrame({'True': y_test_2, 'Predicted': y_pred_2})
df_results_2['Correct'] = df_results_2['True'] == df_results_2['Predicted']

# Создание словаря для второй модели
cuisine_ingredients_2 = {}
for entry in test_data_2:
    cuisine = entry['cuisine']
    ingredients = entry['ingredients']
    if cuisine not in cuisine_ingredients_2:
        cuisine_ingredients_2[cuisine] = []
    cuisine_ingredients_2[cuisine].extend(ingredients)

# Определение максимальной частоты для второй модели
max_frequency = 200  # Измените это значение на нужное вам

# Создание холста для графиков для второй модели
fig3, axs3 = plt.subplots(5, 2, figsize=(12, 15))
fig4, axs4 = plt.subplots(5, 2, figsize=(12, 15))

# Создание графиков для второй модели с наиболее популярными ингредиентами для каждой кухни
for i, (cuisine, ingredients_list) in enumerate(cuisine_ingredients_2.items()):
    counter = Counter(ingredients_list)
    most_common = counter.most_common(5)  # Выбираем 5 наиболее популярных ингредиентов
    labels, values = zip(*most_common)

    if i < 10:
        row, col = divmod(i, 2)
        bars = axs3[row, col].bar(labels, values, color='skyblue')
        axs3[row, col].set_title(f"Топ 5 ингридиентов в {cuisine} кухне (Модель 2)", fontsize=8)
        axs3[row, col].set_ylabel("Частота", fontsize=8)
        axs3[row, col].tick_params(axis='x', rotation=45, labelsize=6)  # Поворот текста
        axs3[row, col].set_ylim(0, max_frequency)
    else:
        row, col = divmod(i - 10, 2)
        bars = axs4[row, col].bar(labels, values, color='skyblue')
        axs4[row, col].set_title(f"Топ 5 ингридиентов в {cuisine} кухне (Модель 2)", fontsize=8)
        axs4[row, col].set_ylabel("Частота", fontsize=8)
        axs4[row, col].tick_params(axis='x', rotation=45, labelsize=6)  # Поворот текста
        axs4[row, col].set_ylim(0, max_frequency)

fig3.tight_layout()
fig4.tight_layout()

# Вывод графиков для второй модели
plt.show()


# Сравнение метрик
print("\n\nСравнение метрик между первой и второй моделью:")
print("Accuracy (доля правильных ответов алгоритма):", accuracy_1 - accuracy_2)
print("Precision (точность):", precision_1 - precision_2)
print("Recall (полнота):", recall_1 - recall_2)
