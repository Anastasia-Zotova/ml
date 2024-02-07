import json
from sklearn.model_selection import train_test_split

# Чтение данных из файла data.json
with open('data.json') as f:
    data = json.load(f)

# Извлечение признаков (ингредиентов) и меток (кухни)
features = [entry['ingredients'] for entry in data]
labels = [entry['cuisine'] for entry in data]

# Разделение данных на обучающую (70%) и тестовую (30%) выборки для первой модели
train_data_1, test_data_1 = train_test_split(
    data, test_size=0.3, random_state=42
)

# Генерация нового значения random_state для второй модели
random_state_2 = 42 + 1  # Просто добавим 1 к предыдущему значению

# Разделение данных на обучающую (70%) и тестовую (30%) выборки для второй модели
train_data_2, test_data_2 = train_test_split(
    data, test_size=0.3, random_state=random_state_2
)

# Создание обучающей выборки для первой модели
with open('train_data_1.json', 'w', encoding='utf-8') as train_file_1:
    json.dump(train_data_1, train_file_1, ensure_ascii=False, indent=4)

# Создание тестовой выборки для первой модели
with open('test_data_1.json', 'w', encoding='utf-8') as test_file_1:
    json.dump(test_data_1, test_file_1, ensure_ascii=False, indent=4)

# Создание обучающей выборки для второй модели
with open('train_data_2.json', 'w', encoding='utf-8') as train_file_2:
    json.dump(train_data_2, train_file_2, ensure_ascii=False, indent=4)

# Создание тестовой выборки для второй модели
with open('test_data_2.json', 'w', encoding='utf-8') as test_file_2:
    json.dump(test_data_2, test_file_2, ensure_ascii=False, indent=4)

# Вывод размеров выборок
print(f"Размер обучающей выборки для первой модели: {len(train_data_1)}")
print(f"Размер тестовой выборки для первой модели: {len(test_data_1)}")
print(f"Размер обучающей выборки для второй модели: {len(train_data_2)}")
print(f"Размер тестовой выборки для второй модели: {len(test_data_2)}")
