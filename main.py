import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
import time

# Генерация данных
A, B = 2, 3
x_train = np.linspace(A, B, 100)
y_train = 0.25 * pow(x_train, 3) - x_train - 1.2502

# Нормализация данных (важно для стабильного обучения)
x_mean, x_std = x_train.mean(), x_train.std()
x_train_normalized = (x_train - x_mean) / x_std

# Создание улучшенной модели
model = Sequential([
    Dense(32, activation='swish', input_shape=(1,)),
    Dense(16, activation='swish'),
    Dense(8, activation='swish'),
    Dense(1)  # Выходной слой без активации для регрессии
])

# Компиляция модели с другим оптимизатором
model.compile(optimizer='adam', loss='mae', metrics=['mae'])

start_time = time.time()
# Обучение модели с большим количеством эпох
history = model.fit(
    x_train_normalized, y_train,
    epochs=16,
    # callbacks = [EarlyStopping(patience=10, min_delta=0.05)],
    verbose=1,
    validation_split=0.1,
    batch_size=8
)
end_time = time.time()
training_time = end_time - start_time

print(f"Время обучения: {training_time:.2f} секунд")
print(f"Время обучения: {training_time/60:.2f} минут")
# Проверка точности
loss, mae = model.evaluate(x_train_normalized, y_train, verbose=1)
print(f"Средняя абсолютная ошибка: {mae:.4f}")

# Визуализация процесса обучения
plt.figure(figsize=(8, 6))

# plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Тренировочный набор')
plt.plot(history.history['val_loss'], label='Проверочный набор')
plt.title('Функция потерь')
plt.ylabel('Значение')
plt.xlabel('Эпохи')
plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history.history['mae'], label='Training MAE')
# plt.plot(history.history['val_mae'], label='Validation MAE')
# plt.title('Model MAE')
# plt.ylabel('MAE')
# plt.xlabel('Epoch')
# plt.legend()
#
plt.tight_layout()
plt.show()

model.save('trained_model.keras')
print("Обученная модель сохранена как 'trained_model.keras'")