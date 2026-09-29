import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("=== ТЕСТИРОВАНИЕ ===")

# Загружаем тестовые данные
A, B = 1, 4
x_test = np.linspace(A, B, 50)
y_test = 0.25 * pow(x_test, 3) - x_test - 1.2502

# Нормализация (используем те же параметры, что и при обучении)
x_train = np.linspace(2, 3, 100)
x_mean, x_std = x_train.mean(), x_train.std()
x_test_normalized = (x_test - x_mean) / x_std

print(f"Создано {len(x_test)} тестовых примеров")

# Загружаем обученную модель
model = tf.keras.models.load_model('trained_model.keras')
print("Модель загружена")

# Тестируем
test_loss, test_mae = model.evaluate(x_test_normalized, y_test, verbose=1)
print(f"Ошибка на тестовых данных: {test_mae:.4f}")

# Визуализация
y_pred = model.predict(x_test_normalized)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x_test, y_test, 'b-', label='Истинная функция', linewidth=2)
plt.plot(x_test, y_pred, 'ro-', label='Предсказание НС', linewidth=1, markersize=4)
plt.legend()
plt.title('Сравнение истинной функции и предсказания')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

plt.subplot(1, 2, 2)
errors = np.abs(y_test - y_pred.flatten())
plt.plot(x_test, errors, 'g-', label='Абсолютная ошибка', linewidth=2)
plt.legend()
plt.title('Абсолютная ошибка предсказания')
plt.xlabel('x')
plt.ylabel('Ошибка')
plt.grid(True)

plt.tight_layout()
plt.show()

# Вывод нескольких примеров
print("\nПримеры предсказаний:")
for i in range(5):
    print(f"x={x_test[i]:.3f}, Истинное y={y_test[i]:.3f}, Предсказанное y={y_pred[i][0]:.3f}")