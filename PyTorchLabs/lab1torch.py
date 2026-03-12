import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt

# GPU or CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device CUDA: ", torch.cuda.is_available())

# Читаем данные
data = pd.read_csv('Used_Car_Price_Prediction.csv')

# Удаляем не нужные колонки
drop_columns = ['city', 'times_viewed', 'assured_buy', 'registered_city', 'registered_state', 'rto', 'is_hot', 'source', 'car_availability', 'ad_created_on', 'emi_starts_from', 'booking_down_pymnt', 'reserved', 'broker_quote']
data.drop(columns=drop_columns, inplace=True)
data = data.dropna()

y = data['sale_price']  # предсказываем цену
X = data.drop('sale_price', axis=1)  # все остальные колонки как признаки

# Разделяем данные на строковые и числовые
categorical_cols = ['car_name', 'fuel_type', 'body_type', 'transmission', 'variant', 'make', 'model', 'car_rating', 'warranty_avail']
numerical_cols = ['yr_mfr', 'kms_run', 'total_owners', 'original_price']

# Маcштабируем числовые данные
min_max_scaler = MinMaxScaler()
X[numerical_cols] = min_max_scaler.fit_transform(X[numerical_cols])

# Преобразуем строковые данные
X = pd.get_dummies(X, columns=categorical_cols).astype(np.float32)

print(X.head()) # вывод первых данных

# Разделяем выборки на тестовую и обучающею части
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание тензоров
X_train = torch.from_numpy(X_train.values.astype(np.float32))
Y_train = torch.from_numpy(y_train.values.astype(np.float32)).unsqueeze(1)

X_test = torch.from_numpy(X_test.values.astype(np.float32))
Y_test = torch.from_numpy(y_test.values.astype(np.float32)).unsqueeze(1)

# Разбиваем датасет на пакеты
train_ds = TensorDataset(X_train, Y_train)
test_ds = TensorDataset(X_test, Y_test)

BATCH_SIZE = 16
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

len_cols = X.shape[1] # количество параметров
print("Count parametrs: ", len_cols)

# Создание модели
model = nn.Sequential(
    nn.Linear(len_cols, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256,128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128,64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64,1),
)

criterion = nn.MSELoss() # определение функции потерь

# Задание алгоритма оптимизации
LEARNING_RATE = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Обучение модели
EPOCHS = 50
train_losses = []
test_losses = []

for epoch in range(EPOCHS):
    running_loss = 0
    for it, (x, y) in enumerate(train_dl):
        optimizer.zero_grad()
        outp = model(x)
        loss = criterion(outp, y)
        running_loss += loss.item()
        loss.backward()

        optimizer.step()

    avg_train_loss = running_loss / len(train_dl)
    train_losses.append(avg_train_loss)

    # Валидационная фаза (на тестовой выборке)
    model.eval()
    running_test_loss = 0
    with torch.no_grad():  # отключаем вычисление градиентов для экономии памяти
        for x, y in test_dl:
            outp = model(x)
            loss = criterion(outp, y)
            running_test_loss += loss.item()

    avg_test_loss = running_test_loss / len(test_dl)
    test_losses.append(avg_test_loss)

    print(f"Epoch {epoch} - Train loss: {avg_train_loss:.6f}, Test loss: {avg_test_loss:.6f}")

# Финальная оценка модели
model.eval()
with torch.no_grad():
    test_predictions = model(X_test)
    test_loss = criterion(test_predictions, Y_test)
    print(f"\nFinal Test Loss: {test_loss:.6f}")

# Визуализация потерь
plt.figure(figsize=(12, 8))
plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
plt.plot(range(len(test_losses)), test_losses, label='Test Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Test Loss")
plt.show()