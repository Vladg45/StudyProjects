import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
from collections import Counter
import os

# Параметры
BATCH_SIZE = 32
EMBEDDING_DIM = 100
HIDDEN_DIM = 64
NUM_LAYERS = 2
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
MAX_VOCAB_SIZE = 5000
MAX_SEQUENCE_LENGTH = 50

class TextDataset(Dataset):
    def __init__(self, texts, labels, word_to_idx, max_length):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        sequence = []
        for word in text.split()[:self.max_length]:
            sequence.append(self.word_to_idx.get(word, self.word_to_idx['<UNK>']))

        if len(sequence) < self.max_length:
            sequence += [self.word_to_idx['<PAD>']] * (self.max_length - len(sequence))

        return torch.tensor(sequence), torch.tensor(label, dtype=torch.long)

class BalancedLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout=0.3):
        super(BalancedLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)

        # Конкатенируем последние скрытые состояния из обоих направлений
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        output = self.dropout(hidden)
        output = self.fc(output)
        return output

def preprocess_text(text):
    if isinstance(text, float):
        return ""

    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_vocab(texts, max_vocab_size):
    all_words = []
    for text in texts:
        all_words.extend(text.split())

    word_counts = Counter(all_words)
    vocab = ['<PAD>', '<UNK>'] + [word for word, count in word_counts.most_common(max_vocab_size - 2)]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    return word_to_idx, vocab

def create_weighted_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

def main():
    print("Загрузка и предобработка данных...")

    try:
        df = pd.read_csv('./text/spam.csv', encoding='latin-1')
    except:
        df = pd.read_csv('./text/spam.csv', encoding='utf-8')

    df = df[['v1', 'v2']].copy()
    df.columns = ['label', 'text']
    df = df.dropna()

    print(f"Размер датасета: {len(df)}")
    print(f"Распределение меток:\n{df['label'].value_counts()}")

    # Предобработка
    df['text'] = df['text'].apply(preprocess_text)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df = df[df['text'].str.len() > 5]

    # Разделение на train/validation/test
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df['label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['label']
    )

    print(f"Обучающая выборка: {len(train_df)}")
    print(f"Валидационная выборка: {len(val_df)}")
    print(f"Тестовая выборка: {len(test_df)}")

    # Создание словаря
    word_to_idx, vocab = build_vocab(train_df['text'], MAX_VOCAB_SIZE)
    print(f"Размер словаря: {len(vocab)}")

    # Даталоадеры
    train_dataset = TextDataset(train_df['text'].values, train_df['label'].values, word_to_idx, MAX_SEQUENCE_LENGTH)
    val_dataset = TextDataset(val_df['text'].values, val_df['label'].values, word_to_idx, MAX_SEQUENCE_LENGTH)
    test_dataset = TextDataset(test_df['text'].values, test_df['label'].values, word_to_idx, MAX_SEQUENCE_LENGTH)

    # Взвешенный семплер для балансировки классов
    train_sampler = create_weighted_sampler(train_df['label'].values)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Модель
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")

    model = BalancedLSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=2,
        dropout=0.3
    ).to(device)

    # Weighted loss для дисбаланса классов
    class_counts = np.bincount(train_df['label'])
    class_weights = torch.tensor([1.0, class_counts[0] / class_counts[1]], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    # Для хранения метрик
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print("\nНачало обучения...")
    best_val_acc = 0
    best_model_path = 'best_spam_classifier.pth'

    for epoch in range(NUM_EPOCHS):
        # Обучение
        model.train()
        total_train_loss = 0
        train_preds, train_labels = [], []

        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(targets.cpu().numpy())

        # Валидация
        model.eval()
        total_val_loss = 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(targets.cpu().numpy())

        # Метрики
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step(val_acc)

        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f'  -> Сохранена лучшая модель с точностью на валидации: {val_acc:.4f}')

        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}]')
        print(f'  Loss: Train {avg_train_loss:.4f}, Val {avg_val_loss:.4f}')
        print(f'  Acc:  Train {train_acc:.4f}, Val {val_acc:.4f}')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')

    # Загрузка лучшей модели для тестирования
    if os.path.exists(best_model_path):
        print(f"\nЗагрузка лучшей модели из {best_model_path}...")
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
    else:
        print("Предупреждение: файл с лучшей моделью не найден, используется последняя модель.")

    # Финальная оценка на тесте
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(targets.cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)

    print(f"\nРезультаты:")
    print(f"Лучшая точность на валидации: {best_val_acc:.4f}")
    print(f"Финальная точность на тесте: {test_acc:.4f}")

    # Детальный отчет
    print("\nДетальный отчет на тестовой выборке:")
    print(classification_report(test_labels, test_preds, target_names=['ham', 'spam']))

    # Матрица ошибок
    cm = confusion_matrix(test_labels, test_preds)
    print("Матрица ошибок:")
    print(cm)

    # Построение графиков
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()