import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
from collections import Counter


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=2, num_classes=2, max_length=100):
        super(TransformerClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_length, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size, seq_len = x.size()

        # Получаем эмбеддинги
        x = self.embedding(x)

        # Позиционные эмбеддинги
        positions = torch.arange(0, seq_len).expand(batch_size, seq_len).to(x.device)
        pos_embeddings = self.pos_embedding(positions)

        # Складываем эмбеддинги с позиционными
        x = x + pos_embeddings
        x = self.dropout(x)

        # Пропускаем через трансформер
        x = self.transformer_encoder(x)

        # Берем embedding первого токена
        cls_embedding = x[:, 0, :]

        # Классификация
        output = self.classifier(cls_embedding)
        return output


class TextPreprocessor:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<CLS>': 2}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<CLS>'}
        self.vocab_size = 3
        self.next_idx = 3

    def build_vocab(self, texts, min_freq=1):
        word_counts = Counter()
        for text in texts:
            words = self.tokenize(text)
            word_counts.update(words)

        for word, count in word_counts.items():
            if count >= min_freq:
                self.word2idx[word] = self.next_idx
                self.idx2word[self.next_idx] = word
                self.next_idx += 1

        self.vocab_size = len(self.word2idx)
        print(f"CLS токен имеет индекс: {self.word2idx['<CLS>']}")

    #
    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        return words

    # Текст в индексы
    def text_to_indices(self, text, max_length=50):
        words = self.tokenize(text)

        indices = [self.word2idx['<CLS>']]
        indices.extend([self.word2idx.get(word, 1) for word in words])

        # Обрезаем или дополняем текст
        if len(indices) > max_length:
            indices = indices[:max_length]
        else:
            indices = indices + [0] * (max_length - len(indices))

        return indices


def load_data():
    try:
        df = pd.read_csv(r'.\text\spam.csv', encoding='latin-1')

        if 'v1' in df.columns and 'v2' in df.columns:
            df = df[['v1', 'v2']]
            df.columns = ['label', 'text']
            df['label'] = df['label'].map({'ham': 0, 'spam': 1})
            df = df.dropna()

            return df['text'].tolist(), df['label'].tolist()

    except Exception as e:
        print(f"Ошибка загрузки: {e}")

    print("Тестовые данные")
    texts = [
        "Free money now!!! Click here to win $1000",
        "Hi John, meeting tomorrow at 3pm",
        "Congratulations you won a prize",
        "Can you send me the documents please",
        "URGENT: Your account has been compromised"
    ]
    labels = [1, 0, 1, 0, 1]
    return texts, labels


def main():
    texts, labels = load_data()

    print(f"\nЗагружено {len(texts)} текстов")
    print(f"Распределение меток: {Counter(labels)}")

    preprocessor = TextPreprocessor()
    preprocessor.build_vocab(texts)
    print(f"Размер словаря: {preprocessor.vocab_size}")


    X = [preprocessor.text_to_indices(text) for text in texts]
    X = torch.tensor(X)
    y = torch.tensor(labels)

    # Разделение данных
    if len(X) > 5:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
    else:
        X_train, y_train = X, y
        X_test, y_test = X, y

    print(f"Размер обучающей выборки: {len(X_train)}")
    print(f"Размер тестовой выборки: {len(X_test)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")

    model = TransformerClassifier(
        vocab_size=preprocessor.vocab_size,
        embed_dim=64,
        num_heads=2,
        num_layers=1,
        num_classes=2
    ).to(device)

    print("Модель создана (CLS в данных)")

    # Обучение
    epochs = 100
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\nНачинаем обучение...")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = model(X_train.to(device))
        loss = criterion(outputs, y_train.to(device))

        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    # Оценка
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test.to(device))
        test_preds = torch.argmax(test_outputs, dim=1)
        accuracy = accuracy_score(y_test.numpy(), test_preds.cpu().numpy())
        print(f'\nТочность на тесте: {accuracy:.4f}')

    test_texts = [
        "HI BABE IM AT HOME NOW WANNA DO SOMETHING? XX",
        "\"I call you later, don't have network. If urgnt, sms me.\"",
        "Do you want a lot of money? MORE MONEY! You just need to call. Find out by phone 8(900)100-20-10",
        "FREE entry into our еЈ250 weekly competition just text the word WIN to 80086 NOW. 18 T&C www.txttowin.com"
    ]

    print("\nТестирование модели:")

    for text in test_texts:
        indices = preprocessor.text_to_indices(text)
        input_tensor = torch.tensor([indices]).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1)
            confidence = torch.nn.functional.softmax(output, dim=1)[0]

        pred_label = "СПАМ" if prediction.item() == 1 else "НЕ СПАМ"
        conf_value = confidence[prediction.item()].item()

        print(f"Текст: {text}")
        print(f"Предсказание: {pred_label} (уверенность: {conf_value:.3f})\n")


if __name__ == "__main__":
    main()