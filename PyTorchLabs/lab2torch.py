import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Устройство
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device CUDA:", torch.cuda.is_available())

# Подготовка данных
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='data', train=False, transform=transform)

BATCH_SIZE = 64
train_dl = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

# Модель
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Dropout(0.25),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Dropout(0.25),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.drop_out = nn.Dropout()

        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


# Обучение
EPOCHS = 20
learning_rate = 1e-3

model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_dl)
train_loss_per_epoch, train_acc_per_epoch, test_loss_per_epoch, test_acc_per_epoch = [], [], [], []

for epoch in range(EPOCHS):
    model.train()
    running_loss, running_correct, total = 0.0, 0, 0

    for i, (images, labels) in enumerate(train_dl):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        running_correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_train_loss = running_loss / total
    epoch_train_acc = running_correct / total
    train_loss_per_epoch.append(epoch_train_loss)
    train_acc_per_epoch.append(epoch_train_acc)

    # Тестирование
    model.eval()
    test_loss, test_correct, test_total = 0.0, 0, 0
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_dl:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

    epoch_test_loss = test_loss / test_total
    epoch_test_acc = test_correct / test_total
    test_loss_per_epoch.append(epoch_test_loss)
    test_acc_per_epoch.append(epoch_test_acc)

    print(f"Epoch [{epoch + 1}/{EPOCHS}] "
          f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc * 100:.2f}% | "
          f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc * 100:.2f}%")


epochs = range(1, EPOCHS + 1)
plt.figure(figsize=(12, 5))

# loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss_per_epoch, 'r-', label='Train Loss')
plt.plot(epochs, test_loss_per_epoch, 'b-', label='Test Loss')
plt.title('Training & Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc_per_epoch, 'r-', label='Train Accuracy')
plt.plot(epochs, test_acc_per_epoch, 'b-', label='Test Accuracy')
plt.title('Training & Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()