import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# КОНФИГУРАЦИЯ И НАСТРОЙКИ
class Config:
    """Класс для хранения всех настроек обучения"""
    # Пути к данным
    DATA_DIR = "./semantic_images"
    CSV_PATH = "./semantic_images/df.csv"

    # Параметры модели
    IMAGE_SIZE = 256
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4

    # Пути для сохранения
    MODEL_SAVE_PATH = "unet_human_segmentation.pth"
    PLOTS_SAVE_PATH = "training_plots.png"


# Проверяем доступность GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")


# ДАТАСЕТ И ЗАГРУЗЧИК ДАННЫХ
class HumanSegmentationDataset(Dataset):
    """Кастомный датасет для сегментации людей"""

    def __init__(self, dataframe, transform=None, image_size=256, is_train=True):
        """
        Args:
            dataframe: DataFrame с путями к изображениям
            transform: трансформации для аугментации
            image_size: размер изображения для resize
            is_train: флаг обучения (для аугментаций)
        """
        self.df = dataframe
        self.image_size = image_size
        self.is_train = is_train

        # Базовые трансформации
        self.resize = T.Resize((image_size, image_size))
        self.to_tensor = T.ToTensor()

        # Аугментации для тренировочных данных
        if is_train and transform is None:
            self.transform = T.Compose([
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.RandomHorizontalFlip(0.5),
            ])
        else:
            self.transform = transform

    def __len__(self):
        """Возвращает количество примеров в датасете"""
        return len(self.df)

    def __getitem__(self, idx):
        """Загружает и возвращает один пример (изображение, маска)"""

        # Получаем пути из DataFrame
        image_path = os.path.join(Config.DATA_DIR, self.df.iloc[idx]['images'])
        mask_path = os.path.join(Config.DATA_DIR, self.df.iloc[idx]['masks'])

        # Загружаем изображение и маску
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 'L' для grayscale (1 канал)

        # Изменяем размер
        image = self.resize(image)
        mask = self.resize(mask)

        # Применяем аугментации (только для тренировочных данных)
        if self.is_train and self.transform:
            image = self.transform(image)
            # Для масок используем те же трансформации, но без цветовых изменений
            if isinstance(self.transform, T.Compose):
                # Применяем только геометрические трансформации
                for transform in self.transform.transforms:
                    if isinstance(transform, (T.RandomHorizontalFlip, T.RandomRotation)):
                        mask = transform(mask)

        # Преобразуем в тензоры
        image = self.to_tensor(image)
        mask = self.to_tensor(mask)

        # Нормализуем изображение (значения от 0 до 1)
        image = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])(image)

        # Для бинарной маски: белый (255) -> 1, черный (0) -> 0
        mask = (mask > 0.5).float()

        return image, mask


# АРХИТЕКТУРА U-Net
class DoubleConv(nn.Module):
    """Блок из двух сверточных слоев (Conv2d -> BatchNorm -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # Первая свертка
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # Вторая свертка
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """Полная архитектура U-Net для семантической сегментации"""

    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # ENCODER (Путь вниз - сжатие)

        # Первый блок энкодера (64 канала)
        self.enc1 = DoubleConv(in_channels, 64)
        # Второй блок энкодера (128 каналов)
        self.enc2 = DoubleConv(64, 128)
        # Третий блок энкодера (256 каналов)
        self.enc3 = DoubleConv(128, 256)
        # Четвертый блок энкодера (512 каналов)
        self.enc4 = DoubleConv(256, 512)

        # Операция пулинга (уменьшение размера в 2 раза)
        self.pool = nn.MaxPool2d(2)

        # BOTTLENECK (Самое узкое место)
        self.bottleneck = DoubleConv(512, 1024)

        # DECODER (Путь вверх - расширение)

        # Восходящая свертка 4 (1024 -> 512)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        # Декодер блок 4 (512 + 512 = 1024 -> 512)
        self.dec4 = DoubleConv(1024, 512)

        # Восходящая свертка 3 (512 -> 256)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # Декодер блок 3 (256 + 256 = 512 -> 256)
        self.dec3 = DoubleConv(512, 256)

        # Восходящая свертка 2 (256 -> 128)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # Декодер блок 2 (128 + 128 = 256 -> 128)
        self.dec2 = DoubleConv(256, 128)

        # Восходящая свертка 1 (128 -> 64)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # Декодер блок 1 (64 + 64 = 128 -> 64)
        self.dec1 = DoubleConv(128, 64)

        # Финальный сверточный слой (64 -> out_channels)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        """Прямой проход через сеть"""

        # ENCODER
        # e1: [B, 64, H, W]
        e1 = self.enc1(x)
        # e2: [B, 128, H/2, W/2]
        e2 = self.enc2(self.pool(e1))
        # e3: [B, 256, H/4, W/4]
        e3 = self.enc3(self.pool(e2))
        # e4: [B, 512, H/8, W/8]
        e4 = self.enc4(self.pool(e3))

        # BOTTLENECK
        # bottleneck: [B, 1024, H/16, W/16]
        bottleneck = self.bottleneck(self.pool(e4))

        # DECODER

        # Шаг 4: [B, 512, H/8, W/8] + skip connection e4
        d4 = self.upconv4(bottleneck)
        d4 = torch.cat([d4, e4], dim=1)  # Конкатенация по каналам
        d4 = self.dec4(d4)

        # Шаг 3: [B, 256, H/4, W/4] + skip connection e3
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        # Шаг 2: [B, 128, H/2, W/2] + skip connection e2
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        # Шаг 1: [B, 64, H, W] + skip connection e1
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Финальный слой + сигмоида для бинарной классификации
        output = torch.sigmoid(self.final_conv(d1))

        return output


# ФУНКЦИИ ДЛЯ ОЦЕНКИ КАЧЕСТВА
def dice_coefficient(pred, target, smooth=1e-6):
    """
    Вычисляет Dice coefficient для оценки качества сегментации
    Dice = (2 * |A ∩ B|) / (|A| + |B|)
    """
    # Выравниваем тензоры в 1D
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    # Вычисляем пересечение и объединение
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

    return dice


def iou_score(pred, target, smooth=1e-6):
    """
    Вычисляет Intersection over Union (IoU)
    IoU = |A ∩ B| / |A ∪ B|
    """
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou


# ПРОЦЕСС ОБУЧЕНИЯ
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Одна эпоха обучения"""
    model.train()  # Режим обучения
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    for batch_idx, (images, masks) in enumerate(dataloader):
        # Перемещаем данные на устройство (GPU/CPU)
        images = images.to(device)
        masks = masks.to(device)

        # Обнуляем градиенты
        optimizer.zero_grad()

        # Прямой проход
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Обратный проход
        loss.backward()
        optimizer.step()

        # Вычисляем метрики
        with torch.no_grad():
            pred_masks = (outputs > 0.5).float()
            dice = dice_coefficient(pred_masks, masks)
            iou = iou_score(pred_masks, masks)

        # Накопление статистики
        running_loss += loss.item()
        running_dice += dice.item()
        running_iou += iou.item()

        # Прогресс каждые 10 батчей
        if batch_idx % 10 == 0:
            print(f'    Батч {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')

    # Средние значения за эпоху
    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    epoch_iou = running_iou / len(dataloader)

    return epoch_loss, epoch_dice, epoch_iou


def validate_epoch(model, dataloader, criterion, device):
    """Одна эпоха валидации"""
    model.eval()  # Режим оценки
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    with torch.no_grad():  # Отключаем вычисление градиентов
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            pred_masks = (outputs > 0.5).float()
            dice = dice_coefficient(pred_masks, masks)
            iou = iou_score(pred_masks, masks)

            running_loss += loss.item()
            running_dice += dice.item()
            running_iou += iou.item()

    # Средние значения за эпоху
    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    epoch_iou = running_iou / len(dataloader)

    return epoch_loss, epoch_dice, epoch_iou


def train_model(model, train_loader, val_loader, num_epochs, device):
    """Полный процесс обучения модели"""

    # Функция потерь и оптимизатор
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Планировщик learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )

    # История обучения
    history = {
        'train_loss': [], 'val_loss': [],
        'train_dice': [], 'val_dice': [],
        'train_iou': [], 'val_iou': []
    }

    print("Начинаем обучение модели...")
    print("-" * 60)

    for epoch in range(num_epochs):
        print(f'Эпоха [{epoch + 1}/{num_epochs}]')

        # Обучение
        train_loss, train_dice, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Валидация
        val_loss, val_dice, val_iou = validate_epoch(
            model, val_loader, criterion, device
        )

        # Обновление learning rate
        scheduler.step(val_loss)

        # Сохранение истории
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)

        # Вывод результатов эпохи
        print(f'  Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}')
        print(f'  Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}')
        print("-" * 50)

        # Сохранение лучшей модели
        if val_dice == max(history['val_dice']):
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  ✅ Сохранена лучшая модель (Dice: {val_dice:.4f})")

    print("Обучение завершено!")
    return history


# ВИЗУАЛИЗАЦИЯ
def plot_training_history(history):
    """Визуализация истории обучения"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_title('Функция потерь')
    axes[0].set_xlabel('Эпоха')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Dice
    axes[1].plot(history['train_dice'], label='Train Dice')
    axes[1].plot(history['val_dice'], label='Val Dice')
    axes[1].set_title('Dice Coefficient')
    axes[1].set_xlabel('Эпоха')
    axes[1].set_ylabel('Dice')
    axes[1].legend()
    axes[1].grid(True)

    # IoU
    axes[2].plot(history['train_iou'], label='Train IoU')
    axes[2].plot(history['val_iou'], label='Val IoU')
    axes[2].set_title('IoU Score')
    axes[2].set_xlabel('Эпоха')
    axes[2].set_ylabel('IoU')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(Config.PLOTS_SAVE_PATH, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_predictions(model, dataloader, device, num_examples=3):
    """Визуализация примеров предсказаний"""
    model.eval()

    # Берем несколько примеров из датасета
    examples = []
    for images, masks in dataloader:
        examples.append((images, masks))
        if len(examples) >= num_examples:
            break

    fig, axes = plt.subplots(num_examples, 3, figsize=(12, 4 * num_examples))

    for i, (images, true_masks) in enumerate(examples):
        if i >= num_examples:
            break

        # Предсказание
        with torch.no_grad():
            images = images.to(device)
            preds = model(images)
            pred_masks = (preds > 0.5).float()

        # Перемещаем обратно на CPU для визуализации
        image = images[0].cpu()
        true_mask = true_masks[0].cpu()
        pred_mask = pred_masks[0].cpu()

        # Денормализация изображения
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
        image = torch.clamp(image, 0, 1)

        # Визуализация
        axes[i, 0].imshow(image.permute(1, 2, 0))
        axes[i, 0].set_title('Исходное изображение')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(true_mask.squeeze(), cmap='gray')
        axes[i, 1].set_title('Истинная маска')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_mask.squeeze(), cmap='gray')
        axes[i, 2].set_title('Предсказанная маска')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig('predictions_examples.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Основная функция для запуска всего пайплайна"""

    print("🚀 Запуск пайплайна обучения U-Net для сегментации людей")
    print("=" * 60)

    # Загрузка и подготовка данных
    print("Загрузка данных...")
    df = pd.read_csv(Config.CSV_PATH)
    print(f"   Загружено {len(df)} примеров")

    # Разделение на train/val
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"   Train: {len(train_df)} примеров")
    print(f"   Val: {len(val_df)} примеров")

    # Создание датасетов и загрузчиков
    print("Создание датасетов...")
    train_dataset = HumanSegmentationDataset(train_df, is_train=True)
    val_dataset = HumanSegmentationDataset(val_df, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    # Создание модели
    print("Создание модели U-Net...")
    model = UNet(in_channels=3, out_channels=1)
    model = model.to(device)

    # Подсчет параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Всего параметров: {total_params:,}")
    print(f"   Обучаемых параметров: {trainable_params:,}")

    # Обучение модели
    print("Начало обучения...")
    history = train_model(model, train_loader, val_loader, Config.NUM_EPOCHS, device)

    # Визуализация результатов
    print("Визуализация результатов...")
    plot_training_history(history)

    # Загружаем лучшую модель для визуализации
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    visualize_predictions(model, val_loader, device)

    # 6. Финальная статистика
    best_dice = max(history['val_dice'])
    best_iou = max(history['val_iou'])
    print("=" * 60)
    print("🎯 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ:")
    print(f"   Лучший Dice на валидации: {best_dice:.4f}")
    print(f"   Лучший IoU на валидации: {best_iou:.4f}")
    print(f"   Модель сохранена: {Config.MODEL_SAVE_PATH}")
    print("=" * 60)


# ФУНКЦИЯ ДЛЯ ПРЕДСКАЗАНИЯ НА НОВЫХ ИЗОБРАЖЕНИЯХ
def predict_single_image(model_path, image_path, output_path=None, device='cuda'):
    """
    Предсказание маски для одного изображения

    Args:
        model_path: путь к сохраненной модели
        image_path: путь к изображению для сегментации
        output_path: путь для сохранения результата (опционально)
        device: устройство для inference
    """

    # Загрузка модели
    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Загрузка и обработка изображения
    original_image = Image.open(image_path).convert('RGB')
    original_size = original_image.size

    # Трансформации
    transform = T.Compose([
        T.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Применяем трансформации
    image_tensor = transform(original_image).unsqueeze(0).to(device)

    # Предсказание
    with torch.no_grad():
        prediction = model(image_tensor)
        pred_mask = (prediction.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    # Возвращаем маску к исходному размеру
    pred_mask_resized = Image.fromarray(pred_mask * 255)
    pred_mask_resized = pred_mask_resized.resize(original_size, Image.NEAREST)

    # Сохранение если указан путь
    if output_path:
        pred_mask_resized.save(output_path)
        print(f"Маска сохранена: {output_path}")

    return pred_mask_resized, original_image


if __name__ == "__main__":
    # Запускаем основной пайплайн
    main()

    # Пример использования для предсказания на новых изображениях:
    # model_path = "unet_human_segmentation.pth"
    # image_path = "ваше_изображение.jpg"
    # mask, original = predict_single_image(model_path, image_path, "результат.png")