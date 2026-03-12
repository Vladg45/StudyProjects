import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
import time

# Настройки
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Параметры для Fashion MNIST
latent_dim = 100
image_size = 32
channels = 1  # MNIST - черно-белые изображения
batch_size = 128
num_epochs = 50
lr = 0.0002
save_interval = 10

# Создание папок для результатов
os.makedirs("generated_fashion", exist_ok=True)
os.makedirs("trained_models", exist_ok=True)


# Архитектура генератора для Fashion MNIST
class Generator(nn.Module):
    def __init__(self, latent_dim, channels, feature_map_size=64):
        super(Generator, self).__init__()

        self.init_size = image_size // 4  # 32//4 = 8
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, feature_map_size * 8 * self.init_size ** 2)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(feature_map_size * 8),
            nn.Upsample(scale_factor=2),  # 8x8 -> 16x16
            nn.Conv2d(feature_map_size * 8, feature_map_size * 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(feature_map_size * 4, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),  # 16x16 -> 32x32
            nn.Conv2d(feature_map_size * 4, feature_map_size * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(feature_map_size * 2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map_size * 2, feature_map_size, 3, stride=1, padding=1),
            nn.BatchNorm2d(feature_map_size, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map_size, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


# Архитектура дискриминатора для Fashion MNIST
class Discriminator(nn.Module):
    def __init__(self, channels, feature_map_size=64):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, feature_map_size, bn=False),  # 32x32 -> 16x16
            *discriminator_block(feature_map_size, feature_map_size * 2),  # 16x16 -> 8x8
            *discriminator_block(feature_map_size * 2, feature_map_size * 4),  # 8x8 -> 4x4
            *discriminator_block(feature_map_size * 4, feature_map_size * 8),  # 4x4 -> 2x2
        )

        # Размер после сверток: 2x2
        ds_size = 2
        self.adv_layer = nn.Sequential(
            nn.Linear(feature_map_size * 8 * ds_size ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


# Загрузка и подготовка данных Fashion MNIST
def prepare_data():
    print("Loading Fashion MNIST dataset...")

    # Трансформации для Fashion MNIST
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Для одного канала
    ])

    try:
        # Загрузка тренировочного датасета
        train_dataset = torchvision.datasets.FashionMNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )

        # DataLoader
        dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        print(f"Training dataset: {len(train_dataset)} images")
        print(f"Image size: {image_size}x{image_size}")
        print(f"Number of classes: 10")

        # Информация о классах
        classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        print("Classes:", classes)

        return dataloader, train_dataset

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None


# Визуализация примеров из датасета
def visualize_dataset_samples(dataloader):
    try:
        batch = next(iter(dataloader))
        images, labels = batch

        # Денормализация для отображения
        images = images * 0.5 + 0.5

        # Классы Fashion MNIST
        classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        for i, ax in enumerate(axes.flat):
            if i < min(10, len(images)):
                img = images[i].squeeze().numpy()  # Убираем dimension канала
                ax.imshow(img, cmap='gray')
                ax.set_title(f"{classes[labels[i].item()]}")
                ax.axis('off')

        plt.suptitle(f"Примеры из Fashion MNIST ({image_size}x{image_size})")
        plt.tight_layout()
        plt.savefig("fashion_dataset_samples.png", dpi=150, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"Error visualizing dataset: {e}")


# Функции для сохранения и визуализации
def save_generated_images(epoch, generator, samples=16):
    """Сохраняет сгенерированные изображения одежды"""
    generator.eval()
    with torch.no_grad():
        z = torch.randn(samples, latent_dim).to(device)
        gen_imgs = generator(z)

        save_image(gen_imgs.data, f"generated_fashion/epoch_{epoch:03d}.png",
                   nrow=4, normalize=True)

    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    gen_imgs_denorm = gen_imgs * 0.5 + 0.5
    for i, ax in enumerate(axes.flat):
        if i < 4:
            img = gen_imgs_denorm[i].cpu().squeeze().numpy()  # Убираем dimension канала
            ax.imshow(img, cmap='gray')
            ax.axis('off')
    plt.suptitle(f"Сгенерированная одежда - Эпоха {epoch} ({image_size}x{image_size})")
    plt.tight_layout()
    plt.savefig(f"generated_fashion/epoch_{epoch:03d}_preview.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved generated fashion images for epoch {epoch}")


def plot_training_progress(g_losses, d_losses):
    """Визуализация процесса обучения"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(g_losses, label='Generator Loss', alpha=0.7, color='blue', linewidth=0.8)
    plt.plot(d_losses, label='Discriminator Loss', alpha=0.7, color='red', linewidth=0.8)
    plt.title('Loss During Training')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    # Сглаживаем для лучшей визуализации
    window = min(50, len(g_losses) // 10)
    if window > 1:
        g_smooth = np.convolve(g_losses, np.ones(window) / window, mode='valid')
        d_smooth = np.convolve(d_losses, np.ones(window) / window, mode='valid')
        plt.plot(g_smooth, label='Generator Loss (smoothed)', color='blue')
        plt.plot(d_smooth, label='Discriminator Loss (smoothed)', color='red')
        plt.title('Smoothed Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("fashion_training_progress.png", dpi=150, bbox_inches='tight')
    plt.show()


# Сохранение моделей
def save_models(generator, discriminator, epoch=None):
    """Сохранение обученных моделей"""
    try:
        if epoch is not None:
            torch.save(generator.state_dict(), f"trained_models/fashion_generator_epoch_{epoch}.pth")
            torch.save(discriminator.state_dict(), f"trained_models/fashion_discriminator_epoch_{epoch}.pth")
        else:
            torch.save(generator.state_dict(), "trained_models/fashion_generator_final.pth")
            torch.save(discriminator.state_dict(), "trained_models/fashion_discriminator_final.pth")

        print("Models saved successfully!")
    except Exception as e:
        print(f"Error saving models: {e}")


# Основная функция обучения
def train_gan():
    # Загрузка данных
    dataloader, dataset = prepare_data()
    if dataloader is None:
        print("Failed to load dataset. Exiting...")
        return None, None, [], []

    # Визуализация примеров данных
    visualize_dataset_samples(dataloader)

    # Инициализация моделей
    generator = Generator(latent_dim, channels).to(device)
    discriminator = Discriminator(channels).to(device)

    # Вывод информации о моделях
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

    # Функция потерь и оптимизаторы
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    g_losses = []
    d_losses = []

    print(f"\nStarting GAN training for {num_epochs} epochs...")
    print(f"Dataset: Fashion MNIST")
    print(f"Image size: {image_size}x{image_size}, Channels: {channels}")
    print(f"Batch size: {batch_size}, Learning rate: {lr}")
    print("-" * 50)

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        num_batches = 0

        for i, (real_imgs, _) in enumerate(dataloader):
            batch_size_current = real_imgs.shape[0]

            # Действительные и фейковые метки
            valid = torch.ones(batch_size_current, 1, device=device)
            fake = torch.zeros(batch_size_current, 1, device=device)

            real_imgs = real_imgs.to(device)

            #  Обучение генератора
            optimizer_G.zero_grad()

            z = torch.randn(batch_size_current, latent_dim, device=device)
            gen_imgs = generator(z)

            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            #  Обучение дискриминатора
            optimizer_D.zero_grad()

            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # Сохранение потерь
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            num_batches += 1

            if i % 50 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

        # Средние потери за эпоху
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches

        print(f"Epoch {epoch} completed - Avg D loss: {avg_d_loss:.4f}, Avg G loss: {avg_g_loss:.4f}")

        # Сохранение и визуализация
        if epoch % save_interval == 0 or epoch == num_epochs - 1:
            save_generated_images(epoch, generator)
            save_models(generator, discriminator, epoch)

    # Завершение обучения
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")

    # Финальное сохранение
    save_generated_images(num_epochs, generator)
    save_models(generator, discriminator)

    # Визуализация прогресса
    plot_training_progress(g_losses, d_losses)

    return generator, discriminator, g_losses, d_losses


# Функция для генерации новых изображений одежды
def generate_new_fashion(generator, num_images=16, save_name="final_generated_fashion"):
    """Генерация новых изображений одежды"""
    if generator is None:
        print("Generator is not available")
        return

    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim).to(device)
        generated_images = generator(z)

        # Сохранение в высоком качестве
        save_image(generated_images.data, f"{save_name}.png",
                   nrow=4, normalize=True)

        # Визуализация
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        generated_images_denorm = generated_images * 0.5 + 0.5

        for i, ax in enumerate(axes.flat):
            if i < num_images:
                img = generated_images_denorm[i].cpu().squeeze().numpy()
                ax.imshow(img, cmap='gray')
                ax.axis('off')

        plt.suptitle(f"Финальная сгенерированная одежда ({image_size}x{image_size})", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{save_name}_large.png", dpi=300, bbox_inches='tight')
        plt.show()

    print(f"Generated {num_images} fashion images saved as '{save_name}.png'")


# Функция для интерполяции между стилями одежды
def interpolate_fashion(generator, num_steps=8):
    """Интерполяция между двумя сгенерированными стилями одежды"""
    generator.eval()
    with torch.no_grad():
        z1 = torch.randn(1, latent_dim).to(device)
        z2 = torch.randn(1, latent_dim).to(device)

        interpolated_images = []
        for alpha in np.linspace(0, 1, num_steps):
            z = alpha * z1 + (1 - alpha) * z2
            img = generator(z)
            interpolated_images.append(img)

        # Объединяем все изображения
        interpolated_images = torch.cat(interpolated_images, 0)
        save_image(interpolated_images.data, "interpolated_fashion.png",
                   nrow=num_steps, normalize=True)

        # Визуализация
        fig, axes = plt.subplots(1, num_steps, figsize=(15, 3))
        interpolated_denorm = interpolated_images * 0.5 + 0.5

        for i, ax in enumerate(axes):
            img = interpolated_denorm[i].cpu().squeeze().numpy()
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            ax.set_title(f"Step {i + 1}")

        plt.suptitle("Интерполяция между стилями одежды", fontsize=16)
        plt.tight_layout()
        plt.savefig("interpolated_fashion_detailed.png", dpi=150, bbox_inches='tight')
        plt.show()

    print("Fashion interpolation completed!")


# Основной блок выполнения
if __name__ == "__main__":
    print("Fashion MNIST GAN Training Script")
    print("=" * 50)

    # Обучение модели
    generator, discriminator, g_losses, d_losses = train_gan()

    if generator is not None:
        # Генерация финальных изображений
        print("\nGenerating final fashion images...")
        generate_new_fashion(generator, num_images=16, save_name="final_generated_fashion")

        # Интерполяция
        print("\nGenerating fashion interpolation...")
        interpolate_fashion(generator)

        print("\n" + "=" * 50)
        print("All tasks completed successfully!")
        print(f"Dataset: Fashion MNIST")
        print(f"Image size: {image_size}x{image_size}")
        print("Generated images saved in 'generated_fashion/' folder")
        print("Models saved in 'trained_models/' folder")
    else:
        print("Training failed. Please check the dataset and try again.")