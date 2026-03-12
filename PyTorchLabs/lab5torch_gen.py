import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import random

image_size = 32

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

def quick_generate(model_path="trained_models/fashion_generator_epoch_30.pth"):
    """Быстрая генерация без лишних настроек"""

    # Загружаем модель
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Создаем генератор (архитектура должна совпадать с обученной моделью)
    generator = Generator(100, 1).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()

    # Генерируем 16 изображений
    with torch.no_grad():
        z = torch.randn(16, 100).to(device)
        images = generator(z)

        file_name = random.randint(1, 1000000)

        # Сохраняем
        save_image(images, f"generated_fashion/{file_name}.png", nrow=4, normalize=True)

        # Показываем
        images = images * 0.5 + 0.5
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            img = images[i].cpu().squeeze().numpy()
            ax.imshow(img, cmap='gray')
            ax.axis('off')
        plt.show()

    print(f"✅ Изображения сгенерированы и сохранены в 'generated_fashion/{file_name}.png'")


# Запуск
if __name__ == "__main__":
    quick_generate()