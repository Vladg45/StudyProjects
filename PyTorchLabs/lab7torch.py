import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import random


# ПОДГОТОВКА ДАННЫХ ДЛЯ YOLO
def prepare_yolo_dataset(csv_path, train_img_dir, test_img_dir, output_dir='./object_detection/yolo_dataset'):
    # Создаем структуру папок
    yolo_dirs = {
        'train': os.path.join(output_dir, 'train'),
        'val': os.path.join(output_dir, 'val'),
        'test': os.path.join(output_dir, 'test')
    }

    for dir_path in yolo_dirs.values():
        os.makedirs(os.path.join(dir_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dir_path, 'labels'), exist_ok=True)

    # Читаем CSV файл
    df = pd.read_csv(csv_path)

    # Получаем список всех уникальных изображений для обучения
    train_images = [f for f in os.listdir(train_img_dir) if f.endswith('.jpg')]
    test_images = [f for f in os.listdir(test_img_dir) if f.endswith('.jpg')]

    # Разделяем train на train и val
    train_images_split, val_images_split = train_test_split(
        train_images, test_size=0.2, random_state=42
    )

    # Функция для обработки одного изображения
    def process_image(img_name, img_dir, label_dir, is_test=False):
        img_path = os.path.join(train_img_dir if not is_test else test_img_dir, img_name)

        # Копируем изображение
        shutil.copy(img_path, os.path.join(img_dir, img_name))

        if not is_test:
            # Создаем аннотации только для train/val
            label_path = os.path.join(label_dir, img_name.replace('.jpg', '.txt'))

            # Получаем все bounding boxes для этого изображения
            img_boxes = df[df['image'] == img_name]

            with open(label_path, 'w') as f:
                for _, row in img_boxes.iterrows():
                    # Загружаем изображение для получения размеров
                    img = cv2.imread(img_path)
                    h, w = img.shape[:2]

                    # Конвертируем координаты в формат YOLO (нормализованные)
                    x_center = ((row['xmin'] + row['xmax']) / 2) / w
                    y_center = ((row['ymin'] + row['ymax']) / 2) / h
                    width = (row['xmax'] - row['xmin']) / w
                    height = (row['ymax'] - row['ymin']) / h

                    # Класс 0 для автомобиля (если только один класс)
                    f.write(f'0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n')

    # Обрабатываем train images
    for img_name in train_images_split:
        process_image(img_name, yolo_dirs['train'] + '/images', yolo_dirs['train'] + '/labels')

    # Обрабатываем val images
    for img_name in val_images_split:
        process_image(img_name, yolo_dirs['val'] + '/images', yolo_dirs['val'] + '/labels')

    # Обрабатываем test images (без аннотаций)
    for img_name in test_images:
        img_path = os.path.join(test_img_dir, img_name)
        shutil.copy(img_path, os.path.join(yolo_dirs['test'] + '/images', img_name))

    print(f"Dataset prepared in {output_dir}")
    print(f"Train: {len(train_images_split)} images")
    print(f"Val: {len(val_images_split)} images")
    print(f"Test: {len(test_images)} images")

    return yolo_dirs


# СОЗДАНИЕ КОНФИГУРАЦИОННОГО ФАЙЛА YAML
def create_yaml_config(data_dirs, output_path='./object_detection/data.yaml'):
    config = {
        'path': os.path.abspath('./object_detection/yolo_dataset'),  # корневой путь к датасету
        'train': os.path.abspath(data_dirs['train'] + '/images'),
        'val': os.path.abspath(data_dirs['val'] + '/images'),
        'test': os.path.abspath(data_dirs['test'] + '/images'),

        # Классы
        'nc': 1,  # количество классов (только автомобили)
        'names': ['car']  # имена классов
    }

    with open(output_path, 'w') as f:
        f.write(f"path: {config['path']}\n")
        f.write(f"train: {config['train']}\n")
        f.write(f"val: {config['val']}\n")
        f.write(f"test: {config['test']}\n\n")
        f.write(f"nc: {config['nc']}\n")
        f.write(f"names: {config['names']}\n")

    print(f"YAML config created at {output_path}")
    return output_path


# ОБУЧЕНИЕ МОДЕЛИ YOLOv8
def train_yolo_model(yaml_path, epochs=50, imgsz=640):
    # Инициализируем модель (предварительно обученную)
    model = YOLO('yolov8n.pt')

    # Обучаем модель
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        patience=10,
        project='car_detection',
        name='yolov8_car_detection',
        visualize=True,
        optimizer='AdamW',
        lr0=0.001,
        augment=True,
    )

    print("Training completed!")
    return model, results


# ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
def plot_training_results(results_path='./car_detection/yolov8_car_detection'):
    # Загружаем результаты обучения
    results_csv = os.path.join(results_path, 'results.csv')

    if not os.path.exists(results_csv):
        print(f"Results CSV not found at {results_csv}")
        return

    df = pd.read_csv(results_csv)

    # Создаем два графика
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Графики loss
    if 'train/box_loss' in df.columns:
        axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', linewidth=2)
    if 'val/box_loss' in df.columns:
        axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', linewidth=2, linestyle='--')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Box Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    if 'train/cls_loss' in df.columns:
        axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss', linewidth=2)
    if 'val/cls_loss' in df.columns:
        axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss', linewidth=2, linestyle='--')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Classification Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Графики метрик
    if 'metrics/mAP50(B)' in df.columns:
        axes[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'],
                        label='mAP@0.5', linewidth=2, color='green')
    if 'metrics/mAP50-95(B)' in df.columns:
        axes[1, 0].plot(df['epoch'], df['metrics/mAP50-95(B)'],
                        label='mAP@0.5:0.95', linewidth=2, color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('mAP')
    axes[1, 0].set_title('Mean Average Precision (mAP)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # График IoU (используем precision и recall)
    if 'metrics/precision(B)' in df.columns:
        axes[1, 1].plot(df['epoch'], df['metrics/precision(B)'],
                        label='Precision', linewidth=2, color='blue')
    if 'metrics/recall(B)' in df.columns:
        axes[1, 1].plot(df['epoch'], df['metrics/recall(B)'],
                        label='Recall', linewidth=2, color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Precision & Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./object_detection/training_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Training plots saved as './object_detection/training_results.png'")


# ТЕСТИРОВАНИЕ И СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
def save_detection_results(model, test_images_dir, output_dir='./object_detection/detection_results'):
    os.makedirs(output_dir, exist_ok=True)

    # Получаем список тестовых изображений
    test_images = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    print(f"Processing {len(test_images)} test images...")

    for i, img_name in enumerate(test_images):
        img_path = os.path.join(test_images_dir, img_name)

        # Предсказание
        results = model(img_path, conf=0.25, iou=0.45)

        # Визуализация результатов
        for result in results:
            # Получаем изображение с bounding boxes
            im_array = result.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image

            # Сохраняем результат
            output_path = os.path.join(output_dir, f"detected_{img_name}")
            im.save(output_path)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(test_images)} images")

    print(f"Detection results saved in {output_dir}")


# ОСНОВНАЯ ФУНКЦИЯ
def main():
    # Параметры
    CSV_PATH = './object_detection/train_solution_bounding_boxes.csv'
    TRAIN_IMG_DIR = './object_detection/training_images/'
    TEST_IMG_DIR = './object_detection/testing_images/'

    # Подготовка данных
    print("Подготовка данных для YOLO")
    data_dirs = prepare_yolo_dataset(CSV_PATH, TRAIN_IMG_DIR, TEST_IMG_DIR)

    # Создание конфигурационного файла
    print("Создание YAML конфигурации")
    yaml_path = create_yaml_config(data_dirs)

    # Обучение модели
    print("Обучение модели YOLOv8")
    model, results = train_yolo_model(yaml_path, epochs=50, imgsz=640)

    # Визуализация результатов обучения
    print("Построение графиков обучения")
    plot_training_results()

    # Тестирование и сохранение результатов
    print("Тестирование на новых изображениях")

    # Загружаем лучшую модель после обучения
    best_model_path = './car_detection/yolov8_car_detection/weights/best.pt'
    if os.path.exists(best_model_path):
        model = YOLO(best_model_path)
        print(f"Loaded best model from {best_model_path}")
    else:
        print("Using the last trained model")

    # Сохраняем результаты детекции
    save_detection_results(model, data_dirs['test'] + '/images')

    # Тест на нескольких случайных изображениях
    print("Пример детекции на случайных изображениях")

    # Выбираем несколько случайных изображений для демонстрации
    test_images_dir = data_dirs['test'] + '/images'
    all_test_images = os.listdir(test_images_dir)
    demo_images = random.sample(all_test_images, min(3, len(all_test_images)))

    for img_name in demo_images:
        img_path = os.path.join(test_images_dir, img_name)
        results = model(img_path, conf=0.3)

        # Показываем результат
        for result in results:
            im_array = result.plot()
            im = Image.fromarray(im_array[..., ::-1])

            # Создаем фигуру для отображения
            plt.figure(figsize=(10, 8))
            plt.imshow(im)
            plt.axis('off')
            plt.title(f'Detection: {img_name}')
            plt.show()

            # Сохраняем отдельно
            demo_path = f'demo_{img_name}'
            im.save(demo_path)
            print(f"Demo result saved as: {demo_path}")
            break


if __name__ == "__main__":
    main()