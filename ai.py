import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import MobileNet_V3_Large_Weights
from PIL import Image
import random
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import shutil
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import torch.nn.functional as F
import pickle
import hashlib


class FeatureCache:
    def __init__(self, cache_dir="./feature_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_index_path = os.path.join(cache_dir, "cache_index.pkl")
        self.load_cache_index()

    def load_cache_index(self):
        """Загружает индекс кэша или создает новый"""
        if os.path.exists(self.cache_index_path):
            with open(self.cache_index_path, 'rb') as f:
                self.cache_index = pickle.load(f)
        else:
            self.cache_index = {}

    def save_cache_index(self):
        """Сохраняет индекс кэша"""
        with open(self.cache_index_path, 'wb') as f:
            pickle.dump(self.cache_index, f)

    def get_file_hash(self, file_path):
        """Вычисляет хеш файла"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read(65536)  # Читаем файл блоками
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()

    def get_cache_path(self, file_hash):
        """Возвращает путь к кэшированному файлу признаков"""
        return os.path.join(self.cache_dir, f"{file_hash}.pkl")

    def get_features(self, image_path):
        """Получает признаки из кэша, если они есть"""
        file_hash = self.get_file_hash(image_path)
        cache_info = self.cache_index.get(image_path)

        if cache_info and cache_info['hash'] == file_hash:
            cache_path = self.get_cache_path(file_hash)
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        return None

    def save_features(self, image_path, features):
        """Сохраняет признаки в кэш"""
        file_hash = self.get_file_hash(image_path)
        cache_path = self.get_cache_path(file_hash)

        with open(cache_path, 'wb') as f:
            pickle.dump(features, f)

        self.cache_index[image_path] = {
            'hash': file_hash,
            'timestamp': datetime.now().timestamp()
        }
        self.save_cache_index()


def load_and_preprocess_image(image_path, transform):
    try:
        image = Image.open(image_path).convert('RGB')
        return transform(image)
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None


def extract_features_batch(image_paths, model, transform, device, cache, batch_size=32):
    """Извлекает признаки из пакета изображений с использованием кэша"""
    features_list = []
    valid_paths = []
    uncached_paths = []
    uncached_indices = []

    # Сначала проверяем кэш для всех изображений
    for i, path in enumerate(image_paths):
        cached_features = cache.get_features(path)
        if cached_features is not None:
            # Убедимся, что кэшированные признаки имеют правильную форму
            cached_features = np.array(cached_features).flatten()
            features_list.append(cached_features)
            valid_paths.append(path)
        else:
            uncached_paths.append(path)
            uncached_indices.append(i)

    # Обрабатываем только некэшированные изображения
    if uncached_paths:
        for i in range(0, len(uncached_paths), batch_size):
            batch_paths = uncached_paths[i:i + batch_size]
            batch_tensors = []
            batch_valid_paths = []

            for path in batch_paths:
                tensor = load_and_preprocess_image(path, transform)
                if tensor is not None:
                    batch_tensors.append(tensor)
                    batch_valid_paths.append(path)

            if batch_tensors:
                batch = torch.stack(batch_tensors).to(device)
                with torch.no_grad():
                    batch_features = model(batch)
                    # Преобразуем в плоский вектор
                    batch_features = batch_features.reshape(batch_features.size(0), -1)
                    batch_features_np = batch_features.cpu().numpy()

                    # Сохраняем признаки в кэш и добавляем их в общий список
                    for path, features in zip(batch_valid_paths, batch_features_np):
                        # Сохраняем уже сплющенные признаки
                        flattened_features = features.flatten()
                        cache.save_features(path, flattened_features)
                        features_list.append(flattened_features)
                        valid_paths.append(path)

    # Преобразуем список в массив numpy, убедившись что все элементы одинаковой длины
    if features_list:
        features_array = np.vstack(features_list)
    else:
        features_array = np.array([])

    return features_array, valid_paths


def combined_similarity_batch(source_features, target_features, weights={'cosine': 0.6, 'euclidean': 0.4}):
    """Вычисляет комбинированное сходство для пакета векторов"""
    # Убедимся, что входные данные плоские и двумерные
    source_features = source_features.reshape(1, -1)
    target_features = target_features.reshape(target_features.shape[0], -1)

    # Нормализация векторов
    source_norm = F.normalize(torch.from_numpy(source_features), p=2, dim=1)
    target_norm = F.normalize(torch.from_numpy(target_features), p=2, dim=1)

    # Косинусное сходство (используем матричные операции)
    cos_sim = torch.mm(source_norm, target_norm.t()).squeeze().numpy()

    # Евклидово расстояние (векторизованное вычисление)
    eucl_dist = np.linalg.norm(target_features - source_features, axis=1)
    eucl_sim = 1 / (1 + eucl_dist)

    return weights['cosine'] * cos_sim + weights['euclidean'] * eucl_sim


def find_similar_images(source_image_path, dataset_folder, output_folder, num_similar=5,
                        weights={'cosine': 0.6, 'euclidean': 0.4}, batch_size=32,
                        cache_dir="./feature_cache"):
    print(f"\n{'=' * 50}")
    print(f"Starting image similarity search at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 50}")

    # Инициализируем кэш
    cache = FeatureCache(cache_dir)

    # Проверяем доступность GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Создаем выходную папку
    os.makedirs(output_folder, exist_ok=True)

    # Настройка предобработки
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Загружаем модель MobileNetV3
    print("Loading MobileNetV3 model...")
    mobilenet = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    # Удаляем последний классификационный слой
    mobilenet = nn.Sequential(*list(mobilenet.children())[:-1])
    mobilenet.to(device)
    mobilenet.eval()

    # Получаем признаки исходного изображения
    print("Extracting features from source image...")
    # Сначала проверяем кэш
    source_features = cache.get_features(source_image_path)
    if source_features is None:
        source_tensor = load_and_preprocess_image(source_image_path, preprocess)
        if source_tensor is None:
            print("Error: Could not process source image")
            return

        with torch.no_grad():
            source_tensor = source_tensor.unsqueeze(0).to(device)
            source_features = mobilenet(source_tensor)
            # Преобразуем в плоский вектор
            source_features = source_features.reshape(source_features.size(0), -1)
            source_features = source_features.cpu().numpy().squeeze()
            # Сохраняем признаки в кэш
            cache.save_features(source_image_path, source_features.flatten())

    # Остальной код остается без изменений...


def find_similar_images(source_image_path, dataset_folder, output_folder, num_similar=5,
                        weights={'cosine': 0.6, 'euclidean': 0.4}, batch_size=32,
                        cache_dir="./feature_cache"):
    print(f"\n{'=' * 50}")
    print(f"Starting image similarity search at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 50}")

    # Инициализируем кэш
    cache = FeatureCache(cache_dir)

    # Проверяем доступность GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Создаем выходную папку
    os.makedirs(output_folder, exist_ok=True)

    # Настройка предобработки
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Загружаем модель MobileNetV3
    print("Loading MobileNetV3 model...")
    mobilenet = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    # Удаляем последний классификационный слой
    mobilenet = nn.Sequential(*list(mobilenet.children())[:-1])
    mobilenet.to(device)
    mobilenet.eval()

    # Получаем признаки исходного изображения
    print("Extracting features from source image...")
    # Сначала проверяем кэш
    source_features = cache.get_features(source_image_path)
    if source_features is None:
        source_tensor = load_and_preprocess_image(source_image_path, preprocess)
        if source_tensor is None:
            print("Error: Could not process source image")
            return

        with torch.no_grad():
            source_tensor = source_tensor.unsqueeze(0).to(device)
            source_features = mobilenet(source_tensor)
            # Flatten features before saving
            source_features = source_features.reshape(source_features.size(0), -1)
            source_features = source_features.cpu().numpy().squeeze()
            # Сохраняем признаки в кэш
            cache.save_features(source_image_path, source_features)

    # Остальной код остается без изменений...


def find_similar_images(source_image_path, dataset_folder, output_folder, num_similar=5,
                        weights={'cosine': 0.6, 'euclidean': 0.4}, batch_size=32,
                        cache_dir="./feature_cache"):
    print(f"\n{'=' * 50}")
    print(f"Starting image similarity search at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 50}")

    # Инициализируем кэш
    cache = FeatureCache(cache_dir)

    # Проверяем доступность GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Создаем выходную папку
    os.makedirs(output_folder, exist_ok=True)

    # Настройка предобработки
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Загружаем модель MobileNetV3
    print("Loading MobileNetV3 model...")
    mobilenet = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    # Удаляем последний классификационный слой
    mobilenet = nn.Sequential(*list(mobilenet.children())[:-1])
    mobilenet.to(device)
    mobilenet.eval()

    # Получаем признаки исходного изображения
    print("Extracting features from source image...")
    # Сначала проверяем кэш
    source_features = cache.get_features(source_image_path)
    if source_features is None:
        source_tensor = load_and_preprocess_image(source_image_path, preprocess)
        if source_tensor is None:
            print("Error: Could not process source image")
            return

        with torch.no_grad():
            source_tensor = source_tensor.unsqueeze(0).to(device)
            source_features = mobilenet(source_tensor).cpu().numpy().squeeze()
            # Сохраняем признаки в кэш
            cache.save_features(source_image_path, source_features)

    # Собираем все изображения из датасета
    print("Collecting images from dataset...")
    dataset_images = []
    for root, _, files in os.walk(dataset_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(root, file)
                if full_path != source_image_path:
                    dataset_images.append(full_path)

    print(f"Total images found: {len(dataset_images)}")

    # Извлекаем признаки всех изображений пакетами
    print("\nExtracting features from dataset images...")
    features_array, valid_paths = extract_features_batch(dataset_images, mobilenet, preprocess,
                                                         device, cache, batch_size)

    # Вычисляем сходство
    print("Computing similarities...")
    similarities = combined_similarity_batch(source_features, features_array, weights)

    # Находим топ-N похожих изображений
    similar_indices = similarities.argsort()[-num_similar:][::-1]

    # Копируем похожие изображения и сохраняем информацию
    print("\nSaving results...")
    similarity_info = []

    for i, idx in enumerate(similar_indices):
        src_path = valid_paths[idx]
        dst_path = os.path.join(output_folder, f'similar_{i + 1}{os.path.splitext(src_path)[1]}')
        shutil.copy2(src_path, dst_path)

        similarity_score = similarities[idx]
        similarity_info.append({
            'rank': i + 1,
            'file': os.path.basename(src_path),
            'similarity_score': similarity_score
        })
        print(f"Image {i + 1}: {os.path.basename(src_path)} (Score: {similarity_score:.4f})")

    # Копируем исходное изображение
    source_filename = os.path.basename(source_image_path)
    source_dst_path = os.path.join(output_folder, f'source_{source_filename}')
    shutil.copy2(source_image_path, source_dst_path)

    # Сохраняем результаты
    similarity_file = os.path.join(output_folder, 'similarity_scores.txt')
    with open(similarity_file, 'w') as f:
        f.write(f"Similarity Search Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source image: {source_filename}\n")
        f.write("-" * 50 + "\n")
        for info in similarity_info:
            f.write(f"Rank {info['rank']}: {info['file']} - Score: {info['similarity_score']:.4f}\n")

    print(f"\nResults saved to: {output_folder}")
    print(f"\n{'=' * 50}")
    print("Search completed successfully!")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    source_image = "uploads/"
    dataset_folder = "train_data_rkn/dataset"
    output_folder = "similar_images/"
    cache_dir = "feature_cache/"  # Директория для кэша

    weights = {
        'cosine': 0.6,
        'euclidean': 0.4
    }

    find_similar_images(
        source_image,
        dataset_folder,
        output_folder,
        num_similar=5,
        weights=weights,
        batch_size=32,
        cache_dir=cache_dir
    )