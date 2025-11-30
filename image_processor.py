import os
import shutil
from pathlib import Path
from database import db

class ImageProcessor:
    def __init__(self, source_dir: str = "manual", output_dir: str = "dataset"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.setup_directories()
    
    def setup_directories(self):
        """Создание необходимых директорий"""
        directories = [
            self.output_dir / "images" / "train",
            self.output_dir / "images" / "val",
            self.output_dir / "labels" / "train", 
            self.output_dir / "labels" / "val"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def process_manual_images(self):
        """Обработка изображений из manual директории"""
        print("🔄 Обработка изображений из manual...")
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        processed_count = 0
        
        for extension in image_extensions:
            for image_path in self.source_dir.glob(extension):
                try:
                    self.process_single_image(image_path)
                    processed_count += 1
                except Exception as e:
                    print(f"❌ Ошибка обработки {image_path}: {e}")
        
        print(f"✅ Обработано {processed_count} изображений")
        return processed_count
    
    def process_single_image(self, image_path: Path):
        """Обработка одного изображения"""
        # Добавляем изображение в базу данных
        image_id = db.add_image(image_path.name, str(image_path))
        
        if image_id:
            # Создаем регионы сканирования (примерные координаты)
            regions = [
                {
                    'name': 'serial_number_1',
                    'x': 0.1, 'y': 0.1, 'width': 0.4, 'height': 0.1
                },
                {
                    'name': 'serial_number_2', 
                    'x': 0.5, 'y': 0.1, 'width': 0.4, 'height': 0.1
                }
            ]
            
            db.add_scan_regions(image_id, regions)
            
            # Извлекаем информацию из имени файла
            metadata = self.extract_metadata_from_filename(image_path.stem)
            db.add_banknote_metadata(image_id, metadata)
            
            # Копируем в датасет с хешированным именем
            self.copy_to_dataset(image_path, image_id)
    
    def extract_metadata_from_filename(self, filename: str) -> dict:
        """Извлечение метаданных из имени файла"""
        # Пример: "USD_100_2020_AB12345678.jpg"
        parts = filename.split('_')
        metadata = {}
        
        if len(parts) >= 4:
            metadata['currency'] = parts[0]  # USD
            metadata['denomination'] = parts[1]  # 100
            metadata['year'] = int(parts[2]) if parts[2].isdigit() else None
            # parts[3] и далее - серийный номер
        elif len(parts) >= 2:
            metadata['denomination'] = parts[0]
        
        # Определяем страну по валюте
        currency_to_country = {
            'USD': 'USA',
            'EUR': 'EUROPE', 
            'RUB': 'RUSSIA',
            'GBP': 'UK',
            'JPY': 'JAPAN',
            'CNY': 'CHINA'
        }
        
        if metadata.get('currency') in currency_to_country:
            metadata['country'] = currency_to_country[metadata['currency']]
        
        return metadata
    
    def copy_to_dataset(self, image_path: Path, image_id: int):
        """Копирование изображения в датасет с хешированным именем"""
        # Получаем хешированное имя из базы данных
        original_name = db.get_image_original_name(image_id)
        if not original_name:
            return
        
        # Создаем хешированное имя файла
        file_hash = db.calculate_file_hash(str(image_path))
        new_filename = f"{file_hash}{image_path.suffix}"
        
        # Копируем в train (80%) или val (20%) директорию
        # Простое распределение по четности ID
        split_dir = "train" if image_id % 5 != 0 else "val"  # 80/20 split
        
        dest_path = self.output_dir / "images" / split_dir / new_filename
        shutil.copy2(image_path, dest_path)
        
        # Создаем пустой файл разметки
        label_path = self.output_dir / "labels" / split_dir / f"{file_hash}.txt"
        label_path.touch()
        
        print(f"📁 Скопировано: {image_path.name} -> {new_filename} ({split_dir})")

def prepare_dataset():
    """Основная функция подготовки датасета"""
    processor = ImageProcessor()
    return processor.process_manual_images()