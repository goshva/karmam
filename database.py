import sqlite3
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

class BanknoteDatabase:
    def __init__(self, db_path: str = "data/banknotes.db"):
        self.db_path = db_path
        Path("data").mkdir(exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Инициализация базы данных"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Таблица изображений
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_name TEXT NOT NULL,
                hash_name TEXT UNIQUE NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                file_hash TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Таблица регионов сканирования
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scan_regions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                region_name TEXT NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                width REAL NOT NULL,
                height REAL NOT NULL,
                FOREIGN KEY (image_id) REFERENCES images (id)
            )
        ''')
        
        # Таблица метаданных банкнот
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS banknote_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                country TEXT,
                denomination TEXT,
                serial_number TEXT,
                currency TEXT,
                year INTEGER,
                additional_info TEXT,
                FOREIGN KEY (image_id) REFERENCES images (id)
            )
        ''')
        
        # Таблица результатов распознавания YOLO
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recognition_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                model_version TEXT NOT NULL,
                region_id INTEGER NOT NULL,
                serial_number TEXT,
                confidence REAL,
                processing_time REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (image_id) REFERENCES images (id),
                FOREIGN KEY (region_id) REFERENCES scan_regions (id)
            )
        ''')
        
        # Таблица символов распознавания
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recognized_symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recognition_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                confidence REAL NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                width REAL NOT NULL,
                height REAL NOT NULL,
                FOREIGN KEY (recognition_id) REFERENCES recognition_results (id)
            )
        ''')
        
        # Таблица сессий обучения
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                start_time DATETIME NOT NULL,
                end_time DATETIME,
                epochs INTEGER NOT NULL,
                batch_size INTEGER NOT NULL,
                learning_rate REAL NOT NULL,
                train_images INTEGER NOT NULL,
                val_images INTEGER NOT NULL,
                best_accuracy REAL,
                best_precision REAL,
                best_recall REAL,
                best_map50 REAL,
                best_map REAL,
                final_loss REAL,
                training_time_minutes REAL,
                status TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ База данных инициализирована")
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Вычисление хеша файла"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def add_image(self, original_name: str, file_path: str) -> int:
        """Добавление изображения в базу"""
        file_hash = self.calculate_file_hash(file_path)
        file_size = Path(file_path).stat().st_size
        hash_name = f"{file_hash}_{Path(original_name).suffix}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO images (original_name, hash_name, file_path, file_size, file_hash)
                VALUES (?, ?, ?, ?, ?)
            ''', (original_name, hash_name, file_path, file_size, file_hash))
            
            image_id = cursor.lastrowid
            conn.commit()
            print(f"✅ Изображение добавлено: {original_name} -> {hash_name}")
            return image_id
            
        except sqlite3.IntegrityError:
            # Если изображение уже существует
            cursor.execute('SELECT id FROM images WHERE file_hash = ?', (file_hash,))
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else None
        finally:
            conn.close()
    
    def add_scan_regions(self, image_id: int, regions: List[Dict]):
        """Добавление регионов сканирования"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for region in regions:
            cursor.execute('''
                INSERT INTO scan_regions (image_id, region_name, x, y, width, height)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                image_id,
                region['name'],
                region['x'],
                region['y'],
                region['width'],
                region['height']
            ))
        
        conn.commit()
        conn.close()
        print(f"✅ Добавлено {len(regions)} регионов для изображения {image_id}")
    
    def add_banknote_metadata(self, image_id: int, metadata: Dict):
        """Добавление метаданных банкноты"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Извлекаем серийный номер из имени файла
        original_name = self.get_image_original_name(image_id)
        serial_number = Path(original_name).stem if original_name else None
        
        cursor.execute('''
            INSERT INTO banknote_metadata 
            (image_id, country, denomination, serial_number, currency, year, additional_info)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            image_id,
            metadata.get('country'),
            metadata.get('denomination'),
            serial_number,  # Используем имя файла как серийный номер
            metadata.get('currency'),
            metadata.get('year'),
            metadata.get('additional_info')
        ))
        
        conn.commit()
        conn.close()
        print(f"✅ Метаданные добавлены для изображения {image_id}")
    
    def add_recognition_result(self, image_id: int, region_id: int, model_version: str, 
                             serial_number: str, confidence: float, processing_time: float,
                             symbols: List[Dict]) -> int:
        """Добавление результата распознавания"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO recognition_results 
            (image_id, region_id, model_version, serial_number, confidence, processing_time)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (image_id, region_id, model_version, serial_number, confidence, processing_time))
        
        recognition_id = cursor.lastrowid
        
        # Добавляем символы
        for symbol in symbols:
            cursor.execute('''
                INSERT INTO recognized_symbols 
                (recognition_id, symbol, confidence, x, y, width, height)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                recognition_id,
                symbol['symbol'],
                symbol['confidence'],
                symbol['x'],
                symbol['y'],
                symbol['width'],
                symbol['height']
            ))
        
        conn.commit()
        conn.close()
        print(f"✅ Результат распознавания добавлен: {serial_number}")
        return recognition_id
    
    def get_image_original_name(self, image_id: int) -> Optional[str]:
        """Получение оригинального имени файла"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT original_name FROM images WHERE id = ?', (image_id,))
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
    
    def get_images_with_metadata(self) -> List[Dict]:
        """Получение всех изображений с метаданными"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                i.id, i.original_name, i.hash_name, i.file_path,
                b.country, b.denomination, b.serial_number, b.currency, b.year
            FROM images i
            LEFT JOIN banknote_metadata b ON i.id = b.image_id
            ORDER BY i.created_at DESC
        ''')
        
        images = []
        for row in cursor.fetchall():
            images.append({
                'id': row[0],
                'original_name': row[1],
                'hash_name': row[2],
                'file_path': row[3],
                'country': row[4],
                'denomination': row[5],
                'serial_number': row[6],
                'currency': row[7],
                'year': row[8]
            })
        
        conn.close()
        return images
    
    def get_recognition_stats(self) -> Dict:
        """Статистика распознавания"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Общая статистика
        cursor.execute('SELECT COUNT(*) FROM recognition_results')
        total_recognition = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT image_id) FROM recognition_results')
        unique_images = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(confidence) FROM recognition_results')
        avg_confidence = cursor.fetchone()[0] or 0
        
        # Статистика по странам
        cursor.execute('''
            SELECT b.country, COUNT(*) as count
            FROM recognition_results r
            JOIN banknote_metadata b ON r.image_id = b.image_id
            WHERE b.country IS NOT NULL
            GROUP BY b.country
        ''')
        country_stats = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Статистика по номиналам
        cursor.execute('''
            SELECT b.denomination, COUNT(*) as count
            FROM recognition_results r
            JOIN banknote_metadata b ON r.image_id = b.image_id
            WHERE b.denomination IS NOT NULL
            GROUP BY b.denomination
        ''')
        denomination_stats = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        return {
            'total_recognition': total_recognition,
            'unique_images': unique_images,
            'avg_confidence': avg_confidence,
            'country_stats': country_stats,
            'denomination_stats': denomination_stats
        }
    
    def get_training_sessions(self, limit: int = 10) -> List[Dict]:
        """Получение сессий обучения"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM training_sessions 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (limit,))
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                'id': row[0],
                'model_name': row[1],
                'start_time': row[2],
                'end_time': row[3],
                'epochs': row[4],
                'batch_size': row[5],
                'learning_rate': row[6],
                'train_images': row[7],
                'val_images': row[8],
                'best_accuracy': row[9],
                'best_precision': row[10],
                'best_recall': row[11],
                'best_map50': row[12],
                'best_map': row[13],
                'final_loss': row[14],
                'training_time_minutes': row[15],
                'status': row[16],
                'created_at': row[17]
            })
        
        conn.close()
        return sessions

# Глобальный экземпляр базы данных
db = BanknoteDatabase()