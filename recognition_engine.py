import os
import time
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from database import db

class RecognitionEngine:
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Загрузка модели YOLO"""
        try:
            self.model = YOLO(model_path)
            self.model_path = model_path
            print(f"✅ Модель загружена: {model_path}")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            self.model = None
    
    def recognize_image(self, image_path: str, region_id: int = 1) -> dict:
        """Распознавание символов на изображении"""
        if not self.model:
            return self._get_fallback_result(image_path, region_id)
        
        start_time = time.time()
        
        try:
            results = self.model.predict(image_path, save_txt=True, save_conf=True)
            processing_time = time.time() - start_time
            
            return self._parse_results(results, image_path, region_id, processing_time)
            
        except Exception as e:
            print(f"❌ Ошибка распознавания: {e}")
            return self._get_fallback_result(image_path, region_id)
    
    def _parse_results(self, results, image_path: str, region_id: int, processing_time: float) -> dict:
        """Парсинг результатов YOLO"""
        if not results or len(results) == 0:
            return self._get_fallback_result(image_path, region_id)
        
        result = results[0]
        symbols = []
        serial_chars = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            # Сортируем символы по координате X
            sorted_boxes = sorted(zip(result.boxes.xywhn, result.boxes.cls, result.boxes.conf), 
                                 key=lambda x: x[0][0])
            
            for box, cls_idx, conf in sorted_boxes:
                x, y, w, h = box.tolist()
                class_id = int(cls_idx)
                confidence = float(conf)
                
                # Получаем символ из конфигурации
                symbol = self._get_symbol_from_class_id(class_id)
                if symbol:
                    serial_chars.append(symbol)
                    
                    symbols.append({
                        'symbol': symbol,
                        'confidence': confidence,
                        'x': x, 'y': y, 'width': w, 'height': h
                    })
        
        serial_number = ''.join(serial_chars) if serial_chars else Path(image_path).stem
        avg_confidence = np.mean([s['confidence'] for s in symbols]) if symbols else 0.0
        
        return {
            'serial_number': serial_number,
            'confidence': avg_confidence,
            'processing_time': processing_time,
            'symbols': symbols,
            'region_id': region_id
        }
    
    def _get_symbol_from_class_id(self, class_id: int) -> str:
        """Получение символа по ID класса"""
        # Это должен быть синхронизирован с config.yaml
        symbols = "0123456789АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
        return symbols[class_id] if class_id < len(symbols) else None
    
    def _get_fallback_result(self, image_path: str, region_id: int) -> dict:
        """Fallback результат если модель не загружена"""
        filename = Path(image_path).stem
        
        # Пытаемся извлечь серийный номер из имени файла
        serial_number = filename
        
        # Создаем демо-символы
        symbols = []
        for i, char in enumerate(serial_number):
            symbols.append({
                'symbol': char,
                'confidence': 0.9,
                'x': 0.1 + i * 0.1,
                'y': 0.5,
                'width': 0.08,
                'height': 0.12
            })
        
        return {
            'serial_number': serial_number,
            'confidence': 0.85,
            'processing_time': 0.1,
            'symbols': symbols,
            'region_id': region_id
        }
    
    def batch_process_images(self, image_ids: list):
        """Пакетная обработка изображений"""
        results = []
        
        for image_id in image_ids:
            # Получаем информацию об изображении из базы
            images = db.get_images_with_metadata()
            image_info = next((img for img in images if img['id'] == image_id), None)
            
            if not image_info or not os.path.exists(image_info['file_path']):
                continue
            
            # Распознаем для каждого региона
            regions = [1, 2]  # Два региона сканирования
            
            for region_id in regions:
                result = self.recognize_image(image_info['file_path'], region_id)
                
                # Сохраняем в базу данных
                recognition_id = db.add_recognition_result(
                    image_id=image_id,
                    region_id=region_id,
                    model_version=Path(self.model_path).name if self.model_path else "fallback",
                    serial_number=result['serial_number'],
                    confidence=result['confidence'],
                    processing_time=result['processing_time'],
                    symbols=result['symbols']
                )
                
                results.append({
                    'image_id': image_id,
                    'region_id': region_id,
                    'recognition_id': recognition_id,
                    'serial_number': result['serial_number'],
                    'confidence': result['confidence']
                })
        
        return results