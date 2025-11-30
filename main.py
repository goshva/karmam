import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import threading
import yaml
from pathlib import Path

from database import db
from image_processor import prepare_dataset
from recognition_engine import RecognitionEngine

# Загрузка конфигурации
with open("config.yaml", 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

app = Flask(__name__, template_folder='templates')

# Инициализация движка распознавания
recognition_engine = RecognitionEngine(config['web']['model_path'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(config['web']['upload_dir'], filename)
        file.save(filepath)
        
        # Добавляем в базу данных
        image_id = db.add_image(filename, filepath)
        
        if image_id:
            # Распознаем изображение
            result = recognition_engine.recognize_image(filepath)
            
            # Сохраняем результат
            recognition_id = db.add_recognition_result(
                image_id=image_id,
                region_id=1,
                model_version=Path(config['web']['model_path']).name if os.path.exists(config['web']['model_path']) else "fallback",
                serial_number=result['serial_number'],
                confidence=result['confidence'],
                processing_time=result['processing_time'],
                symbols=result['symbols']
            )
            
            return jsonify({
                'image_id': image_id,
                'recognition_id': recognition_id,
                'result': result
            })
        else:
            return jsonify({'error': 'Failed to add image to database'}), 500

@app.route('/prepare-dataset', methods=['POST'])
def prepare_dataset_route():
    try:
        count = prepare_dataset()
        return jsonify({'status': 'success', 'processed_count': count})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recognition-stats', methods=['GET'])
def get_recognition_stats():
    try:
        stats = db.get_recognition_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/images', methods=['GET'])
def get_images():
    try:
        images = db.get_images_with_metadata()
        return jsonify({'images': images})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch-recognize', methods=['POST'])
def batch_recognize():
    try:
        data = request.get_json()
        image_ids = data.get('image_ids', [])
        
        def process_batch():
            results = recognition_engine.batch_process_images(image_ids)
            print(f"✅ Пакетная обработка завершена: {len(results)} результатов")
        
        thread = threading.Thread(target=process_batch)
        thread.start()
        
        return jsonify({'status': 'batch_processing_started', 'image_count': len(image_ids)})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/training-sessions', methods=['GET'])
def get_training_sessions():
    try:
        sessions = db.get_training_sessions()
        return jsonify({'sessions': sessions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(config['web']['upload_dir'], filename)

if __name__ == '__main__':
    # Создаем необходимые директории
    Path(config['web']['upload_dir']).mkdir(exist_ok=True)
    Path("templates").mkdir(exist_ok=True)
    
    print("🚀 Запуск приложения...")
    print(f"📊 База данных: {config['database']['path']}")
    print(f"🌐 Порт: {config['web']['port']}")
    
    app.run(debug=True, port=config['web']['port'], host='0.0.0.0')