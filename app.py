from flask import Flask, request, redirect, url_for, render_template, jsonify
import os
from werkzeug.utils import secure_filename
app = Flask(__name__)

# Указываем директорию для сохранения файлов
UPLOAD_FOLDER = 'uploads'
# Создаем папку, если ее нет
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Создаем папку, если ее нет
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Получаем файл из формы
        file = request.files['photo']
        
        # Проверяем, есть ли файл
        if file:
            # Разделяем имя файла на имя и расширение
            filename = secure_filename(file.filename)
            basename, ext = os.path.splitext(filename)
            # Формируем новое безопасное имя файла с оригинальным расширением
            new_filename = "1.jpg"
            # Сохраняем файл в директории UPLOAD_FOLDER
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
            file.save(file_path)
            
            # Перенаправляем на страницу успеха
            return redirect(url_for('koi'))
    
    return 'Ошибка при загрузке файла.'
images = []
current_card_index = 0
# Количество карточек, которые будут показаны сразу
INITIAL_CARDS_COUNT = 5
# Количество карточек, добавляемое по клику на кнопку
CARDS_PER_CLICK = 5

@app.route('/get_more_cards', methods=['POST'])

def get_more_cards():
    global current_card_index
    # Генерируем дополнительные карточки
    cards = generate_cards(CARDS_PER_CLICK)
    return jsonify(cards=cards)


def generate_cards(count):
    global current_card_index
    cards = []
    for _ in range(count):
        cards.append(f'Карточка {current_card_index + 1}')
        current_card_index += 1
    return cards
@app.route('/result')
def koi():
    
    global current_card_index
    # Генерируем начальные карточки
    cards = generate_cards(INITIAL_CARDS_COUNT)
    return render_template('result.html', cards=cards)
@app.route('/success')
def upload_success():
    return '<h1>Файл успешно загружен!</h1>'




current_card_index = 0
if __name__ == '__main__':
    app.run(debug=True)