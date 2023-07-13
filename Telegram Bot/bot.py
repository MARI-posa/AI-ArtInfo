import os
import logging
from aiogram import Bot, Dispatcher, types
from PIL import Image
import torch
import torchvision.transforms as transforms
from aiogram.dispatcher.filters.state import State, StatesGroup
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
# Инициализируем бота и диспетчера, замените API на свой
bot = Bot('5998847624:AAEQ-7P92LeYNsZ4xjZelFioqRaK9pAggoU')
dp = Dispatcher(bot)
with open('/Users/annasavitskaya/Bot-TG_ArtInfo/artists.json', 'r') as file:
    content = file.read()

idx_to_class = json.loads(content)

with open('/Users/annasavitskaya/Bot-TG_ArtInfo/genres.json', 'r') as file:
    content = file.read()

idx_to_style = json.loads(content)
#Словарь для сообщений на двух языках, мб вынести в отельный файл?
messages = {
    'en': {
        'select_language': 'Select a language:',
        'choose_function': 'Choose a function:',
        'function1': 'Guess the artist',
        'function2': 'Guess painting style',
        'function3': 'Guess painting name',
        'back': 'Back',
        'under_construction': 'Under Construction',
        'description1': 'Upload a photo, and I will guess the artist.',
        'prediction1': 'Most likely, this painting was created by {artist}.',
        'description2': 'Upload a photo, and I will guess the painting style.',
        'description3': 'Write a few words to describe the picture and i will suggest you a few pieces of art.',
        'prediction2': 'Most likely, this painting was created in {style} style.',
        'prediction3': 'Probably, you are looking for the one of these paintings.',
    },
    'ru': {
        'select_language': 'Выберите язык:',
        'choose_function': 'Выберите функцию:',
        'function1': 'Узнать художника',
        'function2': 'Узнать стиль картины',
        'function3': 'Узнать название картины',
        'back': 'Назад',
        'under_construction': 'В разработке',
        'description1': 'Загрузите фото картины, и я угадаю автора.',
        'prediction1': 'Скорее всего, эту картину создал {artist}.',
        'description2': 'Загрузите фото картины, и я угадаю стиль.',
        'description3': 'Опиши коротко картину и я предолжу тебе несколько подходящих вариантов.',
        'prediction2': 'Скорее всего, эта картина создана в стиле {style}.',
        'prediction3': 'Возможно, Вы ищете одну из этих картин.',
    }
}

language = None
# # Загрузка дообученной модели с сохраненными весами для функции угадывания автора

model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 51) #Заменяем на актуальное число классов
model.load_state_dict(torch.load('/Users/annasavitskaya/Bot-TG_ArtInfo/artists_model_78.pt', map_location=torch.device('cpu') ))
model.eval()
# # Загрузка дообученной модели с сохраненными весами для функции угадывания стиля

model_style = torch.hub.load('pytorch/vision', 'resnet50', pretrained=False)
model_style.fc = torch.nn.Linear(model_style.fc.in_features, 31)  # Заменяем на актуальное число классов
model_style.load_state_dict(torch.load('/Users/annasavitskaya/Bot-TG_ArtInfo/style_model_epoch_12.pt', map_location=torch.device('cpu') ))
model_style.eval()
# Логирование
logging.basicConfig(level=logging.INFO)

# Функция для выполнения предсказания на основе модели
def make_prediction(save_path,fun_mod):
    # Преобразования данных для аугментации и нормализации
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Загрузка и предобработка изображения
    image = Image.open(save_path)
    image = transform(image).unsqueeze(0)  # Добавляем размерность пакета

    # Выполнение предсказания
    with torch.no_grad():
        output = fun_mod(image)

    # Получение класса с максимальной вероятностью
    _, predicted_class = torch.max(output, 1)
    return predicted_class.item()

 
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
ru_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")

def translate_text(text):
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    outputs = ru_en_model.generate(input_ids)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text
new_df = pd.read_csv('/Users/annasavitskaya/Bot-TG_ArtInfo/description.csv', index_col=False)
#Состояния главного меню: выбор языка-основное меню-меню выбранной функции

# Создание TF-IDF векторизатора
tfidf_vectorizer = TfidfVectorizer()

# Преобразование описаний векторами TF-IDF
tfidf_matrix = tfidf_vectorizer.fit_transform(new_df['description'])

# Функция поиска наиболее похожих изображений
def search_images(query, top_n=5):
    # Преобразование запроса в вектор TF-IDF
    query_vector = tfidf_vectorizer.transform([query])

    # Вычисление косинусной меры сходства между запросом и описаниями изображений
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Получение индексов наиболее похожих изображений
    top_indices = similarity_scores.argsort()[::-1][:top_n]
    results = []

    # Вывод результатов
    for idx in top_indices:
        result = new_df['file_name'].iloc[idx]
        results.append(result)
    return results

states = {'language_selection': 0, 'main_menu': 1, 'function1': 2, 'function2': 3, 'function3': 4}
# При запуске бота сначала выбираем язык
current_state = states['language_selection']
#Добавила счетчик на всякий случай, чтобы параллельно две картинки можно было обработать и сохранить
image_counter = 1
#Запуск бота
@dp.message_handler(commands=['start'])
async def handle_start(message: types.Message):
    await show_language_selection(message)

#Менб с выбором языка
async def show_language_selection(message: types.Message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    itembtn1 = types.KeyboardButton('English')
    itembtn2 = types.KeyboardButton('Русский')
    markup.add(itembtn1, itembtn2)
    await message.answer(messages['en']['select_language'], reply_markup=markup)
    global current_state, language
    current_state = states['language_selection']
    language = 'en'

#Меняем значение language, чтобы знать на каком языке отвечать пользователю
@dp.message_handler(lambda message: current_state == states['language_selection'])
async def handle_language_selection(message: types.Message):
    global language
    if message.text == 'English':
        language = 'en'
    elif message.text == 'Русский':
        language = 'ru'
    else:
        await message.reply(messages[language]['select_language'])

    await show_main_menu(message)

#Выбор функции - 1, 2, 3 или кнопка назад
async def show_main_menu(message: types.Message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    itembtn1 = types.KeyboardButton(messages[language]['function1'])
    itembtn2 = types.KeyboardButton(messages[language]['function2'])
    itembtn3 = types.KeyboardButton(messages[language]['function3'])
    itembtn4 = types.KeyboardButton(messages[language]['back'])
    markup.add(itembtn1, itembtn2, itembtn3, itembtn4)
    await message.answer(messages[language]['choose_function'], reply_markup=markup)
    global current_state
    current_state = states['main_menu']
# Подтверждение, какую функцию выбрал польщователь
@dp.message_handler(lambda message: current_state == states['main_menu'])
async def handle_main_menu(message: types.Message):
    if message.text == messages[language]['function1']:
        await show_function1(message)
    elif message.text == messages[language]['function2']:
        await show_function2(message)
    elif message.text == messages[language]['function3']:
        await show_function3(message)
    elif message.text == messages[language]['back']:
        await show_language_selection(message)
#В случае выбора функции 1 (автор)
async def show_function1(message: types.Message):
    # Логика для функции 1
    await message.answer(messages[language]['description1'])
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    itembtn = types.KeyboardButton(messages[language]['back'])
    markup.add(itembtn)
    #await message.answer(messages[language]['back'], reply_markup=markup)
    global current_state
    current_state = states['function1']
    @dp.message_handler(lambda message: current_state == states['function1'], content_types=types.ContentType.PHOTO)
    async def image_message(message: types.Message):
        # Получение информации об изображении
        photo = message.photo[-1]
        file_id = photo.file_id
        # Загружаем фото с помощью метода get_file
        file_info = await bot.get_file(file_id)
        file_path = file_info.file_path

        # Сохраняем фото и генерируем имя файла с учетом счетчика
        image_filename = f"image{image_counter}.jpg"
        save_path = os.path.join("photo", image_filename)
        await bot.download_file(file_path, save_path)

        # Выполнение предсказания на основе модели
        prediction1 = make_prediction(save_path,model)
        artist=idx_to_class[language][str(prediction1)]
        # Отправка ответа пользователю с предсказанным классом
        await message.reply(messages[language]['prediction1'].format(artist=artist))
        global current_state
        current_state = states['function1']
#В случае выбора функции 3 (подбор по описанию)
async def show_function3(message: types.Message):
    await message.answer(messages[language]['description3'])
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    itembtn = types.KeyboardButton(messages[language]['back'])
    markup.add(itembtn)
    global current_state
    current_state = states['function3']
    @dp.message_handler(lambda message: current_state == states['function3'], content_types=types.ContentType.TEXT)
    # Функция-обработчик для обычных текстовых сообщений
    async def text_message_handler(message: types.Message):
        query = message.text.strip()
        # Проверка наличия кириллицы в сообщении
        if re.search('[а-яА-Я]', query):
            # Запуск функции translate_text, передавая ей текст сообщения
            query = translate_text(query)
            print(query)
        if query:
            matching_images = search_images(query, top_n=5)
            if matching_images:
                print('matching')
                for image_name in matching_images:
                    # Получение значения file_name и artist_name
                    if language == 'en':
                        file_name = new_df[new_df['file_name'] == image_name]['picture_name'].values[0]
                        artist_name = new_df[new_df['file_name'] == image_name]['artist_name'].values[0]
                    else:
                        file_name = new_df[new_df['file_name'] == image_name]['ru_picture_name'].values[0]
                        artist_name = new_df[new_df['file_name'] == image_name]['ru_artist_name'].values[0]

                    # Формирование подписи с учетом значения file_name и artist_name
                    if pd.isnull(file_name):
                        caption = artist_name
                    else:
                        caption = f"{file_name} - {artist_name}"

                        # Отправка изображения пользователю с подписью
                    image_path = os.path.join('/Users/annasavitskaya/Bot-TG_ArtInfo/all_images', str(image_name))
                    with open(image_path, 'rb') as photo:
                        await message.reply_photo(photo, caption=caption)
                        print('step_5')
        else:
            await message.reply("По вашему запросу ничего не найдено.")
            # Отправка ответа пользователю с предсказанным классом      
        global current_state
        current_state = states['function3']
        print('step_6')
#В случае выбора функции 2 (стиль)
async def show_function2(message: types.Message):
    await message.answer(messages[language]['description2'])
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    itembtn = types.KeyboardButton(messages[language]['back'])
    markup.add(itembtn)
    global current_state
    current_state = states['function2']
    @dp.message_handler(lambda message: current_state == states['function2'], content_types=types.ContentType.PHOTO)
    async def image_message(message: types.Message):
        # Получение информации об изображении
        photo = message.photo[-1]
        file_id = photo.file_id
        # Загружаем фото с помощью метода get_file
        file_info = await bot.get_file(file_id)
        file_path = file_info.file_path

        # Сохраняем фото и генерируем имя файла с учетом счетчика
        image_filename = f"image{image_counter}.jpg"
        save_path = os.path.join("photo", image_filename)
        await bot.download_file(file_path, save_path)

        # Выполнение предсказания на основе модели
        prediction2 = make_prediction(save_path,model_style)
        style=idx_to_style[language][str(prediction2)]
        # Отправка ответа пользователю с предсказанным классом
        await message.reply(messages[language]['prediction2'].format(style=style))
        global current_state
        current_state = states['function2']
#В случае выбора кнопки назад - возвращаемся к выбору функции
@dp.message_handler(lambda message: current_state != states['language_selection'] and message.text == messages[language]['back'])
async def handle_back(message: types.Message):
    await show_main_menu(message)

# Запуск бота
if __name__ == '__main__':
    from aiogram import executor
    executor.start_polling(dp, skip_updates=True)