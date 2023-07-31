<!DOCTYPE html>
<html>
<body>
# AI-ArtInfo

![](https://i.gifer.com/origin/78/78013ac9f22c3c8f5713d28fa31c6211.gif)

## Основная идея проекта:<br>
**Разработка телеграмм-бота, который помогает определить автора и стиль картины на основе фотографии, а также находит наиболее подходящую картину по описанию.** <br>
<br>
Полезные особенности и преимущества:<br>
- Интерфейс на двух языках<br>
- Быстрый доступ к интересующей информации<br>
- Доступная информация для изучения искусства<br>
- Отсутствие необходимости ручного поиска<br>

## Что было сделано?<br>
- Этот бот использует датасет, который был частично получен из [сайта](https://allpainters.ru/) с использованием спарсинга данных, а частично собран вручную.<br>
- Затем мы применили [модель](https://huggingface.co/Salesforce/blip-image-captioning-large), чтобы сгенерировать описание для каждой картинки.<br>
- Чтобы обеспечить работу бота на двух языках, мы использовали модели для перевода датасета с русского на английский и наоборот: [RU_EN](https://huggingface.co/Helsinki-NLP/opus-mt-ru-en) и [EN-RU](https://huggingface.co/Helsinki-NLP/opus-mt-ru-en)<br>
- Кроме того, для создания функционала бота мы воспользовались моделями Resnet50 и TfidfVectorizer.<br>

## Алгоритм использования бота<br>
![](img/shema.png)

## Демонстрация работы<br>
[![Watch the video]([https://i.stack.imgur.com/Vp2cE.png])(https://img.freepik.com/free-vector/video-media-player-design_114579-839.jpg)](https://drive.google.com/file/d/1HrpdTcqWAAC-U3Kurz7LRXzwVFvy5ZeH/preview)


## Над проектом работали:<br>
[Анна Савицкая](https://github.com/SaviAnn)<br>
[Мария Козлова](https://github.com/MARI-posa)<br>
[Виктория Князева](https://github.com/vvv-knyazeva)<br>


</body>
</html>
