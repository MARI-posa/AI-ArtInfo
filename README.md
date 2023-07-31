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
[![Watch the video](https://i.stack.imgur.com/Vp2cE.png)](https://drive.google.com/file/d/1HrpdTcqWAAC-U3Kurz7LRXzwVFvy5ZeH/preview)
<iframe src="https://drive.google.com/file/d/1HrpdTcqWAAC-U3Kurz7LRXzwVFvy5ZeH/preview" width="640" height="480" allow="autoplay"></iframe>
<iframe src="https://player.vimeo.com/video/850330264?title=0&amp;byline=0&amp;portrait=0&amp;speed=0&amp;badge=0&amp;autopause=0&amp;player_id=0&amp;app_id=58479" width="400" height="300" frameborder="0" allow="autoplay; fullscreen; picture-in-picture" allowfullscreen title="ArtInfo 2"></iframe>


## Над проектом работали:<br>
[Анна Савицкая](https://github.com/SaviAnn)<br>
[Мария Козлова](https://github.com/MARI-posa)<br>
[Виктория Князева](https://github.com/vvv-knyazeva)<br>


</body>
</html>
