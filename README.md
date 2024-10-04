# Автоматический поворот медицинских справок для OCR

## Описание проекта
Данный проект разработан для компании **DonorSearch** с целью предобработки загружаемых пользователями медицинских справок формы №405. Главной задачей является автоматическое определение и корректировка ориентации справки перед передачей в сервис OCR. Это позволяет системе обрабатывать справки независимо от их изначального положения.

## Данные
В качестве данных для тренировки было использовано 2 набора:
1. RVL-CDIP датасет, содержащий 400000 отсканированных изображений документов. Из них нами было использовано 290000, содержащих печатный текст.
2. 1194 фотографий различных справок и документов на русском языке.
   
Для тестирования моделей использовалась выборка из 99 фотографий донорских справок.

## Тренировка моделей
### Процесс тренировки
Тренировка производидась 2 способами:
1. Использование только 1194 фотографий.
2. Обучение 1 эпоху на датасете RVL-CDIP, затем дообучение несколько эпох на 1194 фотографиях.

Для улучшения сходимости было использовано несколько техник:
1. MixUp
2. RandAugment
3. Уменьшение вероятности аугментации с каждой эпохой
4. OneCycleLR Scheduler

Гиперпараметры обучения подбирались с помощью optuna.

## Результаты моделей
### Размер изображения 128x128 пикселей

| Архитектура       | Стадий тренировки | Размер изображения | Accuracy | Скорость инференса, FPS | 
|-------------------|-------------------|--------------------|----------|-------------------------|
| EfficientNet-B0   | 1                 | 128                | 96.21%   | 26                      | 
| HGNet-V2-B1       | 1                 | 128                | 98.74%   | 30                      | 
| EfficientNet-B0   | 2                 | 128                | 96.97%   | 26                      | 
| HGNet-V2-B1       | 2                 | 128                | 98.99%   | 30                      | 
| ConvNeXt-Femto    | 2                 | 128                | 99.50%   | 31                      | 

Поскольку обучение в 2 стадии даёт лучшие результаты, дальнейшие эксперименты мы проводили только с этим подходом

### Размер изображения 64x64 пикселя

| Архитектура       | Стадий тренировки | Размер изображения | Accuracy | Скорость инференса, FPS | 
|-------------------|-------------------|--------------------|----------|-------------------------|
| EfficientNet-B0   | 2                 | 64                 | 96.72%   | 33                      | 
| HGNet-V2-B1       | 2                 | 64                 | 96.46%   | 38                      | 
| ConvNeXt-Femto    | 2                 | 64                 | 94.70%   | 40                      | 

### Размер изображения 48x48 пикселей

| Архитектура       | Стадий тренировки | Размер изображения | Accuracy | Скорость инференса, FPS | 
|-------------------|-------------------|--------------------|----------|-------------------------|
| ConvNeXt-Femto    | 2                 | 48                 | 93.43%   | 43                      | 

Скорость инференса была измерена на Google Colab CPU - Intel(R) Xeon(R) CPU @ 2.20GHz.

### Матрица ошибок для нашей лучшей модели

<img src='https://github.com/IlyaLion/donorseach-documents-orientation/blob/readme/images/convnext_128-cm.png' height="320" />

<br>

### Матрица ошибок для нашей самой быстрой модели

<img src='https://github.com/IlyaLion/donorseach-documents-orientation/blob/readme/images/convnext_48-cm.png' height="320" />

<br>

## Выводы
 - Задачу определения ориентации справки можно решать с высокой точностью
 - Скорость инференса даже на не очень производительном процессоре достаточно высокая
 - В качестве модели мы рекомендуем использовать ConvNeXt-Femto с разрешением изображения 128x128 пикселей

## Разработчики
* [Илья Гурин](https://github.com/IlyaLion) 
* [Никита Батурин](https://github.com/nktbn)  
