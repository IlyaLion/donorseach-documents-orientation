# Проект: Автоматический поворот медицинских справок для OCR

## Описание проекта
Данный проект разработан для компании **DonorSearch** с целью предобработки загружаемых пользователями медицинских справок формы №405. Главной задачей является автоматическое определение и корректировка ориентации справки перед передачей в сервис OCR. Это позволяет системе обрабатывать справки независимо от их изначального положения.

## Тренировка моделей
### Данные
В качестве данных для тренировки было исполььзовано 2 набора:
1. RVL-CDIP датасет, содержащий 400000 отсканированных изображений документов. Из них нами было использовано 290000, содержащих печатный текст.
2. 1194 фотографий различных справок и документов на русском языке.

Для тестирования моделей использовалась выборка из 99 фотографий донорских справок.

### Процесс тренировки
Тренировка производидась 2 способами:
1. Использование только 1194 фотографий
2. Претрейн на RVL-CDIP, затем файнтюн на 1194 фотографиях.

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
| HGNet-V2-B1       | 1                 | 128                | --.--%   | 30                      | 
| EfficientNet-B0   | 2                 | 128                | 96.97%   | 26                      | 
| HGNet-V2-B1       | 2                 | 128                | 98.99%   | 30                      | 
| ConvNeXt-Femto    | 2                 | 128                | --.--%   | 31                      | 

### Размер изображения 64x64 пикселя

| Архитектура       | Стадий тренировки | Размер изображения | Accuracy | Скорость инференса, FPS | 
|-------------------|-------------------|--------------------|----------|-------------------------|
| EfficientNet-B0   | 2                 | 64                 | 96.72%   | 33                      | 
| HGNet-V2-B1       | 2                 | 64                 | 96.46%   | 38                      | 
| ConvNeXt-Femto    | 2                 | 64                 | --.--%   | 40                      | 

### Размер изображения 48x48 пикселей

| Архитектура       | Стадий тренировки | Размер изображения | Accuracy | Скорость инференса, FPS | 
|-------------------|-------------------|--------------------|----------|-------------------------|
| ConvNeXt-Femto    | 2                 | 48                 | --.--%   | 43                      | 

Скорость инференса была измерена на Google Colab CPU - Intel(R) Xeon(R) CPU @ 2.20GHz.
