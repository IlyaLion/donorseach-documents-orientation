import albumentations as A
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import random
import torch
import torch.nn.functional as F

from .augmentations import get_augmentation

class Dataset(torch.utils.data.Dataset):
    """
    Класс для работы с датасетом.

    Args:
        config (Box): Объект конфигурации.
        image_paths (List[str]): Пути к изображениям.
        mode (str): Режим работы.
        transforms (Dict): Словарь трансформаций.
    """
    def __init__(self, config, image_paths, mode, transforms):
        self.config = config
        self.image_paths = np.array(image_paths)
        self.mode = mode
        self.transforms = transforms

        self.current_step = 0

        self.background_images = None

    def __len__(self):
        """
        Возвращает количество изображений в датасете.

        Returns:
            int: Количество изображений.
        """
        return (len(self.image_paths) 
                if self.mode == 'train'
                else self.config.model.n_classes * len(self.image_paths))

    def __getitem__(self, index):
        """
        Получает изображение и метку по индексу.

        Args:
            index (int): Индекс изображения.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Преобразованное изображение и метка.
        """
        image = self._read_image(self.image_paths[index 
                                                  if self.mode == 'train' 
                                                  else index // self.config.model.n_classes])
        if torch.rand(1).item() < self.config.dataset.insert_background_p:
            image = self._add_background(image)
        image = self._apply_image_transform(image)

        label = (torch.randint(0, self.config.model.n_classes, (1, )).item() 
                 if self.mode == 'train'
                 else index % self.config.model.n_classes)
        
        image = torch.rot90(image, k=label, dims=[1, 2])

        label = F.one_hot(torch.tensor(label), num_classes=self.config.model.n_classes).float()

        return image, label
    
    def _get_augmentations_p(self):    
        """
        Возвращает вероятность применения аугментаций на текущем шаге.

        Returns:
            float: Вероятность аугментаций.
        """ 
        if self.config.training.epochs == 1 or self.mode == 'val':
            return 1.0
        current_p = ((self.config.dataset.augmentation_p_schedule.end_p - self.config.dataset.augmentation_p_schedule.start_p) *
                     (self.current_step / (self.config.training.epochs - 1)) ** self.config.dataset.augmentation_p_schedule.exponent_factor +
                     self.config.dataset.augmentation_p_schedule.start_p)
        return current_p
        

    def _apply_image_transform(self, image):
        """
        Применяет к изображению цепочку пред- и пост-трансформаций, а также аугментации.

        Args:
            image (np.ndarray): Входное изображение.

        Returns:
            torch.Tensor: Трансформированное изображение.
        """
        augmentations_p = self._get_augmentations_p()
        transforms = A.Compose([
            A.Compose(self.transforms['pre_transforms']),
            A.Compose(self.transforms['augmentations'], p=augmentations_p),
            A.Compose(self.transforms['post_transforms'])
        ])

        image = transforms(image=image)["image"]
        return image
    
    def _read_image(self, image_path):
        """
        Читает изображение по заданному пути.

        Args:
            image_path (str): Путь к изображению.

        Returns:
            np.ndarray: Изображение в формате RGB.
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _add_background(self, image):
        """
        Добавляет случайное фоновое изображение к входному изображению.

        Args:
            image (np.ndarray): Входное изображение.

        Returns:
            np.ndarray: Изображение с добавленным фоном.
        """
        if self.background_images is None:
            self.background_images = [self._read_image(os.path.join(self.config.directories.backgrounds, f))
                                      for f in os.listdir(self.config.directories.backgrounds)]
        
        background_image = random.choice(self.background_images)

        image_size = np.array(image.shape[:2])
        background_ratio = random.uniform(1, self.config.dataset.background_max_ratio)
        background_image_size = (background_ratio * image_size).astype(int)

        transforms = A.Compose([
            A.D4(),
            A.LongestMaxSize(max_size=background_image_size.max(), interpolation=cv2.INTER_AREA),
            A.RandomCrop(*background_image_size)
        ])
        background_image = transforms(image=background_image)["image"]

        try:
            x_coord = random.randint(0, background_image_size[0] - image_size[0])
            y_coord = random.randint(0, background_image_size[1] - image_size[1])
        except:
            x_coord, y_coord = 0, 0

        background_image[x_coord:x_coord + image_size[0], y_coord:y_coord + image_size[1]] = image
        return background_image
    
    def step(self):
        """
        Увеличивает текущий шаг для корректного расчета вероятности аугментаций.
        """
        self.current_step += 1
    

class DataModule:
    """
    Класс для работы с загрузкой данных, включает в себя train и val режимы.

    Args:
        config (Box): Объект конфигурации.
    """
    def __init__(self, config):
        self.config = config
        self.train_image_paths, self.val_image_paths = self._get_data()

    def _get_data(self):
        """
        Собирает пути к изображениям и разделяет их на тренировочные и валидационные.

        Returns:
            Tuple[List[str], List[str]]: Пути к тренировочным и валидационным изображениям.
        """
        file_paths = []
        for subdir, _, files in os.walk(self.config.directories.images):
            for file in files:
                file_path = os.path.join(subdir, file)
                file_paths.append(file_path)

        file_paths = random.sample(file_paths, 
                                   int(len(file_paths) * self.config.dataset.data_fraction))

        train_paths, val_paths = train_test_split(file_paths, 
                                                        test_size=self.config.dataset.val_part, 
                                                        random_state=self.config.seed)
        return train_paths, val_paths


    def _get_dataloader(self, mode):
        """
        Возвращает dataloader для заданного режима.

        Args:
            mode (str): Режим загрузки данных.

        Returns:
            torch.utils.data.DataLoader: Dataloader для выбранного режима.
        """
        if mode == 'train':
            transforms = {
                'pre_transforms': self.config.dataset.pre_transforms,
                'augmentations': self.config.dataset.train_augmentations,
                'post_transforms': self.config.dataset.post_transforms
            }
            batch_size = self.config.dataset.data_loaders.train_batch_size
            image_paths = self.train_image_paths
            shuffle = True
            drop_last = True
        elif mode == 'val':
            transforms = {
                'pre_transforms': self.config.dataset.pre_transforms,
                'augmentations': self.config.dataset.val_augmentations,
                'post_transforms': self.config.dataset.post_transforms
            }
            batch_size = self.config.dataset.data_loaders.val_batch_size
            image_paths = self.val_image_paths
            shuffle = False
            drop_last = False
        else:
            raise ValueError(f'Unknown data mode: {mode}')
        
        transforms = {key: list(map(lambda t: get_augmentation(self.config, t), value))
                      for key, value in transforms.items()}

        dataset = Dataset(config=self.config,
                          image_paths=image_paths,
                          mode=mode,
                          transforms=transforms)
        return torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           num_workers=os.cpu_count(),
                                           drop_last=drop_last)          

    @property
    def train_dataloader(self):
        """
        Возвращает DataLoader для тренировочного режима.

        Returns:
            torch.utils.data.DataLoader: Dataloader для тренировки.
        """
        return self._get_dataloader(mode='train')

    @property
    def val_dataloader(self):
        """
        Возвращает DataLoader для валидационного режима.

        Returns:
            torch.utils.data.DataLoader: Dataloader для валидации.
        """
        return self._get_dataloader(mode='val')