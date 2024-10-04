import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import random
    
def get_augmentation(config, transform):
    """
    Возвращает аугментацию на основе конфигурации и типа преобразования.

    Args:
        config (Box): Объект конфигурации.
        transform (Box): Описание параметров конкретной аугментации.

    Returns:
        A.BasicTransform: Аугментация из библиотеки Albumentations.

    Raises:
        ValueError: Если указан неизвестный тип аугментации.
    """
    match transform.type:
        case 'normalize':
            return A.Normalize()
        case 'longest_max_size':
            return A.LongestMaxSize(max_size=config.dataset.image_size,
                                    interpolation=cv2.INTER_AREA)
        case 'pad_if_needed':
            return A.PadIfNeeded(min_height=config.dataset.image_size, 
                                 min_width=config.dataset.image_size,
                                 border_mode=3)
        case 'random_crop_from_borders':
            return A.RandomCropFromBorders(crop_left=transform.crop_ratio, 
                                           crop_right=transform.crop_ratio, 
                                           crop_top=transform.crop_ratio, 
                                           crop_bottom=transform.crop_ratio,
                                           p=transform.p)
        case 'shift_scale_rotate':
            return A.ShiftScaleRotate(shift_limit=transform.shift_limit,
                                      scale_limit=transform.scale_limit,
                                      rotate_limit=transform.rotate_limit,
                                      border_mode=3,
                                      p=transform.p)
        case 'elastic_transform':
            return A.ElasticTransform(alpha=transform.alpha,
                                      sigma=transform.sigma,
                                      border_mode=3,
                                      approximate=True,
                                      same_dxdy=True,
                                      p=transform.p)
        case 'grid_destortion':
            return A.GridDistortion(num_steps=transform.num_steps,
                                    distort_limit=transform.distort_limit,
                                    p=transform.p)
        case 'random_brightness_contrast':
            return A.RandomBrightnessContrast(brightness_limit=transform.brightness_limit,
                                              contrast_limit=transform.contrast_limit,
                                              p=transform.p)
        case 'median_blur':
            return A.MedianBlur(blur_limit=transform.blur_limit, 
                                p=transform.p)
        case 'rgb_shift':
            return A.RGBShift(r_shift_limit=transform.r_shift_limit, 
                              g_shift_limit=transform.g_shift_limit,
                              b_shift_limit=transform.b_shift_limit,
                              p=transform.p)
        case 'planckian_jitter':
            return A.PlanckianJitter(temperature_limit=transform.temperature_limit,
                                     p=transform.p)
        case 'coarse_dropout':
            return A.CoarseDropout(min_holes=transform.min_holes,
                                   max_holes=transform.max_holes,
                                   min_height=transform.min_height,
                                   max_height=transform.max_height,
                                   min_width=transform.min_width,
                                   max_width=transform.max_width,
                                   fill_value=transform.fill_value,
                                   p=transform.p)
        case 'randaugment':
            return RandAugment(num_transforms=transform.num_transforms,
                               magnitude=transform.magnitude,
                               p=transform.p)
        case 'trivialaugment':
            return TrivialAugment(p=transform.p)
        case 'to_tensor':
            return ToTensorV2()
        case _:
            raise ValueError(f'Unknown augmentation type: {transform.type}')
        

class RandAugment(A.BaseCompose):
    """
    Применяет серию случайных преобразований, как описано в статье RandAugment.

    Args:
        num_transforms (int): Количество преобразований для применения. 
        magnitude (int): Величина каждого преобразования (от 0 до 10).
        p (float): Вероятность применения всей серии преобразований.
    """

    def __init__(self, num_transforms = 3, magnitude = 3, p = 1.0):
        assert 0 <= magnitude <= 10, "Magnitude must be between 0 and 10."

        super().__init__(randaugment_transforms(magnitude), p)

        assert 1 <= num_transforms <= len(self.transforms), (
            "Number of transforms must be between 1 and the number of available transforms."
        )

        self.num_transforms = num_transforms
        self.magnitude = magnitude

    def __call__(self, *arg, force_apply = False, **data):
        if force_apply or random.random() < self.p:
            transforms = random.sample(self.transforms, self.num_transforms)
            for t in transforms:
                data = t(force_apply=True, **data)

        return data

class TrivialAugment(A.ImageOnlyTransform):
    """
    Применяет одно случайное преобразование с случайной величиной, как описано в статье TrivialAugment.

    Args:
        p (float): Вероятность применения преобразования. По умолчанию: 1.0.
    """

    def __init__(self, p: float = 1.0):
        super().__init__(p=p)

    def __call__(self, *arg, force_apply = False, **data):
        if force_apply or random.random() < self.p:
            magnitude = random.randint(0, 10)
            transform = random.choice(randaugment_transforms(magnitude))
            data = transform(force_apply=True, **data)

        return data


def randaugment_transforms(magnitude = 4):
    """
    Возвращает список преобразований albumentations для RandAugment или TrivialAugment с заданной величиной.

    Args:
        magnitude (int): Величина каждого преобразования в возвращаемом списке.

    Returns:
        List[A.BasicTransform]: Список аугментаций для RandAugment.
    """

    transform_list = [
        A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=.1+magnitude*.15, always_apply=True),
        A.RandomBrightnessContrast(brightness_limit=.1+magnitude*.15, contrast_limit=0, always_apply=True),
        A.Solarize(threshold=255-magnitude*25, always_apply=True),
        A.Equalize(always_apply=True),
        A.RGBShift(r_shift_limit=magnitude*10, g_shift_limit=magnitude*10, b_shift_limit=magnitude*10, always_apply=True),
        A.Sharpen((0, magnitude/10), always_apply=True),
        A.Posterize(num_bits=8-int(magnitude*4/10), always_apply=True),
        A.Rotate(limit=magnitude*5, always_apply=True),
        A.Affine(shear={"x": (-magnitude*10, magnitude*10)}, always_apply=True),
        A.Affine(shear={"y": (-magnitude*10, magnitude*10)}, always_apply=True),
        A.Affine(translate_percent={"x": (-magnitude*.05, magnitude*.05)}, always_apply=True),
        A.Affine(translate_percent={"y": (-magnitude*.05, magnitude*.05)}, always_apply=True),
        A.NoOp(always_apply=True),
    ]
    return transform_list