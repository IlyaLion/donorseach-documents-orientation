import gc
import json
import numpy as np
import os
import random
import torch
from tqdm import tqdm

from criterion.metrics import calc_accuracy
from criterion.criterion import get_criterion
from dataset.dataset import DataModule
from model.model import get_model
from optimizer.optimizer import get_optimizer
from scheduler.scheduler import get_scheduler

class AverageMeter:
    """
    Класс для вычисления средних значений и статистики во время обучения.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Установка значения в 0.
        """
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Обновление среднего значения.

        Args:
            val (float): Значение для обновления.
            n (int): Количество значений, которое соответствует val.
        """
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def seed_everything(seed):
    """Установка random seed для воспроизводимости.

    Args:
        seed (int): Значение random seed для инициализации генераторов случайных чисел.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cutmix(batch, targets, alpha):
    """
    Применение CutMix к данным.

    Args:
        batch (torch.Tensor): Тензор входных данных.
        targets (torch.Tensor): Целевые метки для данных.
        alpha (float): Параметр для распределения Бета.

    Returns:
        Tuple: Измененный тензор данных и целевые метки.
    """
    indices = torch.randperm(batch.size(0))
    shuffled_data = batch[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)

    image_h, image_w = batch.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    batch[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    targets = (targets, shuffled_targets, lam)

    return batch, targets

def mixup(batch, targets, alpha):
    """
    Применение MixUp к данным.

    Args:
        batch (torch.Tensor): Тензор входных данных.
        targets (torch.Tensor): Целевые метки для данных.
        alpha (float): Параметр для распределения Бета.

    Returns:
        Tuple: Измененный тензор данных и целевые метки.
    """
    indices = torch.randperm(batch.size(0))
    shuffled_data = batch[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    new_batch = batch * lam + shuffled_data * (1 - lam)
    new_targets = [targets, shuffled_targets, lam]
    return new_batch, new_targets

def train_one_epoch(config, model, optimizer, scheduler, criterion, train_dataloader, val_dataloader):
    """
    Обучение модели одну эпоху.

    Args:
        config (Box): Объект конфигурации.
        model (torch.nn.Module): Модель для обучения.
        optimizer (torch.optim.Optimizer): Оптимизатор.
        scheduler (torch.optim.lr_scheduler): Планировщик обучения.
        criterion (torch.nn.Module): Функция потерь.
        train_dataloader (DataLoader): Даталоадер для обучающего набора данных.
        val_dataloader (DataLoader): Даталоадер для валидационного набора данных.

    Returns:
        Tuple: Средняя ошибка обучения, среднияя ошибка валидации, точность валидации.
    """
    if config.training.precision == 16:
        use_amp = True
    elif config.training.precision == 32:
        use_amp = False
    else:
        raise ValueError(f'Incorrect precision value: {config.training.precision}')
        
    if config.training.mixing.strategy == 'mixup':
        mixing_f = mixup
    elif config.training.mixing.strategy == 'cutmix':
        mixing_f = cutmix
    elif config.training.mixing.strategy is None:
        mixing_f = None
    else:
        raise ValueError(f'Incorrect mixing strategy: {config.training.mixing.strategy}')

    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    train_losses = AverageMeter()
    
    for step, (images, labels) in enumerate(tqdm(train_dataloader, desc='training', smoothing=0.05)):
        images, labels = images.cuda(), labels.cuda()
        
        mixing_on_this_batch = mixing_f is not None and torch.rand(1).item() < config.training.mixing.p
        if mixing_on_this_batch:
            images, labels = mixing_f(images, labels, alpha=config.training.mixing.alpha)
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            y_pred = model(images)
            if mixing_on_this_batch:
                labels, shuffled_labels, lambda_ = labels
                loss = lambda_ * criterion(y_pred, labels) + (1 - lambda_) * criterion(y_pred, shuffled_labels)
            else:
                loss = criterion(y_pred, labels)
        
        if config.training.accumulate_grad_batches > 1:
            loss = loss / config.training.accumulate_grad_batches
        
        batch_size = labels.size(0)   
        train_losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        
        if (step + 1) % config.training.accumulate_grad_batches == 0:
            scaler.unscale_(optimizer)
            if config.training.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
       
            
    val_losses = AverageMeter()
    model.eval()
    y_trues, y_preds = [], []       
    for step, (images, labels) in enumerate(tqdm(val_dataloader, desc='validation', smoothing=0.05)):
        images, labels = images.cuda(), labels.cuda()

        batch_size = labels.size(0)

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16), torch.no_grad():
            y_pred = model(images)
            loss = criterion(y_pred, labels)

        if config.training.accumulate_grad_batches > 1:
            loss = loss / config.training.accumulate_grad_batches
    
        val_losses.update(loss.item(), batch_size)
        y_pred = y_pred.softmax(dim=1).cpu().numpy()
        for i in range(batch_size):
            y_preds.append(y_pred[i])
            y_trues.append(labels[i].cpu().numpy())
        
    y_trues = np.array(y_trues)
    y_preds = np.array(y_preds)
    val_accuracy = calc_accuracy(y_trues, y_preds)
    return train_losses.avg, val_losses.avg, val_accuracy

def train_model(config):
    """Основная функция для обучения модели.

    Args:
        config (Box): Объект конфигурации.
    """
    seed_everything(config.seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_float32_matmul_precision('medium')

    dm = DataModule(config)
    train_dataloader = dm.train_dataloader
    val_dataloader = dm.val_dataloader

    model = get_model(config)
    if config.training.compile:
        model = torch.compile(model)
    model.cuda()
    
    optimizer = get_optimizer(model, config)
    criterion = get_criterion(config)

    train_steps_per_epoch = len(train_dataloader) // config.training.accumulate_grad_batches
    scheduler = get_scheduler(optimizer, config, train_steps_per_epoch)
    
    metrics = {}
    for epoch in range(1, config.training.epochs + 1):
        print(f'Epoch {epoch}/{config.training.epochs}')
        train_loss, val_loss, val_accuracy = train_one_epoch(config=config,
                                                             model=model,
                                                             optimizer=optimizer,
                                                             scheduler=scheduler, 
                                                             criterion=criterion, 
                                                             train_dataloader=train_dataloader, 
                                                             val_dataloader=val_dataloader)
        info_string = f'{train_loss=:.5f}-{val_loss=:.5f}-{val_accuracy=:.5f}'
        print(info_string)
        metrics[epoch] = {'train_loss': train_loss,
                          'val_loss': val_loss,
                          'val_accuracy': val_accuracy}
        
        checkpoint_dir = os.path.join(config.directories.models, config.experiment_name) 
        os.makedirs(checkpoint_dir, exist_ok=True)
        if config.training.save_checkpoints:
            model_filename = os.path.join(checkpoint_dir, f'{epoch=}-{info_string}.ckpt')
            state_dict = model.state_dict()
            if config.training.precision == 16:
                state_dict = {k: state_dict[k].to(torch.float16)
                              for k in state_dict.keys()}
            torch.save(state_dict, model_filename)

        train_dataloader.dataset.step()

    if config.training.save_metrics:
        metrics_filename = os.path.join(checkpoint_dir, 'metrics.json')
        with open(metrics_filename, 'w') as f:
            json.dump(metrics, f)

    del model
    gc.collect()
    torch.cuda.empty_cache()  