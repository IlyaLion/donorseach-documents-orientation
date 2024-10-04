from .parameters import get_optimizer_one_lr_params, get_optimizer_two_lrs_params
from torch.optim import AdamW
from bitsandbytes.optim import AdamW8bit


def get_optimizer(model, config):
    """
    Создает и возвращает оптимизатор для обучения модели на основе конфигурации.

    Args:
        model (nn.Module): Модель, для которой будет создан оптимизатор.
        config (Box): Объект конфигурации.

    Returns:
        torch.optim.Optimizer: Инициализированный оптимизатор.
    """
    match config.optimizer.lr_type:
        case 'one_lr':
            optimizer_parameters = get_optimizer_one_lr_params(model,
                                                               lr=config.optimizer.lr,
                                                               weight_decay=config.optimizer.weight_decay)
            lr = config.optimizer.lr
        case 'two_lrs':
            optimizer_parameters = get_optimizer_two_lrs_params(model,
                                                                backbone_lr=config.optimizer.backbone_lr,
                                                                fc_lr=config.optimizer.fc_lr,
                                                                weight_decay=config.optimizer.weight_decay)
            lr = config.optimizer.backbone_lr
        case _:
            raise ValueError(f'Invalid optimizer lr_type: {config.optimizer.lr_type}')
    
    match config.optimizer.type:
        case 'adamw':
            optimizer_f = AdamW
        case 'adamw_8bit':
            optimizer_f = AdamW8bit
        case _:
            raise ValueError(f'Invalid optimizer type: {config.optimizer.type}')
    

    optimizer = optimizer_f(optimizer_parameters, 
                            lr=lr,
                            eps=config.optimizer.eps,
                            betas=config.optimizer.betas)
    return optimizer