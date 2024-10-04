from torch import nn
    
def get_criterion(config):
    """
    Возвращает функцию потерь на основе конфигурации.

    Args:
        config (Box): Объект конфигурации.

    Returns:
        nn.Module: Функция потерь.
    """
    if config.criterion.type == 'cross_entropy':
        return nn.CrossEntropyLoss(reduction='mean')
    
    raise ValueError(f'Invalid criterion type: {config.criterion.type}')