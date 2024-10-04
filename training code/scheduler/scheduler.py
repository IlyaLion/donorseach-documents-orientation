from torch.optim.lr_scheduler import OneCycleLR

def get_scheduler(optimizer, config, steps_per_epoch):
    """
    Создает и возвращает планировщик изменения скорости обучения для оптимизатора.

    Args:
        optimizer (torch.optim.Optimizer): Оптимизатор, для которого будет создан планировщик.
        config (Box): Объект конфигурации.
        steps_per_epoch (int): Количество шагов в каждой эпохе.

    Returns:
        torch.optim.lr_scheduler.OneCycleLR: Инициализированный планировщик изменения скорости обучения.
    """
    scheduler = OneCycleLR(optimizer, 
                           max_lr=[g['lr'] for g in optimizer.param_groups],
                           epochs=config.training.epochs,
                           steps_per_epoch=steps_per_epoch,
                           anneal_strategy=config.scheduler.anneal_strategy,
                           pct_start=config.scheduler.pct_start,
                           div_factor=config.scheduler.div_factor,
                           final_div_factor=config.scheduler.final_div_factor)

    return scheduler