def get_optimizer_one_lr_params(model, lr, weight_decay):
    """
    Получает параметры для оптимизатора с одним значением скорости обучения.

    Args:
        model (nn.Module): Модель, для которой будут получены параметры.
        lr (float): Скорость обучения.
        weight_decay (float): Значение весового распада.

    Returns:
        list: Список словарей с параметрами для оптимизатора.
    """
    no_decay = ['bias']
    optimizer_parameters = [
        {'params': [p for n, p in model.backbone.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'lr': lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.backbone.named_parameters() 
                    if any(nd in n for nd in no_decay)],
         'lr': lr, 'weight_decay': 0.0},
        {'params': [p for p in model.fc.parameters()],
         'lr': lr, 'weight_decay': 0.0}
    ]
    return optimizer_parameters

def get_optimizer_two_lrs_params(model, backbone_lr, fc_lr, weight_decay):
    """
    Получает параметры для оптимизатора с двумя значениями скорости обучения.

    Args:
        model (nn.Module): Модель, для которой будут получены параметры.
        backbone_lr (float): Скорость обучения для backbone.
        fc_lr (float): Скорость обучения для полносвязного слоя.
        weight_decay (float): Значение весового распада.

    Returns:
        list: Список словарей с параметрами для оптимизатора.
    """
    no_decay = ['bias']
    optimizer_parameters = [
        {'params': [p for n, p in model.backbone.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'lr': backbone_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.backbone.named_parameters() 
                    if any(nd in n for nd in no_decay)],
         'lr': backbone_lr, 'weight_decay': 0.0},
        {'params': [p for p in model.fc.parameters()],
         'lr': fc_lr, 'weight_decay': 0.0}
    ]
    return optimizer_parameters