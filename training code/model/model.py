import os
import timm
import torch
from torch import nn

class Model(nn.Module):
    """
    Модель нейронной сети, использующая предобученную архитектуру timm для извлечения признаков.

    Args:
        config (Box): Объект конфигурации.
    """
    def __init__(self, config):
        super().__init__()
        self.backbone = timm.create_model(model_name=config.model.backbone_name,
                                          pretrained=config.model.pretrained,
                                          num_classes=0,#config.model.n_classes,
                                          in_chans=config.model.in_channels)
        self.fc = nn.Linear(self.backbone.head_hidden_size, config.model.n_classes, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x     
    
def get_model(config):
    """
    Создает и возвращает модель на основе конфигурации. 

    Args:
        config (Box): Объект конфигурации.

    Returns:
        Model: Инициализированная модель.
    """
    model = Model(config)

    if config.model.load_from is not None:
        model_weights_file = os.path.join(config.directories.models, config.model.load_from)
        model.load_state_dict(torch.load(model_weights_file))
    return model