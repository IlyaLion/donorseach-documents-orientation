import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import timm
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, backbone_name):
        super().__init__()
        self.backbone = timm.create_model(model_name=backbone_name,
                                          pretrained=False,
                                          num_classes=0,
                                          in_chans=3)
        self.fc = nn.Linear(self.backbone.head_hidden_size, 4, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x     
    
class Rotator:
    def __init__(self, backbone_name, model_weights_filename, image_size):
        self.image_size = image_size
        self._init_model(backbone_name, model_weights_filename)

    def _init_model(self, backbone_name, model_weights_filename):
        self.model = Model(backbone_name=backbone_name)
        #  add   , map_location=torch.device('cpu')
        self.model.load_state_dict(torch.load(model_weights_filename, map_location=torch.device('cpu')))
        self.model.eval()

    def rotate(self, image):
        rotation_angle = self._get_rotation_angle(image)
        rotated_image = np.rot90(image, k=-rotation_angle, axes=(0, 1))
        return rotated_image

    def _get_rotation_angle(self, image):
        tensor = self._image_to_tensor(image)
        with torch.no_grad():
            prediction = self.model(tensor)[0].argmax(dim=0).item()
        return prediction


    def _image_to_tensor(self, image):
        transforms = A.Compose([A.LongestMaxSize(self.image_size, interpolation=cv2.INTER_AREA),
                                A.PadIfNeeded(min_height=self.image_size, min_width=self.image_size, border_mode=3),
                                A.Normalize(),
                                ToTensorV2()])
        tensor = transforms(image=image)["image"].unsqueeze(0)
        return tensor