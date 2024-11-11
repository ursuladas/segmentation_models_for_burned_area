import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class BaseSegmentationModel(nn.Module):  # Inherit from nn.Module
    def __init__(self, model_name, encoder_name='resnet34', in_channels=3, classes=1, encoder_weights=None):
        super(BaseSegmentationModel, self).__init__()
        self.model_name = model_name
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.classes = classes
        self.encoder_weights = encoder_weights
        self.model = self._build_model()
    
    def _build_model(self):
        if self.model_name == "Unet":
            return smp.Unet(encoder_name=self.encoder_name, 
                            in_channels=self.in_channels, 
                            classes=self.classes, 
                            encoder_weights=self.encoder_weights)
        elif self.model_name == "Unet++":
            return smp.UnetPlusPlus(encoder_name=self.encoder_name, 
                           in_channels=self.in_channels, 
                           classes=self.classes, 
                           encoder_weights=self.encoder_weights)
        elif self.model_name == "DeepLabV3+":
            return smp.DeepLabV3Plus(encoder_name=self.encoder_name, 
                           in_channels=self.in_channels, 
                           classes=self.classes, 
                           encoder_weights=self.encoder_weights)
        elif self.model_name == "MAnet":
            return smp.MAnet(encoder_name=self.encoder_name, 
                           in_channels=self.in_channels, 
                           classes=self.classes, 
                           encoder_weights=self.encoder_weights)
        elif self.model_name == "FPN":
            return smp.FPN(encoder_name=self.encoder_name, 
                           in_channels=self.in_channels, 
                           classes=self.classes, 
                           encoder_weights=self.encoder_weights)
        elif self.model_name == "PSPNet":
            return smp.PSPNet(encoder_name=self.encoder_name, 
                              in_channels=self.in_channels, 
                              classes=self.classes, 
                              encoder_weights=self.encoder_weights)
        elif self.model_name == "DeepLabV3":
            return smp.DeepLabV3(encoder_name=self.encoder_name, 
                                 in_channels=self.in_channels, 
                                 classes=self.classes, 
                                 encoder_weights=self.encoder_weights)
        elif self.model_name == "PAN":
            return smp.PAN(encoder_name=self.encoder_name, 
                           in_channels=self.in_channels, 
                           classes=self.classes, 
                           encoder_weights=self.encoder_weights)
        else:
            raise ValueError(f"Model {self.model_name} not supported.")
    
    def forward(self, x):
        return self.model(x)

# Model-specific classes
class UnetModel(BaseSegmentationModel):
    def __init__(self, encoder_name='resnet34', in_channels=3, classes=1, encoder_weights=None):
        super(UnetModel, self).__init__(model_name="Unet", encoder_name=encoder_name, in_channels=in_channels, classes=classes, encoder_weights=encoder_weights)

class UnetPlusPlusModel(BaseSegmentationModel):
    def __init__(self, encoder_name='resnet34', in_channels=3, classes=1, encoder_weights=None):
        super(UnetPlusPlusModel, self).__init__(model_name="Unet++", encoder_name=encoder_name, in_channels=in_channels, classes=classes, encoder_weights=encoder_weights)

class FPNModel(BaseSegmentationModel):
    def __init__(self, encoder_name='resnet34', in_channels=3, classes=1, encoder_weights=None):
        super(FPNModel, self).__init__(model_name="FPN", encoder_name=encoder_name, in_channels=in_channels, classes=classes, encoder_weights=encoder_weights)

class PSPNetModel(BaseSegmentationModel):
    def __init__(self, encoder_name='resnet34', in_channels=3, classes=1, encoder_weights=None):
        super(PSPNetModel, self).__init__(model_name="PSPNet", encoder_name=encoder_name, in_channels=in_channels, classes=classes, encoder_weights=encoder_weights)

class DeepLabV3Model(BaseSegmentationModel):
    def __init__(self, encoder_name='resnet34', in_channels=3, classes=1, encoder_weights=None):
        super(DeepLabV3Model, self).__init__(model_name="DeepLabV3", encoder_name=encoder_name, in_channels=in_channels, classes=classes, encoder_weights=encoder_weights)

class DeepLabV3PlusModel(BaseSegmentationModel):
    def __init__(self, encoder_name='resnet34', in_channels=3, classes=1, encoder_weights=None):
        super(DeepLabV3PlusModel, self).__init__(model_name="DeepLabV3+", encoder_name=encoder_name, in_channels=in_channels, classes=classes, encoder_weights=encoder_weights)

class PANModel(BaseSegmentationModel):
    def __init__(self, encoder_name='resnet34', in_channels=3, classes=1, encoder_weights=None):
        super(PANModel, self).__init__(model_name="PAN", encoder_name=encoder_name, in_channels=in_channels, classes=classes, encoder_weights=encoder_weights)

class MAnetModel(BaseSegmentationModel):
    def __init__(self, encoder_name='resnet34', in_channels=3, classes=1, encoder_weights=None):
        super(MAnetModel, self).__init__(model_name="MAnet", encoder_name=encoder_name, in_channels=in_channels, classes=classes, encoder_weights=encoder_weights)
