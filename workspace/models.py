from torchvision.models import densenet161, DenseNet161_Weights, ResNet152_Weights, resnet152, googlenet
import torch.nn as nn

class GoogleNet(nn.Module):
    def __init__(self, num_classes):
        super(GoogleNet, self).__init__()
        self.model = googlenet(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)

class DenseNet(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet, self).__init__()
        self.model = densenet161(weights=DenseNet161_Weights.IMAGENET1K_V1)
        num_features = self.model.classifier.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)