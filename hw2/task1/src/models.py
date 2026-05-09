import torch
import torch.nn as nn
from torchvision import models
import config

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def build_model(num_classes=37):
    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1 if config.USE_PRETRAINED else None
    )

    if config.USE_ATTENTION:
        for layer in [model.layer3, model.layer4]:
            for block in layer:
                block.se = SEBlock(block.bn2.num_features)
                old_forward = block.forward

                def new_forward(x, fwd=old_forward, b=block):
                    identity = x
                    out = b.conv1(x)
                    out = b.bn1(out)
                    out = b.relu(out)
                    out = b.conv2(out)
                    out = b.bn2(out)
                    if b.downsample is not None:
                        identity = b.downsample(x)
                    out += identity
                    out = b.se(out)
                    out = b.relu(out)
                    return out

                block.forward = new_forward

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(config.DEVICE)

    params = [
        {"params": [p for n, p in model.named_parameters() if "fc" not in n], "lr": config.BASE_LR},
        {"params": model.fc.parameters(), "lr": config.HEAD_LR}
    ]

    return model, params




# import torch
# import torch.nn as nn
# from torchvision import models
# import config

# class SEBlock(nn.Module):
#     def __init__(self, channel, reduction=8):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction),
#             nn.ReLU(),
#             nn.Linear(channel // reduction, channel),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)

# def build_model():
#     if config.USE_ATTENTION:
#         model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if config.USE_PRETRAINED else None)
#         for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
#             for block in layer:
#                 block.add_module("se", SEBlock(block.bn2.num_features))
#     else:
#         model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if config.USE_PRETRAINED else None)

#     model.fc = nn.Linear(model.fc.in_features, 37)
#     model = model.to(config.DEVICE)

#     params = [
#         {"params": [p for n, p in model.named_parameters() if "fc" not in n], "lr": config.BASE_LR},
#         {"params": model.fc.parameters(), "lr": config.HEAD_LR}
#     ]
#     return model, params