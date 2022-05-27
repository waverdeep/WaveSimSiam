import torchvision
import torch.nn as nn


def select_feature_encoder_model(model_name, pretrain=True):
    feature_encoder = nn.Sequential()
    if model_name == 'h2-0':
        feature_encoder.add_module(
            "encoder_layer",
            nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.Conv2d(64, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(128, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),

                nn.Conv2d(256, 512, 3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(512, 512, 3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),

                nn.Conv2d(512, 512, 3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(512, 1024, 3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),

                nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
            )
        )

    elif model_name == 'h2':
        feature_encoder.add_module(
            "encoder_layer",
            nn.Sequential(
                nn.Conv2d(3, 64, 10, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.PReLU(),

                nn.Conv2d(64, 128, 8, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.PReLU(),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(128, 256, 4, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.PReLU(),

                nn.Conv2d(256, 512, 4, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.PReLU(),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(512, 512, 3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.PReLU(),

                nn.Conv2d(512, 1024, 3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(1024, 1024, 2, stride=1, padding=1),
                nn.ReLU(),

                nn.Conv2d(1024, 2048, 2, stride=1, padding=1),
                nn.ReLU(),
            )
        )

    elif model_name == 'h5':
        feature_encoder.add_module(
            "encoder_layer",
            nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(128, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),

                nn.Conv2d(256, 256, 3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 512, 3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(512, 512, 3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),

                nn.Conv2d(512, 512, 3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),

                nn.Conv2d(512, 1024, 3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),

                nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(),
                nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
            )
        )
    elif model_name == 'resnet50':
        feature_encoder.add_module(
            "encoder_layer",
            nn.Sequential(
                torchvision.models.resnet50(pretrained=pretrain).conv1,
                torchvision.models.resnet50(pretrained=pretrain).bn1,
                torchvision.models.resnet50(pretrained=pretrain).relu,
                torchvision.models.resnet50(pretrained=pretrain).layer1,
                torchvision.models.resnet50(pretrained=pretrain).layer2,
                torchvision.models.resnet50(pretrained=pretrain).layer3,
                torchvision.models.resnet50(pretrained=pretrain).layer4,

            )
        )
    elif model_name == 'resnet50m':
        feature_encoder.add_module(
            "encoder_layer",
            nn.Sequential(
                torchvision.models.resnet50(pretrained=pretrain).conv1,
                torchvision.models.resnet50(pretrained=pretrain).bn1,
                torchvision.models.resnet50(pretrained=pretrain).relu,
                torchvision.models.resnet50(pretrained=pretrain).maxpool,
                torchvision.models.resnet50(pretrained=pretrain).layer1,
                torchvision.models.resnet50(pretrained=pretrain).layer2,
                torchvision.models.resnet50(pretrained=pretrain).layer3,
                torchvision.models.resnet50(pretrained=pretrain).layer4,

            )
        )
    elif model_name == 'resnet34m':
        feature_encoder.add_module(
            "encoder_layer",
            nn.Sequential(
                torchvision.models.resnet34(pretrained=pretrain).conv1,
                torchvision.models.resnet34(pretrained=pretrain).bn1,
                torchvision.models.resnet34(pretrained=pretrain).relu,
                torchvision.models.resnet34(pretrained=pretrain).maxpool,
                torchvision.models.resnet34(pretrained=pretrain).layer1,
                torchvision.models.resnet34(pretrained=pretrain).layer2,
                torchvision.models.resnet34(pretrained=pretrain).layer3,
                torchvision.models.resnet34(pretrained=pretrain).layer4,

            )
        )
    elif model_name == 'resnet18m':
        feature_encoder.add_module(
            "encoder_layer",
            nn.Sequential(
                torchvision.models.resnet18(pretrained=pretrain).conv1,
                torchvision.models.resnet18(pretrained=pretrain).bn1,
                torchvision.models.resnet18(pretrained=pretrain).relu,
                torchvision.models.resnet18(pretrained=pretrain).maxpool,
                torchvision.models.resnet18(pretrained=pretrain).layer1,
                torchvision.models.resnet18(pretrained=pretrain).layer2,
                torchvision.models.resnet18(pretrained=pretrain).layer3,
                torchvision.models.resnet18(pretrained=pretrain).layer4,

            )
        )
    elif model_name == 'resnet101m':
        print("ResNet101M")
        feature_encoder.add_module(
            "encoder_layer",
            nn.Sequential(
                torchvision.models.resnet101(pretrained=pretrain).conv1,
                torchvision.models.resnet101(pretrained=pretrain).bn1,
                torchvision.models.resnet101(pretrained=pretrain).relu,
                torchvision.models.resnet101(pretrained=pretrain).maxpool,
                torchvision.models.resnet101(pretrained=pretrain).layer1,
                torchvision.models.resnet101(pretrained=pretrain).layer2,
                torchvision.models.resnet101(pretrained=pretrain).layer3,
                torchvision.models.resnet101(pretrained=pretrain).layer4,

            )
        )
    elif model_name == 'resnet152m':
        print("ResNet152M")
        feature_encoder.add_module(
            "encoder_layer",
            nn.Sequential(
                torchvision.models.resnet152(pretrained=pretrain).conv1,
                torchvision.models.resnet152(pretrained=pretrain).bn1,
                torchvision.models.resnet152(pretrained=pretrain).relu,
                torchvision.models.resnet152(pretrained=pretrain).maxpool,
                torchvision.models.resnet152(pretrained=pretrain).layer1,
                torchvision.models.resnet152(pretrained=pretrain).layer2,
                torchvision.models.resnet152(pretrained=pretrain).layer3,
                torchvision.models.resnet152(pretrained=pretrain).layer4,

            )
        )
    elif model_name == 'resnet18m':
        print("ResNet152M")
        feature_encoder.add_module(
            "encoder_layer",
            nn.Sequential(
                torchvision.models.resnet18(pretrained=pretrain).conv1,
                torchvision.models.resnet18(pretrained=pretrain).bn1,
                torchvision.models.resnet18(pretrained=pretrain).relu,
                torchvision.models.resnet18(pretrained=pretrain).maxpool,
                torchvision.models.resnet18(pretrained=pretrain).layer1,
                torchvision.models.resnet18(pretrained=pretrain).layer2,
                torchvision.models.resnet18(pretrained=pretrain).layer3,
                torchvision.models.resnet18(pretrained=pretrain).layer4,

            )
        )
    elif model_name == 'resnet34m':
        print("ResNet152M")
        feature_encoder.add_module(
            "encoder_layer",
            nn.Sequential(
                torchvision.models.resnet34(pretrained=pretrain).conv1,
                torchvision.models.resnet34(pretrained=pretrain).bn1,
                torchvision.models.resnet34(pretrained=pretrain).relu,
                torchvision.models.resnet34(pretrained=pretrain).maxpool,
                torchvision.models.resnet34(pretrained=pretrain).layer1,
                torchvision.models.resnet34(pretrained=pretrain).layer2,
                torchvision.models.resnet34(pretrained=pretrain).layer3,
                torchvision.models.resnet34(pretrained=pretrain).layer4,

            )
        )
    elif model_name == 'mobile3large':
        feature_encoder.add_module(
            "encoder_layer",
            nn.Sequential(
                torchvision.models.mobilenet_v3_large(pretrained=pretrain).features
            )
        )
    elif model_name == 'mobile2':
        feature_encoder.add_module(
            "encoder_layer",
            nn.Sequential(
                torchvision.models.mobilenet_v2(pretrained=pretrain).features
            )
        )
    return feature_encoder
