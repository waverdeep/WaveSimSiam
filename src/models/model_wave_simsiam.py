import copy
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def set_requires_grad(model, requires):
    for parameter in model.parameters():
        parameter.requires_grad = requires


class MLPNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, stride, filter_size, padding):
        super(Encoder, self).__init__()
        self.feature_extractor = nn.Sequential()
        self.feature_encoder = nn.Sequential()
        for index, (stride, filter_size, padding) in enumerate(zip(stride, filter_size, padding)):
            self.feature_extractor.add_module(
                "extractor_layer{}".format(index),
                nn.Sequential(
                    nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim,
                              kernel_size=filter_size, stride=stride, padding=padding),
                    nn.BatchNorm1d(hidden_dim),
                    nn.PReLU(),
                )
            )
            input_dim = hidden_dim
        self.feature_encoder.add_module(
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

        self.adaptive_average_pooling = nn.AdaptiveAvgPool2d(1)
        self.adaptive_max_pooling = nn.AdaptiveMaxPool2d(1)


    def forward(self, x):
        x = F.normalize(x, dim=-1, p=2)
        feature_x = self.feature_extractor(x)
        print(feature_x.size())

        feature_chunks = feature_x.chunk(3, dim=1)
        feature_cat = torch.stack(feature_chunks, dim=1)
        feature_cat = F.normalize(feature_cat, dim=-1, p=2)

        representation_x = self.feature_encoder(feature_cat)
        print(representation_x.size())

        out01 = self.adaptive_max_pooling(representation_x)
        B, T, D, C = out01.shape
        out01 = out01.reshape((B, T * D * C))

        out02 = self.adaptive_average_pooling(representation_x)
        B, T, D, C = out02.shape
        out02 = out02.reshape((B, T * D * C))

        out_merge = out01 + out02
        out_merge = F.normalize(out_merge, dim=1, p=2)
        return out_merge, [feature_x, representation_x]


class WaveSimSiam(nn.Module):
    def __init__(self, encoder_input_dim, encoder_hidden_dim, encoder_stride, encoder_filter_size, encoder_padding):
        super(WaveSimSiam, self).__init__()
        self.encoder = Encoder(input_dim=encoder_input_dim, hidden_dim=encoder_hidden_dim,
                               stride=encoder_stride, filter_size=encoder_filter_size, padding=encoder_padding)

if __name__ == '__main__':
    test_encoder = Encoder(
        input_dim=1,
        hidden_dim=513,
        filter_size=[10, 8, 6, 4, 2],
        stride=[5, 4, 2, 2, 2],
        padding=[2, 2, 2, 2, 1],
    )

    sample_data = torch.rand(2, 1, 15200)
    out = test_encoder(sample_data)
