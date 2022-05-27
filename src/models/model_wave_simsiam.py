import copy
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import src.models.model_feature_encoder as model_feature_encoder
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def loss_function(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def simsiam_loss_function(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return -(x * y).sum(dim=-1).mean()


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
    def __init__(self, input_dim, hidden_dim, stride, filter_size, padding, feature_extractor_model, pretrain, dropout=None):
        super(Encoder, self).__init__()
        assert (
                len(stride) == len(filter_size) == len(padding)
        ), "Inconsistent length of strides, filter sizes and padding"
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

        if dropout is not None:
            self.feature_extractor.add_module(
                "dropout_layer",
                nn.Sequential(
                    nn.Dropout(dropout)
                )
            )

        self.feature_encoder = model_feature_encoder.select_feature_encoder_model(
            model_name=feature_extractor_model, pretrain=pretrain)

        self.adaptive_average_pooling = nn.AdaptiveAvgPool2d(1)
        self.adaptive_max_pooling = nn.AdaptiveMaxPool2d(1)


    def forward(self, x):
        x = F.normalize(x, dim=-1, p=2)
        feature_x = self.feature_extractor(x)

        feature_chunks = feature_x.chunk(3, dim=1)
        feature_cat = torch.stack(feature_chunks, dim=1)
        feature_cat = F.normalize(feature_cat, dim=-1, p=2)

        representation_x = self.feature_encoder(feature_cat)

        out01 = self.adaptive_max_pooling(representation_x)
        B, T, D, C = out01.shape
        out01 = out01.reshape((B, T * D * C))

        out02 = self.adaptive_average_pooling(representation_x)
        B, T, D, C = out02.shape
        out02 = out02.reshape((B, T * D * C))

        out_merge = out01 + out02
        out_merge = F.normalize(out_merge, dim=1, p=2)
        return out_merge, [representation_x, feature_x]


class WaveSimSiam(nn.Module):
    def __init__(self, config, encoder_input_dim, encoder_hidden_dim, encoder_stride, encoder_filter_size, encoder_padding,
                 mlp_input_dim, mlp_hidden_dim, mlp_output_dim, feature_extractor_model, pretrain, dropout):
        super(WaveSimSiam, self).__init__()
        self.encoder = Encoder(input_dim=encoder_input_dim, hidden_dim=encoder_hidden_dim,
                               stride=encoder_stride, filter_size=encoder_filter_size, padding=encoder_padding,
                               feature_extractor_model=feature_extractor_model, pretrain=pretrain, dropout=dropout)
        self.projection = nn.Sequential(MLPNetwork(mlp_input_dim, mlp_hidden_dim, mlp_hidden_dim),
                                        MLPNetwork(mlp_hidden_dim, mlp_hidden_dim, mlp_hidden_dim),
                                        MLPNetwork(mlp_hidden_dim, mlp_hidden_dim, mlp_hidden_dim),)

        self.prediction = nn.Sequential(MLPNetwork(mlp_hidden_dim, mlp_hidden_dim, mlp_hidden_dim),
                                        MLPNetwork(mlp_hidden_dim, mlp_hidden_dim, mlp_output_dim),)

    def forward(self, x01, x02):
        input01 = x01
        input02 = x02

        encoder_output01, _ = self.encoder(input01)
        encoder_output02, _ = self.encoder(input02)

        projection_output01 = self.projection(encoder_output01)
        projection_output02 = self.projection(encoder_output02)

        prediction_output01 = self.prediction(projection_output01)
        prediction_output02 = self.prediction(projection_output02)

        target_output01 = projection_output01.detach()
        target_output02 = projection_output02.detach()

        loss01 = loss_function(prediction_output01, target_output02)
        loss02 = loss_function(prediction_output02, target_output01)
        loss = loss01 + loss02

        # loss01 = simsiam_loss_function(prediction_output01, target_output02)
        # loss02 = simsiam_loss_function(prediction_output02, target_output01)
        # loss = loss01/2 + loss02/2

        return loss.mean(), [encoder_output01, encoder_output02, encoder_output01, encoder_output02]

    def get_representation(self, x):
        online, online_rep = self.encoder(x)
        return online_rep
        # online_projection = self.projection(online)
        # return online_projection


if __name__ == '__main__':
    config={}
    test_encoder = WaveSimSiam(
        config=config,
        encoder_input_dim=1,
        encoder_hidden_dim=513,
        encoder_filter_size=[10, 8, 6, 4, 2],
        encoder_stride=[5, 4, 3, 2, 2],
        encoder_padding=[2, 2, 2, 2, 1],
        mlp_input_dim=2048,
        mlp_hidden_dim=4096,
        mlp_output_dim=4096,
        feature_extractor_model='resnet101m',
        pretrain=True,
        dropout=None
    )

    sample_data = torch.rand(2, 1, 20480)
    out = test_encoder(sample_data, sample_data)
    rep = test_encoder.get_representation(sample_data)
    print(rep[0].size())
    print(rep[1].size())
    print(out[0])
