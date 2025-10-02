# -*- coding: utf-8 -*-
# @Time    : 22/3/21 8:43
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    : TextNet.py
import math
from models.BaseModel import BaseModel
from module.init_weights import weights_init_kaiming
import torch
import torch.nn as nn
from sarcopenia_data.SarcopeniaDataLoader import TEXT_COLS


class MetadataNetFeature(BaseModel):
    def __init__(self, backbone, n_channels, num_classes, pretrained):
        super(MetadataNetFeature, self).__init__(backbone, n_channels, num_classes, pretrained)
        self.hidden = [32, 128, 512]
        self.layer1 = nn.Sequential(
            nn.Conv2d(len(TEXT_COLS), self.hidden[0], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.hidden[0]),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.hidden[0], self.hidden[1], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.hidden[1]),
            nn.Sigmoid()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(self.hidden[1], self.hidden[2], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.hidden[2]),
            nn.SiLU(inplace=True)
        )
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        layer1 = self.layer1(x.view(-1, len(TEXT_COLS), 1, 1))
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        return layer3


class TextNetFeature(BaseModel):
    def __init__(self, backbone, n_channels, num_classes, pretrained):
        super(TextNetFeature, self).__init__(backbone, n_channels, num_classes, pretrained)
        in_planes2 = len(TEXT_COLS)
        self.hidden = [512]
        self.num_conv1d = nn.Conv1d(in_planes2, self.hidden[-1], kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm1d(self.hidden[-1])
        self.silu2 = nn.SiLU(inplace=True)
        # self.fc = nn.Linear(self.hidden[-1], num_classes)
        self.fc = nn.Linear(self.hidden[-1], self.hidden[-1])
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, num_x2):
        # batch_size × text_len × embedding_size -> batch_size × embedding_size × text_len
        num_x = self.num_conv1d(num_x2.permute(0, 2, 1))
        num_x = self.silu2(self.bn2(num_x))
        num_x = self.fc(num_x.squeeze(-1)).unsqueeze(-1)  # 全连接临床特征
        return num_x


class TextNet(BaseModel):
    def __init__(self, backbone, n_channels, num_classes, pretrained):
        super(TextNet, self).__init__(backbone, n_channels, num_classes, pretrained)
        in_planes2 = len(TEXT_COLS)
        out_planes2 = 64
        self.num = nn.Conv1d(in_planes2, out_planes2, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_planes2)
        self.silu2 = nn.SiLU(inplace=True)
        self.fc = nn.Linear(out_planes2, num_classes)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, num_x2):
        num_x = self.num(num_x2.permute(0, 2, 1))
        self.feats = self.silu2(self.bn2(num_x))
        cls_out = self.fc(self.feats.squeeze(-1))
        return cls_out, None

    def get_feats(self):
        return self.feats.squeeze(-1)


############
# Omic Model
############
class MaxNet(BaseModel):
    def __init__(self, backbone, n_channels, num_classes, pretrained, omic_dim=32, dropout_rate=0.25, act=None,
                 init_max=True):
        super(MaxNet, self).__init__(backbone, n_channels, num_classes, pretrained)
        self.hidden = [64, 48, 32, 32]
        self.act = act
        input_dim = len(TEXT_COLS)
        encoder1 = nn.Sequential(
            nn.Linear(input_dim, self.hidden[0]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder2 = nn.Sequential(
            nn.Linear(self.hidden[0], self.hidden[1]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder3 = nn.Sequential(
            nn.Linear(self.hidden[1], self.hidden[2]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder4 = nn.Sequential(
            nn.Linear(self.hidden[2], omic_dim),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        self.encoder = nn.Sequential(encoder1, encoder2, encoder3, encoder4)
        self.classifier = nn.Sequential(nn.Linear(omic_dim, num_classes))

        if init_max:
            weights_init_kaiming(self)

        self.output_range = nn.Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = nn.Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def get_features(self, num_x2):
        num_x2 = num_x2.squeeze(1)
        features = self.encoder(num_x2)
        return features

    def forward(self, num_x2):
        num_x2 = num_x2.squeeze(1)
        features = self.encoder(num_x2)
        out = self.classifier(features)
        if self.act is not None:
            out = self.act(out)

            if isinstance(self.act, nn.Sigmoid):
                out = out * self.output_range + self.output_shift

        # return features, out
        return out, None


############
# Omic Model
############
class MaxNetFeature(BaseModel):
    def __init__(self, backbone, n_channels, num_classes, pretrained, omic_dim=32, dropout_rate=0.25, act=None,
                 init_max=True):
        super(MaxNetFeature, self).__init__(backbone, n_channels, num_classes, pretrained)
        self.hidden = [64, 48, 32, 32]
        self.act = act
        input_dim = len(TEXT_COLS)
        encoder1 = nn.Sequential(
            nn.Linear(input_dim, self.hidden[0]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder2 = nn.Sequential(
            nn.Linear(self.hidden[0], self.hidden[1]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder3 = nn.Sequential(
            nn.Linear(self.hidden[1], self.hidden[2]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder4 = nn.Sequential(
            nn.Linear(self.hidden[2], omic_dim),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        self.encoder = nn.Sequential(encoder1, encoder2, encoder3, encoder4)
        self.classifier = nn.Sequential(nn.Linear(omic_dim, num_classes))

        if init_max:
            weights_init_kaiming(self)

        self.output_range = nn.Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = nn.Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, num_x2):
        num_x2 = num_x2.squeeze(1)
        features = self.encoder(num_x2)
        return features


class SNN(nn.Module):
    """
    Self-normalizing neural network.

    Literature:
    'Self-Normalizing Neural Networks' by Klambauer et al (arXiv:1706.02515)

    """

    def __init__(self, n_in, n_hidden, dropout={}):
        """
        Initialize SNN.

        n_in - number of input coordinates
        n_hidden - list of ints, number of hidden units per layer
        dropout - dict, {'idx': p_drop}, i.e.
                  keys give the index of the hidden layer AFTER which
                  AlphaDropout is applied and the respective dropout
                  probabilities are given by the values
        """
        super().__init__()
        self.call_kwargs = {'n_in': n_in,
                            'n_hidden': n_hidden,
                            'dropout': dropout,
                            }
        self.n_out = n_hidden[-1]
        n_units = [n_in] + list(n_hidden)
        self.hidden_layers = nn.ModuleList([nn.Linear(n_units[i], n_units[i + 1])
                                            for i in range(len(n_units) - 1)
                                            ])
        # TODO: do we have a better way of doing this?
        # we could use None instead of the Identity,
        # but then we would have to check in forward if it is None
        # now we can just apply without 'thinking'
        dropout_list = [nn.Identity() for _ in range(len(self.hidden_layers))]
        for key, val in dropout.items():
            idx = int(key)
            dropout_list[idx] = nn.AlphaDropout(val)
        self.dropout_layers = nn.ModuleList(dropout_list)
        self.activation = nn.SELU()
        self.reset_parameters()  # initialize weights

    def forward(self, x):
        for h, d in zip(self.hidden_layers, self.dropout_layers):
            x = d(self.activation(h(x)))
        return x

    def reset_parameters(self):
        # properly initialize weights/biases
        for lay in self.dropout_layers:
            # reset the dropout layer params (if any)
            # check if it has a reset-parameters function first
            reset_func = getattr(lay, "reset_parameters", None)
            if reset_func is not None:
                lay.reset_parameters()
        # NOTE: I think we do not need to check:
        # we can only have nn.Linear layers in there
        # TODO? for the biases we keep the pytorch standard, i.e.
        # uniform \in [-1/sqrt(N_in), + 1/sqrt(N_in)]
        for lay in self.hidden_layers:
            lay.reset_parameters()  # reset biases (and weights)
            fan_out = lay.out_features
            nn.init.normal_(lay.weight, mean=0., std=1. / math.sqrt(fan_out))


if __name__ == '__main__':
    with torch.no_grad():
        x = torch.rand((1, 27 * 5))
        # model = MaxNet(backbone='none',
        #                n_channels=3, num_classes=1)
        model = SNN()
        y = model(x)
        print(y.shape)
