import torch.nn as nn
import torch.nn.functional as F
import torch
from tornado.concurrent import future_set_exc_info
from models.BaseModel import BaseModel
from module.backbone import BACKBONE
from models.cls_models.TextNet import TextNetFeature
from torch.nn.functional import normalize
from module.init_weights import weights_init_normal


class ResClsLessCNN(nn.Module):
    # initializers
    def __init__(self, filter_num=32, scale=16, num_class=2):
        super(ResClsLessCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Linear(filter_num * scale, num_class),
        )
        self.apply(weights_init_normal)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.conv1(x)
        return x


class ResNet(BaseModel):
    def __init__(self, backbone, n_channels, num_classes, pretrained):
        super(ResNet, self).__init__(backbone, n_channels, num_classes, pretrained)
        self.backbone = BACKBONE[backbone](backbone=backbone, pretrained=pretrained)  # pretrained ImageNet
        # self.backbone = BACKBONE[backbone](backbone=backbone, pretrained=pretrained, channels=4)
        scale = 16
        if backbone[:3] == 'vgg':
            scale = 16
        elif int(backbone[6:]) > 34:
            scale = 64
        elif int(backbone[6:]) <= 34:
            scale = 16
        else:
            raise Exception('Unknown backbone')
        self.cls_branch = ResClsLessCNN(scale=scale, num_class=num_classes)

    def get_backbone_layers(self):
        small_lr_layers = []
        small_lr_layers.append(self.backbone)
        return small_lr_layers

    def forward(self, image):
        self.res = self.backbone(image)
        out = self.cls_branch(self.res[-1])
        return out, None

    def get_feats(self):
        feats = F.adaptive_avg_pool2d(self.res[-1], (1, 1))
        feats = feats.view(feats.size(0), -1)
        return feats


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation=None):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        # self.back_conv = nn.Conv2d(in_channels=in_dim //8 , out_channels=in_dim, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # Attn^T
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class ResNetFusionTextNet(BaseModel):
    def __init__(self, backbone, n_channels, num_classes, pretrained):
        super(ResNetFusionTextNet, self).__init__(backbone, n_channels, num_classes, pretrained)

        self.feature_dim = 512
        self.filter_num = 32

        # ResNet、VGG
        if backbone[:3] == 'vgg':
            self.scale1 = 16
            self.conv1 = nn.Sequential(
                nn.Linear(self.filter_num * self.scale1, num_classes),
            )
        elif int(backbone[6:]) > 34:
            self.scale1 = 64
            self.conv1 = nn.Sequential(
                nn.Linear(self.filter_num * self.scale1, num_classes),
            )
        elif int(backbone[6:]) <= 34:
            self.scale1 = 16
            self.conv1 = nn.Sequential(
                nn.Linear(self.filter_num * self.scale1, num_classes),
            )
        else:
            raise Exception('Unknown backbone')
        # resnet18
        self.backbone = BACKBONE[backbone](backbone=backbone, pretrained=pretrained, channels=n_channels)
        self.dropout = nn.Dropout(p=0.5)
        self.text_net = TextNetFeature(backbone=backbone, n_channels=n_channels, num_classes=num_classes,
                                       pretrained=pretrained)
        self.text_dim = self.text_net.hidden[-1]
        self.attn512 = Self_Attn(self.feature_dim)
        self.mh_attn = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=1, batch_first=True, dropout=0.5)
        self.aw = nn.Parameter(torch.zeros(2))
        self.softmax2d = nn.Softmax2d()
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fusion_projector = nn.Sequential(
            nn.Linear(self.feature_dim * 1, self.feature_dim + self.text_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim + self.text_dim, self.feature_dim // 2),
        )

        self.classifier3 = nn.Sequential(
            nn.Linear(self.feature_dim * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2),
        )

    def get_backbone_layers(self):
        small_lr_layers = []
        small_lr_layers.append(self.backbone)
        return small_lr_layers

    def forward(self, x, mask, text, text_included=True):
        h = self.backbone(mask)
        nlb = self.dropout(h[-1])
        text_w = self.aw[0]
        text_feature = self.text_net(text)
        text = self.text_net.num_conv1d(text.permute(0, 2, 1))
        text_feature = text_feature + text_w * self.softmax2d(text_feature) * text
        text_imbedding = text_feature.squeeze(-1)
        attn = self.attn512(nlb)[0]
        vis_embedding = self.maxpool(attn).squeeze(-1).squeeze(-1)
        cross_attn_embedding, attn_weights = self.mh_attn(text_imbedding, vis_embedding, vis_embedding)
        fusion_embeddings = torch.cat((vis_embedding, cross_attn_embedding, text_imbedding), 1)
        contrs_learn = vis_embedding
        self.feats = self.maxpool(nlb).squeeze(-1).squeeze(-1)
        out = self.classifier3(fusion_embeddings)
        z_i = normalize(self.fusion_projector(contrs_learn), dim=1)
        return out, z_i

    def get_feats(self):
        return self.feats


if __name__ == '__main__':
    # ResMax = ResNetFusionMaxNet('resnet18', 3, 2, True)
    # print(ResMax)
    import torch

    model = ResNetFusionTextNet(backbone='resnet18', n_channels=4, num_classes=2, pretrained=True)
    text_f64 = torch.randn(32, 64)
    text_f512 = torch.randn(32, 512)
    image_f512 = torch.randn(32, 512)
    out, out_w = model.mh_attn(text_f512, image_f512, image_f512)
    img = torch.randn(32, 4, 224, 224)
    text = torch.randn(32, 1, 5)
    out, _ = model(img, text, text_included=True)

    # tf = torch.randn(32, 512, 1, 1)
    # cn = ChannelAtt(512, 512)
    # tfo, _ = cn(tf)
    # print(tfo.shape)
    pass
