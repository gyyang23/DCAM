import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50


class Net(nn.Module):

    def __init__(self, stride=16, n_classes=20):
        super(Net, self).__init__()
        if stride == 16:
            self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))
            self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu,
                                        self.resnet50.maxpool, self.resnet50.layer1)
        else:
            self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 1, 1), dilations=(1, 1, 2, 2))
            self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu,
                                        self.resnet50.maxpool, self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.n_classes = n_classes

        self.classifier = nn.Conv2d(2048, self.n_classes, 1, bias=False)
        self.classifier2 = nn.Conv2d(2048, self.n_classes, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.dcam = nn.ModuleList([self.classifier])
        self.dcam_sce = nn.ModuleList([self.classifier2])

    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = torchutils.gap2d(x, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, self.n_classes)

        return x

    def train(self, mode=True):
        super(Net, self).train(mode)
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False
        # for p in self.backbone.parameters():
        #     p.requires_grad = False

    def trainable_parameters(self):

        return (list(self.backbone.parameters()),
                list(self.dcam.parameters()),
                list(self.dcam_sce.parameters()))


class Net_CAM(Net):

    def __init__(self, stride=16, n_classes=20):
        super(Net_CAM, self).__init__(stride=stride, n_classes=n_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        feature = self.stage4(x)

        x = torchutils.gap2d(feature, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, self.n_classes)

        cams = F.conv2d(feature, self.classifier.weight)
        cams = F.relu(cams)

        return x, cams, feature


class Net_CAM_Feature(Net):

    def __init__(self, stride=16, n_classes=20):
        super(Net_CAM_Feature, self).__init__(stride=stride, n_classes=n_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        feature = self.stage4(x)  # bs*2048*32*32

        x = torchutils.gap2d(feature, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, self.n_classes)

        cams = F.conv2d(feature, self.classifier.weight)
        cams = F.relu(cams)
        cams = cams / (F.adaptive_max_pool2d(cams, (1, 1)) + 1e-5)
        cams_feature = cams.unsqueeze(2) * feature.unsqueeze(1)  # bs*20*2048*32*32
        cams_feature = cams_feature.view(cams_feature.size(0), cams_feature.size(1), cams_feature.size(2), -1)
        cams_feature = torch.mean(cams_feature, -1)

        return x, cams_feature, cams


class CAM(Net):

    def __init__(self, stride=16, n_classes=20):
        super(CAM, self).__init__(stride=stride, n_classes=n_classes)

    def forward(self, x, separate=False):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.conv2d(x, self.classifier.weight)

        if separate:
            return x
        x = F.relu(x)
        x = x[0] + x[1].flip(-1)

        return x

    def forward1(self, x, weights, separate=False):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.conv2d(x, weights)

        if separate:
            return x
        x = F.relu(x)
        x = x[0] + x[1].flip(-1)

        return x

    def forward2(self, x, weight, separate=False):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.conv2d(x,  weight * self.classifier.weight)

        if separate:
            return x
        x = F.relu(x)
        x = x[0] + x[1].flip(-1)

        return x


class DCAM_SCE(Net):

    def __init__(self, stride=16, n_classes=20):
        super(DCAM_SCE, self).__init__(stride=stride, n_classes=n_classes)

        for p in self.backbone.parameters():
            p.requires_grad = False

        for p in self.classifier.parameters():
            p.requires_grad = False

    def forward(self, x, label):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        feature = self.stage4(x)  # bs*2048*32*32

        x = torchutils.gap2d(feature, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, self.n_classes)

        loss_cls = torch.tensor(0.0, device='cuda:0')
        # loss_cls = F.multilabel_soft_margin_loss(x, label)

        cams = F.conv2d(feature, self.classifier.weight)
        cams = F.relu(cams)
        cams = cams / (F.adaptive_max_pool2d(cams, (1, 1)) + 1e-5)
        cams_feature = cams.unsqueeze(2) * feature.unsqueeze(1)  # bs*20*2048*32*32
        cams_feature = cams_feature.view(cams_feature.size(0), cams_feature.size(1), cams_feature.size(2), -1)
        cams_feature = torch.mean(cams_feature, -1)

        batch_size = cams_feature.shape[0]
        x2 = cams_feature.reshape(batch_size, self.n_classes, -1)  # bs*20*2048
        mask = label > 0  # bs*20

        feature_list = [x2[i][mask[i]] for i in range(batch_size)]  # bs*n*2048
        prediction = [self.classifier2(y.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1) for y in feature_list]
        labels = [torch.nonzero(label[i]).squeeze(1) for i in range(label.shape[0])]

        loss_ce = 0
        acc = 0
        num = 0
        for logit, label in zip(prediction, labels):
            if label.shape[0] == 0:
                continue
            loss = F.cross_entropy(logit, label)
            loss_ce += loss
            acc += (logit.argmax(dim=1) == label.view(-1)).sum().float()
            num += label.size(0)

        loss_ce = loss_ce / batch_size

        return loss_cls, loss_ce, acc / num


if __name__ == '__main__':
    import os

    dcam = DCAM_SCE()
    dcam.load_state_dict(torch.load(os.path.join('../cam_weight', 'checkpoint.pth'))['model'])
    print('-----')
