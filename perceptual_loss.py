import torch
import torch.nn as nn
from torchvision import models


"""
VGG(features): Sequential(
1->(0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
2->(2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
3->(5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
4->(7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
5->(10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
6->(12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
7->(14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))d
8->(16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
9->(19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
10->(21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
11->(23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
12->(25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
13->(28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
14->(30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
15->(32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
16->(34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
(classifier): Sequential(
17->(0): Linear(in_features=25088, out_features=4096, bias=True)
18->(3): Linear(in_features=4096, out_features=4096, bias=True)
19->(6): Linear(in_features=4096, out_features=1000, bias=True)))
"""


class Vgg19(nn.Module):
    def __init__(self, layer_no=30, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg = models.vgg19(pretrained=True)
        # vgg.load_state_dict(torch.load(os.path.dirname(os.path.realpath(__file__)) + "/vgg19-dcbb9e9d.pth"))
        vgg_pretrained_features = vgg.features
        self.vgg = vgg
        self.slice_1 = torch.nn.Sequential()

        # Selecting features from 14 layer, without activation (ESRGAN Paper, AutoRetouch Paper)
        for x in range(30):
            self.slice_1.add_module(str(x), vgg_pretrained_features[x])
        self.slice_1.add_module(str(30), vgg_pretrained_features[30])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        out = self.slice_1(X)
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19(layer_no=30)
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = self.criterion(x_vgg, y_vgg)
        return loss
