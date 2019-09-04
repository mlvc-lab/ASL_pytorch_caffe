'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.asl_module import ActiveShiftLayer
#from asl_module import ActiveShiftLayer

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, groups=2, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.asl = ActiveShiftLayer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=stride, padding=0, groups=3, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(self.asl(F.relu(self.bn2(out))))
        out += self.shortcut(x)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, base_width, num_classes=10):
        super(ResNet, self).__init__()
        self.base_width = base_width
        w = self.base_width

        self.conv1 = nn.Conv2d(3, w, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(block, w, w, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, w, w * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, w * 2, w * 2**2, num_blocks[2], stride=2)

        self.last_bn = nn.BatchNorm2d(w * 2**2)
        self.linear = nn.Linear(w * 2**2, num_classes)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_planes, out_planes, num_blocks, stride):

        layers = []
        layers.append(block(in_planes, out_planes, stride=stride))
        for i in range(num_blocks-1):
            layers.append(block(out_planes, out_planes, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.last_bn(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def hgc_asl_resnet18(base_width, num_classes=10):
    return ResNet(BasicBlock, [3,3,3], base_width, num_classes=num_classes)

#def ResNet34():
#    return ResNet(BasicBlock, [3,4,6,3])
#
#def ResNet50():
#    return ResNet(Bottleneck, [3,4,6,3])
#
#def ResNet101():
#    return ResNet(Bottleneck, [3,4,23,3])
#
#def ResNet152():
#    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = asl_resnet18().cuda()
    y = net(torch.randn(128,3,32,32).cuda())
    import pdb
    pdb.set_trace()
    print(y.size())

if __name__=='__main__':
    test()
