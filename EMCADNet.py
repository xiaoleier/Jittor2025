# Jittor version of EMCADNet (backbone + decoder + out_head)
import jittor as jt
import jittor.nn as nn
from emcad_decoder import EMCAD  # Assume previous decoder code is saved in emcad_decoder.py
from jittor.models import resnet18, resnet34, resnet50, resnet101

class ModifiedResNet(nn.Module):
    def __init__(self, resnet_model):
        super().__init__()
        self.conv1 = resnet_model.conv1
        self.bn1 = resnet_model.bn1
        self.relu = resnet_model.relu
        self.maxpool = resnet_model.maxpool
        self.layer1 = resnet_model.layer1
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4

    def execute(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4

class EMCADNet(nn.Module):
    def __init__(self, num_classes=2, encoder='resnet18', pretrained=True,
                 kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True,
                 add=True, lgag_ks=3, activation='relu'):
        super().__init__()

        # Gray to RGB conv block
        self.conv1 = nn.Sequential(
            nn.Conv(1, 3, 1),
            nn.BatchNorm(3),
            nn.ReLU()
        )

        # Backbone setup
        if encoder == 'resnet18':
            backbone = resnet18(pretrained=pretrained)
            channels = [512, 256, 128, 64]
        elif encoder == 'resnet34':
            backbone = resnet34(pretrained=pretrained)
            channels = [512, 256, 128, 64]
        elif encoder == 'resnet50':
            backbone = resnet50(pretrained=pretrained)
            channels = [2048, 1024, 512, 256]
        elif encoder == 'resnet101':
            backbone = resnet101(pretrained=pretrained)
            channels = [2048, 1024, 512, 256]
        else:
            raise NotImplementedError("Unsupported encoder")

        self.backbone = ModifiedResNet(backbone)

        # Decoder
        self.decoder = EMCAD(
            channels=channels,
            kernel_sizes=kernel_sizes,
            expansion_factor=expansion_factor,
            dw_parallel=dw_parallel,
            add=add,
            lgag_ks=lgag_ks,
            activation=activation
        )

        # Output heads
        self.head4 = nn.Conv(channels[0], num_classes, 1)
        self.head3 = nn.Conv(channels[1], num_classes, 1)
        self.head2 = nn.Conv(channels[2], num_classes, 1)
        self.head1 = nn.Conv(channels[3], num_classes, 1)

    def execute(self, x):
        if x.shape[1] == 1:
            x = self.conv1(x)

        # Backbone forward
        x1, x2, x3, x4 = self.backbone(x)

        # Decoder forward
        d4, d3, d2, d1 = self.decoder(x4, [x3, x2, x1])

        # Prediction
        p4 = nn.interpolate(self.head4(d4), scale_factor=32, mode='bilinear')
        p3 = nn.interpolate(self.head3(d3), scale_factor=16, mode='bilinear')
        p2 = nn.interpolate(self.head2(d2), scale_factor=8, mode='bilinear')
        p1 = nn.interpolate(self.head1(d1), scale_factor=4, mode='bilinear')

        return [p4, p3, p2, p1]



if __name__ == '__main__':
    model = EMCADNet(encoder='resnet101')
    x = jt.randn(1, 1, 352, 352)
    out = model(x)
    for i, o in enumerate(out):
        print(f"Output p{i+1} shape: ", o.shape)
