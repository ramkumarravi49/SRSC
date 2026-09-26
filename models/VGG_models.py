

# VGG_models.py
import random
from models.layers import *
import torch
import torch.nn as nn

# ----------------------------------------------------------------------
# VGGSNN and VGGSNNwoAP (QCFS-style SNN wrappers)
# ----------------------------------------------------------------------

class VGGSNN(nn.Module):
    def __init__(self):
        super(VGGSNN, self).__init__()
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        # pool = APLayer(2)
        self.features = nn.Sequential(
            Layer(2, 64, 3, 1, 1),
            Layer(64, 128, 3, 1, 1),
            pool,
            Layer(128, 256, 3, 1, 1),
            Layer(256, 256, 3, 1, 1),
            pool,
            Layer(256, 512, 3, 1, 1),
            Layer(512, 512, 3, 1, 1),
            pool,
            Layer(512, 512, 3, 1, 1),
            Layer(512, 512, 3, 1, 1),
            pool,
        )
        # W was computed from 48 input: 48/2/2/2/2 -> 3
        W = int(48 / 2 / 2 / 2 / 2)
        # self.T = 4
        self.classifier = SeqToANNContainer(nn.Linear(512 * W * W, 10))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x


class VGGSNNwoAP(nn.Module):
    def __init__(self):
        super(VGGSNNwoAP, self).__init__()
        self.features = nn.Sequential(
            Layer(2, 64, 3, 1, 1),
            Layer(64, 128, 3, 2, 1),
            Layer(128, 256, 3, 1, 1),
            Layer(256, 256, 3, 2, 1),
            Layer(256, 512, 3, 1, 1),
            Layer(512, 512, 3, 2, 1),
            Layer(512, 512, 3, 1, 1),
            Layer(512, 512, 3, 2, 1),
        )
        W = int(48 / 2 / 2 / 2 / 2)
        # self.T = 4
        self.classifier = SeqToANNContainer(nn.Linear(512 * W * W, 10))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x


# ----------------------------------------------------------------------
# VGG utilities & canonical SNN VGGBlocks
# ----------------------------------------------------------------------

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super(VGGBlock, self).__init__()
        layers = []
        for i in range(num_convs):
            conv_layer = nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            )
            bn_layer = tdBatchNorm(out_channels)
            layers.append(tdLayer(conv_layer, bn_layer))
            layers.append(LIFSpike())

        # NOTE: .pyc produced a plain list here; that was a bug.
        # Use Sequential so forward(x) works as expected:
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ----------------------------------------------------------------------
# vgg11 (SNN-styled) - preserved from your human / GitHub file
# ----------------------------------------------------------------------
class vgg11(nn.Module):
    def __init__(self, num_classes=10):
        super(vgg11, self).__init__()
        self.T = 1

        self.conv1 = nn.Conv2d(
            3, 128, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = tdBatchNorm(128)
        self.spike = LIFSpike()
        self.features = nn.Sequential(
            tdLayer(self.conv1, self.bn1),
            LIFSpike(),

            VGGBlock(128, 128, 1),
            tdLayer(nn.MaxPool2d(kernel_size=2, stride=2)),

            VGGBlock(128, 256, 3),
            tdLayer(nn.MaxPool2d(kernel_size=2, stride=2)),

            VGGBlock(256, 512, 3),
            tdLayer(nn.MaxPool2d(kernel_size=2, stride=2)),
        )

        self.classifier = nn.Sequential(
            tdLayer(nn.Linear(512 * 4 * 4, 2048)),
            LIFSpike(),
            tdLayer(nn.Linear(2048, 2048)),
            LIFSpike(),
            tdLayer(nn.Linear(2048, num_classes))
        )

    def _forward_impl(self, x):
        x = self.features(x)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x

    def forward(self, x):
        x = add_dimention(x, self.T)  # Add time dimension: [T, B, C, H, W]
        return self._forward_impl(x)






# TET-compatible VGG16 that is architecturally identical to QCFS VGG16   - VGG 16 NEW 2
# Paste into models/VGG_models.py (replace or add alongside existing classes)

import torch
import torch.nn as nn
from models.layers import tdLayer, tdBatchNorm, SeqToANNContainer, LIFSpike, add_dimention

class VGG16_QCFS_Compat(nn.Module):
    """
    TET-style VGG16 that mirrors QCFS VGG16 architecture exactly
    (block layout and classifier sizing). Uses tdLayer + tdBatchNorm + LIFSpike.
    """
    def __init__(self, num_classes=10, dropout=0.0):
        super(VGG16_QCFS_Compat, self).__init__()
        self.T = 1  # default temporal length for TET forward; caller may override
        self.dropout = dropout

        # Feature blocks (matches QCFS cfg for VGG16)
        # Block1: 64,64, M
        self.block1 = nn.Sequential(
            tdLayer(nn.Conv2d(3, 64, kernel_size=3, padding=1), tdBatchNorm(64)),
            LIFSpike(),
            tdLayer(nn.Conv2d(64, 64, kernel_size=3, padding=1), tdBatchNorm(64)),
            LIFSpike(),
            tdLayer(nn.MaxPool2d(kernel_size=2, stride=2))
        )

        # Block2: 128,128, M
        self.block2 = nn.Sequential(
            tdLayer(nn.Conv2d(64, 128, kernel_size=3, padding=1), tdBatchNorm(128)),
            LIFSpike(),
            tdLayer(nn.Conv2d(128, 128, kernel_size=3, padding=1), tdBatchNorm(128)),
            LIFSpike(),
            tdLayer(nn.MaxPool2d(kernel_size=2, stride=2))
        )

        # Block3: 256,256,256, M
        self.block3 = nn.Sequential(
            tdLayer(nn.Conv2d(128, 256, kernel_size=3, padding=1), tdBatchNorm(256)),
            LIFSpike(),
            tdLayer(nn.Conv2d(256, 256, kernel_size=3, padding=1), tdBatchNorm(256)),
            LIFSpike(),
            tdLayer(nn.Conv2d(256, 256, kernel_size=3, padding=1), tdBatchNorm(256)),
            LIFSpike(),
            tdLayer(nn.MaxPool2d(kernel_size=2, stride=2))
        )

        # Block4: 512,512,512, M
        self.block4 = nn.Sequential(
            tdLayer(nn.Conv2d(256, 512, kernel_size=3, padding=1), tdBatchNorm(512)),
            LIFSpike(),
            tdLayer(nn.Conv2d(512, 512, kernel_size=3, padding=1), tdBatchNorm(512)),
            LIFSpike(),
            tdLayer(nn.Conv2d(512, 512, kernel_size=3, padding=1), tdBatchNorm(512)),
            LIFSpike(),
            tdLayer(nn.MaxPool2d(kernel_size=2, stride=2))
        )

        # Block5: 512,512,512, M
        self.block5 = nn.Sequential(
            tdLayer(nn.Conv2d(512, 512, kernel_size=3, padding=1), tdBatchNorm(512)),
            LIFSpike(),
            tdLayer(nn.Conv2d(512, 512, kernel_size=3, padding=1), tdBatchNorm(512)),
            LIFSpike(),
            tdLayer(nn.Conv2d(512, 512, kernel_size=3, padding=1), tdBatchNorm(512)),
            LIFSpike(),
            tdLayer(nn.MaxPool2d(kernel_size=2, stride=2))
        )

        # Adaptive pooling to produce 1x1 feature map for CIFAR-like inputs
        self.avgpool = tdLayer(nn.AdaptiveAvgPool2d((1, 1)))

        # Classifier (matches QCFS classifier layout)
        # For CIFAR (num_classes != 1000) QCFS used Linear(512 -> 4096) ...
        self.classifier = nn.Sequential(
            tdLayer(nn.Linear(512, 4096)),
            LIFSpike(),
            nn.Dropout(p=self.dropout),
            tdLayer(nn.Linear(4096, 4096)),
            LIFSpike(),
            nn.Dropout(p=self.dropout),
            tdLayer(nn.Linear(4096, num_classes))
        )

        # weight init for convs kept consistent with QCFS style
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # BatchNorm and Linear biases follow QCFS defaults (they already are reasonable)

    def _forward_impl(self, x):
        # x is expected: [B, C, H, W] -> TET will add time dim below
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        # pool -> flatten spatial dims keeping time & batch
        x = self.avgpool(x)             # tdLayer wrapper handles time flattening internally
        x = torch.flatten(x, 2)         # keep [T, B, C]
        x = self.classifier(x)          # classifier is tdLayer-wrapped, so accepts time-first input
        return x

    def forward(self, x):
        # TET convention: add temporal dimension when T > 0
        if getattr(self, "T", 0) > 0:
            x = add_dimention(x, self.T)   # produces [T, B, C, H, W] compatible with tdLayer
        return self._forward_impl(x)


# factory
def vgg16_qcfs_compat(num_classes=10, dropout=0.0):
    return VGG16_QCFS_Compat(num_classes=num_classes, dropout=dropout)

# alias to keep your existing calls working; replace existing vgg16() if desired
def vgg16(num_classes=10, dropout=0.0):
    return vgg16_qcfs_compat(num_classes=num_classes, dropout=dropout)


# ----------------------------------------------------------------------
# Small test / debug harness (kept from your original)
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # Quick smoke test that constructs a model and runs a forward
    # NOTE: this is not a unit test, just a quick shape check.
    model = VGGSNNwoAP()
    # model = resnet20(num_classes=10)   # example if you want to switch
    x = torch.rand(2, 3, 32, 32)
    y = model(x)
    print("Output shape:", y.shape)













# # ----------------------------------------------------------------------
# # VGG16 (QCFS / CIFAR-style) - careful: use CIFAR classifier shape
# # ----------------------------------------------------------------------
# class VGG16(nn.Module):
#     def __init__(self, num_classes=10):
#         super(VGG16, self).__init__()
#         self.T = 1
#         # VGG16-like architecture (matching QCFS structure)
#         self.features = nn.Sequential(
#             # Block 1: [64, 64, 'M']
#             VGGBlock(3, 64, 2),
#             tdLayer(nn.MaxPool2d(kernel_size=2, stride=2)),

#             # Block 2: [128, 128, 'M']
#             VGGBlock(64, 128, 2),
#             tdLayer(nn.MaxPool2d(kernel_size=2, stride=2)),

#             # Block 3: [256, 256, 256, 'M']
#             VGGBlock(128, 256, 3),
#             tdLayer(nn.MaxPool2d(kernel_size=2, stride=2)),

#             # Block 4: [512, 512, 512, 'M']
#             VGGBlock(256, 512, 3),
#             tdLayer(nn.MaxPool2d(kernel_size=2, stride=2)),

#             # Block 5: [512, 512, 512, 'M']
#             VGGBlock(512, 512, 3),
#             tdLayer(nn.MaxPool2d(kernel_size=2, stride=2)),
#         )

#         # Compute feature map size for classifier dynamically.
#         # For CIFAR10/CIFAR100 (input 32x32) after 5 pools -> 1x1 feature map.
#         # Keep your CIFAR-style classifier (this was your chosen option D).
#         # NOTE: your .pyc had a simplified Linear(512 -> 2048), but we expand
#         # to be explicit about the flattened channel dimension.
#         self.classifier = nn.Sequential(
#             tdLayer(nn.Linear(512 * 1 * 1, 2048)),
#             LIFSpike(),
#             tdLayer(nn.Linear(2048, 2048)),
#             LIFSpike(),
#             tdLayer(nn.Linear(2048, num_classes))
#         )

        
#         # self.classifier = nn.Sequential(
#         #     tdLayer(nn.Linear(512 * 1 * 1, 4096)),
#         #     LIFSpike(),
#         #     tdLayer(nn.Linear(4096, 4096)),
#         #     LIFSpike(),
#         #     tdLayer(nn.Linear(4096, num_classes))
#         # )


#         # Initialize weights like QCFS for conv layers
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

#     def _forward_impl(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, 2)  # flatten spatial dims only (keep time/batch)
#         x = self.classifier(x)
#         return x

#     def forward(self, x):
#         x = add_dimention(x, self.T)            # [B,T,C,H,W]
#         return self._forward_impl(x)

#     # def forward(self, x):
#     #     x = add_dimention(x, self.T)              # [B,T,C,H,W]
#     #     x = x.permute(1,0,2,3,4).contiguous()     # [T,B,C,H,W]
#     #     out = self._forward_impl(x)               # [T,B,10]
#     #     return out.permute(1,0,2).contiguous()    # [B,T,10]


# def vgg16(num_classes=10):
#     return VGG16(num_classes=num_classes)



















# import random
# from models.layers import *



# class VGGSNN(nn.Module):
#     def __init__(self):
#         super(VGGSNN, self).__init__()
#         pool = SeqToANNContainer(nn.AvgPool2d(2))
#         #pool = APLayer(2)
#         self.features = nn.Sequential(
#             Layer(2,64,3,1,1),
#             Layer(64,128,3,1,1),
#             pool,
#             Layer(128,256,3,1,1),
#             Layer(256,256,3,1,1),
#             pool,
#             Layer(256,512,3,1,1),
#             Layer(512,512,3,1,1),
#             pool,
#             Layer(512,512,3,1,1),
#             Layer(512,512,3,1,1),
#             pool,
#         )
#         W = int(48/2/2/2/2)
#         # self.T = 4
#         self.classifier = SeqToANNContainer(nn.Linear(512*W*W,10))

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

#     def forward(self, input):
#         # input = add_dimention(input, self.T)
#         x = self.features(input)
#         x = torch.flatten(x, 2)
#         x = self.classifier(x)
#         return x

# class VGGSNNwoAP(nn.Module):
#     def __init__(self):
#         super(VGGSNNwoAP, self).__init__()
#         self.features = nn.Sequential(
#             Layer(2,64,3,1,1),
#             Layer(64,128,3,2,1),
#             Layer(128,256,3,1,1),
#             Layer(256,256,3,2,1),
#             Layer(256,512,3,1,1),
#             Layer(512,512,3,2,1),
#             Layer(512,512,3,1,1),
#             Layer(512,512,3,2,1),
#         )
#         W = int(48/2/2/2/2)
#         # self.T = 4
#         self.classifier = SeqToANNContainer(nn.Linear(512*W*W,10))

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

#     def forward(self, input):
#         # input = add_dimention(input, self.T)
#         x = self.features(input)
#         x = torch.flatten(x, 2)
#         x = self.classifier(x)
#         return x


# class VGGBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, num_convs):
#         super(VGGBlock, self).__init__()
#         layers = []
#         for i in range(num_convs):
#             conv_layer = nn.Conv2d(in_channels if i == 0 else out_channels,
#                                      out_channels, kernel_size=3, padding=1, bias=False)
#             bn_layer = tdBatchNorm(out_channels)
#             layers.append(tdLayer(conv_layer, bn_layer))
#             layers.append(LIFSpike())
#         self.block = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.block(x)


# class vgg11(nn.Module):
#     def __init__(self, num_classes=10):
#         super(vgg11, self).__init__()
#         self.T = 1
        
#         self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1,
#                                bias=False)
#         self.bn1 = tdBatchNorm(128)
#         self.spike = LIFSpike()
#         self.features = nn.Sequential(
#             tdLayer(self.conv1, self.bn1),
#             LIFSpike(),

#             VGGBlock(128, 128, 1),
#             tdLayer(nn.MaxPool2d(kernel_size=2, stride=2)),

#             VGGBlock(128, 256, 3),
#             tdLayer(nn.MaxPool2d(kernel_size=2, stride=2)),

#             VGGBlock(256, 512, 3),
#             tdLayer(nn.MaxPool2d(kernel_size=2, stride=2)),

#         )


#         self.classifier = nn.Sequential(
#             tdLayer(nn.Linear(512 * 4 * 4, 2048)),
#             LIFSpike(),
#             tdLayer(nn.Linear(2048, 2048)),
#             LIFSpike(),
#             tdLayer(nn.Linear(2048, num_classes))
#         )

#     def _forward_impl(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, 2)
#         x = self.classifier(x)
#         return x

#     def forward(self, x):
#         x = add_dimention(x, self.T)  # Add time dimension: [T, B, C, H, W]
#         return self._forward_impl(x)

# if __name__ == '__main__':
#     model = VGGSNNwoAP()