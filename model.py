import timm
import torch
import torch.nn as nn
from ultralytics.nn.modules import C2f, SPPF
from ultralytics.nn.modules import OBB

class SwinBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=True,
        features_only=True,
        out_indices=(1, 2, 3)
        )
        self.adapters = nn.ModuleList([
        nn.Conv2d(192, 256, 1),
        nn.Conv2d(384, 512, 1),
        nn.Conv2d(768, 1024, 1),
        ])


    def forward(self, x):
        feats = self.backbone(x)
        return [a(f) for a, f in zip(self.adapters, feats)]




class YOLONeck(nn.Module):
    def __init__(self, ch=(256, 512, 1024)):
        super().__init__()
        c3, c4, c5 = ch


        self.sppf = SPPF(c5, c5)
        self.reduce_p5 = nn.Conv2d(c5, c4, 1)
        self.c2f_p4 = C2f(c4 * 2, c4)


        self.reduce_p4 = nn.Conv2d(c4, c3, 1)
        self.c2f_p3 = C2f(c3 * 2, c3)


        self.down_p3 = nn.Conv2d(c3, c3, 3, 2, 1)
        self.c2f_n4 = C2f(c3 + c4, c4)


        self.down_p4 = nn.Conv2d(c4, c4, 3, 2, 1)
        self.c2f_n5 = C2f(c4 + c5, c5)


    def forward(self, x):
        p3, p4, p5 = x


        p5 = self.sppf(p5)
        p4 = self.c2f_p4(
        torch.cat([nn.functional.interpolate(self.reduce_p5(p5), scale_factor=2), p4], 1)
        )
        p3 = self.c2f_p3(
        torch.cat([nn.functional.interpolate(self.reduce_p4(p4), scale_factor=2), p3], 1)
        )


        n4 = self.c2f_n4(torch.cat([self.down_p3(p3), p4], 1))
        n5 = self.c2f_n5(torch.cat([self.down_p4(n4), p5], 1))


        return [p3, n4, n5]




class YOLO_Swin_OBB(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.backbone = SwinBackbone()
        self.neck = YOLONeck()
        self.head = OBB(nc=num_classes, ch=[256, 512, 1024])


    def forward(self, x):
        return self.head(self.neck(self.backbone(x)))