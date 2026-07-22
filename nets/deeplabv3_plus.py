"""
DA-DLV3+: 基于可变形卷积与注意力机制增强的DeepLabV3+
==========================================================
论文架构说明：
  本项目在标准DeepLabV3+基础上进行两项核心改进：

  1. 可变形卷积 (DCN, Deformable Convolution)
     - 替换ASPP融合阶段的标准卷积为可变形卷积(DeformASPP)
     - 替换解码器中的标准卷积为可变形卷积
     - 使网络能适应遥感图像中不规则形状的地物（弯曲河流、不规则建筑等）

  2. 多级注意力机制增强
     - 浅层CA (Coordinate Attention): 增强浅层特征的位置感知能力
     - ASPP后注意力（二选一）:
       * DAPSA (Direction-Aware Polarized Self-Attention): 方向感知极化自注意力（推荐）
       * CBAM (Convolutional Block Attention Module): 通道+空间串行注意力
         CBAM = Channel Attention → Spatial Attention

  注意力位置示意：
    Input → Backbone → ASPP → [DAPSA/CBAM ASPP后] → Upsample
      ↓                                      ↑
    Low-level → shortcut → [浅层CA] ---------┘→ Cat → Decoder → Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d as TorchDeformConv2d
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2
from nets.dapsa_attention import DAPSA


# ============================================================
#  可变形卷积模块 (DCNv2)
#  论文: Deformable ConvNets v2 (CVPR 2019)
# ============================================================
class DeformConv2d(nn.Module):
    """DCNv2 可变形卷积（带调制机制）

    通过学习采样偏移量(offset)和调制因子(mask)，使卷积核能自适应不规则形状。
    - offset: 每个采样点的2D偏移 (x, y)
    - mask: 每个采样点的重要性权重 (0~1)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2 * kernel_size * kernel_size, kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.modulator_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1 * kernel_size * kernel_size, kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.conv = TorchDeformConv2d(in_channels, out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, bias=bias)
        self._init_weights()

    def _init_weights(self):
        for name in ['offset_conv', 'modulator_conv']:
            layers = [m for m in getattr(self, name).modules() if isinstance(m, nn.Conv2d)]
            if len(layers) >= 1:
                nn.init.kaiming_normal_(layers[0].weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layers[0].bias, 0)
            if len(layers) >= 2:
                nn.init.constant_(layers[1].weight, 0)
                nn.init.constant_(layers[1].bias, 0)

    def forward(self, x):
        offset = self.offset_conv(x)
        modulator = torch.sigmoid(self.modulator_conv(x))
        return self.conv(x, offset, modulator)


class DeformConvBlock(nn.Module):
    """可变形卷积块: DeformConv → BN → ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size,
                                        stride, padding, dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.deform_conv(x)))


# ============================================================
#  CBAM: 卷积块注意力模块
#  论文: CBAM: Convolutional Block Attention Module (ECCV 2018)
#  结构: Channel Attention → Spatial Attention（串行）
# ============================================================
class ChannelAttention(nn.Module):
    """通道注意力: GAP + GMP → 共享MLP → 相加 → Sigmoid"""
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    """空间注意力: 通道维mean/max → concat → 7x7conv → Sigmoid"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = torch.cat([torch.mean(x, dim=1, keepdim=True),
                         torch.max(x, dim=1, keepdim=True)[0]], dim=1)
        return self.sigmoid(self.conv(out))


class CBAM(nn.Module):
    """CBAM: 先通道注意力，后空间注意力"""
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x) * x) * x


# ============================================================
#  CA: 坐标注意力 (Coordinate Attention)
#  论文: Coordinate Attention for Efficient Mobile Network Design (CVPR 2021)
#  核心: 将位置信息嵌入通道注意力，沿H/W方向分别池化
#  在本架构中仅用于浅层特征分支
# ============================================================
class CoordAtt(nn.Module):
    """坐标注意力 - 轻量且具位置感知能力

    沿H和W两个方向分别进行全局池化 → 共享卷积压缩 → 分别生成方向注意力权重
    优势: 比SE多位置信息，比CBAM更轻量
    """
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_w = nn.AdaptiveAvgPool2d((None, 1))   # 保留H
        self.pool_h = nn.AdaptiveAvgPool2d((1, None))    # 保留W
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, 1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, oup, 1)  # H方向注意力
        self.conv_w = nn.Conv2d(mip, oup, 1)  # W方向注意力

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_w(x)                          # (B,C,H,1)
        x_w = self.pool_h(x).permute(0, 1, 3, 2)      # (B,C,W,1)
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        return identity * torch.sigmoid(self.conv_h(x_h)) * torch.sigmoid(self.conv_w(x_w))


# ============================================================
#  主干网络封装
# ============================================================
class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super().__init__()
        from functools import partial
        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]
        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]
        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))

    @staticmethod
    def _nostride_dilate(m, dilate):
        if m.__class__.__name__.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            elif m.kernel_size == (3, 3):
                m.dilation = (dilate, dilate)
                m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x


# ============================================================
#  ASPP: 空间金字塔池化模块（标准版）
# ============================================================
class ASPP(nn.Module):
    """标准ASPP: 5分支（1x1conv + 三个不同膨胀率的3x3conv + 全局池化）"""
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, 0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom), nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, 6*rate, dilation=6*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom), nn.ReLU(inplace=True))
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, 12*rate, dilation=12*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom), nn.ReLU(inplace=True))
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, 18*rate, dilation=18*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom), nn.ReLU(inplace=True))
        self.branch5 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom), nn.ReLU(inplace=True))
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out*5, dim_out, 1, 1, 0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom), nn.ReLU(inplace=True))

    def forward(self, x):
        _, _, row, col = x.size()
        global_feat = F.interpolate(self.branch5(F.adaptive_avg_pool2d(x, 1)), (row, col),
                                     mode='bilinear', align_corners=True)
        return self.conv_cat(torch.cat([self.branch1(x), self.branch2(x),
                                        self.branch3(x), self.branch4(x), global_feat], dim=1))


# ============================================================
#  DeformASPP: 可变形ASPP（DA-DLV3+ 核心改进之一）
#  在ASPP融合阶段使用双路径：标准卷积 + 可变形卷积
# ============================================================
class DeformASPP(nn.Module):
    """可变形ASPP - DCN增强版

    与标准ASPP的区别：特征融合阶段使用双路径
    - 路径1: 标准1x1卷积（稳定基准）
    - 路径2: 可变形3x3卷积（适应不规则形状）
    - 加权融合（可学习参数）
    """
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super().__init__()
        # 5分支（同标准ASPP）
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, 0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom), nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, 6*rate, dilation=6*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom), nn.ReLU(inplace=True))
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, 12*rate, dilation=12*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom), nn.ReLU(inplace=True))
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, 18*rate, dilation=18*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom), nn.ReLU(inplace=True))
        self.branch5 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom), nn.ReLU(inplace=True))
        # 双路径融合
        self.conv_cat_standard = nn.Sequential(
            nn.Conv2d(dim_out*5, dim_out, 1, 1, 0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom), nn.ReLU(inplace=True))
        self.conv_cat_deform = DeformConvBlock(dim_out*5, dim_out, kernel_size=3, padding=1)
        self.fusion_weight = nn.Parameter(torch.ones(2) / 2)

    def forward(self, x):
        _, _, row, col = x.size()
        global_feat = F.interpolate(self.branch5(F.adaptive_avg_pool2d(x, 1)), (row, col),
                                     mode='bilinear', align_corners=True)
        feat_cat = torch.cat([self.branch1(x), self.branch2(x),
                              self.branch3(x), self.branch4(x), global_feat], dim=1)
        weights = F.softmax(self.fusion_weight, dim=0)
        return weights[0] * self.conv_cat_standard(feat_cat) + weights[1] * self.conv_cat_deform(feat_cat)


# ============================================================
#  DA-DLV3+: 完整模型
# ============================================================
class DeepLab(nn.Module):
    """DA-DLV3+: 基于可变形卷积与注意力增强的DeepLabV3+

    架构组成:
    ┌─────────────────────────────────────────────────────┐
    │  Input Image                                        │
    │    ↓                                                │
    │  Backbone (MobileNetV2 / Xception)                   │
    │    ├── low_level_features (高分辨率, 1/4 or 1/8)     │
    │    └── x (低分辨率, 1/16)                            │
    │    ↓                                                │
    │  ASPP / DeformASPP (多尺度上下文提取)                 │
    │    ↓                                                │
    │  [ASPP后注意力] ← 二选一: DAPSA / CBAM               │
    │    ↓                                                │
    │  Upsample + Concat                                  │
    │    ↑ 浅层shortcut → shortcut_conv → [浅层CA(可选)]   │
    │    ↓                                                │
    │  Decoder (标准/可变形卷积)                            │
    │    ↓                                                │
    │  Classification Head → Output                        │
    └─────────────────────────────────────────────────────┘

    Args:
        num_classes: 分类数（含背景）
        backbone: 主干网络 ('mobilenet' | 'xception')
        pretrained: 是否使用预训练主干
        downsample_factor: 下采样倍数 (8 | 16)
        use_dcn: 是否使用可变形卷积 (DCN/DeformASPP/DeformDecoder)
        use_dapsa: ASPP后使用DAPSA方向感知极化自注意力（推荐）
        use_cbam: ASPP后使用CBAM注意力
        use_ca: 浅层特征后使用CA坐标注意力，增强位置感知能力
    """

    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16,
                 use_dcn=False, use_dapsa=False, use_cbam=False, use_ca=False):
        super().__init__()

        # ---- 主干网络 ----
        if backbone == "xception":
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels, low_level_channels = 2048, 256
        elif backbone == "mobilenet":
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels, low_level_channels = 320, 24
        else:
            raise ValueError(f'Unsupported backbone: `{backbone}`')

        # ---- ASPP模块 ----
        self.aspp = (DeformASPP if use_dcn else ASPP)(
            dim_in=in_channels, dim_out=256, rate=16 // downsample_factor)

        # ---- 浅层特征处理 ----
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1), nn.BatchNorm2d(48), nn.ReLU(inplace=True))

        # ---- 注意力模块配置 ----
        # 浅层CA（独立开关，增强位置感知）
        self.low_level_attention = CoordAtt(48, 48) if use_ca else None

        # ASPP后注意力（二选一: DAPSA / CBAM）
        if use_dapsa:
            self.aspp_attention = DAPSA(256, reduction=16)
        elif use_cbam:
            self.aspp_attention = CBAM(256)
        else:
            self.aspp_attention = None

        # ---- 解码头 ----
        decoder_in = 48 + 256
        if use_dcn:
            self.cat_conv = nn.Sequential(
                DeformConvBlock(decoder_in, 256, 3, 1, 1), nn.Dropout(0.5),
                DeformConvBlock(256, 256, 3, 1, 1), nn.Dropout(0.1))
        else:
            self.cat_conv = nn.Sequential(
                nn.Conv2d(decoder_in, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.Dropout(0.1))

        # ---- 分类头 ----
        self.cls_conv = nn.Conv2d(256, num_classes, 1, 1)

        # 打印配置摘要
        attn_name = {DAPSA: 'DAPSA', CBAM: 'CBAM'}
        aspp_attn = attn_name.get(type(self.aspp_attention), 'None') if self.aspp_attention else 'None'
        print(f"\n{'='*50}")
        print(f"  DA-DLV3+ Model Config")
        print(f"{'='*50}")
        print(f"  Backbone   : {backbone}")
        print(f"  Classes    : {num_classes}")
        print(f"  DCN        : {'DeformASPP+DeformDecoder' if use_dcn else 'Standard'}")
        print(f"  ASPP_Attn  : {aspp_attn}")
        print(f"  Shallow_CA : {'ON' if use_ca else 'OFF'}")
        print(f"{'='*50}\n")

    def forward(self, x):
        H, W = x.size(2), x.size(3)

        # 1. 主干网络提取特征
        low_level_features, x = self.backbone(x)

        # 2. ASPP多尺度特征提取
        x = self.aspp(x)

        # 3. ASPP后注意力增强
        if self.aspp_attention is not None:
            x = self.aspp_attention(x)

        # 4. 浅层特征压缩 + 浅层CA（增强位置感知）
        low_level_features = self.shortcut_conv(low_level_features)
        if self.low_level_attention is not None:
            low_level_features = self.low_level_attention(low_level_features)

        # 5. 特征融合 + 解码
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)),
                          mode='bilinear', align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))

        # 6. 分类 + 上采样
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x
