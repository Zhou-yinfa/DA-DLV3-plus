# DA-DLV3+: 基于可变形卷积与注意力机制增强的DeepLabV3+遥感地物分割

> **Deformable Attention-enhanced DeepLabV3+ for Remote Sensing Land Cover Segmentation**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-ee4c2c.svg)](https://pytorch.org/)

---

## 📌 方法简介

本项目在标准DeepLabV3+基础上提出 **DA-DLV3+**，核心改进包括：

1. **可变形卷积 (DCNv2)**：在ASPP融合阶段和解码器中使用可变形卷积替代标准卷积，使网络能够自适应遥感图像中不规则形状的地物（弯曲河流、不规则建筑等）

2. **多级注意力增强**：
   - **浅层CA**：坐标注意力注入浅层特征，增强位置感知能力
   - **ASPP后注意力**：DAPSA（方向感知极化自注意力）或 CBAM，增强多尺度语义特征的判别力

```
Input → Backbone → ASPP/DeformASPP → [DAPSA/CBAM] → Upsample
  ↓                                         ↑
Low-level → shortcut_conv → [浅层CA] --------┘→ Cat → Decoder → Output
```

---

## 🗂️ 数据集准备

采用 **VOC格式** 数据集。

```
VOCdevkit/
└── VOC2007/
    ├── JPEGImages/         # 输入图像 (.jpg)
    ├── SegmentationClass/  # 标签图像 (.png, 像素值=类别ID)
    ├── ImageSets/
    │   └── Segmentation/
    │       ├── train.txt   # 训练集文件列表
    │       └── val.txt     # 验证集文件列表
    └── SegmentationClassRaw/  # 原始标签（用于生成JSON）
```

1. 将图片放入 `JPEGImages/`，标签放入 `SegmentationClass/`
2. 修改 `voc_annotation.py` 中的类别名称后运行，生成 `train.txt` 和 `val.txt`
3. 修改 `train.py` 中的 `num_classes` 为**类别数 + 1**（含背景）

---

## ⚙️ 环境安装

```bash
# 创建虚拟环境（推荐）
conda create -n da-dlv3 python=3.8 -y
conda activate da-dlv3

# 安装依赖
pip install -r requirements.txt
```

**依赖项**：`torch >= 1.7.0`, `torchvision`, `numpy`, `opencv-python`, `tqdm`, `pillow`, `tensorboard`

---

## 🚀 训练 & 测试

### 训练

在 `train.py` 中配置注意力模块：

```python
# DA-DLV3+ 注意力配置
use_dcn   = True     # 可变形卷积 (DeformASPP + DeformDecoder)
use_dapsa = False    # DAPSA 方向感知极化自注意力
use_cbam  = False    # CBAM 通道+空间注意力
use_ca    = True     # 浅层CA坐标注意力

# 损失函数
focal_loss = True    # Focal Loss
dice_loss  = True    # Dice Loss
```

**推荐配置**：

| 配置 | 说明 |
|------|------|
| `use_dcn=True, use_ca=True` | 轻量版，DCN + 浅层CA |
| `use_dcn=True, use_dapsa=True, use_ca=True` | 完整DA-DLV3+ |

```bash
python train.py
```

训练日志和权重保存在 `logs/` 目录下。

### 预测

修改 `deeplab.py` 中的 `_defaults`，设置 `model_path`、`backbone`、`num_classes` 与训练一致：

```python
_defaults = {
    "model_path"  : "logs/best_epoch_weights.pth",
    "num_classes" : 6,           # 类别数 + 1
    "backbone"    : "mobilenet",
    ...
}
```

```bash
python predict.py
```

### mIoU 评估

```bash
python get_miou.py
```

---

## 📁 仓库结构

```
DA-DLV3+
├── nets/
│   ├── deeplabv3_plus.py      # DA-DLV3+ 完整模型（DCN, CBAM, CA, DAPSA）
│   ├── dapsa_attention.py     # DAPSA 方向感知极化自注意力
│   ├── deeplabv3_training.py  # 损失函数（CE, Focal, Dice）
│   ├── mobilenetv2.py         # MobileNetV2 主干网络
│   └── xception.py            # Xception 主干网络
├── utils/
│   ├── utils_fit.py           # 训练循环
│   ├── dataloader.py          # 数据加载
│   ├── voc_config.py          # 数据集配置
│   └── ...
├── train.py                   # 训练入口（注意力开关配置）
├── predict.py                 # 预测/可视化
├── deeplab.py                 # 模型封装（预测接口）
├── get_miou.py                # mIoU 评估
├── voc_annotation.py          # 数据集标注生成
├── summary.py                 # 模型参数量统计
├── requirements.txt           # 依赖列表
└── img/                       # 示例图片
```

---

## 🙏 致谢

本项目基于以下开源工作：

- [DeepLabV3+ (bubbliiiing)](https://github.com/bubbliiiing/deeplabv3-plus-pytorch) — PyTorch实现的基础框架
- [DCNv2](https://arxiv.org/abs/1811.11168) — Deformable ConvNets v2 (Zhu et al., CVPR 2019)
- [CBAM](https://arxiv.org/abs/1807.06521) — Convolutional Block Attention Module (Woo et al., ECCV 2018)
- [Coordinate Attention](https://arxiv.org/abs/2103.02907) — Hou et al., CVPR 2021
- [Polarized Self-Attention](https://arxiv.org/abs/2107.00782) — Liu et al., CVPR 2021

---

## 📄 License

本项目基于 [Apache 2.0](LICENSE) 协议开源。
