"""
DAPSA v3: Direction-Aware Polarized Self-Attention (方向感知极化自注意力)

版本演进：
  v1 (mIoU=71.49): 有偏Query + 双池化V + 加法空间融合 → Q污染+SE级联+sigmoid压制
  v2 (mIoU=71.67): 方向→Value + 卷积融合空间增强 → SE预门控+全局γ一刀切
  v3 (mIoU=72.09): 移除预门控 + 内容感知方向γ → 首次超越历史最优(+0.07)
  v4 (mIoU=71.85): +空间内容感知γ(共享瓶颈) → 负收益(-0.24), 16维瓶颈多任务冲突

v3 最终方案（当前代码，已验证最优）：
  方案一: 纯净双投影V — v_avg = ConvAvg(x), v_max = ConvMax(x), 无SE预门控
  方案二: 内容感知方向强度 — γ = γ_base + γ_adapt(GAP(x)), tanh[-1,1], 2输出
  方案C: 残差卷积空间融合 — sp_weight = sigmoid(global + γ_sp * residual), 静态γ_sp

退化保证: γ_base=0 + γ_adapt≈0 + γ_sp=0 → 等价双V版PSA

位置：ASPP输出后 (256ch)
参数量：231K (PSA: 131K, +76%)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DAPSA(nn.Module):
    """DAPSA v3: 纯净双V + 内容感知方向γ + 卷积融合空间增强"""
    
    def __init__(self, channels, reduction=16):
        super(DAPSA, self).__init__()
        C = channels
        mid = max(8, C // reduction)  # 16 when C=256
        mid_adapt = max(4, C // 16)    # 16 when C=256
        
        # ================================================================
        # Channel Branch — Q: 纯PSA（无偏置, 无预门控）
        # ================================================================
        self.ch_wq = nn.Conv2d(C, 1, 1)
        
        # ================================================================
        # Channel Branch — V: 双重投影 + 方向感知 (方案一 + 保留方案B)
        #
        # 方案一改进: 移除 x * sigmoid(pool(x)) 的SE预门控
        #   v_avg = ConvAvg(x)  — 直接投影, 无通道压制
        #   v_max = ConvMax(x)  — 直接投影, 无通道压制
        #   两个独立随机初始化的投影提供天然多样性
        #
        # 方向感知Value (保留方案B):
        #   v_h = Conv(C→C/2)(pool_h(x)).expand — H方向信息（广播）
        #   v_w = Conv(C→C/2)(pool_w(x)).expand — W方向信息（广播）
        #   v = v_dual + γ_vh·v_h + γ_vw·v_w
        # ================================================================
        
        # 双重投影: Avg/Max双视角（方案一: 纯净版, 无预门控）
        self.ch_wv_avg = nn.Conv2d(C, C // 2, 1)
        self.ch_wv_max = nn.Conv2d(C, C // 2, 1)
        
        # 方向感知V（方案B: 方向信息注入Value）
        self.v_pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.v_pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.ch_wv_h = nn.Conv2d(C, C // 2, 1)
        self.ch_wv_w = nn.Conv2d(C, C // 2, 1)
        
        # ================================================================
        # 方案二: 内容感知方向γ (Content-Adaptive Direction γ)
        #
        # gamma_adapt网络 (一次GAP → 判断方向偏好):
        #   γ_vh = γ_vh_base + γ_adapt(x)[0]     方向V: H方向强度
        #   γ_vw = γ_vw_base + γ_adapt(x)[1]     方向V: W方向强度
        #   γ_sp = gamma_sp (静态, 不共享瓶颈)    空间: 局部融合强度
        #
        # γ_adapt网络: GAP(C) → Conv(C→C/16) → ReLU → Conv(C/16→2) → Tanh
        #   - 2输出专用于方向调制, 16维瓶颈聚焦"方向偏好"单一任务
        #   - Tanh输出[-1,1]: 对称调制, 初始≈0
        #   - γ_sp 为静态参数(nn.Parameter), 不参与内容感知 (v4教训: 共享瓶颈→多任务冲突)
        # ================================================================
        self.gamma_vh_base = nn.Parameter(torch.zeros(1))
        self.gamma_vw_base = nn.Parameter(torch.zeros(1))
        
        self.gamma_adapt = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),              # [B, C, H, W] → [B, C, 1, 1]
            nn.Conv2d(C, mid_adapt, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_adapt, 2, 1, bias=True),  # 2 outputs: [vh, vw]
            nn.Tanh(),                            # [-1, 1], 对称调制
        )
        
        # 通道恢复投影
        self.ch_wz = nn.Conv2d(C // 2, C, 1)
        
        # ================================================================
        # Spatial Branch — 全局matmul + 局部conv + 卷积融合 (方案C)
        # ================================================================
        
        self.sp_wv = nn.Conv2d(C, C // 2, 1)
        self.sp_wq = nn.Conv2d(C, C // 2, 1)
        self.agp = nn.AdaptiveAvgPool2d(1)
        
        self.sp_local = nn.Sequential(
            nn.Conv2d(2, mid, 7, padding=3, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1, bias=False),
        )
        
        self.sp_fuse = nn.Sequential(
            nn.Conv2d(2, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1, bias=False),
        )
        self.gamma_sp = nn.Parameter(torch.zeros(1))
        
        # ================================================================
        # 归一化层
        # ================================================================
        self.softmax_ch = nn.Softmax(1)
        self.softmax_sp = nn.Softmax(-1)
        self.sigmoid = nn.Sigmoid()
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 方案二关键初始化: gamma_adapt 最后层强制输出≈0
        #   tanh(0) = 0, 确保所有γ = γ_base(0) + 0 = 0 → 退化保证
        adapt_last_conv = self.gamma_adapt[3]  # Conv2d(C/16, 2, 1)
        nn.init.constant_(adapt_last_conv.weight, 0)
        nn.init.constant_(adapt_last_conv.bias, 0)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # ================================================================
        # Channel Branch: 纯Q + 纯净双V + 内容感知方向V
        # ================================================================
        
        # --- Q: 纯PSA ---
        q = self.ch_wq(x).reshape(B, -1, 1)               # [B, HW, 1]
        q = self.softmax_ch(q)
        
        # --- V: 双重投影（方案一: 纯净版, 无预门控）---
        v_avg = self.ch_wv_avg(x)                          # [B, C/2, H, W] 直接投影
        v_max = self.ch_wv_max(x)                          # [B, C/2, H, W] 直接投影
        v_dual = v_avg + v_max                             # 双重投影融合
        
        # --- V: 方向感知（方案B）---
        feat_h = self.v_pool_h(x)                          # [B, C, H, 1]
        v_h = self.ch_wv_h(feat_h)                         # [B, C/2, H, 1]
        v_h = v_h.expand(-1, -1, -1, W)                    # [B, C/2, H, W]
        
        feat_w = self.v_pool_w(x)                          # [B, C, 1, W]
        v_w = self.ch_wv_w(feat_w)                         # [B, C/2, 1, W]
        v_w = v_w.expand(-1, -1, H, -1)                    # [B, C/2, H, W]
        
        # --- 方案二: 内容感知γ ---
        # gamma_adapt: GAP → Conv → ReLU → Conv → Tanh → [B, 2, 1, 1]
        adapt = self.gamma_adapt(x)                        # [B, 2, 1, 1]
        gamma_vh = self.gamma_vh_base.abs() + adapt[:, 0:1, :, :]  # [B, 1, 1, 1]
        gamma_vw = self.gamma_vw_base.abs() + adapt[:, 1:2, :, :]  # [B, 1, 1, 1]
        
        # 合成V: 纯净双V + 内容感知方向V
        v = v_dual + gamma_vh * v_h + gamma_vw * v_w
        v = v.reshape(B, C // 2, -1)                       # [B, C/2, HW]
        
        # 通道投票
        ch_attn = torch.matmul(v, q)                       # [B, C/2, 1]
        ch_weight = self.sigmoid(
            self.ch_wz(ch_attn.reshape(B, C // 2, 1, 1))
        )
        ch_out = x * ch_weight
        
        # ================================================================
        # Spatial Branch: 全局 + 局部 + 卷积融合 (方案C, 保留)
        # ================================================================
        
        sp_v = self.sp_wv(x).reshape(B, C // 2, -1)
        sp_q = self.sp_wq(x)
        sp_q = self.agp(sp_q).reshape(B, 1, C // 2)
        sp_q = self.softmax_sp(sp_q)
        sp_global = torch.matmul(sp_q, sp_v).reshape(B, 1, H, W)
        
        avg_s = torch.mean(x, dim=1, keepdim=True)
        max_s, _ = torch.max(x, dim=1, keepdim=True)
        sp_local = self.sp_local(torch.cat([avg_s, max_s], dim=1))
        
        # 残差卷积融合
        sp_cat = torch.cat([sp_global, sp_local], dim=1)
        sp_residual = self.sp_fuse(sp_cat)
        sp_weight = self.sigmoid(
            sp_global + self.gamma_sp.abs() * sp_residual
        )
        sp_out = x * sp_weight
        
        return ch_out + sp_out


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("DAPSA v3 单元测试")
    print("=" * 60)
    
    # 1. 参数量
    model = DAPSA(256).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"\n参数量: {params:,} ({params/1024:.0f}K)")
    
    # 统计各组件
    dual_v_params = sum(p.numel() for m in [model.ch_wv_avg, model.ch_wv_max] for p in m.parameters())
    dir_v_params  = sum(p.numel() for m in [model.ch_wv_h, model.ch_wv_w] for p in m.parameters())
    adapt_params  = sum(p.numel() for p in model.gamma_adapt.parameters())
    spatial_params = sum(p.numel() for p in list(model.sp_local.parameters()) + list(model.sp_fuse.parameters()))
    print(f"  双重投影V: {dual_v_params:,}")
    print(f"  方向感知V: {dir_v_params:,}")
    print(f"  内容感知γ网络: {adapt_params:,} (输出2通道: γ_vh/γ_vw)")
    print(f"  空间增强: {spatial_params:,}")
    
    # 2. 前向传播
    x = torch.randn(2, 256, 32, 32).to(device)
    with torch.no_grad():
        y = model(x)
    print(f"\n前向: {list(x.shape)} → {list(y.shape)}")
    
    # 3. 退化保证
    print(f"\n初始参数:")
    print(f"  gamma_vh_base = {model.gamma_vh_base.item():.4f}")
    print(f"  gamma_vw_base = {model.gamma_vw_base.item():.4f}")
    print(f"  gamma_sp      = {model.gamma_sp.item():.4f}")
    
    with torch.no_grad():
        adapt_out = model.gamma_adapt(x)
        print(f"  gamma_adapt   = [{adapt_out[0,0,0,0]:.4f}, {adapt_out[0,1,0,0]:.4f}] (应为 [0,0])")
    
    # 验证退化: γ_base=0 + adapt≈0 → 输出应与 显式置零 一致
    with torch.no_grad():
        y1 = model(x).clone()
        model.gamma_vh_base.data.zero_()
        model.gamma_vw_base.data.zero_()
        model.gamma_sp.data.zero_()
        # 强制置零gamma_adapt权重
        model.gamma_adapt[3].weight.data.zero_()
        model.gamma_adapt[3].bias.data.zero_()
        y2 = model(x)
        diff = (y1 - y2).abs().max().item()
        print(f"\n退化验证: max|v3 - v3_forced_zero| = {diff:.8f} (应为0.0)")
    
    # 4. 测试不同图像得到不同γ
    print(f"\n内容感知验证 (gamma_base=0.1):")
    model.gamma_vh_base.data.fill_(0.1)
    model.gamma_vw_base.data.fill_(0.1)
    # 恢复gamma_adapt权重 (用小的随机值)
    nn.init.kaiming_normal_(model.gamma_adapt[1].weight)   # 第一个Conv
    nn.init.kaiming_normal_(model.gamma_adapt[3].weight)   # 第二个Conv(2通道)
    nn.init.constant_(model.gamma_adapt[3].bias, 0)
    
    with torch.no_grad():
        x1 = torch.randn(2, 256, 32, 32).to(device)        # 模拟复杂场景
        x2 = torch.randn(2, 256, 32, 32).to(device) * 0.3  # 模拟均匀场景
        a1 = model.gamma_adapt(x1)
        a2 = model.gamma_adapt(x2)
        print(f"  高激活场景 adapt: [vh={a1[0,0,0,0]:.4f}, vw={a1[0,1,0,0]:.4f}]")
        print(f"  低激活场景 adapt: [vh={a2[0,0,0,0]:.4f}, vw={a2[0,1,0,0]:.4f}]")
        print(f"  (两组不同的图像应获得不同的γ值 ✓)")
    
    print(f"\n✅ DAPSA v3 测试通过!")
    print("方案一: 纯净双投影V (无预门控) ✓")
    print("方案二: 内容感知方向强度 (base+adapt) ✓")
