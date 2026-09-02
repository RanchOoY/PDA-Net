import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops  
import torchvision.models as models
from pytorch_msssim import SSIM


class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.conv_offset = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size,
                                     kernel_size=3, padding=1, stride=stride)

        self.conv_mask = nn.Conv2d(in_channels, kernel_size * kernel_size,
                                   kernel_size=3, padding=1, stride=stride)

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=1)
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.conv_offset.weight, 0)
        nn.init.constant_(self.conv_offset.bias, 0)
        nn.init.constant_(self.conv_mask.weight, 0)
        nn.init.constant_(self.conv_mask.bias, 0)

    def forward(self, x):
        offset = self.conv_offset(x)
        mask = torch.sigmoid(self.conv_mask(x)) 

        return ops.deform_conv2d(input=x, offset=offset, weight=self.weight, bias=self.bias,
                                 stride=self.stride, padding=self.padding, mask=mask)


class NoiseEstimator(nn.Module):
    def __init__(self):
        super(NoiseEstimator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class DCRB(nn.Module):

    def __init__(self, dim):
        super(DCRB, self).__init__()

        self.conv1 = DCNv2(dim, dim, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = DCNv2(dim, dim, 3, 1, 1)

        self.noise_gate = nn.Sequential(
            nn.Conv2d(1, dim, 1),
            nn.Tanh()
        )

    def forward(self, x, noise_map):
        shortcut = x

        x = self.conv1(x)
        x = self.act(x)

        gate = self.noise_gate(noise_map)
        x = x * (1 + gate)

        x = self.conv2(x)

        return x + shortcut


class TCA(nn.Module):
    def __init__(self, dim, num_heads=4):
        super(TCA, self).__init__()
        self.num_heads = num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(b, self.num_heads, -1, h * w)
        k = k.view(b, self.num_heads, -1, h * w)
        v = v.view(b, self.num_heads, -1, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).view(b, -1, h, w)
        return self.proj(out) + x


class EnhancedYPath(nn.Module):
    def __init__(self, dim):
        super(EnhancedYPath, self).__init__()
        self.entry = nn.Conv2d(1, dim, 3, 1, 1)  
        self.block1 = DCRB(dim)
        self.attn1 = TCA(dim)
        self.block2 = DCRB(dim)
        self.attn2 = TCA(dim)
        self.block3 = DCRB(dim)

    def forward(self, y, noise_map, early_exit=False):
        x = self.entry(y)
        x = self.block1(x, noise_map)
        x = self.attn1(x)

        if early_exit:
            return x

        x = self.block2(x, noise_map)
        x = self.attn2(x)
        x = self.block3(x, noise_map)
        return x


class EnhancedCWD(nn.Module):
    def __init__(self, dim):
        super(EnhancedCWD, self).__init__()
        self.fusion = nn.Conv2d(dim * 2, dim, 1)
        self.enc1 = DCRB(dim)
        self.enc2 = DCRB(dim)
        self.mid_attn = TCA(dim)
        self.dec1 = DCRB(dim)
        self.dec2 = DCRB(dim)

    def forward(self, uv, y_feat, noise_map, early_exit=False):
        x = torch.cat([uv, y_feat], dim=1)
        x = self.fusion(x)
        shortcut = x
        x = self.enc1(x, noise_map)

        if early_exit:
            return x + shortcut

        x = self.enc2(x, noise_map)
        x = self.mid_attn(x)
        x = self.dec1(x, noise_map)
        x = self.dec2(x, noise_map)
        return x + shortcut


class PDA(nn.Module):
    def __init__(self, filters=64):
        super(PDA, self).__init__()
        self.noise_net = NoiseEstimator()
        self.y_net = EnhancedYPath(filters)
        self.uv_net = EnhancedCWD(filters)
        self.uv_entry = nn.Conv2d(2, filters, 3, 1, 1)

        self.fusion_conv = nn.Conv2d(filters * 2, filters, 1)
        self.tail = nn.Sequential(
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(filters, 3, 3, 1, 1)
        )

        self.early_fusion_conv = nn.Conv2d(filters * 2, filters, 1)
        self.early_tail = nn.Sequential(
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(filters, 3, 3, 1, 1)
        )

    def _rgb_to_ycbcr(self, image):
        y = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        u = -0.147 * image[:, 0, :, :] - 0.289 * image[:, 1, :, :] + 0.436 * image[:, 2, :, :]
        v = 0.615 * image[:, 0, :, :] - 0.515 * image[:, 1, :, :] - 0.100 * image[:, 2, :, :]
        return y.unsqueeze(1), torch.stack((u, v), dim=1)

    def forward(self, x):
        y_img, uv_img = self._rgb_to_ycbcr(x)
        noise_map = self.noise_net(y_img)

        is_simple = False
        y_feat = self.y_net(y_img, noise_map, early_exit=is_simple)
        uv_feat_in = self.uv_entry(uv_img)
        uv_feat = self.uv_net(uv_feat_in, y_feat, noise_map, early_exit=is_simple)

        concat = torch.cat([y_feat, uv_feat], dim=1)
        if is_simple:
            fused = self.early_fusion_conv(concat)
            residual = self.early_tail(fused)
        else:
            fused = self.fusion_conv(concat)
            residual = self.tail(fused)

        out = torch.sigmoid(x + residual)
        return out
