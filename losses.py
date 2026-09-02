import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops 
import torchvision.models as models
from pytorch_msssim import SSIM


class FFTLoss(nn.Module):
    def __init__(self):
        super(FFTLoss, self).__init__()

    def forward(self, pred, target):
        pred_f32 = pred.to(torch.float32)
        target_f32 = target.to(torch.float32)

        pred_fft = torch.fft.rfft2(pred_f32, norm='backward')
        target_fft = torch.fft.rfft2(target_f32, norm='backward')

        loss = torch.mean(torch.abs(pred_fft - target_fft))
        return loss


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss


class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:16]
        self.loss_model = vgg.to(device).eval()
        for param in self.loss_model.parameters():
            param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, y_true, y_pred):
        y_true = (y_true - self.mean) / self.std
        y_pred = (y_pred - self.mean) / self.std
        return F.mse_loss(self.loss_model(y_true), self.loss_model(y_pred))


def color_loss(y_true, y_pred):
    b, c, h, w = y_true.shape
    y_true = y_true + 1e-6
    y_pred = y_pred + 1e-6
    true_flat = y_true.view(b, c, -1).mean(dim=2)
    pred_flat = y_pred.view(b, c, -1).mean(dim=2)
    return (1 - F.cosine_similarity(true_flat, pred_flat, dim=1).mean()) * 5.0


def gaussian_kernel(x, mu, sigma):
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)


def histogram_loss(y_true, y_pred, bins=64, sigma=0.01):
    bin_edges = torch.linspace(0.0, 1.0, bins, device=y_true.device)
    y_true_hist = torch.sum(gaussian_kernel(y_true.unsqueeze(-1), bin_edges, sigma), dim=0)
    y_pred_hist = torch.sum(gaussian_kernel(y_pred.unsqueeze(-1), bin_edges, sigma), dim=0)
    y_true_hist /= (y_true_hist.sum() + 1e-6)
    y_pred_hist /= (y_pred_hist.sum() + 1e-6)
    return torch.mean(torch.abs(y_true_hist - y_pred_hist))


class CombinedLoss(nn.Module):
    def __init__(self, device, warmup_epochs=50):
        super(CombinedLoss, self).__init__()
        self.device = device
        self.warmup_epochs = warmup_epochs

        self.charbonnier = CharbonnierLoss().to(device)
        self.ssim_module = SSIM(data_range=1.0, size_average=True, channel=3).to(device)
        self.perceptual = VGGPerceptualLoss(device)
        self.fft_loss = FFTLoss().to(device)

        self.alpha_char = 1.0
        self.alpha_ssim = 0.2
        self.alpha_perc = 0.06
        self.alpha_col = 0.2
        self.alpha_fft = 0.05

    def forward(self, y_true, y_pred, current_epoch=None):

        loss_pixel = self.charbonnier(y_pred, y_true)
        loss_ssim = 1.0 - self.ssim_module(y_pred, y_true)
        loss_perc = self.perceptual(y_true, y_pred)
        loss_col = color_loss(y_true, y_pred)
        loss_fft = self.fft_loss(y_pred, y_true)

        losses_stack = torch.stack([loss_pixel, loss_ssim, loss_perc, loss_col, loss_fft]).detach()
        mean_loss = losses_stack.mean() + 1e-6
        dynamic_weights = losses_stack / mean_loss
        dynamic_weights = torch.clamp(dynamic_weights, min=0.8, max=1.2)

        total_loss = (self.alpha_char * dynamic_weights[0] * loss_pixel +
                      self.alpha_ssim * dynamic_weights[1] * loss_ssim +
                      self.alpha_perc * dynamic_weights[2] * loss_perc +
                      self.alpha_col * dynamic_weights[3] * loss_col +
                      self.alpha_fft * dynamic_weights[4] * loss_fft)

        return total_loss
