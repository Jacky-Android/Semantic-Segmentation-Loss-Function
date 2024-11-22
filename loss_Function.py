import torch
import torch.nn.functional as F

# Dice Loss
class DiceLoss:
    def __init__(self, smooth=1e-5):
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        y_true_flat = y_true.view(-1)
        y_pred_flat = y_pred.view(-1)
        intersection = (y_true_flat * y_pred_flat).sum()
        return 1 - (2. * intersection + self.smooth) / \
                   (y_true_flat.sum() + y_pred_flat.sum() + self.smooth)


# Binary Cross-Entropy + Dice Loss
class BCEDiceLoss:
    def __init__(self, smooth=1e-5):
        self.dice_loss = DiceLoss(smooth)

    def __call__(self, y_true, y_pred):
        bce_loss = F.binary_cross_entropy(y_pred, y_true)
        dice_loss = self.dice_loss(y_true, y_pred)
        return (bce_loss + dice_loss) / 2.0


# Focal Loss
class FocalLoss:
    def __init__(self, alpha=0.25, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, y_true, y_pred):
        y_pred = y_pred.clamp(1e-5, 1 - 1e-5)
        ce_loss = - (y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        focal_loss = self.alpha * (1 - y_pred) ** self.gamma * y_true * ce_loss + \
                     (1 - self.alpha) * y_pred ** self.gamma * (1 - y_true) * ce_loss
        return focal_loss.mean()


# Tversky Loss
class TverskyLoss:
    def __init__(self, alpha=0.7, smooth=1e-5):
        self.alpha = alpha
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        y_true_flat = y_true.view(-1)
        y_pred_flat = y_pred.view(-1)
        true_pos = (y_true_flat * y_pred_flat).sum()
        false_neg = (y_true_flat * (1 - y_pred_flat)).sum()
        false_pos = ((1 - y_true_flat) * y_pred_flat).sum()
        return 1 - (true_pos + self.smooth) / \
                   (true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + self.smooth)


# Structural Similarity Index (SSIM) Loss
class SSIMLoss:
    def __call__(self, y_true, y_pred, max_val=1):
        ssim = torch.clamp((1 - y_true) * (1 - y_pred) + y_true * y_pred, 0, 1)
        return 1 - ssim.mean()


# Jaccard Loss
class JaccardLoss:
    def __init__(self, smooth=1e-5):
        self.smooth = smooth

    def __call__(self, y_true, y_pred):
        y_true_flat = y_true.view(-1)
        y_pred_flat = y_pred.view(-1)
        intersection = (y_true_flat * y_pred_flat).sum()
        union = y_true_flat.sum() + y_pred_flat.sum() - intersection
        return 1 - (intersection + self.smooth) / (union + self.smooth)


# Hybrid Loss: Combination of Focal, SSIM, and Jaccard Loss
class HybridLoss:
    def __init__(self, alpha=0.25, gamma=2.0, smooth=1e-5):
        self.focal_loss = FocalLoss(alpha, gamma)
        self.ssim_loss = SSIMLoss()
        self.jaccard_loss = JaccardLoss(smooth)

    def __call__(self, y_true, y_pred):
        focal = self.focal_loss(y_true, y_pred)
        ssim = self.ssim_loss(y_true, y_pred)
        jaccard = self.jaccard_loss(y_true, y_pred)
        return focal + ssim + jaccard
