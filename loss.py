import torch
import torch.nn as nn
import torch.nn.functional as F

def filter_valid_label(scores, labels, num_classes, ignored_label_inds, device):
    """Loss functions for semantic segmentation."""
    valid_scores = scores.reshape(-1, num_classes)
    valid_labels = labels.reshape(-1).to(device)
    ignored_bool = torch.zeros_like(valid_labels, dtype=torch.bool)
    for ign_label in ignored_label_inds:
        ignored_bool = torch.logical_or(ignored_bool, torch.eq(valid_labels, ign_label))
    valid_idx = torch.where(torch.logical_not(ignored_bool))[0].to(device)
    valid_scores = torch.gather(valid_scores, 0, valid_idx.unsqueeze(-1).expand(-1, num_classes))
    valid_labels = torch.gather(valid_labels, 0, valid_idx)
    # Reduce label values in the range of logit shape
    reducing_list = torch.arange(0, num_classes, dtype=torch.int64)
    inserted_value = torch.zeros([1], dtype=torch.int64)
    for ign_label in ignored_label_inds:
        if ign_label >= 0:
            reducing_list = torch.cat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
    valid_labels = torch.gather(reducing_list.to(device), 0, valid_labels.long())
    return valid_scores, valid_labels

def one_hot(index, classes):
    out_idx = torch.arange(classes, device=index.device)
    out_idx = torch.unsqueeze(out_idx, 0)
    index = torch.unsqueeze(index, -1)
    return (index == out_idx).float()

class CrossEntropyLoss(nn.Module):

    def __init__(self, loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, cls_score, label, weight=None, avg_factor=None, **kwargs):
        """Forward function.
        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.

        Returns:
            torch.Tensor: The calculated loss
        """
        loss = F.cross_entropy(cls_score, label, reduction='none')

        if weight is not None:
            loss = F.cross_entropy(cls_score, label, weight = weight, reduction='none')

        if avg_factor:
            return loss.sum() / avg_factor
        else:
            return loss.mean()

class FocalLoss(nn.Module):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

    Args:
        gamma (float, optional): The gamma for calculating the modulating factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss. Defaults to 0.25.
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, gamma=2.0, alpha=0.25, loss_weight=1.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None):
        pred_sigmoid = pred.sigmoid()
        if len(pred.shape) > 1:
            target = one_hot(target, int(pred.shape[-1]))

        target = target.type_as(pred)

        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) *(1 - target)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight

        if weight is not None:
            loss = loss * weight

        loss = loss * self.loss_weight

        if avg_factor is None:
            return loss.mean()
        elif avg_factor > 0:
            return loss.sum() / avg_factor
        else:
            return loss
