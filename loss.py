import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, pred, gt):
        """calculate loss

        Args:
            pred (torch.Tensor): (B, gt_len, vocab_size)
            gt (torch.Tensor): (B, gt_len, vocab_size)
        """
        _, _, vocab_size = pred.shape

        gt = gt[:, 1:]
        gt = gt.flatten()

        pred = pred[:, :-1, :]
        
        gt = F.one_hot(gt.to(torch.int64), num_classes=vocab_size).to(torch.bfloat16)
        pred = pred.contiguous().view(-1, vocab_size)

        return self.ce_loss(gt, pred)
        
