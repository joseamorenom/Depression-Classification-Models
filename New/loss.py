# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (float, optional): Peso para la clase minoritaria (ej. 0.25). 
                                     Si es None, no aplica pesos directos, solo focal.
            gamma (float): Factor de enfoque. Mayor gamma = más enfoque en difíciles. 
                           gamma=2.0 es el estándar.
            reduction (str): 'mean' o 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [N, C] logits
        # targets: [N] labels
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # pt es la probabilidad de la clase correcta
        
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            # Manejo simple de alpha para 2 clases (0 y 1)
            # alpha aplica a la clase 1, (1-alpha) a la clase 0
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss