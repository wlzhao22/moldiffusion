import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenization import PAD_ID, PAD, EOS_ID, EOS

class GraphLoss(nn.Module):
    """
    Loss function for graph-based predictions (edges).
    This is retained from the original MolNexTR for edge prediction.
    """
    def __init__(self):
        super(GraphLoss, self).__init__()
        # Custom weighting to penalize wrong bond types more than predicting no bond.
        weight = torch.ones(7) * 10
        weight[0] = 1 # 'no bond' has lower weight
        weight[5] = 20 # 'solid wedge bond' has higher weight
        weight[6] = 20 # 'dashed wedge bond' has higher weight
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=-100)

    def forward(self, outputs, targets):
        """
        Args:
            outputs (dict): Dictionary containing 'edges' predictions.
            targets (dict): Dictionary containing 'edges' ground truth.
        Returns:
            dict: Dictionary with 'edges' loss.
        """
        results = {}
        if 'edges' in outputs:
            pred = outputs['edges']
            max_len = pred.size(-1)
            target = targets['edges'][:, :max_len, :max_len]
            results['edges'] = self.criterion(pred, target)
        return results

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-100, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha)
            else:
                self.alpha = alpha
        else:
            self.alpha = None
            
    def forward(self, logits, labels):
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        
        valid_mask = (labels != self.ignore_index)
        logits = logits[valid_mask]
        labels = labels[valid_mask]
        
        if len(logits) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        probs = F.softmax(logits, dim=-1)
        
        p_t = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        log_p_t = F.log_softmax(logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
        loss = -1 * focal_weight * log_p_t
        
        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            alpha_t = self.alpha.gather(0, labels)
            loss = loss * alpha_t
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class WeightedCELoss(nn.Module):
    """
    Claculate thhe loss for a Masked Language Model.
    It computes the CrossEntropyLoss only on positions that are masked.
    """
    def __init__(self, pad_weight = 0.1, content_weight=1.0, eos_weight=2.0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        self.pad_id = PAD_ID
        self.eos_id = EOS_ID
        self.pad_weight = pad_weight
        self.content_weight = content_weight
        self.eos_weight = eos_weight
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            logits (torch.Tensor): The Model's output logits. Shape: [B, L, K].
            labels (torch.Tensor): The target labels. Non-masked positions are -100. Shape: [B, L].
        """
        B, L, K = logits.shape
        
        # Flatten logits and labels for loss calculation
        flat_logits = logits.view(-1, K)
        flat_labels = labels.view(-1)
        
        # Calculate unreduced loss for all masked positions
        unreduced_loss = self.criterion(flat_logits, flat_labels)
        
        weights = torch.ones_like(flat_labels)
        is_pad_mask = (flat_labels == self.pad_id)
        is_content_mask = (flat_labels != self.pad_id) & (flat_labels != -100)
        is_eos_mask = (flat_labels == self.eos_id)
        weights[is_pad_mask] = self.pad_weight
        weights[is_content_mask] = self.content_weight
        weights[is_eos_mask] = self.eos_weight
        weighted_loss = unreduced_loss * weights
        
        num_active_tokens = (flat_labels != -100).sum().clamp(min=1)
        
        return weighted_loss.sum() / num_active_tokens


class Criterion(nn.Module):
    """
    A wrapper for all loss functions.
    It orchestrates the calculation of sequence diffusion loss and graph edge loss.
    """
    def __init__(self, args, tokenizer):
        super(Criterion, self).__init__()
        self.args = args
        self.criterion = nn.ModuleDict()

        # Add mlm loss for the primary sequence generation task
        if 'chartok_coords' in args.formats:
            self.criterion['diffusion'] = FocalLoss(
                gamma=2.0,
                ignore_index=-100,
                reduction='mean'
            )
            # self.criterion['diffusion'] = WeightedCELoss(
            #     pad_weight=1.0,
            #     content_weight=10.0,
            #     eos_weight=20.0
            # )

        # Add graph loss if edge prediction is required
        if 'edges' in args.formats:
            self.criterion['graph'] = GraphLoss()

    def forward(self, results: dict, refs: dict):
        """
        Args:
            results (dict): The output from the model's `Decoder` module.
                            For diffusion: {'chartok_coords': (logits, target_x0, dec_out), 'edges': ...}
            refs (dict): The ground truth data from the dataloader.
                         Contains 'x0', 'edges', etc.
        
        Returns:
            dict: A dictionary of computed losses, e.g., {'diffusion': 0.5, 'edges': 0.2}.
        """
        losses = {}
        
        # 1. Calculate mlm Loss for the generated sequence
        if 'diffusion' in self.criterion and 'chartok_coords' in results:
            logits, x0, xt, labels, dec_out = results['chartok_coords'] # Unpack
            losses['diffusion'] = self.criterion['diffusion'](logits, labels)

        # 2. Calculate Graph Loss for edge prediction
        if 'graph' in self.criterion and 'edges' in results:
            if refs['edges'].dim() < 3: # skip dummy edge predictions
                pass
            else:
                edge_predictions = results['edges']
                graph_losses = self.criterion['graph'](edge_predictions, refs)
                losses.update(graph_losses)
            
        return losses