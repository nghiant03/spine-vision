"""Loss functions for training.

Provides custom loss functions for various training scenarios:
- FocalLoss: Focal loss for handling class imbalance in binary classification

Usage:
    from spine_vision.training.losses import FocalLoss

    # Basic usage with default gamma
    loss_fn = FocalLoss()
    loss = loss_fn(logits, targets)

    # With custom gamma and optional alpha
    loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
    loss = loss_fn(logits, targets)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for binary classification.

    Focal loss down-weights well-classified examples and focuses on hard,
    misclassified examples. This is particularly useful for imbalanced datasets.

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
        https://arxiv.org/abs/1708.02002

    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t is the model's estimated probability for the ground truth class.

    Args:
        gamma: Focusing parameter. Higher values increase focus on hard examples.
            gamma=0 is equivalent to standard cross-entropy. Default: 2.0.
        alpha: Optional weighting factor for the positive class.
            If None, no class weighting is applied. Default: None.
        pos_weight: Optional weight for positive examples (alternative to alpha).
            If None, no positive weighting is applied. Default: None.
        reduction: Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'.

    Note:
        - Accepts logits (not probabilities) for numerical stability.
        - Uses F.binary_cross_entropy_with_logits internally.
        - alpha and pos_weight are set to None by default to avoid
          "double compensation" when using weighted sampling.

    Example:
        >>> loss_fn = FocalLoss(gamma=2.0)
        >>> logits = torch.randn(32, 1)  # Raw model output
        >>> targets = torch.randint(0, 2, (32, 1)).float()
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = None,
        pos_weight: float | None = None,
        reduction: str = "mean",
    ) -> None:
        """Initialize FocalLoss.

        Args:
            gamma: Focusing parameter (default: 2.0).
            alpha: Optional class weight for positive class (default: None).
            pos_weight: Optional weight for positive examples (default: None).
            reduction: Reduction method ('none', 'mean', 'sum').
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.reduction = reduction

        # Validate reduction
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(
                f"Invalid reduction: {reduction}. Must be 'none', 'mean', or 'sum'."
            )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw model outputs (before sigmoid), shape [B, *].
            targets: Ground truth labels (0 or 1), shape [B, *].

        Returns:
            Focal loss value. Shape depends on reduction:
                - 'none': same shape as input
                - 'mean' or 'sum': scalar
        """
        # Compute probabilities from logits
        probs = torch.sigmoid(logits)

        # Compute p_t (probability of the true class)
        # For positive class (target=1): p_t = p
        # For negative class (target=0): p_t = 1 - p
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Compute binary cross entropy with logits (for numerical stability)
        # reduction='none' to apply focal weight per-element
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=torch.tensor(self.pos_weight).to(logits.device)
            if self.pos_weight is not None
            else None,
        )

        # Apply focal weight
        focal_loss = focal_weight * bce_loss

        # Apply alpha weighting if specified
        if self.alpha is not None:
            # alpha_t: alpha for positive class, (1 - alpha) for negative class
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss

    def extra_repr(self) -> str:
        """Return extra representation string."""
        parts = [f"gamma={self.gamma}"]
        if self.alpha is not None:
            parts.append(f"alpha={self.alpha}")
        if self.pos_weight is not None:
            parts.append(f"pos_weight={self.pos_weight}")
        parts.append(f"reduction={self.reduction!r}")
        return ", ".join(parts)
