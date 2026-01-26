"""Configurable head architectures for models.

Provides modular head components that can be configured via config,
enabling easy experimentation with different head architectures.

Available head types:
- MLP: Multi-layer perceptron (default)
- Linear: Single linear layer
- Attention: Self-attention based head
- Conv: 1x1 convolution based head

Usage:
    from spine_vision.training.heads import HeadConfig, create_head

    # Via config
    config = HeadConfig(head_type="mlp", hidden_dims=[512, 256])
    head = create_head(config, in_features=2048, out_features=2)

    # Or use factory directly
    head = HeadFactory.create("mlp", in_features=2048, out_features=2)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn

HeadType = Literal["mlp", "linear", "attention", "conv", "residual"]


@dataclass
class HeadConfig:
    """Configuration for model heads.

    Attributes:
        head_type: Type of head architecture.
        hidden_dims: Hidden layer dimensions for MLP/residual heads.
        dropout: Dropout rate.
        activation: Activation function name.
        use_layer_norm: Whether to use layer normalization.
        num_attention_heads: Number of attention heads (for attention head).
        output_activation: Activation for output layer (e.g., "sigmoid", "softmax", "none").
    """

    head_type: HeadType = "mlp"
    hidden_dims: list[int] = field(default_factory=lambda: [256])
    dropout: float = 0.2
    activation: str = "gelu"
    use_layer_norm: bool = True
    num_attention_heads: int = 4
    output_activation: str = "none"


def get_activation(name: str) -> nn.Module:
    """Get activation module by name.

    Args:
        name: Activation name (relu, gelu, silu, tanh, sigmoid, none).

    Returns:
        Activation module.
    """
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "softmax": nn.Softmax(dim=-1),
        "none": nn.Identity(),
    }
    if name.lower() not in activations:
        raise ValueError(
            f"Unknown activation: {name}. Available: {list(activations.keys())}"
        )
    return activations[name.lower()]


class BaseHead(nn.Module, ABC):
    """Abstract base class for model heads."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features [B, in_features].

        Returns:
            Output tensor [B, out_features].
        """
        ...

    @property
    @abstractmethod
    def out_features(self) -> int:
        """Number of output features."""
        ...


class LinearHead(BaseHead):
    """Simple linear head with optional normalization and dropout."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        output_activation: str = "none",
    ) -> None:
        super().__init__()
        self._out_features = out_features

        layers: list[nn.Module] = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(in_features))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_features, out_features))
        layers.append(get_activation(output_activation))

        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

    @property
    def out_features(self) -> int:
        return self._out_features


class MLPHead(BaseHead):
    """Multi-layer perceptron head with configurable hidden layers."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        activation: str = "gelu",
        use_layer_norm: bool = True,
        output_activation: str = "none",
    ) -> None:
        super().__init__()
        self._out_features = out_features
        hidden_dims = hidden_dims or [256]

        layers: list[nn.Module] = []

        # Input normalization
        if use_layer_norm:
            layers.append(nn.LayerNorm(in_features))

        # Hidden layers
        prev_dim = in_features
        for i, hidden_dim in enumerate(hidden_dims):
            if dropout > 0:
                layers.append(nn.Dropout(dropout if i == 0 else dropout / 2))
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(get_activation(activation))
            prev_dim = hidden_dim

        # Output layer
        if dropout > 0:
            layers.append(nn.Dropout(dropout / 2))
        layers.append(nn.Linear(prev_dim, out_features))
        layers.append(get_activation(output_activation))

        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

    @property
    def out_features(self) -> int:
        return self._out_features


class AttentionHead(BaseHead):
    """Self-attention based head for global feature aggregation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.2,
        use_layer_norm: bool = True,
        output_activation: str = "none",
    ) -> None:
        super().__init__()
        self._out_features = out_features

        self.norm = nn.LayerNorm(in_features) if use_layer_norm else nn.Identity()
        self.attention = nn.MultiheadAttention(
            embed_dim=in_features,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, out_features)
        self.output_act = get_activation(output_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add sequence dimension if not present
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, D]

        x = self.norm(x)
        attn_out, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_out)

        # Pool and project
        x = x.mean(dim=1)  # [B, D]
        x = self.fc(x)
        return self.output_act(x)

    @property
    def out_features(self) -> int:
        return self._out_features


class ResidualHead(BaseHead):
    """Residual MLP head with skip connections."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        activation: str = "gelu",
        use_layer_norm: bool = True,
        output_activation: str = "none",
    ) -> None:
        super().__init__()
        self._out_features = out_features
        hidden_dims = hidden_dims or [256]

        # Input projection if needed
        self.input_proj = (
            nn.Linear(in_features, hidden_dims[0]) if hidden_dims else nn.Identity()
        )
        self.input_norm = nn.LayerNorm(in_features) if use_layer_norm else nn.Identity()

        # Residual blocks
        self.blocks = nn.ModuleList()
        prev_dim = hidden_dims[0] if hidden_dims else in_features

        for hidden_dim in hidden_dims:
            block = nn.Sequential(
                nn.LayerNorm(prev_dim) if use_layer_norm else nn.Identity(),
                nn.Dropout(dropout),
                nn.Linear(prev_dim, hidden_dim),
                get_activation(activation),
                nn.Dropout(dropout / 2),
                nn.Linear(hidden_dim, prev_dim),
            )
            self.blocks.append(block)

        # Output layer
        self.output_norm = nn.LayerNorm(prev_dim) if use_layer_norm else nn.Identity()
        self.output_dropout = nn.Dropout(dropout)
        self.output_fc = nn.Linear(prev_dim, out_features)
        self.output_act = get_activation(output_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.input_proj(x)

        for block in self.blocks:
            x = x + block(x)  # Residual connection

        x = self.output_norm(x)
        x = self.output_dropout(x)
        x = self.output_fc(x)
        return self.output_act(x)

    @property
    def out_features(self) -> int:
        return self._out_features


class ConvHead(BaseHead):
    """1x1 convolution based head (for spatial features)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        activation: str = "gelu",
        output_activation: str = "none",
    ) -> None:
        super().__init__()
        self._out_features = out_features
        hidden_dims = hidden_dims or [256]

        layers: list[nn.Module] = []
        prev_dim = in_features

        for hidden_dim in hidden_dims:
            layers.append(nn.Conv1d(prev_dim, hidden_dim, kernel_size=1))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Conv1d(prev_dim, out_features, kernel_size=1))

        self.conv_layers = nn.Sequential(*layers)
        self.output_act = get_activation(output_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect [B, C] or [B, C, L]
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B, C, 1]

        x = self.conv_layers(x)
        x = x.squeeze(-1)  # [B, out_features]

        return self.output_act(x)

    @property
    def out_features(self) -> int:
        return self._out_features


class HeadFactory:
    """Factory for creating head modules.

    Supports registration of custom head types.
    """

    _heads: dict[str, type[BaseHead]] = {
        "linear": LinearHead,
        "mlp": MLPHead,
        "attention": AttentionHead,
        "residual": ResidualHead,
        "conv": ConvHead,
    }

    @classmethod
    def register(cls, name: str) -> Any:
        """Decorator to register a custom head type.

        Args:
            name: Unique identifier for the head type.

        Returns:
            Decorator function.
        """

        def decorator(head_cls: type[BaseHead]) -> type[BaseHead]:
            cls._heads[name] = head_cls
            return head_cls

        return decorator

    @classmethod
    def create(
        cls,
        head_type: str,
        in_features: int,
        out_features: int,
        **kwargs: Any,
    ) -> BaseHead:
        """Create a head module by type.

        Args:
            head_type: Registered head type name.
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            **kwargs: Additional arguments for the head constructor.

        Returns:
            Head module instance.

        Raises:
            KeyError: If head type not found.
        """
        if head_type not in cls._heads:
            available = ", ".join(cls._heads.keys())
            raise KeyError(f"Unknown head type: {head_type}. Available: {available}")

        head_cls = cls._heads[head_type]
        return head_cls(in_features=in_features, out_features=out_features, **kwargs)

    @classmethod
    def list_heads(cls) -> list[str]:
        """List available head types."""
        return list(cls._heads.keys())


def create_head(
    config: HeadConfig,
    in_features: int,
    out_features: int,
) -> BaseHead:
    """Create a head module from configuration.

    Args:
        config: Head configuration.
        in_features: Input feature dimension.
        out_features: Output feature dimension.

    Returns:
        Head module instance.
    """
    kwargs: dict[str, Any] = {
        "dropout": config.dropout,
        "output_activation": config.output_activation,
    }

    if config.head_type in ("mlp", "residual", "conv"):
        kwargs["hidden_dims"] = config.hidden_dims
        kwargs["activation"] = config.activation

    if config.head_type in ("mlp", "linear", "residual", "attention"):
        kwargs["use_layer_norm"] = config.use_layer_norm

    if config.head_type == "attention":
        kwargs["num_heads"] = config.num_attention_heads

    return HeadFactory.create(
        config.head_type,
        in_features=in_features,
        out_features=out_features,
        **kwargs,
    )


# Multi-task head for MTL models
class MultiTaskHead(nn.Module):
    """Multi-task head that creates separate heads for each task.

    Useful for multi-task learning where each task has different output
    dimensions and possibly different head architectures.
    """

    def __init__(
        self,
        in_features: int,
        task_configs: dict[str, tuple[int, HeadConfig]],
    ) -> None:
        """Initialize multi-task head.

        Args:
            in_features: Input feature dimension (shared).
            task_configs: Dict mapping task name to (out_features, HeadConfig).
        """
        super().__init__()

        self.heads = nn.ModuleDict()
        self._task_out_features: dict[str, int] = {}

        for task_name, (out_features, head_config) in task_configs.items():
            head = create_head(head_config, in_features, out_features)
            self.heads[task_name] = head
            self._task_out_features[task_name] = out_features

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through all task heads.

        Args:
            x: Input features [B, in_features].

        Returns:
            Dictionary mapping task name to output tensor.
        """
        return {name: head(x) for name, head in self.heads.items()}

    def forward_task(self, x: torch.Tensor, task_name: str) -> torch.Tensor:
        """Forward pass through a specific task head.

        Args:
            x: Input features [B, in_features].
            task_name: Name of the task.

        Returns:
            Output tensor for the task.
        """
        if task_name not in self.heads:
            raise KeyError(f"Unknown task: {task_name}")
        return self.heads[task_name](x)

    @property
    def task_names(self) -> list[str]:
        """List of task names."""
        return list(self.heads.keys())

    def get_out_features(self, task_name: str) -> int:
        """Get output features for a task."""
        return self._task_out_features[task_name]
