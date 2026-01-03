"""
Graph Neural Network baselines for molecular property prediction.

This module provides wrappers around popular 3D GNN architectures
(SchNet, DimeNet++, SphereNet) with conformer ensemble support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math

# Note: These implementations are simplified versions.
# For production use, consider using PyTorch Geometric implementations.


class RadialBasisFunctions(nn.Module):
    """Radial basis functions for distance embedding."""

    def __init__(
        self,
        num_rbf: int = 50,
        cutoff: float = 10.0,
        rbf_type: str = "gaussian",
    ):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.rbf_type = rbf_type

        # Centers and widths for Gaussian RBF
        self.register_buffer(
            "centers",
            torch.linspace(0, cutoff, num_rbf)
        )
        self.register_buffer(
            "widths",
            torch.FloatTensor([cutoff / num_rbf] * num_rbf)
        )

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Embed distances using radial basis functions.

        Args:
            distances: Pairwise distances (..., )

        Returns:
            RBF embeddings (..., num_rbf)
        """
        if self.rbf_type == "gaussian":
            return torch.exp(
                -((distances.unsqueeze(-1) - self.centers) ** 2)
                / (2 * self.widths ** 2)
            )
        else:
            raise ValueError(f"Unknown RBF type: {self.rbf_type}")


class CutoffFunction(nn.Module):
    """Smooth cutoff function for distance-based interactions."""

    def __init__(self, cutoff: float = 10.0):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Apply smooth cutoff."""
        # Cosine cutoff
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs


class SchNetInteraction(nn.Module):
    """SchNet continuous-filter convolution block."""

    def __init__(
        self,
        hidden_dim: int,
        num_filters: int,
        num_rbf: int,
        cutoff: float,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_filters = num_filters

        # Filter-generating network
        self.filter_network = nn.Sequential(
            nn.Linear(num_rbf, num_filters),
            nn.SiLU(),
            nn.Linear(num_filters, num_filters),
        )

        # Atom-wise layers
        self.atom_dense1 = nn.Linear(hidden_dim, num_filters)
        self.atom_dense2 = nn.Linear(num_filters, hidden_dim)

        self.cutoff_fn = CutoffFunction(cutoff)

    def forward(
        self,
        x: torch.Tensor,
        rbf: torch.Tensor,
        distances: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of SchNet interaction.

        Args:
            x: Node features (num_atoms, hidden_dim)
            rbf: RBF-embedded distances (num_edges, num_rbf)
            distances: Raw distances (num_edges,)
            edge_index: Edge indices (2, num_edges)

        Returns:
            Updated node features
        """
        # Generate continuous filters
        filters = self.filter_network(rbf)  # (num_edges, num_filters)
        filters = filters * self.cutoff_fn(distances).unsqueeze(-1)

        # Message passing
        x_j = x[edge_index[0]]  # Source node features
        x_j = self.atom_dense1(x_j)  # (num_edges, num_filters)

        # Apply filters
        messages = x_j * filters  # (num_edges, num_filters)

        # Aggregate messages
        out = torch.zeros_like(x[:, :self.num_filters])
        out.scatter_add_(0, edge_index[1].unsqueeze(-1).expand_as(messages), messages)

        # Output transformation
        out = self.atom_dense2(out)

        return x + out


class SchNet(nn.Module):
    """
    SchNet: Continuous-filter convolutional neural network.

    Simplified implementation for molecular property prediction
    from 3D structures.

    Reference:
        Schütt et al. "SchNet: A continuous-filter convolutional neural
        network for modeling quantum interactions" (2017)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_rbf: int = 50,
        cutoff: float = 10.0,
        max_atomic_num: int = 100,
        num_outputs: int = 1,
        conformer_aggregation: str = "mean",
    ):
        """
        Initialize SchNet.

        Args:
            hidden_dim: Hidden dimension
            num_filters: Number of filters in CFConv
            num_interactions: Number of interaction blocks
            num_rbf: Number of radial basis functions
            cutoff: Distance cutoff
            max_atomic_num: Maximum atomic number
            num_outputs: Number of outputs
            conformer_aggregation: How to aggregate conformers
        """
        super().__init__()

        self.cutoff = cutoff
        self.conformer_aggregation = conformer_aggregation

        # Atom embedding
        self.atom_embedding = nn.Embedding(max_atomic_num, hidden_dim)

        # Radial basis functions
        self.rbf = RadialBasisFunctions(num_rbf, cutoff)

        # Interaction blocks
        self.interactions = nn.ModuleList([
            SchNetInteraction(hidden_dim, num_filters, num_rbf, cutoff)
            for _ in range(num_interactions)
        ])

        # Output network
        self.output_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_outputs),
        )

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        conformer_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            atomic_numbers: Atomic numbers (num_atoms,)
            positions: 3D coordinates (num_atoms, 3)
            batch: Batch indices (num_atoms,)
            conformer_batch: Conformer indices for ensemble aggregation

        Returns:
            Predictions
        """
        # Initial embedding
        x = self.atom_embedding(atomic_numbers)

        # Compute pairwise distances and edges (simplified: all pairs within cutoff)
        # In practice, use neighbor list for efficiency
        num_atoms = positions.shape[0]
        row = torch.arange(num_atoms, device=positions.device).repeat_interleave(num_atoms)
        col = torch.arange(num_atoms, device=positions.device).repeat(num_atoms)

        # Remove self-loops
        mask = row != col
        row, col = row[mask], col[mask]

        # Compute distances
        diff = positions[row] - positions[col]
        distances = torch.norm(diff, dim=-1)

        # Apply cutoff
        cutoff_mask = distances < self.cutoff
        row, col = row[cutoff_mask], col[cutoff_mask]
        distances = distances[cutoff_mask]

        edge_index = torch.stack([row, col], dim=0)

        # RBF embedding
        rbf = self.rbf(distances)

        # Interaction blocks
        for interaction in self.interactions:
            x = interaction(x, rbf, distances, edge_index)

        # Aggregate per molecule
        num_molecules = batch.max().item() + 1
        molecule_features = torch.zeros(
            num_molecules, x.shape[-1],
            device=x.device, dtype=x.dtype
        )
        molecule_features.scatter_add_(
            0,
            batch.unsqueeze(-1).expand_as(x),
            x
        )

        # Count atoms per molecule for mean aggregation
        atom_counts = torch.zeros(num_molecules, device=x.device)
        atom_counts.scatter_add_(0, batch, torch.ones_like(batch, dtype=torch.float))
        molecule_features = molecule_features / atom_counts.unsqueeze(-1).clamp(min=1)

        # Output
        predictions = self.output_network(molecule_features)

        return predictions


class DimeNetPP(nn.Module):
    """
    Simplified DimeNet++ implementation.

    For full implementation, use PyTorch Geometric's DimeNet++.

    Reference:
        Klicpera et al. "Fast and Uncertainty-Aware Directional Message
        Passing for Non-Equilibrium Molecules" (2020)
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        out_channels: int = 1,
        num_blocks: int = 4,
        num_bilinear: int = 8,
        num_spherical: int = 7,
        num_radial: int = 6,
        cutoff: float = 5.0,
        envelope_exponent: int = 5,
        conformer_aggregation: str = "mean",
    ):
        """
        Initialize DimeNet++.

        Note: This is a simplified placeholder. Full DimeNet++ requires
        spherical harmonics and bilinear layers.
        """
        super().__init__()

        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        self.conformer_aggregation = conformer_aggregation

        # Simplified: use MLP instead of full DimeNet architecture
        self.embedding = nn.Linear(100, hidden_channels)  # Atomic number one-hot

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            for _ in range(num_blocks)
        ])

        self.output = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass (simplified)."""
        # One-hot encoding (simplified)
        x = F.one_hot(atomic_numbers.clamp(0, 99), 100).float()
        x = self.embedding(x)

        for layer in self.layers:
            x = x + layer(x)

        # Aggregate
        num_molecules = batch.max().item() + 1
        out = torch.zeros(num_molecules, x.shape[-1], device=x.device, dtype=x.dtype)
        out.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x)

        # Mean
        counts = torch.zeros(num_molecules, device=x.device)
        counts.scatter_add_(0, batch, torch.ones_like(batch, dtype=torch.float))
        out = out / counts.unsqueeze(-1).clamp(min=1)

        return self.output(out)


class SphereNet(nn.Module):
    """
    Simplified SphereNet implementation.

    For full implementation, use PyTorch Geometric's SphereNet.

    Reference:
        Liu et al. "Spherical Message Passing for 3D Molecular Graphs" (2022)
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        out_channels: int = 1,
        num_layers: int = 4,
        cutoff: float = 5.0,
        lmax: int = 2,
        conformer_aggregation: str = "mean",
    ):
        """Initialize SphereNet (simplified)."""
        super().__init__()

        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        self.conformer_aggregation = conformer_aggregation

        # Simplified architecture
        self.embedding = nn.Linear(100, hidden_channels)

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            for _ in range(num_layers)
        ])

        self.output = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass (simplified)."""
        x = F.one_hot(atomic_numbers.clamp(0, 99), 100).float()
        x = self.embedding(x)

        for layer in self.layers:
            x = x + layer(x)

        num_molecules = batch.max().item() + 1
        out = torch.zeros(num_molecules, x.shape[-1], device=x.device, dtype=x.dtype)
        out.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x)

        counts = torch.zeros(num_molecules, device=x.device)
        counts.scatter_add_(0, batch, torch.ones_like(batch, dtype=torch.float))
        out = out / counts.unsqueeze(-1).clamp(min=1)

        return self.output(out)


class GNNWithConformerAggregation(nn.Module):
    """
    Wrapper to add conformer aggregation to any GNN.

    Processes each conformer through the base GNN, then aggregates
    the per-conformer predictions.
    """

    def __init__(
        self,
        base_gnn: nn.Module,
        aggregation: str = "mean",
        use_attention: bool = False,
        attention_hidden_dim: int = 64,
    ):
        """
        Initialize wrapper.

        Args:
            base_gnn: Base GNN model
            aggregation: 'mean', 'max', 'attention', or 'boltzmann'
            use_attention: Whether to use attention aggregation
            attention_hidden_dim: Hidden dim for attention
        """
        super().__init__()

        self.base_gnn = base_gnn
        self.aggregation = aggregation
        self.use_attention = use_attention

        if use_attention:
            # Get output dim from base GNN (assume it's stored)
            out_dim = getattr(base_gnn, 'out_channels', 128)
            self.attention = nn.Sequential(
                nn.Linear(out_dim, attention_hidden_dim),
                nn.ReLU(),
                nn.Linear(attention_hidden_dim, 1),
            )

    def forward(
        self,
        conformer_data: List[Dict],
        energies: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with conformer aggregation.

        Args:
            conformer_data: List of conformer data dicts
            energies: Optional conformer energies for Boltzmann weighting

        Returns:
            Aggregated predictions
        """
        # Process each conformer
        conformer_outputs = []
        for conf in conformer_data:
            out = self.base_gnn(
                conf['atomic_numbers'],
                conf['positions'],
                conf['batch'],
            )
            conformer_outputs.append(out)

        outputs = torch.stack(conformer_outputs, dim=1)  # (batch, n_conf, out_dim)

        # Aggregate
        if self.aggregation == "mean":
            return outputs.mean(dim=1)
        elif self.aggregation == "max":
            return outputs.max(dim=1)[0]
        elif self.use_attention:
            attn = self.attention(outputs).squeeze(-1)
            attn = F.softmax(attn, dim=1)
            return (outputs * attn.unsqueeze(-1)).sum(dim=1)
        elif self.aggregation == "boltzmann" and energies is not None:
            weights = F.softmax(-energies, dim=1)
            return (outputs * weights.unsqueeze(-1)).sum(dim=1)
        else:
            return outputs.mean(dim=1)
