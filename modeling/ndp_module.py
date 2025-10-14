

import abc
from typing import Dict, Optional, Tuple

import torch


class NDPModule(torch.nn.Module):

    def forward(
        self,
        input_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        item_sideinfo: Optional[torch.Tensor],
        precomputed_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_embeddings: (B, input_embedding_dim) x float
            item_embeddings: (1/B, X, item_embedding_dim) x float
            item_sideinfo: (1/B, X, item_sideinfo_dim) x float
        
        Returns:
            Tuple of (B, X,) similarity values, keyed outputs
        """
        pass
