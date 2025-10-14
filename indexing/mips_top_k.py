

from typing import Tuple

import torch

from indexing.candidate_index import TopKModule


class MIPSTopKModule(TopKModule):

    def __init__(
        self,
        item_embeddings: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> None:
        """
        Args:
            item_embeddings: (1, X, D)
            item_ids: (1, X,)
        """
        super().__init__()

        self._item_embeddings: torch.Tensor = item_embeddings
        self._item_ids: torch.Tensor = item_ids


class MIPSBruteForceTopK(MIPSTopKModule):

    def __init__(
        self,
        item_embeddings: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> None:
        super().__init__(
            item_embeddings=item_embeddings,
            item_ids=item_ids,
        )
        del self._item_embeddings
        self._item_embeddings_t: torch.Tensor = item_embeddings.permute(2, 1, 0).squeeze(2)

    def forward(
        self,
        query_embeddings: torch.Tensor,
        k: int,
        sorted: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_embeddings: (B, ...). Implementation-specific.
            k: int. final top-k to return.
            sorted: bool. whether to sort final top-k results or not.

        Returns:
            Tuple of (top_k_scores x float, top_k_ids x int), both of shape (B, K,)
        """
        # (B, X,)
        all_logits = torch.mm(query_embeddings, self._item_embeddings_t)
        top_k_logits, top_k_indices = torch.topk(
            all_logits, dim=1, k=k, sorted=sorted, largest=True,
        )  # (B, k,)
        return top_k_logits, self._item_ids.squeeze(0)[top_k_indices]
