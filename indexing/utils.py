

import torch

from indexing.candidate_index import CandidateIndex, TopKModule
from indexing.mips_top_k import MIPSBruteForceTopK


def get_top_k_module(top_k_method: str, model: torch.nn.Module, item_embeddings: torch.Tensor, item_ids: torch.Tensor) -> TopKModule:
    if top_k_method == "MIPSBruteForceTopK":
        top_k_module = MIPSBruteForceTopK(
            item_embeddings=item_embeddings,
            item_ids=item_ids,
        )
    else:
        raise ValueError(f"Invalid top-k method {top_k_method}")
    return top_k_module
