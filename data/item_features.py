

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class ItemFeatures:
    num_items: int
    max_jagged_dimension: int
    max_ind_range: List[int]     # [(,)] x num_features
    lengths: List[torch.Tensor]  # [(num_items,)] x num_features
    values: List[torch.Tensor]   # [(num_items, max_jagged_dimension)] x num_features
