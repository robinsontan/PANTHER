

import os
from typing import Optional, Tuple

import gin
import torch

from trainer.data_collator import behseq_iceberg_collator
from trainer.tokenizer import get_tokenizer


@gin.configurable
def create_data_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    world_size: int,
    rank: int,
    shuffle: bool,
    prefetch_factor: int = 128,
    num_workers: int = os.cpu_count(),
    drop_last: bool = False,
    tokenizer_path: str = "prune95",
    max_sequence_length: int = 256,
    n_targets: int = 1,
) -> Tuple[Optional[torch.utils.data.distributed.DistributedSampler], torch.utils.data.DataLoader]:
    # print(f"num_workers={num_workers}")
    if shuffle:
        sampler = None
        pass
        # sampler = torch.utils.data.distributed.DistributedSampler(
        #     dataset,
        #     num_replicas=world_size,
        #     rank=rank,
        #     shuffle=True,
        #     seed=0,
        #     drop_last=drop_last,
        # )
    else:
        sampler = None
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        # shuffle=True, cannot use with sampler
        num_workers=num_workers,
        sampler=sampler,
        prefetch_factor=prefetch_factor,
        persistent_workers=False,
        drop_last=drop_last,
    )
    data_loader.collate_fn = behseq_iceberg_collator(
        tokenizer= get_tokenizer(path=tokenizer_path),
        ignore_last_n=1, 
        padding_length=max_sequence_length+1, 
        chronological=True, 
        shift_id_by=0,
        n_targets=n_targets
    )

    return sampler, data_loader
