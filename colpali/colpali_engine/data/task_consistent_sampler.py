"""Distributed sampler that keeps every global micro-batch in one task."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterator, Sequence

import torch
from torch.utils.data import Sampler


class TaskConsistentDistributedSampler(Sampler[int]):
    """Yield rank-local slices of globally task-consistent batches.

    With ``shard_by_rank=True``, every process selects its fixed local slice.
    HuggingFace Trainer instead passes ``False`` because Accelerate shards the
    DataLoader's consecutive local batches itself. In both modes one optimizer
    microstep sees one source globally. Small task tails are dropped when
    ``drop_last`` is true; otherwise they are padded within the same task.
    """

    def __init__(
        self,
        task_ids: Sequence[int],
        local_batch_size: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
        drop_last: bool = True,
        shard_by_rank: bool = True,
    ):
        if local_batch_size <= 0 or num_replicas <= 0:
            raise ValueError("batch size and num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} outside [0, {num_replicas})")
        self.task_ids = task_ids
        self.local_batch_size = int(local_batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.shard_by_rank = bool(shard_by_rank)
        self.epoch = 0
        self.global_batch_size = self.local_batch_size * self.num_replicas

        counts: dict[int, int] = defaultdict(int)
        for task_id in task_ids:
            counts[int(task_id)] += 1
        if not counts:
            raise ValueError("task_ids must not be empty")
        if self.drop_last:
            global_batches = sum(count // self.global_batch_size for count in counts.values())
        else:
            global_batches = sum(
                math.ceil(count / self.global_batch_size) for count in counts.values()
            )
        if global_batches <= 0:
            raise ValueError(
                f"no task has enough rows for a global batch of {self.global_batch_size}"
            )
        self.num_samples = global_batches * (
            self.local_batch_size if self.shard_by_rank else self.global_batch_size
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        groups: dict[int, list[int]] = defaultdict(list)
        for index, task_id in enumerate(self.task_ids):
            groups[int(task_id)].append(index)

        global_batches: list[list[int]] = []
        for task_id in sorted(groups):
            indices = groups[task_id]
            order = torch.randperm(len(indices), generator=generator).tolist()
            shuffled = [indices[i] for i in order]
            if self.drop_last:
                usable = len(shuffled) // self.global_batch_size * self.global_batch_size
                shuffled = shuffled[:usable]
            else:
                target = math.ceil(len(shuffled) / self.global_batch_size) * self.global_batch_size
                if target > len(shuffled):
                    repeats = (target - len(shuffled) + len(shuffled) - 1) // len(shuffled)
                    shuffled.extend((shuffled * repeats)[: target - len(shuffled)])
            global_batches.extend(
                shuffled[start : start + self.global_batch_size]
                for start in range(0, len(shuffled), self.global_batch_size)
            )

        batch_order = torch.randperm(len(global_batches), generator=generator).tolist()
        start = self.rank * self.local_batch_size
        stop = start + self.local_batch_size
        for batch_index in batch_order:
            batch = global_batches[batch_index]
            yield from batch[start:stop] if self.shard_by_rank else batch

