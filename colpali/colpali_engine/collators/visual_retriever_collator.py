import random
from typing import Any, Dict, List, Union

import torch
from PIL.Image import Image

from colpali_engine.data.dataset import ColPaliEngineDataset
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor

N_AUGMENTATION_TOKENS = 10


def prefix_keys(data: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in data.items()}


def _maybe_stack(examples, key, dtype):
    vals = [ex.get(key) for ex in examples]
    if all(v is None for v in vals):
        return None
    return torch.tensor(vals, dtype=dtype)


class VisualRetrieverCollator:
    query_prefix = "query_"
    pos_doc_prefix = "doc_"
    neg_doc_prefix = "neg_doc_"

    def __init__(self, processor: BaseVisualRetrieverProcessor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        queries, pos_targets, neg_targets, listwise_grades = [], [], [], []
        for example in examples:
            query = example[ColPaliEngineDataset.QUERY_KEY]
            queries.append(random.choice(query) if isinstance(query, list) else query)
            pos_tgt = example[ColPaliEngineDataset.POS_TARGET_KEY]
            pos_targets.append(random.choice(pos_tgt) if isinstance(pos_tgt, list) else pos_tgt)
            neg_tgt = example.get(ColPaliEngineDataset.NEG_TARGET_KEY)
            if neg_tgt is not None:
                neg_targets.append(neg_tgt)
            grades = example.get(ColPaliEngineDataset.LISTWISE_GRADES_KEY)
            if grades is not None:
                listwise_grades.append(grades)

        queries = [
            self.processor.query_prefix + q + self.processor.query_augmentation_token * N_AUGMENTATION_TOKENS
            for q in queries
        ]
        batch = {
            **self.auto_collate(queries, self.query_prefix),
            **self.auto_collate(pos_targets, self.pos_doc_prefix),
            **(self.auto_collate(neg_targets, self.neg_doc_prefix) if neg_targets else {}),
        }
        if listwise_grades:
            batch[ColPaliEngineDataset.LISTWISE_GRADES_KEY] = torch.tensor(listwise_grades, dtype=torch.float32)
        mask = _maybe_stack(examples, ColPaliEngineDataset.MASK_KEY, torch.float32)
        if mask is not None:
            batch[ColPaliEngineDataset.MASK_KEY] = mask
        weight = _maybe_stack(examples, ColPaliEngineDataset.WEIGHT_KEY, torch.float32)
        if weight is not None:
            batch[ColPaliEngineDataset.WEIGHT_KEY] = weight
        group = _maybe_stack(examples, ColPaliEngineDataset.GROUP_KEY, torch.long)
        if group is not None:
            batch[ColPaliEngineDataset.GROUP_KEY] = group
        return batch

    def auto_collate(self, batch: List[Union[str, Image]], key_prefix: str = "") -> Dict[str, Any]:
        if isinstance(batch[0], str):
            proc_batch = self.processor.process_texts(texts=batch)
        elif isinstance(batch[0], Image):
            proc_batch = self.processor.process_images(images=batch)
        elif isinstance(batch[0], list):
            batch_size = len(batch)
            if isinstance(batch[0][0], str):
                all_items = [t for texts in batch for t in texts]
                num_negatives = len(all_items) // batch_size
                proc_batch = self.processor.process_texts(texts=all_items)
            else:
                all_items = [img for imgs in batch for img in imgs]
                num_negatives = len(batch[0])
                proc_batch = self.processor.process_images(images=all_items)
            for k, v in proc_batch.items():
                # Qwen-VL pixel_values is a variable-length concat; only view tensors that are one row per image.
                if isinstance(v, torch.Tensor) and v.shape[0] == batch_size * num_negatives:
                    proc_batch[k] = v.view(batch_size, num_negatives, *v.shape[1:])
        else:
            raise ValueError(f"Unsupported batch type: {type(batch[0])}")
        return prefix_keys(proc_batch, key_prefix)
