import random
from typing import Any, Dict, List, Optional, Union

from PIL import Image
from torch.utils.data import Dataset

Document = Union[str, Image.Image]


class Corpus:
    def __init__(
        self,
        corpus_data: List[Dict[str, Any]],
        docid_to_idx_mapping: Optional[Dict[str, int]] = None,
        doc_column_name: str = "doc",
    ):
        self.corpus_data = corpus_data
        self.docid_to_idx_mapping = docid_to_idx_mapping
        self.doc_column_name = doc_column_name

    def __len__(self) -> int:
        return len(self.corpus_data)

    def retrieve(self, docid: Any) -> Document:
        idx = self.docid_to_idx_mapping[docid] if self.docid_to_idx_mapping is not None else docid
        return self.corpus_data[idx][self.doc_column_name]


class ColPaliEngineDataset(Dataset):
    QUERY_KEY = "query"
    POS_TARGET_KEY = "pos_target"
    NEG_TARGET_KEY = "neg_target"
    LISTWISE_GRADES_KEY = "listwise_grades"
    MASK_KEY = "hardneg_slot_mask"
    POSITIVE_IDS_COLUMN = "positive_doc_ids"
    MASK_COLUMN = "negative_mask"
    WEIGHT_COLUMN = "sample_weight"
    GROUP_COLUMN = "query_group_id"
    WEIGHT_KEY = "sample_weight"
    GROUP_KEY = "query_group_id"

    def __init__(
        self,
        data: List[Dict[str, Any]],
        corpus: Optional[Corpus] = None,
        query_column_name: str = "query",
        pos_target_column_name: str = "pos_target",
        neg_target_column_name: str = None,
        num_negatives: int = 3,
        listwise_doc_ids_column_name: str = None,
        listwise_grades_column_name: str = None,
    ):
        self.data = data
        self.corpus = corpus
        self.query_column_name = query_column_name
        self.pos_target_column_name = pos_target_column_name
        self.neg_target_column_name = neg_target_column_name
        self.listwise_doc_ids_column_name = listwise_doc_ids_column_name
        self.listwise_grades_column_name = listwise_grades_column_name
        self.num_negatives = num_negatives

        names = getattr(self.data, "column_names", None) or list(self.data[0].keys())
        has_pool = self.POSITIVE_IDS_COLUMN in names
        has_mask = self.MASK_COLUMN in names
        has_group = self.GROUP_COLUMN in names
        # all-pos: one fixed positive per row + mask + group/weight. judged-pos: sample from a pool.
        self.all_pos = has_mask and not has_pool and has_group
        self.judged_pos = has_pool and has_mask and not self.all_pos

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        query = sample[self.query_column_name]
        pos_targets = sample[self.pos_target_column_name]
        if not isinstance(pos_targets, list):
            pos_targets = [pos_targets]
        neg_targets = None
        if self.neg_target_column_name is not None:
            neg_targets = sample[self.neg_target_column_name]
            if not isinstance(neg_targets, list):
                neg_targets = [neg_targets]

        if self.listwise_doc_ids_column_name is not None:
            doc_ids = sample[self.listwise_doc_ids_column_name]
            grades = sample[self.listwise_grades_column_name]
            return {
                self.QUERY_KEY: query,
                self.POS_TARGET_KEY: [self.corpus.retrieve(doc_ids[0])],
                self.NEG_TARGET_KEY: [self.corpus.retrieve(doc_id) for doc_id in doc_ids[1:]],
                self.LISTWISE_GRADES_KEY: grades,
            }

        if self.judged_pos:
            pool = [int(x) for x in sample[self.POSITIVE_IDS_COLUMN]]
            k = self.num_negatives or len(sample[self.MASK_COLUMN])
            mask = [int(x) for x in sample[self.MASK_COLUMN][:k]]
            neg_ids = [int(x) for x in sample[self.neg_target_column_name][:k]]
            return {
                self.QUERY_KEY: query,
                self.POS_TARGET_KEY: [self.corpus.retrieve(random.choice(pool))],
                self.NEG_TARGET_KEY: [self.corpus.retrieve(doc_id) for doc_id in neg_ids],
                self.MASK_KEY: mask,
            }

        if self.all_pos:
            chosen = int(pos_targets[0])
            k = self.num_negatives or len(sample[self.MASK_COLUMN])
            mask = [int(x) for x in sample[self.MASK_COLUMN][:k]]
            neg_ids = [int(x) for x in neg_targets[:k]]
            return {
                self.QUERY_KEY: query,
                self.POS_TARGET_KEY: [self.corpus.retrieve(chosen)],
                self.NEG_TARGET_KEY: [self.corpus.retrieve(doc_id) for doc_id in neg_ids],
                self.MASK_KEY: mask,
                self.WEIGHT_KEY: float(sample[self.WEIGHT_COLUMN]),
                self.GROUP_KEY: int(sample[self.GROUP_COLUMN]),
            }

        if self.corpus is not None:
            pos_targets = [self.corpus.retrieve(doc_id) for doc_id in pos_targets]
            if neg_targets is not None:
                if len(neg_targets) > self.num_negatives:
                    neg_targets = neg_targets[: self.num_negatives]
                neg_targets = [self.corpus.retrieve(doc_id) for doc_id in neg_targets]

        return {
            self.QUERY_KEY: query,
            self.POS_TARGET_KEY: pos_targets,
            self.NEG_TARGET_KEY: neg_targets,
        }

    def take(self, n: int) -> "ColPaliEngineDataset":
        return self.__class__(
            self.data.take(n),
            self.corpus,
            self.query_column_name,
            self.pos_target_column_name,
            self.neg_target_column_name,
            self.num_negatives,
            self.listwise_doc_ids_column_name,
            self.listwise_grades_column_name,
        )
