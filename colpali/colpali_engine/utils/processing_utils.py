from abc import ABC, abstractmethod
from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature

from colpali_engine.utils.maxsim import maxsim_inbatch
from colpali_engine.utils.torch_utils import get_torch_device


class BaseVisualRetrieverProcessor(ABC):
    query_prefix: ClassVar[str] = ""

    @abstractmethod
    def process_images(self, images: List[Image.Image]) -> Union[BatchFeature, BatchEncoding]:
        pass

    @abstractmethod
    def process_texts(self, texts: List[str]) -> Union[BatchFeature, BatchEncoding]:
        pass

    def process_queries(
        self,
        texts: Optional[List[str]] = None,
        queries: Optional[List[str]] = None,
        max_length: int = 50,
        contexts: Optional[List[str]] = None,
        suffix: Optional[str] = None,
    ) -> Union[BatchFeature, BatchEncoding]:
        if queries is not None:
            texts = queries
        if texts is None:
            raise ValueError("No texts or queries provided.")
        if suffix is None:
            suffix = self.query_augmentation_token * 10
        texts = [self.query_prefix + text + suffix for text in texts]
        return self.process_texts(texts=texts)

    @abstractmethod
    def score(
        self,
        qs: Union[torch.Tensor, List[torch.Tensor]],
        ps: Union[torch.Tensor, List[torch.Tensor]],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        pass

    @staticmethod
    def score_multi_vector(
        qs: Union[torch.Tensor, List[torch.Tensor]],
        ps: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: int = 128,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        device = device or get_torch_device("auto")
        scores_list: List[torch.Tensor] = []
        for i in range(0, len(qs), batch_size):
            scores_batch = []
            qs_batch = torch.nn.utils.rnn.pad_sequence(
                qs[i : i + batch_size], batch_first=True, padding_value=0
            ).to(device)
            for j in range(0, len(ps), batch_size):
                ps_batch = torch.nn.utils.rnn.pad_sequence(
                    ps[j : j + batch_size], batch_first=True, padding_value=0
                ).to(device)
                scores_batch.append(maxsim_inbatch(qs_batch, ps_batch))
            scores_list.append(torch.cat(scores_batch, dim=1).cpu())
        return torch.cat(scores_list, dim=0).to(torch.float32)

    @abstractmethod
    def get_n_patches(self, image_size: Tuple[int, int], *args, **kwargs) -> Tuple[int, int]:
        pass
