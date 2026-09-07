import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union

from peft import LoraConfig, PeftModel, get_peft_model
from transformers import PreTrainedModel, TrainingArguments

from colpali_engine.collators import VisualRetrieverCollator
from colpali_engine.data.dataset import ColPaliEngineDataset
from colpali_engine.loss.late_interaction_losses import ColbertLoss
from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


@dataclass
class ColModelTrainingConfig:
    model: Union[PreTrainedModel, PeftModel]
    processor: BaseVisualRetrieverProcessor
    train_dataset: Union[ColPaliEngineDataset, List[ColPaliEngineDataset]]
    eval_dataset: Optional[Union[ColPaliEngineDataset, Dict[str, ColPaliEngineDataset]]] = None
    tr_args: Optional[TrainingArguments] = None
    output_dir: Optional[str] = None
    max_length: int = 256
    run_eval: bool = True
    run_train: bool = True
    peft_config: Optional[LoraConfig] = None
    loss_func: Optional[Callable] = ColbertLoss()
    pretrained_peft_model_name_or_path: Optional[str] = None
    trainer_cls: Optional[Type[ContrastiveTrainer]] = None
    trainer_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.output_dir is None:
            raise ValueError(
                "ColModelTrainingConfig.output_dir is required "
                "(do not dump into ./models or the Python env)"
            )
        if self.tr_args is None:
            self.tr_args = TrainingArguments(output_dir=self.output_dir)
        elif self.tr_args.output_dir is None or self.tr_args.output_dir == "trainer_output":
            self.tr_args.output_dir = self.output_dir
        if isinstance(self.tr_args.learning_rate, str):
            self.tr_args.learning_rate = float(self.tr_args.learning_rate)
        self.tr_args.remove_unused_columns = False
        if self.pretrained_peft_model_name_or_path is not None:
            self.model.load_adapter(self.pretrained_peft_model_name_or_path, is_trainable=True)
        elif self.peft_config is not None:
            self.model = get_peft_model(self.model, self.peft_config)
            self.model.print_trainable_parameters()


class ColModelTraining:
    def __init__(self, config: ColModelTrainingConfig) -> None:
        self.config = config
        self.model = config.model
        self.train_dataset = config.train_dataset
        self.eval_dataset = config.eval_dataset
        self.collator = VisualRetrieverCollator(processor=config.processor, max_length=config.max_length)

    def train(self) -> None:
        trainer_cls = self.config.trainer_cls or ContrastiveTrainer
        trainer = trainer_cls(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            args=self.config.tr_args,
            data_collator=self.collator,
            loss_func=self.config.loss_func,
            is_vision_model=self.config.processor is not None,
            **self.config.trainer_kwargs,
        )
        trainer.args.remove_unused_columns = False
        trainer.train(resume_from_checkpoint=self.config.tr_args.resume_from_checkpoint)

    def save(self):
        rank = (os.environ.get("RANK") or "").strip()
        if rank.isdigit() and int(rank) != 0:
            return
        self.model.save_pretrained(self.config.output_dir)
        self.config.processor.save_pretrained(self.config.output_dir)
