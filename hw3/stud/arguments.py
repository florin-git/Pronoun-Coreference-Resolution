from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=False, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    verbose_logging: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to log verbose messages or not."},
    )
    alpha: Optional[float] = field(
        default=0.1,
        metadata={"help": "loss_cls + alpha * loss_ctc"},
    )
    tokenizer: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained tokenizer"}
    )


@dataclass
class CustomTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

    resume_from_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "The path to a file with a valid checkpoint for your model."}
    )

    save_model: bool = field(
        default=False, metadata={"help": "Whether you want to save the model during training."}
    )

    num_train_epochs: int = field(
        default=3, metadata={"help": "Total number of training epochs to perform."}
    )

    logging_steps: int = field(
        default=250, metadata={"help": "Log every X updates steps."}
    )

    learning_rate: float = field(
        default=0.0005, metadata={"help": "The initial learning rate."}
    )

    grad_clipping: Optional[float] = field(
        default=None, metadata={"help": "Clip value if you want to use gradient clipping."}
    )

    early_stopping: bool = field(
        default=False, metadata={"help": "Whether to use early stopping."}
    )

    early_stopping_mode: str = field(
        default="max", metadata={"help": "Early stopping mode, selected in ['min', 'max']."},
    )

    early_stopping_patience: int = field(
        default=0, metadata={"help": "How many epochs to wait before early stop if no progress on the validation set."}
    )

    

    
    