from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class BertOutputStrategy(Enum):
    LAST = "last"
    CONCAT = "concat"
    SUM = "sum"

"""
Using `HfArgumentParser` we can turn these classes
into argparse arguments to be able to specify them on
the command line.
"""

@dataclass
class ModelArguments:
    """
    Arguments about the model/config/tokenizer.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained tokenizer or tokenizer identifier from huggingface.co/models"}
    )

    resize_embeddings: bool = field(
        default=False,
        metadata={"help": "Resize the transformer embeddings dimensions for the introduction of new tokens"}
    )

    bert_hidden_size: int = field(
        default=768,
        metadata={"help": "Hidden size of the bert model."}
    )

    head_hidden_size: int = field(
        default=512,
        metadata={"help": "Hidden size of the head model."}
    )

    linear_hidden_size: int = field(
        default=256,
        metadata={"help": "Linear hidden size of the mention score model."}
    )

    dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout probability."}
    )

    num_output: int = field(
        default=3,
        metadata={"help": "The dimension of the model output (e.g., number of classes for classification)."}
    )

    output_strategy: BertOutputStrategy = field(
        default="last",
        metadata={"help": "How to manage the transformer output."}
    )


@dataclass
class CustomTrainingArguments:
    """
    Arguments pertaining the training and evaluation process.
    """
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

    task_type: int = field(
        metadata={"help": "The type of the coreference resolution task: 3 is entity resolution; 2 is entity identification and resolution; 1 is end-to-end coreference resolution."}
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

    use_early_stopping: bool = field(
        default=False, metadata={"help": "Whether to use early stopping."}
    )

    early_stopping_mode: str = field(
        default="max", metadata={"help": "Early stopping mode."},
    )

    early_stopping_patience: int = field(
        default=0, metadata={"help": "How many epochs to wait before early stop if no progress on the validation set."}
    )

    use_scaler: bool = field(
        default=True, metadata={"help": "If you want to use gradient scaling or not."}
    )

    

    
    