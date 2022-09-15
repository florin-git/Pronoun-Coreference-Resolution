from transformers import (
    AutoTokenizer,
    BertModel,
    logging
)

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader

import os
import random
import yaml

from typing import *

from trainer import Trainer
from arguments import *
from gap_dataset import *
from gap_utils import *


SEED = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


logging.set_verbosity_error()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yaml_file = "./train.yaml"
# Read configuration file with all the necessary parameters
with open(yaml_file) as file:
    config = yaml.safe_load(file)

training_args = CustomTrainingArguments(**config['training_args'])
model_args = ModelArguments(**config['model_args'])
model_name_or_path = model_args.model_name_or_path

train_clean_path = "../../model/data/train_clean.tsv"
valid_clean_path = "../../model/data/valid_clean.tsv"

df_train = pd.read_csv(filepath_or_buffer=train_clean_path, sep="\t")
df_valid = pd.read_csv(filepath_or_buffer=valid_clean_path, sep="\t")


tag_labels = {
    "pronoun_tag": "<p>",
    "start_A_tag": "<a>",
    "end_A_tag": "</a>",
    "start_B_tag": "<b>",
    "end_B_tag": "</b>"
}

tokenizer_name_or_path = model_args.tokenizer
if tokenizer_name_or_path is None:
    tokenizer_name_or_path = model_name_or_path

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_name_or_path, never_split=list(tag_labels.values()))
tokenizer.add_tokens(list(tag_labels.values()), special_tokens=True)

train_ds = GAP_Dataset(df_train, tokenizer, tag_labels, truncate_up_to_pron=False)
valid_ds = GAP_Dataset(df_valid, tokenizer, tag_labels, truncate_up_to_pron=False)


class Entity_Resolution_Head(nn.Module):
    def __init__(self, bert_hidden_size: int, args: ModelArguments):
        super().__init__()

        self.args = args
        self.bert_hidden_size = bert_hidden_size


        input_size_pronoun = bert_hidden_size
        input_size_entities = bert_hidden_size * 6
        if self.args.output_strategy == "concat":
            input_size_pronoun *= 4
            input_size_entities *= 4

        self.ffnn_pronoun = nn.Sequential(
            nn.Linear(input_size_pronoun, bert_hidden_size),
            nn.LayerNorm(bert_hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(bert_hidden_size, args.head_hidden_size),
        )

        self.ffnn_entities = nn.Sequential(
            nn.Linear(input_size_entities, bert_hidden_size),
            nn.LayerNorm(bert_hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(bert_hidden_size, args.head_hidden_size),
        )


        linear_hidden_size = args.linear_hidden_size

        # self.bilinear = torch.nn.Bilinear(args.head_hidden_size, args.head_hidden_size, bilinear_hidden_size, bias=False)
        # self.bilinear = torch.nn.Bilinear(bert_hidden_size, bert_hidden_size, bilinear_hidden_size, bias=False)

        self.linear = torch.nn.Linear(args.head_hidden_size * 2, 
                                      linear_hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(args.dropout)
        self.classifier = nn.Linear(linear_hidden_size, args.num_output)

    def forward(self, bert_outputs, offsets):

        pron_emb, ent_emb = self._retrieve_pron_and_ent_embeddings(
            bert_outputs, offsets)

        x_pron = self.ffnn_pronoun(pron_emb)
        x_ent = self.ffnn_entities(ent_emb)

        # x = self.bilinear(x_pron, x_ent)
        x = self.linear(torch.cat([x_pron, x_ent], dim=-1))
        x = self.activation(x)
        x = self.dropout(x)

        output = self.classifier(x)
        return output

    def _retrieve_pron_and_ent_embeddings(self, bert_embeddings: torch.Tensor, 
                                          offsets: torch.Tensor):
        embeddings_A = []
        embeddings_B = []
        pronouns_embedding = []

        # Consider embeddings and offsets in each batch separately
        for embeddings, off in zip(bert_embeddings, offsets):
            start_A = off[0]
            end_A = off[1]
            start_B = off[2]
            end_B = off[3]
            pron_off = off[-1]

            # The representation of an entity is the concatenation
            # of the embedding of its first subtoken, its last subtoken
            # and the average of all the subtokens embeddings that 
            # compose the entire entity 
            emb_A = torch.cat([embeddings[start_A], embeddings[end_A-1],
                               embeddings[start_A:end_A].mean(dim=0)], dim=-1)
            embeddings_A.append(emb_A)

            emb_B = torch.cat([embeddings[start_B], embeddings[end_B-1],
                              embeddings[start_B:end_B].mean(dim=0)], dim=-1)
            embeddings_B.append(emb_B)

            pronouns_embedding.append(embeddings[pron_off])

        # batch_size x (embedding_dim * 3 * 2)
        merged_entities_embeddings = torch.cat([
            torch.stack(embeddings_A, dim=0),
            torch.stack(embeddings_B, dim=0),
        ], dim=1)

        # batch_size x embedding_dim
        stacked_pronouns_embedding = torch.stack(pronouns_embedding, dim=0)
        return stacked_pronouns_embedding, merged_entities_embeddings


class CR_Model(nn.Module):
    """The main model."""

    def __init__(self, bert_model: str, tokenizer, args: ModelArguments):
        super().__init__()

        self.args = args

        if bert_model in {"bert-base-uncased", "bert-base-cased"}:
            self.bert_hidden_size = 768
        elif bert_model in {"bert-large-uncased", "bert-large-cased"}:
            self.bert_hidden_size = 1024
        else:
            self.bert_hidden_size = args.bert_hidden_size

        self.bert = BertModel.from_pretrained(
            bert_model).to(device, non_blocking=True)

        # If the tag tokens (e.g., <p>, <a> etc.) are present in the features,
        # the embedding dimension of the bert embeddings must be changed
        # to be compliant with the new size of the tokenizer vocabulary. 
        if args.resize_embeddings:
            self.bert.resize_token_embeddings(len(tokenizer.vocab))

        self.head = Entity_Resolution_Head(self.bert_hidden_size, args).to(
            device, non_blocking=True)

    def forward(self, sample):
        x = sample['features']
        x_offsets = sample['offsets']

        bert_outputs = self.bert(
            x, attention_mask=(x > 0).long(),
            token_type_ids=None, output_hidden_states=True)

        if self.args.output_strategy == "last":
            out = bert_outputs.last_hidden_state

        elif self.args.output_strategy == "concat":
            out = torch.cat([bert_outputs.hidden_states[x] for x in [-1, -2, -3, -4]], dim=-1)

        elif self.args.output_strategy == "sum":
            layers_to_sum = torch.stack([bert_outputs.hidden_states[x] for x in [-1, -2, -3, -4]], dim=0)
            out = torch.sum(layers_to_sum, dim=0)

        else:
            raise ValueError("Unsupported output strategy.")
        head_outputs = self.head(out, x_offsets)
        return head_outputs


def predict(model, sentences: List[Dict], tokenizer, tag_labels, device) -> List[Tuple[Tuple[str, int], Tuple[str, int]]]:
    df = pd.DataFrame(sentences)

    # tokenizer will be self.tokenizer in the final implementation
    dataset = GAP_Dataset(df, tokenizer, tag_labels, truncate_up_to_pron=False, labeled=False, cleaned=False)
    collator = Collator(device, labeled=False)

    predictions = []

    model.eval()
    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size=1,
                                collate_fn=collator, shuffle=False)

        for sample, sentence in zip(dataloader, sentences):

            predicted_label_id = model(sample).argmax(1).item()
            pred_entity, pred_entity_offset = get_entity_and_offset_from_id(
                predicted_label_id, sentence)
            pron, pron_offset = sentence['pron'], sentence['p_offset']

            predictions.append(
                ((pron, pron_offset), (pred_entity, pred_entity_offset)))

    return predictions


model = CR_Model(model_name_or_path, tokenizer,
                 model_args).to(device, non_blocking=True)

# last_frozen_layer = 4
# modules = [model.bert.embeddings, *model.bert.encoder.layer[:last_frozen_layer]]
# modules = [*model.bert.encoder.layer[:last_frozen_layer]]
# freeze_weights(modules)

batch_size = 4

collator = Collator(device)
train_dataloader = DataLoader(train_ds, batch_size=batch_size,
                              collate_fn=collator, shuffle=True)
valid_dataloader = DataLoader(valid_ds, batch_size=batch_size,
                              collate_fn=collator, shuffle=False)


# Make sure that the learning rate is read as a number and not as a string
training_args.learning_rate = float(training_args.learning_rate)
print(training_args)

criterion = torch.nn.CrossEntropyLoss().to(device=device, non_blocking=True)
optimizer = torch.optim.Adam(
    model.parameters(), lr=training_args.learning_rate)
# scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
# scheduler = None


trainer = Trainer(str(device), model, training_args,
                  train_dataloader, valid_dataloader,
                  criterion, optimizer, scheduler)

trainer.train()
