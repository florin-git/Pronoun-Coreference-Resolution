import os
from transformers import (
    Trainer,
    TrainingArguments,
    BertTokenizer,
    BertModel
)
import transformers
import numpy as np
import pandas as pd

import random
import yaml
from tqdm import tqdm
from typing import *

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler

SEED = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# Display the entire text
pd.set_option("display.max_colwidth", None)


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_clean_path = "../../model/data/train_clean.tsv"
valid_clean_path = "../../model/data/valid_clean.tsv"

df_train = pd.read_csv(filepath_or_buffer=train_clean_path, sep="\t")
df_valid = pd.read_csv(filepath_or_buffer=valid_clean_path, sep="\t")


model_name_or_path = "bert-base-uncased"


def get_class_label(is_coref_A: str, is_coref_B: str):
    if is_coref_A == "TRUE" or is_coref_A is True:
        return 0
    elif is_coref_B == "TRUE" or is_coref_B is True:
        return 1
    else:
        return 2


FEMININE = 0
MASCULINE = 1
UNKNOWN = 2


def get_gender(pronoun: str):
    gender_mapping = {
        'she': FEMININE,
        'her': FEMININE,
        'he': MASCULINE,
        'his': MASCULINE,
        'him': MASCULINE,
    }

    return gender_mapping.get(pronoun.lower(), UNKNOWN)


FEMININE = 0
MASCULINE = 1
UNKNOWN = 2


def get_gender(pronoun: str):
    gender_mapping = {
        'she': FEMININE,
        'her': FEMININE,
        'he': MASCULINE,
        'his': MASCULINE,
        'him': MASCULINE,
    }

    return gender_mapping.get(pronoun.lower(), UNKNOWN)


class GAPDataset(Dataset):
    """Custom GAP Dataset class"""

    def __init__(self, df, tokenizer, labeled=True):
        self.df = df

        self.labeled = labeled
        self.tokenizer = tokenizer
        self.offsets, self.tokens = [], []

        if labeled:
            self.labels = df.target.values.astype("uint8")

        self._convert_tokens_to_ids()

#     @staticmethod
#     def get_class_label(is_coref_A: str, is_coref_B: str):
#         if is_coref_A == "TRUE":
#                 return 0
#         elif is_coref_B == "TRUE":
#             return 1
#         else:
#             return 2

    def _convert_tokens_to_ids(self):
        CLS = [self.tokenizer.cls_token]
        SEP = [self.tokenizer.sep_token]

        for _, row in self.df.iterrows():
            tokens, offsets = self._tokenize(row)
            self.offsets.append(offsets)
            self.tokens.append(self.tokenizer.convert_tokens_to_ids(
                CLS + tokens + SEP))

    def _tokenize(self, row):
        # The order is important because we want that the pronoun comes after all the
        # coreferenced entities in the output, even if B could come after the pronoun.
        break_points = sorted([
            ("A", row['offset_A'], row['entity_A']),
            ("B", row['offset_B'], row['entity_B']),
            ("P", row['p_offset'], row['pron'])
        ], key=lambda x: x[0])

        tokens, spans, current_pos = [], {}, 0
        for name, offset, text in break_points:
            tokens.extend(self.tokenizer.tokenize(
                row["text"][current_pos:offset]))
            # Make sure we do not get it wrong
            assert row["text"][offset:offset+len(text)] == text
            # Tokenize the target
            tmp_tokens = self.tokenizer.tokenize(
                row["text"][offset:offset+len(text)])

            # [num_tokens until entity, num_tokens including the entity]
            spans[name] = [len(tokens), len(tokens) +
                           len(tmp_tokens) - 1]  # inclusive
            # In the last iteration, the pronoun is appended to the end
            tokens.extend(tmp_tokens)
            current_pos = offset + len(text)

        tokens.extend(self.tokenizer.tokenize(row["text"][current_pos:offset]))

        # The pronoun is a single token, so the span is the same
        assert spans["P"][0] == spans["P"][1]
        return tokens, (spans["A"] + spans["B"] + [spans["P"][0]])

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        if self.labeled:
            return self.tokens[idx], self.offsets[idx], self.labels[idx]
        return self.tokens[idx], self.offsets[idx], None


df_train['target'] = [get_class_label(is_coref_A, is_coref_B) for is_coref_A, is_coref_B in zip(
    df_train['is_coref_A'],  df_train['is_coref_B'])]
df_valid['target'] = [get_class_label(is_coref_A, is_coref_B) for is_coref_A, is_coref_B in zip(
    df_valid['is_coref_A'],  df_valid['is_coref_B'])]

tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
train_ds = GAPDataset(df_train, tokenizer)
valid_ds = GAPDataset(df_valid, tokenizer)


def collate_batch(batch, truncate_len=400):
    """Batch preparation.

    1. Pad the sequences
    2. Transform the target.
    """
    batch_features, batch_offsets, batch_labels = zip(*batch)

    max_len = min(
        max((len(x) for x in batch_features)),
        truncate_len
    )

    # Features
    features = np.zeros((len(batch), max_len), dtype=np.int64)

    # Padding
    for i, row in enumerate(batch_features):
        features[i, :len(row)] = row

    features_tensor = torch.tensor(features, device=device)

    # Offsets
    offsets_tensor = torch.stack([
        torch.tensor(x, dtype=torch.int64, device=device) for x in batch_offsets
    ], dim=0) + 1  # Account for the [CLS] token

    # Labels
    if batch_labels[0] is None:
        return features_tensor, offsets_tensor, None

    labels_tensor = torch.tensor(
        batch_labels, dtype=torch.uint8, device=device)
    return features_tensor, offsets_tensor, labels_tensor


def retrieve_entities_and_pron_embeddings(bert_embeddings, entities_and_pron_offsets):
    embeddings_A = []
    embeddings_B = []
    embeddings_pron = []

    # Consider embeddings and offsets in each batch separately
    for embeddings, off in zip(bert_embeddings, entities_and_pron_offsets):
        # The offsets of mention A are the first and the second
        # in the 'off' tensor
        offsets_ent_A = range(off[0], off[1]+1)
        # The offsets of mention B are the third and the fourth
        # in the 'off' tensor
        offsets_ent_B = range(off[2], off[3]+1)
        # The offset of the pronoun is the last in the 'off' tensor
        offset_pron = off[-1]

        # The embedding of a mention is the mean of
        # all the subtokens embeddings that represent it
        embeddings_A.append(embeddings[offsets_ent_A].mean(dim=0))
        embeddings_B.append(embeddings[offsets_ent_B].mean(dim=0))
        embeddings_pron.append(embeddings[offset_pron])

    # Merge outputs
    merged_entities_and_pron_embeddings = torch.cat([
        torch.stack(embeddings_A, dim=0),
        torch.stack(embeddings_B, dim=0),
        torch.stack(embeddings_pron, dim=0)
    ], dim=1)
    # print(torch.stack(outputs_A, dim=0))
    # torch.stack(outputs_B, dim=0)
    # print(torch.stack(outputs_pron, dim=0))

    # shape: batch_size x (embedding_dim * 3)
    return merged_entities_and_pron_embeddings


class CorefHead(nn.Module):
    def __init__(self, bert_hidden_size: int):
        super().__init__()
        self.bert_hidden_size = bert_hidden_size
        self.head_hidden_size = 512

#         self.fc = nn.Sequential(
#             nn.Dropout(0.1),
#             nn.Linear(bert_hidden_size * 3, 512),
#             nn.ReLU(),
#             nn.Linear(512, 3)
#         )
        self.fc = nn.Sequential(
            #             nn.BatchNorm1d(bert_hidden_size * 3),
            #             nn.Dropout(0.5),
            #             nn.LeakyReLU(),
            #             nn.Linear(bert_hidden_size * 3, self.head_hidden_size),
            #             nn.BatchNorm1d(self.head_hidden_size),
            #             nn.Dropout(0.5),
            #             nn.Linear(self.head_hidden_size, self.head_hidden_size),
            #             nn.ReLU(),
            #             nn.BatchNorm1d(self.head_hidden_size),
            #             nn.Dropout(0.5),
            nn.Dropout(0.1),
            nn.Linear(bert_hidden_size * 3, self.head_hidden_size),
            nn.BatchNorm1d(self.head_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.head_hidden_size, 3)
        )

    def forward(self, bert_outputs, offsets):
        assert bert_outputs.shape[2] == self.bert_hidden_size
        embeddings = retrieve_entities_and_pron_embeddings(bert_outputs,
                                                           offsets)

        return self.fc(embeddings)


class GAPModel(nn.Module):
    """The main model."""

    def __init__(self, bert_model: str):
        super().__init__()

        if bert_model in ("bert-base-uncased", "bert-base-cased"):
            self.bert_hidden_size = 768
        elif bert_model in ("bert-large-uncased", "bert-large-cased"):
            self.bert_hidden_size = 1024
        else:
            raise ValueError("Unsupported BERT model.")

        self.bert = BertModel.from_pretrained(
            bert_model).to(device, non_blocking=True)
        self.head = CorefHead(self.bert_hidden_size).to(
            device, non_blocking=True)

    def forward(self, x, offsets):
        bert_outputs = self.bert(
            x, attention_mask=(x > 0).long(),
            token_type_ids=None, output_hidden_states=True)
#         concat_bert = torch.cat((bert_outputs[-1],bert_outputs[-2],bert_outputs[-3]),dim=-1)

        last_layer = bert_outputs.last_hidden_state
        head_outputs = self.head(last_layer, offsets)
#         return concat_bert
        return head_outputs

class Trainer:
    
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        criterion: torch.nn,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        
    ):
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        if args is None:
            output_dir = "../../model/tmp_trainer"
            print(f"No 'TrainingArguments' passed, using 'output_dir={output_dir}'.")
            args = TrainingArguments(output_dir=output_dir)
        
        self.args = args
        
    def train(self, grad_clipping):
        args = self.args
        train_dataloader = self.train_dataloader
        valid_dataloader = self.valid_dataloader
        
        train_losses = []
        train_acc_list = []
        valid_losses = []
        valid_acc_list = []
        
        epochs = args.num_train_epochs
        train_loss = 0.0
        train_acc, total_count = 0.0, 0.0
        
        scaler = GradScaler()
        self.model.train()
        for epoch in range(epochs):
            
            epoch_loss = 0.0
            
            for step, (features, offsets, labels) in enumerate(train_dataloader):
                # Empty gradients
                self.optimizer.zero_grad(set_to_none=True)
                
                # Forward
                predictions = self.model(features, offsets)
                
                
                
                loss = self.criterion(predictions, labels)
                train_acc += (predictions.argmax(1) == labels).sum().item()
                total_count += labels.shape[0]
                
#                 # Backward  
#                 loss.backward()
                # Backward pass without mixed precision
                # It's not recommended to use mixed precision for backward pass
                # Because we need more precise loss
                scaler.scale(loss).backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clipping)
                
                # Update weights 
#                 self.optimizer.step()
                scaler.step(self.optimizer)
                scaler.update()
        
                
                epoch_loss += loss.tolist()

                if step % args.logging_steps == args.logging_steps - 1:
                    mid_loss = epoch_loss / (step + 1)
                    mid_acc = train_acc / total_count
#                     print('\t[E: {:2d} @ step {}] current avg loss = {:0.4f}'.format(epoch, step, mid_loss))
                    print(f'\t| step {step+1:3d}/{len(train_dataloader):d} | train_loss: {mid_loss:.3f} | ' \
                    f'train_acc: {mid_acc:.3f} |')
            
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            train_loss += avg_epoch_loss
            train_losses.append(train_loss)
            train_acc_list.append(train_acc / total_count)
            
#             print('\t[E: {:2d}] train loss = {:0.4f}'.format(epoch, avg_epoch_loss))  # print train loss at the end of the epoch
            
    
            valid_loss, valid_acc = self.evaluate(valid_dataloader)
            valid_losses.append(valid_loss)
            valid_acc_list.append(valid_acc)
            
#             print('  [E: {:2d}] valid loss = {:0.4f}'.format(epoch, valid_loss))
            print('-' * 75)
            print(f'| epoch {epoch+1:3d}/{epochs:d} | train_loss: {avg_epoch_loss:.3f} | ' \
                    f'valid_loss: {valid_loss:.3f} | valid_acc: {valid_acc:.3f} |')
            print('-' * 75)
            
        avg_epoch_loss = train_loss / epochs
        histories = {
            "train_losses": train_losses,
            "train_acc": train_acc_list,
            "valid_losses": valid_losses,
            "valid_acc": valid_acc_list,

        }
#         print(histories)
        
        return #avg_epoch_loss, histories
            
    def evaluate(self, eval_dataloader):
        valid_loss = 0.0
        eval_acc, total_count = 0, 0
        
        self.model.eval()
        with torch.no_grad():
            for (features, offsets, labels) in eval_dataloader:
                
                predictions = self.model(features, offsets)
                loss = self.criterion(predictions, labels)
                valid_loss += loss.tolist()

                eval_acc += (predictions.argmax(1) == labels).sum().item()
                total_count += labels.shape[0]
        
        return valid_loss / len(eval_dataloader), eval_acc / total_count


yaml_file = "./train.yaml"
# Read configuration file with all the necessary parameters
with open(yaml_file) as file:
    config = yaml.safe_load(file)
    
training_args = TrainingArguments(**config['training_args'])

# Make sure that the learning rate is read as a number and not as a string
training_args.learning_rate = float(training_args.learning_rate)

model = GAPModel(model_name_or_path).to(device, non_blocking=True)

criterion = torch.nn.CrossEntropyLoss().to(device=device, non_blocking=True)
optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)

batch_size = 4

train_dataloader = DataLoader(train_ds, batch_size=batch_size, 
                              collate_fn=collate_batch, shuffle=True)
valid_dataloader = DataLoader(valid_ds, batch_size=batch_size, 
                              collate_fn=collate_batch, shuffle=False)

trainer = Trainer(model, training_args, 
                  train_dataloader, valid_dataloader, 
                  criterion, optimizer)


grad_clipping = 0.7

trainer.train(grad_clipping)