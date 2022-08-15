import os
from transformers import (
    Trainer,
    BertTokenizer,
    BertModel
)
import transformers
import numpy as np
import pandas as pd

import random
import time
import yaml
from tqdm import tqdm
from typing import *
from arguments import CustomTrainingArguments

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
#         self.original_offsets = []

        if labeled:
            self._extract_target()
            self.labels = df.target.values.astype("uint8")

        self._convert_tokens_to_ids()

    @staticmethod
    def get_class_id(is_coref_A: Union[str, bool], is_coref_B: Union[str, bool]) -> int:
        if is_coref_A == "TRUE" or is_coref_A is True:
            return 0
        elif is_coref_B == "TRUE" or is_coref_B is True:
            return 1
        else:
            return 2
        
    def _extract_target(self):
        self.df['target'] = [self.get_class_id(is_coref_A, is_coref_B) 
                             for is_coref_A, is_coref_B in zip(self.df['is_coref_A'],  
                                                               self.df['is_coref_B'])]
        
    def _convert_tokens_to_ids(self):
        CLS = [self.tokenizer.cls_token]
        SEP = [self.tokenizer.sep_token]

        for _, row in self.df.iterrows():
            tokens, offsets = self._tokenize(row)
#             self.original_offsets.append([row.offset_A, row.offset_B, row.p_offset])
            self.offsets.append(offsets)
            self.tokens.append(self.tokenizer.convert_tokens_to_ids(
                CLS + tokens + SEP))

    def _tokenize(self, row):
        # The order is important because we want the pronoun to come after all the
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

        # a) Always BN -> AC, (Nothing b/w them).
        # b) BN -> Dropout over Dropout -> BN, but try both. [Newer research, finds 1st better ]
        # c) BN eliminates the need of Dropout, no need to use Dropout.
        # e) BN before Dropout is data Leakage.
        # f) Best thing is to try every combination.
        # SO CALLED BEST METHOD -
        # Layer -> BN -> AC -> Dropout ->Layer
        
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
        embeddings = self._retrieve_entities_and_pron_embeddings(bert_outputs,
                                                           offsets)

        return self.fc(embeddings)
    
    def _retrieve_entities_and_pron_embeddings(self, bert_embeddings, entities_and_pron_offsets):
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


class GAPModel(nn.Module):
    """The main model."""

    def __init__(self, bert_model: str):
        super().__init__()

        if bert_model in {"bert-base-uncased", "bert-base-cased"}:
            self.bert_hidden_size = 768
        elif bert_model in {"bert-large-uncased", "bert-large-cased"}:
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
        args: CustomTrainingArguments,
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
            args = CustomTrainingArguments(output_dir=output_dir)
        
        self.args = args
        
    def train(self):
        args = self.args
        valid_dataloader = self.valid_dataloader
        epochs = args.num_train_epochs
        
        train_losses = []
        train_acc_list = []
        valid_losses = []
        valid_acc_list = []
        
        if args.early_stopping:
            patience_counter = 0 

        scaler = GradScaler()

        if args.resume_from_checkpoint is not None:
            self._resume_model(args.resume_from_checkpoint, scaler)

        training_start_time = time.time()
        print("\nTraining...")
        for epoch in range(epochs):
            train_loss, train_acc = self._inner_training_loop(scaler)
            train_losses.append(train_loss)
            train_acc_list.append(train_acc)

            valid_loss, valid_acc = self.evaluate(valid_dataloader)
            valid_losses.append(valid_loss)
            valid_acc_list.append(valid_acc)

            if self.scheduler is not None:
                print('-' * 14)
                print(f"  LR: {self.scheduler.get_last_lr()[0]:.2e}")
                self.scheduler.step()

            self._print_epoch_log(epoch, epochs, train_loss, valid_loss, valid_acc)

            if args.early_stopping and len(valid_acc_list) >= 2:
                # stop = args.early_stopping_mode == 'min' and epoch > 0 and valid_acc_list[-1] > valid_acc_list[-2]
                stop = args.early_stopping_mode == 'max' and epoch > 0 and valid_acc_list[-1] < valid_acc_list[-2]
                if stop:
                    if patience_counter >= args.early_stopping_patience:
                        print('Early stop.')
                        break
                    else:
                        print('-- Patience.\n')
                        patience_counter += 1
        
        training_time = time.time() - training_start_time
        print(f'Training time: {training_time:.2f}s')

        metrics_history = {
            "train_losses": train_losses,
            "train_acc": train_acc_list,
            "valid_losses": valid_losses,
            "valid_acc": valid_acc_list,
        }
        print(metrics_history)
        if args.save_model:
            self._save_model(epoch, valid_acc, scaler)
    
        return #metrics_history

    def _inner_training_loop(self, scaler):
        args = self.args
        train_dataloader = self.train_dataloader
        
        train_loss = 0.0
        train_acc, total_count = 0.0, 0.0

        self.model.train()
        for step, (features, offsets, labels) in enumerate(train_dataloader):
            # Empty gradients
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward
            predictions = self.model(features, offsets)
            loss = self.criterion(predictions, labels)
            # with torch.cuda.amp.autocast(): # autocast as a context manager
            #     predictions = self.model(features, offsets)
            #     loss = self.criterion(predictions, labels)

            train_acc += (predictions.argmax(1) == labels).sum().item()
            total_count += labels.shape[0]
            
#                 # Backward  
            # loss.backward()
            # Backward pass without mixed precision
            # It's not recommended to use mixed precision for backward pass
            # Because we need more precise loss
            scaler.scale(loss).backward()
            
            if args.grad_clipping is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clipping)
            
            # Update weights 
            # self.optimizer.step()
            scaler.step(self.optimizer)
            scaler.update()

            train_loss += loss.item()

            if step % args.logging_steps == args.logging_steps - 1:
                running_loss = train_loss / (step + 1)
                running_acc = train_acc / total_count
                self._print_step_log(step, running_loss, running_acc)
                
        return train_loss / len(train_dataloader), train_acc / total_count


    def evaluate(self, eval_dataloader):
        valid_loss = 0.0
        eval_acc, total_count = 0, 0
        
        self.model.eval()
        with torch.no_grad():
            for (features, offsets, labels) in eval_dataloader:
                
                predictions = self.model(features, offsets)
                loss = self.criterion(predictions, labels)
                valid_loss += loss.item()

                eval_acc += (predictions.argmax(1) == labels).sum().item()
                total_count += labels.shape[0]
        
        return valid_loss / len(eval_dataloader), eval_acc / total_count


    def _print_step_log(self, step, running_loss, running_acc):
        print(f'\t| step {step+1:3d}/{len(self.train_dataloader):d} | train_loss: {running_loss:.3f} | ' \
                f'train_acc: {running_acc:.3f} |')

    def _print_epoch_log(self, epoch, epochs, train_loss, valid_loss, valid_acc):
        print('-' * 74)
        print(f'| epoch {epoch+1:3d}/{epochs:d} | train_loss: {train_loss:.3f} | ' \
                f'valid_loss: {valid_loss:.3f} | valid_acc: {valid_acc:.3f} |')
        print('-' * 74)
        
    
    def _save_model(self, epoch, valid_acc, scaler):
        print("Saving model...")
        torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            }, f"{self.args.output_dir}my_model_{str(valid_acc)[2:5]}_{epoch+1}.pth")
        print("Model saved.")

    def _resume_model(self, path, scaler):
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])




yaml_file = "./train.yaml"
# Read configuration file with all the necessary parameters
with open(yaml_file) as file:
    config = yaml.safe_load(file)
    
training_args = CustomTrainingArguments(**config['training_args'])

# Make sure that the learning rate is read as a number and not as a string
training_args.learning_rate = float(training_args.learning_rate)

model = GAPModel(model_name_or_path).to(device, non_blocking=True)

criterion = torch.nn.CrossEntropyLoss().to(device=device, non_blocking=True)
optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

batch_size = 4

train_dataloader = DataLoader(train_ds, batch_size=batch_size, 
                            collate_fn=collate_batch, shuffle=True)
                            
valid_dataloader = DataLoader(valid_ds, batch_size=batch_size, 
                            collate_fn=collate_batch, shuffle=False)

trainer = Trainer(model, training_args, 
                train_dataloader, valid_dataloader, 
                criterion, optimizer, scheduler)


trainer.train()